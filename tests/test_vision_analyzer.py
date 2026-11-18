"""Tests for ``src/vision_analyzer.py``.

The analyzer must work without OpenCV: frame decoding, dHash computation and
frame extraction are all delegated to FFmpeg. Tests either use pure helpers or
mock FFmpeg/Ollama, with one small synthetic-video integration test that is
skipped when FFmpeg/ffprobe are unavailable.
"""
import base64
import json
import logging
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from src import vision_analyzer as va
from src.models import AnalysisError
from src.utils import ToolNotFound, file_fingerprint, find_tool


def _resolve_tool(name: str) -> str:
    try:
        return str(Path(find_tool(name)).resolve())
    except ToolNotFound:
        return ""


FFMPEG = _resolve_tool("ffmpeg")
FFPROBE = _resolve_tool("ffprobe")
HAS_FFMPEG = bool(FFMPEG and FFPROBE)
requires_ffmpeg = pytest.mark.skipif(
    not HAS_FFMPEG, reason="ffmpeg/ffprobe are not available")


# --------------------------------------------------------------------- fixtures
@pytest.fixture
def synthetic_video(tmp_path):
    """A 2 s clip with a hard cut at 1.0 s, so scene detection has a target."""
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg/ffprobe are not available")
    path = tmp_path / "synthetic.mp4"
    command = [
        FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=10:duration=1",
        "-f", "lavfi", "-i", "smptebars=size=160x120:rate=10:duration=1",
        "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0",
        "-c:v", "mpeg4", "-q:v", "3", "-pix_fmt", "yuv420p", str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        pytest.skip(f"cannot generate synthetic video: {completed.stderr[-200:]}")
    return path


def make_cfg(**overrides):
    values = {
        "analyze_vision": True,
        "frame_interval_sec": 0.5,
        "min_frames": 2,
        "max_frames": 6,
        "scene_threshold": 0.3,
        "vision_batch_size": 1,
        "vision_frame_width": 160,
        "ollama_url": "http://ollama.test:11434",
        "ollama_vision_model": "llava:test",
        "ollama_text_model": "llama3.2:test",
        "ollama_timeout": 5.0,
        "ollama_keep_alive": "2m",
        "ollama_num_ctx": 2048,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def make_frame_files(tmp_path, timestamps):
    """Create dummy JPEG files outside ``work/frames`` and map path -> time."""
    frames_dir = tmp_path / "fake_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_times = {}
    for index, timestamp in enumerate(timestamps):
        path = frames_dir / f"frame_{index}.jpg"
        path.write_bytes(f"fake-jpeg-bytes-{index}".encode("ascii"))
        frame_times[path] = float(timestamp)
    return frame_times


def patch_pipeline(monkeypatch, frame_times, *, duration=9.0, scenes=(),
                   hashes=None, ledger=None):
    """Replace the FFmpeg-facing helpers with fakes; return a call ledger."""
    counts = ledger if ledger is not None else {
        "probe": 0, "scenes": 0, "extract": 0, "dhash": 0}
    # Distant hashes so dedupe keeps every fake frame by default.
    distinct = ("0000000000000000", "ffffffffffffffff",
                "0f0f0f0f0f0f0f0f", "f0f0f0f0f0f0f0f0")
    default_hashes = {path: distinct[index % len(distinct)]
                      for index, path in enumerate(frame_times)}

    def probe(_source):
        counts["probe"] += 1
        return duration

    def detect(_source, _threshold, _duration=0.0):
        counts["scenes"] += 1
        return list(scenes)

    def extract(_source, _frames_dir, _plan, _width):
        counts["extract"] += 1
        return dict(frame_times)

    def dhash(path):
        counts["dhash"] += 1
        return (hashes or default_hashes)[path]

    monkeypatch.setattr(va, "probe_duration", probe)
    monkeypatch.setattr(va, "detect_scene_changes", detect)
    monkeypatch.setattr(va, "extract_frames", extract)
    monkeypatch.setattr(va, "frame_dhash", dhash)
    return counts


class FakeResponse:
    def __init__(self, response="", status_code=200, payload=None):
        self.status_code = status_code
        if payload is not None:
            self._payload = dict(payload)
        else:
            self._payload = {"response": response}
        self.text = json.dumps(self._payload)

    def json(self):
        return self._payload


def patch_post(monkeypatch, handler):
    """Replace ``requests.post`` with ``handler(body, call_number)``."""
    calls = []

    def fake_post(url, json=None, timeout=None):
        calls.append({"url": url, "body": json, "timeout": timeout})
        return handler(json or {}, len(calls))

    monkeypatch.setattr(va.requests, "post", fake_post)
    return calls


def simple_ollama(cfg, *, vision="A player runs across the map.",
                  summary="The player runs, then the round ends."):
    def handler(body, _attempt):
        if body.get("model") == cfg.ollama_text_model:
            return FakeResponse(summary)
        images = body.get("images") or []
        if len(images) > 1:
            return FakeResponse("\n".join(
                f"{index}. Frame {index} description."
                for index in range(1, len(images) + 1)))
        return FakeResponse(vision)

    return handler


# --------------------------------------------------------------- sampling plans
def test_plan_grid_covers_whole_long_clip():
    long_times = va.plan_grid_times(600.0, 1.5, 4, 20)
    assert len(long_times) == 20
    assert long_times == sorted(long_times)
    assert len(set(long_times)) == len(long_times)
    assert long_times[0] > 0.0
    assert long_times[-1] > 600.0 - 600.0 / 20.0  # last sample near the end
    assert long_times[-1] >= 0.9 * 600.0

    # A 90 s clip is capped by max_frames yet still reaches the end.
    times_90 = va.plan_grid_times(90.0, 1.5, 4, 20)
    assert len(times_90) == 20
    assert times_90[-1] > 0.9 * 90.0

    # Short clips are densified up to min_frames.
    short = va.plan_grid_times(3.0, 1.5, 4, 20)
    assert len(short) == 4
    assert short[-1] > 1.5

    # Moderate clips keep roughly the requested interval.
    assert len(va.plan_grid_times(20.0, 1.5, 4, 20)) == 13
    # Degenerate durations produce no frames.
    assert va.plan_grid_times(0.0, 1.5, 4, 20) == []


def test_merge_plan_keeps_scene_frames_and_caps_total():
    grid = va.plan_grid_times(20.0, 1.5, 4, 8)
    plan = va.merge_sampling_plan(grid, [4.9, 4.95, 17.2], 20.0, 8, 4)
    assert len(plan) <= 8
    assert plan == sorted(plan, key=lambda point: point.timestamp)
    assert len({point.timestamp for point in plan}) == len(plan)
    # 4.9 and 17.2 are kept as scene changes; 4.95 merges into 4.9.
    scene_times = [point.timestamp for point in plan if point.scene]
    assert scene_times == pytest.approx([4.9, 17.2])
    assert any(abs(point.timestamp - 4.9) < 0.01 for point in plan)


def test_split_batch_response_handles_numbered_lines():
    assert va._split_batch_response("1. One\n2. Two", 2) == ["One", "Two"]
    assert va._split_batch_response(
        "Frame 1 - First\nFrame 2 - Second", 2) == ["First", "Second"]
    assert va._split_batch_response("one rambling sentence", 2) is None


# -------------------------------------------------------------- deduplication
def test_dedupe_preserves_original_timestamps():
    frames = {Path("a.jpg"): 1.1, Path("b.jpg"): 1.6, Path("c.jpg"): 7.3}
    hashes = {
        Path("a.jpg"): "0000000000000000",
        Path("b.jpg"): "0000000000000000",  # visually identical to a
        Path("c.jpg"): "ffffffffffffffff",
    }
    kept = va.dedupe_frames(frames, hashes=hashes, max_hamming=5)
    assert [path.name for path, _timestamp in kept] == ["a.jpg", "c.jpg"]
    # 7.3 comes from the stored mapping, not from index * interval.
    assert dict(kept)[Path("c.jpg")] == 7.3


def test_dedupe_always_keeps_scene_change_frames():
    frames = {Path("a.jpg"): 1.0, Path("b.jpg"): 2.0}
    hashes = {Path("a.jpg"): "0" * 16, Path("b.jpg"): "0" * 16}
    kept = va.dedupe_frames(
        frames, hashes=hashes, always_keep=[Path("b.jpg")])
    assert [path.name for path, _timestamp in kept] == ["a.jpg", "b.jpg"]


def test_dhash_flags_tiny_differences_as_near_duplicates():
    base = bytes(range(72))
    tweaked = bytearray(base)
    tweaked[4] += 3
    alternating = bytes([0, 255] * 36)

    base_hash = va.dhash_from_gray_bytes(base)
    assert len(base_hash) == 16
    assert va.dhash_hamming(base_hash,
                            va.dhash_from_gray_bytes(bytes(range(72)))) == 0
    tiny = va.dhash_hamming(base_hash,
                            va.dhash_from_gray_bytes(bytes(tweaked)))
    assert 0 < tiny <= 4
    assert (va.dhash_hamming(base_hash,
                             va.dhash_from_gray_bytes(alternating))
            > va.DEFAULT_DEDUPE_HAMMING)


def test_scene_parser_reads_showinfo_pts_time():
    output = (
        "[Parsed_showinfo_2 @ 000002a0] config in time_base: 1/1000, "
        "frame_rate: 10/1\n"
        "[Parsed_showinfo_2 @ 000002a0] n:   0 pts:    100 pts_time:0.1   "
        "duration:1\n"
        "[Parsed_showinfo_2 @ 000002a0] n:   1 pts:   1000 pts_time:1.0   "
        "duration:1\n"
        "[Parsed_showinfo_2 @ 000002a0] n:   2 pts:   1000 pts_time:1.000 "
        "duration:1\n"
        "[Parsed_showinfo_2 @ 000002a0] n:   3 pts:   2345 pts_time:2.345 "
        "duration:1\n"
    )
    assert va.parse_showinfo_pts(output) == [0.1, 1.0, 2.345]
    assert va.parse_showinfo_pts("") == []
    assert va.parse_showinfo_pts("no timestamps here") == []


# ----------------------------------------------------------------------- cache
def test_cache_roundtrip_skips_ffmpeg_and_ollama(monkeypatch, tmp_path):
    cfg = make_cfg()
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"fake video bytes")
    work = tmp_path / "work"
    frame_times = make_frame_files(tmp_path, [1.25, 4.5])
    ledger = patch_pipeline(monkeypatch, frame_times, duration=9.0,
                            scenes=[3.0])
    calls = patch_post(monkeypatch, simple_ollama(cfg))

    first = va.analyze_vision(source, work, cfg)

    assert first.duration == pytest.approx(9.0)
    assert [(frame.timestamp, frame.description) for frame in first.frames] == [
        (1.25, "A player runs across the map."),
        (4.5, "A player runs across the map."),
    ]
    assert [frame.path for frame in first.frames] == [str(path)
                                                      for path in frame_times]
    assert first.sequence_summary == "The player runs, then the round ends."
    assert first.model == cfg.ollama_vision_model
    assert first.source_fingerprint == file_fingerprint(source)
    assert first.scene_changes == [3.0]
    assert ledger == {"probe": 1, "scenes": 1, "extract": 1, "dhash": 2}
    assert len(calls) == 3  # two frame requests + one summary request

    cache = json.loads((work / "vision.json").read_text(encoding="utf-8"))
    assert cache["version"] == va.CACHE_VERSION
    assert cache["model"] == cfg.ollama_vision_model
    assert cache["text_model"] == cfg.ollama_text_model
    assert cache["source_fingerprint"] == first.source_fingerprint
    assert cache["duration"] == pytest.approx(9.0)
    assert [item["path"] for item in cache["frames"]] == [
        str(path) for path in frame_times]
    assert cache["scene_changes"] == [3.0]
    assert cache["sequence_summary"] == first.sequence_summary

    def boom(*_args, **_kwargs):
        raise AssertionError("FFmpeg/Ollama must not run on a cache hit")

    monkeypatch.setattr(va, "probe_duration", boom)
    monkeypatch.setattr(va, "detect_scene_changes", boom)
    monkeypatch.setattr(va, "extract_frames", boom)
    monkeypatch.setattr(va, "frame_dhash", boom)
    monkeypatch.setattr(va.requests, "post", boom)

    second = va.analyze_vision(source, work, cfg)
    assert second == first


def test_stale_fingerprint_recomputes_and_rewrites_cache(monkeypatch, tmp_path):
    cfg = make_cfg()
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"video one")
    work = tmp_path / "work"
    frame_times = make_frame_files(tmp_path, [1.0, 3.0])
    patch_pipeline(monkeypatch, frame_times)
    patch_post(monkeypatch, simple_ollama(cfg))
    va.analyze_vision(source, work, cfg)

    cache_path = work / "vision.json"
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    cache["source_fingerprint"] = "0:0"
    cache_path.write_text(json.dumps(cache), encoding="utf-8")

    ledger = {"probe": 0, "scenes": 0, "extract": 0, "dhash": 0}
    patch_pipeline(monkeypatch, frame_times, ledger=ledger)
    calls = patch_post(monkeypatch, simple_ollama(cfg))
    refreshed = va.analyze_vision(source, work, cfg)

    assert ledger["extract"] == 1
    assert len(calls) == 3
    assert refreshed.source_fingerprint == file_fingerprint(source)
    new_cache = json.loads(cache_path.read_text(encoding="utf-8"))
    assert new_cache["source_fingerprint"] == refreshed.source_fingerprint


def test_stale_model_recomputes_and_rewrites_cache(monkeypatch, tmp_path):
    cfg = make_cfg()
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"video two")
    work = tmp_path / "work"
    frame_times = make_frame_files(tmp_path, [1.0, 3.0])
    patch_pipeline(monkeypatch, frame_times)
    patch_post(monkeypatch, simple_ollama(cfg))
    va.analyze_vision(source, work, cfg)

    other = make_cfg(ollama_vision_model="llava:other")
    ledger = {"probe": 0, "scenes": 0, "extract": 0, "dhash": 0}
    patch_pipeline(monkeypatch, frame_times, ledger=ledger)
    calls = patch_post(monkeypatch, simple_ollama(other))
    rebuilt = va.analyze_vision(source, work, other)

    assert ledger["extract"] == 1
    assert len(calls) == 3
    assert rebuilt.model == "llava:other"
    cache = json.loads((work / "vision.json").read_text(encoding="utf-8"))
    assert cache["model"] == "llava:other"


def test_corrupt_cache_is_recomputed(monkeypatch, tmp_path):
    cfg = make_cfg()
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"video three")
    work = tmp_path / "work"
    work.mkdir(parents=True, exist_ok=True)
    (work / "vision.json").write_text("{not valid json", encoding="utf-8")

    frame_times = make_frame_files(tmp_path, [1.0])
    ledger = {"probe": 0, "scenes": 0, "extract": 0, "dhash": 0}
    patch_pipeline(monkeypatch, frame_times, ledger=ledger)
    patch_post(monkeypatch, simple_ollama(cfg))
    result = va.analyze_vision(source, work, cfg)

    assert ledger["extract"] == 1
    assert len(result.frames) == 1
    cache = json.loads((work / "vision.json").read_text(encoding="utf-8"))
    assert cache["version"] == va.CACHE_VERSION


# ------------------------------------------------------------- Ollama requests
def test_ollama_request_shape_and_sequence_summary(monkeypatch, tmp_path):
    cfg = make_cfg()
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"payload")
    frame_times = make_frame_files(tmp_path, [1.0, 4.5])
    patch_pipeline(monkeypatch, frame_times)
    calls = patch_post(monkeypatch, simple_ollama(
        cfg, vision="Action at the objective.",
        summary="The team pushes the objective and wins the round."))

    result = va.analyze_vision(source, tmp_path / "work", cfg)

    vision_calls = [call for call in calls
                    if call["body"].get("model") == cfg.ollama_vision_model]
    summary_calls = [call for call in calls
                     if call["body"].get("model") == cfg.ollama_text_model]
    assert len(vision_calls) == 2
    assert len(summary_calls) == 1
    for call in vision_calls:
        body = call["body"]
        assert call["url"].endswith("/api/generate")
        assert body["stream"] is False
        assert body["keep_alive"] == cfg.ollama_keep_alive
        assert body["options"]["temperature"] == 0.2
        assert body["options"]["num_ctx"] == cfg.ollama_num_ctx
        assert len(body["images"]) == 1
        assert base64.b64decode(body["images"][0]).startswith(b"fake-jpeg-bytes")
        assert "1-2 sentences" in body["prompt"]

    summary_body = summary_calls[0]["body"]
    assert "images" not in summary_body
    assert "[1.00s]" in summary_body["prompt"]
    assert "[4.50s]" in summary_body["prompt"]
    assert "narrative" in summary_body["prompt"]
    assert result.sequence_summary == (
        "The team pushes the objective and wins the round.")


def test_batch_vision_sends_multiple_images_and_maps_lines(monkeypatch,
                                                           tmp_path):
    cfg = make_cfg(vision_batch_size=3)
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"payload")
    frame_times = make_frame_files(tmp_path, [1.0, 2.0])
    patch_pipeline(monkeypatch, frame_times)

    def handler(body, _attempt):
        if body.get("model") == cfg.ollama_text_model:
            return FakeResponse("Narrative.")
        return FakeResponse("1. First beat.\n2. Second beat.")

    calls = patch_post(monkeypatch, handler)
    result = va.analyze_vision(source, tmp_path / "work", cfg)

    vision_calls = [call for call in calls
                    if call["body"].get("model") == cfg.ollama_vision_model]
    assert len(vision_calls) == 1
    assert len(vision_calls[0]["body"]["images"]) == 2
    assert [frame.description for frame in result.frames] == [
        "First beat.", "Second beat."]
    assert [frame.timestamp for frame in result.frames] == [1.0, 2.0]


def test_batch_vision_falls_back_to_single_requests(monkeypatch, tmp_path):
    cfg = make_cfg(vision_batch_size=2)
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"payload")
    frame_times = make_frame_files(tmp_path, [1.0, 2.0])
    patch_pipeline(monkeypatch, frame_times)

    def handler(body, attempt):
        if body.get("model") == cfg.ollama_text_model:
            return FakeResponse("Narrative.")
        if len(body.get("images") or []) > 1:
            return FakeResponse("one rambling sentence")
        return FakeResponse(f"Single frame {attempt}.")

    calls = patch_post(monkeypatch, handler)
    result = va.analyze_vision(source, tmp_path / "work", cfg)

    vision_calls = [call for call in calls
                    if call["body"].get("model") == cfg.ollama_vision_model]
    assert len(vision_calls) == 3  # one rejected batch + two single requests
    assert len(result.frames) == 2
    assert all(frame.description.startswith("Single frame")
               for frame in result.frames)


def test_ollama_retries_transient_errors(monkeypatch):
    cfg = make_cfg()
    monkeypatch.setattr(va.time, "sleep", lambda _seconds: None)

    def handler(_body, attempt):
        if attempt <= 2:
            return FakeResponse("", status_code=500)
        return FakeResponse("Recovered description.")

    calls = patch_post(monkeypatch, handler)
    text = va.ollama_generate(cfg, cfg.ollama_vision_model, "prompt")
    assert text == "Recovered description."
    assert len(calls) == 3


def test_ollama_does_not_retry_client_errors(monkeypatch):
    cfg = make_cfg()
    calls = patch_post(
        monkeypatch,
        lambda _body, _attempt: FakeResponse("bad request", status_code=400))
    with pytest.raises(AnalysisError, match="HTTP 400"):
        va.ollama_generate(cfg, cfg.ollama_vision_model, "prompt")
    assert len(calls) == 1


def test_all_frames_failing_raises_analysis_error(monkeypatch, tmp_path):
    cfg = make_cfg()
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"payload")
    frame_times = make_frame_files(tmp_path, [1.0, 3.0])
    patch_pipeline(monkeypatch, frame_times)
    monkeypatch.setattr(va.time, "sleep", lambda _seconds: None)

    def refuse(_body, _attempt):
        raise requests.ConnectionError("connection refused")

    calls = patch_post(monkeypatch, refuse)
    with pytest.raises(AnalysisError, match="could not describe any"):
        va.analyze_vision(source, tmp_path / "work", cfg)
    assert len(calls) == 6  # 2 frames x (1 attempt + 2 retries)
    assert not (tmp_path / "work" / "vision.json").exists()


def test_summary_failure_keeps_frames_without_inventing(monkeypatch, tmp_path,
                                                        caplog):
    cfg = make_cfg()
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"payload")
    frame_times = make_frame_files(tmp_path, [1.0, 3.0])
    patch_pipeline(monkeypatch, frame_times)
    monkeypatch.setattr(va.time, "sleep", lambda _seconds: None)

    def handler(body, _attempt):
        if body.get("model") == cfg.ollama_text_model:
            raise requests.ConnectionError("summary model down")
        return FakeResponse("A player is visible.")

    patch_post(monkeypatch, handler)
    with caplog.at_level(logging.ERROR):
        result = va.analyze_vision(source, tmp_path / "work", cfg)

    assert len(result.frames) == 2
    assert result.sequence_summary == ""
    cache = json.loads(
        (tmp_path / "work" / "vision.json").read_text(encoding="utf-8"))
    assert cache["sequence_summary"] == ""
    assert "summary" in caplog.text.lower()


# --------------------------------------------------------------- failure paths
def test_no_extracted_frames_raises_analysis_error(monkeypatch, tmp_path):
    cfg = make_cfg()
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"payload")
    patch_pipeline(monkeypatch, {})

    with pytest.raises(AnalysisError, match="could not extract any frame"):
        va.analyze_vision(source, tmp_path / "work", cfg)


def test_scene_detection_failure_returns_empty_list(tmp_path):
    missing = tmp_path / "missing.mp4"
    # FFmpeg failure must not raise: scene detection is an optional enhancement.
    assert va.detect_scene_changes(missing, 0.3, 2.0) == []
    # An unusable threshold disables detection instead of selecting everything.
    assert va.detect_scene_changes(missing, 0.0, 2.0) == []


# --------------------------------------------------------- real FFmpeg integration
@requires_ffmpeg
def test_real_clip_extraction_dedupe_and_timestamps(monkeypatch, tmp_path,
                                                    synthetic_video):
    cfg = make_cfg(frame_interval_sec=0.5, min_frames=2, max_frames=5,
                   scene_threshold=0.3, vision_frame_width=160)
    work = tmp_path / "work"
    patch_post(monkeypatch, simple_ollama(cfg))

    result = va.analyze_vision(synthetic_video, work, cfg)

    assert result.duration == pytest.approx(2.0, abs=0.15)
    assert result.frames
    assert 0 < len(result.frames) <= cfg.max_frames

    # Extraction covered the whole clip, including near the end.
    extracted = list((work / "frames").glob("*.jpg"))
    assert len(extracted) >= 2
    extracted_times = sorted(
        int(path.stem.rsplit("_", 1)[1]) / 1000.0 for path in extracted)
    assert extracted_times[-1] > result.duration * 0.7

    # Each reported timestamp is the frame's real timestamp, not an index guess.
    for frame in result.frames:
        path = Path(frame.path)
        assert path.is_file() and path.stat().st_size > 0
        file_timestamp = int(path.stem.rsplit("_", 1)[1]) / 1000.0
        assert frame.timestamp == pytest.approx(file_timestamp)
        assert 0.0 <= frame.timestamp <= result.duration

    # Dedupe drops the near-identical tail frame but keeps the scene change.
    assert len(result.frames) < len(extracted)
    assert any("scene_" in Path(frame.path).name for frame in result.frames)
    assert result.scene_changes
    assert any(abs(timestamp - 1.0) < 0.35 for timestamp in result.scene_changes)
    assert result.sequence_summary
    assert (work / "vision.json").is_file()


@requires_ffmpeg
def test_real_dhash_dedupes_tiny_different_jpegs(tmp_path, synthetic_video):
    q3 = tmp_path / "q3.jpg"
    q31 = tmp_path / "q31.jpg"
    for quality, target in ((3, q3), (31, q31)):
        completed = subprocess.run(
            [FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
             "-ss", "0.4", "-i", str(synthetic_video),
             "-frames:v", "1", "-an", "-vf", "scale=160:-2",
             "-q:v", str(quality), str(target)],
            capture_output=True, text=True)
        assert completed.returncode == 0, completed.stderr[-300:]

    q3_hash = va.frame_dhash(q3)
    q31_hash = va.frame_dhash(q31)
    assert len(q3_hash) == 16
    assert va.dhash_hamming(q3_hash, q31_hash) <= va.DEFAULT_DEDUPE_HAMMING

    kept = va.dedupe_frames({q3: 0.4, q31: 0.4}, {q3: q3_hash, q31: q31_hash})
    assert [path.name for path, _timestamp in kept] == ["q3.jpg"]


# ----------------------------------------------------------------------- no cv2
def test_module_does_not_import_opencv():
    source = Path(va.__file__).read_text(encoding="utf-8")
    assert "import cv2" not in source
    assert "cv2." not in source
