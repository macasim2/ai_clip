import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.config import Config
from src.database import Database
from src.editor import EditorError
from src.models import (AudioAnalysisResult, CuratorDecision, DownloadResult,
                        ProviderHealth, TranscriptSegment, TranscriptWord,
                        TwitchClip, VisionAnalysisResult)
from src.pipeline import Pipeline


@pytest.fixture
def sample_video(tmp_path):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg is not available")
    path = tmp_path / "sample.mp4"
    result = subprocess.run(
        [
            shutil.which("ffmpeg"), "-y", "-f", "lavfi",
            "-i", "testsrc=size=320x240:rate=10", "-f", "lavfi",
            "-i", "sine=frequency=440:sample_rate=16000", "-t", "2",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            "-shortest", str(path),
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        pytest.skip("cannot generate sample video")
    return path


def make_cfg(tmp_path):
    cfg = Config(
        raw_clips=tmp_path / "raw",
        processed=tmp_path / "processed",
        exports=tmp_path / "exports",
        logs=tmp_path / "logs",
        data=tmp_path / "data",
        db_path=tmp_path / "data" / "clips.db",
        db_backups=tmp_path / "data" / "backups",
        min_clip_duration=2.0,
        max_clip_duration=10.0,
        subtitles_enabled=False,
        analyze_vision=True,
        max_parallel=1,
    ).validate()
    cfg.ensure_dirs()
    return cfg


@pytest.fixture
def db(tmp_path):
    cfg = make_cfg(tmp_path)
    return cfg, Database(cfg.db_path)


def make_clip(clip_id="c1", game="Test Game"):
    return TwitchClip(
        id=clip_id, title="title", game_name=game, broadcaster_name="b",
        url="https://example/clip", created_at="", duration=2.0, view_count=1,
    )


def make_audio(duration=2.0):
    words = [TranscriptWord(0.2, 0.6, "wow"), TranscriptWord(0.7, 1.0, "yes!")]
    return AudioAnalysisResult(
        language="ru", duration=duration,
        segments=[TranscriptSegment(0.2, 1.0, "wow yes!")],
        words=words,
        events=[],
        full_text="wow yes!",
    )


def make_vision(duration=2.0):
    return VisionAnalysisResult(duration, [], "player wins")


def make_decision(score=85, start=0.2, end=1.2, decision="candidate"):
    return CuratorDecision(
        category="Экшен", event_description="win",
        unexpectedness=0.9, emotionality=0.8, payoff_strength=0.9,
        event_clarity=0.9, visual_interest=0.8, hook_strength=0.8,
        context_independence=0.8, overall_score=score, confidence=0.9,
        decision=decision, best_moment_start=start, best_moment_end=end,
        reason="test",
    )


def patch_success(monkeypatch, sample_video, decision=None, render=None):
    calls = {"download": 0, "audio": 0, "vision": 0, "curator": 0, "render": 0}

    def fake_download(clip, raw_dir, attempts=3, min_free_mb=1024):
        calls["download"] += 1
        return DownloadResult(path=sample_video, duration=2.0,
                              size=sample_video.stat().st_size)

    def fake_audio(source, work, cfg):
        calls["audio"] += 1
        return make_audio()

    def fake_vision(source, work, cfg):
        calls["vision"] += 1
        return make_vision()

    def fake_curator(audio, vision, duration, cfg, **kwargs):
        calls["curator"] += 1
        return decision or make_decision()

    def fake_render(clip_id, source, out_path, start, end, cfg, subtitles=None):
        calls["render"] += 1
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"x" * 4096)
        return out_path

    monkeypatch.setattr("src.pipeline.download", fake_download)
    monkeypatch.setattr("src.pipeline.analyze_audio", fake_audio)
    monkeypatch.setattr("src.pipeline.analyze_vision", fake_vision)
    monkeypatch.setattr("src.pipeline.analyze_clip", fake_curator)
    monkeypatch.setattr("src.pipeline.render_vertical", render or fake_render)
    return calls


def test_discovered_to_ready(db, sample_video, monkeypatch):
    cfg, database = db
    database.upsert_clip(make_clip())
    calls = patch_success(monkeypatch, sample_video)
    pipeline = Pipeline(cfg, database, twitch=object())
    result = pipeline.process_one("c1")
    assert result["ok"] is True
    assert result["decision"] == "candidate"
    row = database.get_clip("c1")
    assert row["status"] == "READY"
    assert row["export_path"] and Path(row["export_path"]).is_file()
    assert calls == {"download": 1, "audio": 1, "vision": 1,
                     "curator": 1, "render": 1}


def test_resume_from_analyzed_reuses_analysis(db, sample_video, monkeypatch):
    cfg, database = db
    database.upsert_clip(make_clip())
    database.update_status("c1", "DOWNLOADING")
    database.set_local_path("c1", sample_video)
    database.update_status("c1", "DOWNLOADED")
    database.update_status("c1", "ANALYZING")
    database.save_analysis(
        "c1",
        json.dumps({"language": "ru", "duration": 2.0, "full_text": "wow",
                    "events": [], "segments": [],
                    "words": [{"start": 0.2, "end": 0.6, "word": "wow"}]}),
        json.dumps({"duration": 2.0, "frames": [], "sequence_summary": "win"}),
    )
    database.update_status("c1", "ANALYZED")

    calls = patch_success(monkeypatch, sample_video)
    # Analysis must not run again; only curator + render.
    monkeypatch.setattr("src.pipeline.analyze_audio",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr("src.pipeline.analyze_vision",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    pipeline = Pipeline(cfg, database, twitch=object())
    result = pipeline.process_one("c1")
    assert result["ok"] is True
    assert calls["audio"] == 0 and calls["vision"] == 0
    assert calls["curator"] == 1
    assert database.get_clip("c1")["status"] == "READY"


def test_resume_from_approved_reuses_decision(db, sample_video, monkeypatch):
    cfg, database = db
    database.upsert_clip(make_clip())
    database.update_status("c1", "DOWNLOADING")
    database.set_local_path("c1", sample_video)
    database.update_status("c1", "DOWNLOADED")
    database.update_status("c1", "ANALYZING")
    database.save_analysis(
        "c1",
        json.dumps({"language": "ru", "duration": 2.0, "full_text": "",
                    "events": [], "segments": [],
                    "words": [{"start": 0.2, "end": 0.6, "word": "wow"}]}),
        json.dumps({"duration": 2.0, "frames": [], "sequence_summary": ""}),
    )
    database.update_status("c1", "ANALYZED")
    database.save_decision("c1", {
        "category": "Экшен", "overall_score": 85,
        "best_moment_start": 0.2, "best_moment_end": 1.2,
        "decision": "candidate", "confidence": 0.9, "reason": "cached",
    }, "APPROVED")

    calls = patch_success(monkeypatch, sample_video)
    monkeypatch.setattr("src.pipeline.analyze_clip",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    pipeline = Pipeline(cfg, database, twitch=object())
    result = pipeline.process_one("c1")
    assert result["ok"] is True
    assert calls["curator"] == 0
    assert database.get_clip("c1")["status"] == "READY"


def test_curator_failure_is_analysis_error(db, sample_video, monkeypatch):
    cfg, database = db
    database.upsert_clip(make_clip())
    calls = patch_success(monkeypatch, sample_video)

    def fail(*args, **kwargs):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr("src.pipeline.analyze_clip", fail)
    pipeline = Pipeline(cfg, database, twitch=object())
    result = pipeline.process_one("c1")
    assert result["ok"] is False
    assert database.get_clip("c1")["status"] == "ANALYSIS_ERROR"


def test_missing_render_file_is_not_ready(db, sample_video, monkeypatch):
    cfg, database = db
    database.upsert_clip(make_clip())
    patch_success(monkeypatch, sample_video,
                  render=lambda *a, **k: None)
    pipeline = Pipeline(cfg, database, twitch=object())
    result = pipeline.process_one("c1")
    assert result["ok"] is False
    assert database.get_clip("c1")["status"] == "FAILED"


def test_queue_continues_after_one_failure(db, sample_video, monkeypatch):
    cfg, database = db
    database.upsert_clip(make_clip("a"))
    database.upsert_clip(make_clip("b"))
    patch_success(monkeypatch, sample_video)
    counter = {"n": 0}

    def flaky_render(clip_id, source, out_path, start, end, cfg, subtitles=None):
        counter["n"] += 1
        if clip_id == "a":
            raise EditorError("boom")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"x" * 4096)
        return Path(out_path)

    monkeypatch.setattr("src.pipeline.render_vertical", flaky_render)
    pipeline = Pipeline(cfg, database, twitch=object())
    results = pipeline.run_queue(limit=2)
    assert len(results) == 2
    assert database.get_clip("a")["status"] == "FAILED"
    assert database.get_clip("b")["status"] == "READY"
    assert database.get_clip("b")["locked_by"] is None


def test_review_is_not_processed_automatically(db, sample_video, monkeypatch):
    cfg, database = db
    database.upsert_clip(make_clip())
    database.update_status("c1", "DOWNLOADING")
    database.update_status("c1", "DOWNLOADED")
    database.update_status("c1", "ANALYZING")
    database.update_status("c1", "ANALYZED")
    database.save_decision("c1", {
        "category": "Скучное", "overall_score": 30,
        "best_moment_start": 0, "best_moment_end": 0, "reason": "review",
    }, "REVIEW")
    patch_success(monkeypatch, sample_video)
    pipeline = Pipeline(cfg, database, twitch=object())
    assert pipeline.run_queue(limit=1) == []
    assert database.get_clip("c1")["status"] == "REVIEW"


def test_discovery_failure_continues_queue(db, sample_video, monkeypatch):
    cfg, database = db
    database.upsert_clip(make_clip())
    patch_success(monkeypatch, sample_video)
    fake_twitch = SimpleNamespace(
        check_health=lambda: ProviderHealth(False, "fake", "offline"),
        provider_name="fake",
        get_clips=lambda *a, **k: (_ for _ in ()).throw(AssertionError()),
    )
    pipeline = Pipeline(cfg, database, twitch=fake_twitch)
    results = pipeline.run("Test Game", 1)
    assert results and results[0]["ok"] is True
    assert database.get_clip("c1")["status"] == "READY"


def test_parallel_queue_respects_global_limit(db, sample_video, monkeypatch):
    cfg, database = db
    cfg.max_parallel = 2
    for clip_id in ("a", "b", "c", "d"):
        database.upsert_clip(make_clip(clip_id))
    patch_success(monkeypatch, sample_video)
    pipeline = Pipeline(cfg, database, twitch=object())
    results = pipeline.run_queue(limit=3)
    assert len(results) == 3
    stats = database.stats()
    assert stats["READY"] == 3
    assert stats["DISCOVERED"] == 1


def test_reject_with_high_weighted_score_goes_to_review(db, sample_video,
                                                       monkeypatch):
    cfg, database = db
    database.upsert_clip(make_clip())
    patch_success(monkeypatch, sample_video,
                  decision=make_decision(score=80, decision="reject"))
    pipeline = Pipeline(cfg, database, twitch=object())
    result = pipeline.process_one("c1")
    assert result["decision"] == "review"
    assert database.get_clip("c1")["status"] == "REVIEW"


def test_reject_with_low_weighted_score_is_rejected(db, sample_video,
                                                    monkeypatch):
    cfg, database = db
    database.upsert_clip(make_clip())
    patch_success(monkeypatch, sample_video,
                  decision=make_decision(score=10, decision="reject"))
    pipeline = Pipeline(cfg, database, twitch=object())
    result = pipeline.process_one("c1")
    assert result["decision"] == "reject"
    assert database.get_clip("c1")["status"] == "REJECTED"


