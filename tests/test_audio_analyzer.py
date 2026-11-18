"""Tests for audio analysis: acoustic detectors, text events and cache."""
import json
import logging
import math
import sys
import threading
import time
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src import audio_analyzer as aa
from src.models import (AnalysisError, AudioEvent, TranscriptWord)
from src.utils import file_fingerprint, read_json, write_json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_cfg(**overrides):
    values = {
        "whisper_model": "small",
        "whisper_device": "cpu",
        "whisper_compute": "int8",
        "whisper_beam_size": 5,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def make_word(start, end, word):
    return TranscriptWord(float(start), float(end), word)


def fake_word(start, end, word):
    return SimpleNamespace(start=float(start), end=float(end), word=word)


class FakeSegment:
    def __init__(self, start, end, text, words=None):
        self.start = float(start)
        self.end = float(end)
        self.text = text
        self.words = list(words or [])


class FakeModel:
    """Minimal faster-whisper stand-in (no downloads, no network)."""

    def __init__(self, segments=None, info=None, error=None,
                 iteration_error=None):
        self.segments = list(segments or [])
        self.info = info if info is not None else SimpleNamespace(
            language="ru", duration=2.0)
        self.error = error
        self.iteration_error = iteration_error
        self.calls = []

    def transcribe(self, path, **kwargs):
        self.calls.append({"path": path, **kwargs})
        if self.error is not None:
            raise self.error

        def generate():
            if self.iteration_error is not None:
                raise self.iteration_error
            for segment in self.segments:
                yield segment

        return generate(), self.info


def simple_segments():
    return [FakeSegment(0.5, 1.0, "привет",
                        [fake_word(0.5, 0.8, "привет")])]


def write_wav(path, duration=4.0, sample_rate=16000, loud_span=(1.0, 2.0),
              amplitude=0.8):
    """Write a small 16 kHz mono PCM WAV with one loud tone burst."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = int(round(duration * sample_rate))
    timeline = np.arange(count, dtype=np.float64) / sample_rate
    signal = np.zeros(count, dtype=np.float64)
    if loud_span:
        start = max(0, int(loud_span[0] * sample_rate))
        end = min(count, int(loud_span[1] * sample_rate))
        signal[start:end] = amplitude * np.sin(
            2.0 * np.pi * 220.0 * timeline[start:end])
    pcm = (np.clip(signal, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return path


def make_source(tmp_path, name="clip.mp4", payload=b"x" * 4096):
    source = tmp_path / name
    source.write_bytes(payload)
    return source


def cache_payload(fingerprint, **overrides):
    payload = {
        "version": aa.CACHE_VERSION,
        "whisper_model": "small",
        "whisper_device": "cpu",
        "whisper_compute": "int8",
        "source_fingerprint": fingerprint,
        "model": "small",
        "language": "ru",
        "duration": 2.0,
        "full_text": "привет мир",
        "segments": [{"start": 0.0, "end": 2.0, "text": "привет мир"}],
        "words": [
            {"start": 0.2, "end": 0.6, "word": "привет"},
            {"start": 0.7, "end": 1.1, "word": "мир"},
        ],
        "events": [
            {"kind": "laughter", "start": 0.2, "end": 1.0,
             "intensity": 1.25, "speech_overlap": True},
        ],
    }
    payload.update(overrides)
    return payload


def forbidden(name, calls):
    def fail(*args, **kwargs):
        calls[name] = calls.get(name, 0) + 1
        raise AssertionError(f"{name} must not be called")
    return fail


def patch_recompute(monkeypatch, calls, model=None, duration=2.0):
    """Patch the expensive stages with fakes that record their calls."""
    def fake_extract(source, out_wav):
        calls["extract"] = calls.get("extract", 0) + 1
        write_wav(out_wav, duration=duration)

    fake_model = model or FakeModel(simple_segments(),
                                    info=SimpleNamespace(language="ru",
                                                         duration=duration))
    monkeypatch.setattr(aa, "_extract_wav", fake_extract)
    monkeypatch.setattr(aa, "_get_whisper", lambda *a, **k: fake_model)
    return fake_model


def event_dicts(events):
    return [(event.kind, round(event.start, 2), round(event.end, 2))
            for event in events]


# ---------------------------------------------------------------------------
# Acoustic detectors (pure functions over synthetic RMS arrays)
# ---------------------------------------------------------------------------
def test_loud_vs_noise_classification_from_rms():
    energy = np.full(40, 0.01)
    energy[10:12] = 0.6  # loud burst at 2.5-3.0 s

    without_speech = aa._detect_acoustic_events(energy, 0.25, 10.0, [])
    kinds = {event.kind for event in without_speech}
    assert "noise" in kinds
    assert "loud" not in kinds
    assert "reaction" not in kinds and "scream" not in kinds
    noise = next(event for event in without_speech if event.kind == "noise")
    assert noise.start == pytest.approx(2.5)
    assert noise.end == pytest.approx(3.0)
    assert noise.speech_overlap is False
    assert noise.intensity > 1.0

    with_speech = aa._detect_acoustic_events(energy, 0.25, 10.0,
                                             [(2.4, 3.1)])
    kinds = {event.kind for event in with_speech}
    assert "loud" in kinds and "noise" not in kinds
    loud = next(event for event in with_speech if event.kind == "loud")
    assert loud.speech_overlap is True


def test_onset_detection_from_rms():
    energy = np.full(40, 0.01)
    energy[20] = 0.5
    energy[21] = 0.55
    events = aa._detect_acoustic_events(energy, 0.25, 10.0, [])
    onsets = [event for event in events if event.kind == "onset"]
    assert onsets, event_dicts(events)
    assert onsets[0].start == pytest.approx(5.0)
    assert onsets[0].intensity >= 2.0


def test_pause_detection_from_rms():
    energy = np.concatenate([
        np.full(10, 0.08),
        np.full(8, 0.001),   # 2.0 s of silence
        np.full(8, 0.08),
    ])
    events = aa._detect_acoustic_events(energy, 0.25, 6.5, [])
    pauses = [event for event in events if event.kind == "pause"]
    assert len(pauses) == 1
    assert pauses[0].start == pytest.approx(2.5)
    assert pauses[0].end == pytest.approx(4.5)
    assert pauses[0].speech_overlap is False

    # A word overlapping the silence trims the gap instead of dropping it.
    trimmed = aa._detect_acoustic_events(energy, 0.25, 6.5, [(2.5, 2.75)])
    trimmed_pauses = [event for event in trimmed if event.kind == "pause"]
    assert trimmed_pauses
    assert trimmed_pauses[0].start == pytest.approx(2.75)


def test_acoustic_events_are_finite_clamped_and_sorted():
    energy = np.array([0.01] * 5 + [np.nan, np.inf, 0.6, 0.6]
                      + [0.01] * 9, dtype=np.float64)
    events = aa._detect_acoustic_events(energy, 0.25, 4.0, [])
    assert events
    previous_start = -1.0
    for event in events:
        assert math.isfinite(event.start)
        assert math.isfinite(event.end)
        assert math.isfinite(event.intensity)
        assert event.intensity > 0
        assert 0.0 <= event.start < event.end <= 4.0
        assert event.start >= previous_start
        previous_start = event.start


def test_speechless_loud_burst_never_becomes_reaction():
    energy = np.full(40, 0.01)
    energy[10:14] = 0.6
    events = aa._detect_acoustic_events(energy, 0.25, 10.0, [])
    promoted = aa._apply_text_emphasis(events, [])
    kinds = {event.kind for event in promoted}
    assert "noise" in kinds
    assert not kinds & {"reaction", "scream"}


def test_loud_with_emphasis_becomes_reaction_or_scream():
    energy = np.full(40, 0.01)
    energy[10:14] = 0.6
    intervals = [(2.5, 3.5)]

    plain = aa._apply_text_emphasis(
        aa._detect_acoustic_events(energy, 0.25, 10.0, intervals),
        [make_word(2.5, 3.0, "wow")],
    )
    assert any(event.kind == "loud" for event in plain)
    assert not any(event.kind == "reaction" for event in plain)

    reaction = aa._apply_text_emphasis(
        aa._detect_acoustic_events(energy, 0.25, 10.0, intervals),
        [make_word(2.5, 3.0, "wow!")],
    )
    assert any(event.kind == "reaction" for event in reaction)
    # Reclassified, not duplicated: loud and reaction never coexist.
    assert not any(event.kind == "loud" for event in reaction)

    scream = aa._apply_text_emphasis(
        aa._detect_acoustic_events(energy, 0.25, 10.0, intervals),
        [make_word(2.5, 3.0, "WOW!")],
    )
    assert any(event.kind == "scream" for event in scream)
    assert not any(event.kind == "loud" for event in scream)


# ---------------------------------------------------------------------------
# Event normalisation
# ---------------------------------------------------------------------------
def test_normalize_events_clamps_sorts_and_sanitizes():
    raw = [
        AudioEvent("loud", -2.0, 99.0, float("nan"), False),
        AudioEvent("pause", 4.9, 5.1, float("inf"), False),
        AudioEvent("laughter", 2.0, 1.0, 1.0, False),  # reversed interval
        AudioEvent("onset", 0.0, 0.04, 1.0, False),    # too short
    ]
    events = aa._normalize_events(raw, 5.0)
    assert [event.kind for event in events] == ["loud", "laughter", "pause"]
    previous_start = -1.0
    for event in events:
        assert math.isfinite(event.start)
        assert math.isfinite(event.end)
        assert math.isfinite(event.intensity)
        assert event.intensity > 0
        assert 0.0 <= event.start <= event.end <= 5.0
        assert event.start >= previous_start
        previous_start = event.start
    assert events[1].start == pytest.approx(1.0)
    assert events[1].end == pytest.approx(2.0)


def test_normalize_prefers_reaction_over_loud():
    raw = [
        AudioEvent("loud", 1.0, 3.0, 2.0),
        AudioEvent("reaction", 1.2, 2.8, 2.5),
    ]
    events = aa._normalize_events(raw, 5.0)
    assert [event.kind for event in events] == ["reaction"]


def test_normalize_merges_adjacent_same_kind():
    raw = [
        AudioEvent("laughter", 0.0, 0.5, 1.0),
        AudioEvent("laughter", 0.7, 1.2, 2.0),
    ]
    events = aa._normalize_events(raw, 2.0)
    assert len(events) == 1
    assert events[0].end == pytest.approx(1.2)
    assert events[0].intensity == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Text classification (RU + EN)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("token", [
    "ха-ха", "хаха", "ХА-ХА", "лол", "ахаха", "хех",
    "haha", "hahaha", "hehe", "lol", "lmao", "rofl", "xd",
])
def test_laughter_tokens_ru_en(token):
    events = aa._classify_text_group([make_word(1.0, 1.5, token)])
    assert [event.kind for event in events] == ["laughter"], token
    assert events[0].speech_overlap is True


def test_single_weak_laugh_syllable_is_not_laughter():
    assert aa._classify_text_group([make_word(0.0, 0.4, "ха")]) == []
    events = aa._classify_text_group(
        [make_word(0.0, 0.3, "ха"), make_word(0.4, 0.7, "ха")])
    assert [event.kind for event in events] == ["laughter"]


def test_scream_and_reaction_thresholds_ru_en():
    def kinds(words):
        return {event.kind for event in aa._classify_text_group(words)}

    # One "!" alone is a mild reaction, not a scream.
    assert kinds([make_word(0.0, 0.5, "wow!")]) == {"reaction"}
    assert kinds([make_word(0.0, 0.5, "Вау!")]) == {"reaction"}
    # Two emphasis markers escalate to a scream.
    assert kinds([make_word(0.0, 0.5, "WOW!")]) == {"scream"}
    assert kinds([make_word(0.0, 0.5, "WOW!!")]) == {"scream"}
    assert kinds([make_word(0.0, 0.5, "НЕТ"), make_word(0.5, 1.0, "НЕТ")]) \
        == {"scream"}
    assert kinds([make_word(0.0, 0.5, "ЧТО?!")]) == {"scream"}
    # Elongated vowels are screams even without punctuation.
    assert kinds([make_word(0.0, 0.6, "noooo")]) == {"scream"}
    assert kinds([make_word(0.0, 0.6, "ААААА")]) == {"scream"}
    # Plain speech is not an event; "?" is a mild reaction.
    assert kinds([make_word(0.0, 0.5, "wow")]) == set()
    assert kinds([make_word(0.0, 0.5, "что?")]) == {"reaction"}


# ---------------------------------------------------------------------------
# Whisper model cache
# ---------------------------------------------------------------------------
def test_get_whisper_loads_each_model_once_with_lock(monkeypatch):
    created = []

    class SlowWhisperModel:
        def __init__(self, model_name, device="cpu", compute_type="int8"):
            time.sleep(0.03)
            created.append((model_name, device, compute_type))

    monkeypatch.setitem(sys.modules, "faster_whisper",
                        SimpleNamespace(WhisperModel=SlowWhisperModel))
    monkeypatch.setattr(aa, "_MODEL_CACHE", {})

    results = []
    errors = []
    barrier = threading.Barrier(4)

    def worker():
        try:
            barrier.wait()
            results.append(aa._get_whisper("tiny", "cpu", "int8"))
        except BaseException as exc:  # pragma: no cover - test diagnostics
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert len(created) == 1
    assert len(results) == 4
    assert all(result is results[0] for result in results)
    assert aa._get_whisper("tiny", "cpu", "int8") is results[0]


def test_model_load_failure_raises_analysis_error(monkeypatch):
    class BrokenWhisperModel:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("model files are unavailable")

    monkeypatch.setitem(sys.modules, "faster_whisper",
                        SimpleNamespace(WhisperModel=BrokenWhisperModel))
    monkeypatch.setattr(aa, "_MODEL_CACHE", {})

    with pytest.raises(AnalysisError, match="Could not load Whisper model"):
        aa._get_whisper("tiny", "cpu", "int8")


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
def test_cache_roundtrip_skips_whisper_and_ffmpeg(tmp_path, monkeypatch):
    source = make_source(tmp_path)
    work = tmp_path / "work"
    fingerprint = file_fingerprint(source)
    write_json(work / "transcript.json", cache_payload(fingerprint))

    calls = {}
    monkeypatch.setattr(aa, "_get_whisper", forbidden("whisper", calls))
    monkeypatch.setattr(aa, "_extract_wav", forbidden("extract", calls))

    result = aa.analyze_audio(source, work, make_cfg())

    assert calls == {}
    assert result.language == "ru"
    assert result.duration == pytest.approx(2.0)
    assert result.full_text == "привет мир"
    assert result.model == "small"
    assert result.source_fingerprint == fingerprint
    assert [word.word for word in result.words] == ["привет", "мир"]
    assert [event.kind for event in result.events] == ["laughter"]
    # A cache hit must not even extract the WAV.
    assert not (work / "audio.wav").exists()


def test_stale_source_fingerprint_forces_recompute(tmp_path, monkeypatch,
                                                  caplog):
    source = make_source(tmp_path)
    work = tmp_path / "work"
    write_json(work / "transcript.json",
               cache_payload(fingerprint="123456:1"))
    calls = {}
    model = patch_recompute(monkeypatch, calls)
    cfg = make_cfg()

    with caplog.at_level(logging.WARNING, logger=aa.__name__):
        result = aa.analyze_audio(source, work, cfg)

    assert calls["extract"] == 1
    assert len(model.calls) == 1
    assert result.full_text == "привет"
    stored = read_json(work / "transcript.json")
    assert stored["source_fingerprint"] == file_fingerprint(source)
    assert any("stale" in record.getMessage().lower()
               for record in caplog.records)


def test_changed_whisper_model_forces_recompute(tmp_path, monkeypatch,
                                                caplog):
    source = make_source(tmp_path)
    work = tmp_path / "work"
    fingerprint = file_fingerprint(source)
    write_json(work / "transcript.json",
               cache_payload(fingerprint, whisper_model="large-v3"))
    calls = {}
    model = patch_recompute(monkeypatch, calls)

    with caplog.at_level(logging.WARNING, logger=aa.__name__):
        aa.analyze_audio(source, work, make_cfg())

    assert calls["extract"] == 1
    assert len(model.calls) == 1
    stored = read_json(work / "transcript.json")
    assert stored["whisper_model"] == "small"
    assert any("model" in record.getMessage().lower()
               for record in caplog.records)


@pytest.mark.parametrize("bad_content", [
    "this is not json at all",
    json.dumps({"version": aa.CACHE_VERSION}),
])
def test_corrupt_cache_forces_recompute_with_warning(tmp_path, monkeypatch,
                                                     caplog, bad_content):
    source = make_source(tmp_path)
    work = tmp_path / "work"
    work.mkdir(parents=True)
    (work / "transcript.json").write_text(bad_content, encoding="utf-8")
    calls = {}
    model = patch_recompute(monkeypatch, calls)

    with caplog.at_level(logging.WARNING, logger=aa.__name__):
        result = aa.analyze_audio(source, work, make_cfg())

    assert calls["extract"] == 1
    assert len(model.calls) == 1
    assert result.full_text == "привет"
    assert any("cache" in record.getMessage().lower()
               for record in caplog.records)


def test_old_cache_version_forces_recompute(tmp_path, monkeypatch):
    source = make_source(tmp_path)
    work = tmp_path / "work"
    write_json(work / "transcript.json",
               cache_payload(file_fingerprint(source), version=1))
    calls = {}
    model = patch_recompute(monkeypatch, calls)

    aa.analyze_audio(source, work, make_cfg())

    assert calls["extract"] == 1
    assert len(model.calls) == 1
    assert read_json(work / "transcript.json")["version"] == aa.CACHE_VERSION


def test_wav_reuse_requires_matching_source_fingerprint(tmp_path, monkeypatch):
    source = make_source(tmp_path)
    work = tmp_path / "work"
    work.mkdir(parents=True)
    fingerprint = file_fingerprint(source)
    wav = work / "audio.wav"
    write_wav(wav, duration=2.0)
    stat = wav.stat()
    write_json(aa._wav_meta_path(wav), {
        "source_fingerprint": fingerprint,
        "wav_size": stat.st_size,
        "wav_mtime_ns": stat.st_mtime_ns,
    })

    calls = {}
    monkeypatch.setattr(aa, "_extract_wav", forbidden("extract", calls))
    model = FakeModel(simple_segments(),
                      info=SimpleNamespace(language="ru", duration=2.0))
    monkeypatch.setattr(aa, "_get_whisper", lambda *a, **k: model)
    result = aa.analyze_audio(source, work, make_cfg())
    assert calls == {}  # the provenance-matched WAV was reused
    assert result.full_text == "привет"

    # A different source version must re-extract, not reuse the old WAV.
    source.write_bytes(b"y" * 5000)
    calls = {}
    fake_model = patch_recompute(monkeypatch, calls)
    result = aa.analyze_audio(source, work, make_cfg())
    assert calls["extract"] == 1
    assert len(fake_model.calls) == 1
    meta = read_json(aa._wav_meta_path(wav))
    assert meta["source_fingerprint"] == file_fingerprint(source)


# ---------------------------------------------------------------------------
# Full analysis with a fake faster-whisper model
# ---------------------------------------------------------------------------
def test_fake_whisper_result_and_transcript_cache(tmp_path, monkeypatch):
    source = make_source(tmp_path)
    work = tmp_path / "processed" / "c1"
    calls = {"extract": 0}

    def fake_extract(source_path, out_wav):
        calls["extract"] += 1
        write_wav(out_wav, duration=4.0)

    monkeypatch.setattr(aa, "_extract_wav", fake_extract)

    segments = [
        FakeSegment(1.0, 1.7, "ХА-ХА",
                    [fake_word(1.0, 1.3, "ха"),
                     fake_word(1.3, 1.7, "ха")]),
        FakeSegment(2.0, 2.7, "WOW! НЕТ",
                    [fake_word(2.0, 2.3, "WOW!"),
                     fake_word(2.3, 2.7, "НЕТ")]),
    ]
    model = FakeModel(segments, info=SimpleNamespace(language="ru",
                                                    duration=4.0))
    monkeypatch.setattr(aa, "_get_whisper", lambda *a, **k: model)
    cfg = make_cfg()

    result = aa.analyze_audio(source, work, cfg)

    # Transcription settings requested from faster-whisper.
    assert len(model.calls) == 1
    call = model.calls[0]
    assert call["path"] == str(work / "audio.wav")
    assert call["word_timestamps"] is True
    assert call["vad_filter"] is True
    assert call["beam_size"] == cfg.whisper_beam_size == 5

    # Result fields.
    assert calls["extract"] == 1
    assert result.language == "ru"
    assert result.duration == pytest.approx(4.0)
    assert result.model == "small"
    assert result.source_fingerprint == file_fingerprint(source)
    assert result.full_text == "ХА-ХА WOW! НЕТ"
    assert [(round(w.start, 2), round(w.end, 2), w.word)
            for w in result.words] == [
        (1.0, 1.3, "ха"), (1.3, 1.7, "ха"),
        (2.0, 2.3, "WOW!"), (2.3, 2.7, "НЕТ"),
    ]
    assert [(round(s.start, 2), round(s.end, 2), s.text)
            for s in result.segments] == [
        (1.0, 1.7, "ХА-ХА"), (2.0, 2.7, "WOW! НЕТ"),
    ]

    # Events: text laughter/scream plus acoustic onsets and pauses; all valid.
    kinds = {event.kind for event in result.events}
    assert {"laughter", "scream", "onset", "pause"} <= kinds
    assert "noise" not in kinds
    previous_start = -1.0
    for event in result.events:
        assert math.isfinite(event.start)
        assert math.isfinite(event.end)
        assert math.isfinite(event.intensity)
        assert event.intensity > 0
        assert 0.0 <= event.start < event.end <= result.duration
        assert event.start >= previous_start
        previous_start = event.start

    # transcript.json contents.
    cache_path = work / "transcript.json"
    stored = read_json(cache_path)
    assert stored["version"] == aa.CACHE_VERSION
    assert stored["whisper_model"] == "small"
    assert stored["whisper_device"] == "cpu"
    assert stored["whisper_compute"] == "int8"
    assert stored["source_fingerprint"] == file_fingerprint(source)
    assert stored["language"] == "ru"
    assert stored["duration"] == pytest.approx(4.0)
    assert stored["full_text"] == "ХА-ХА WOW! НЕТ"
    assert len(stored["segments"]) == len(result.segments)
    assert len(stored["words"]) == len(result.words)
    assert len(stored["events"]) == len(result.events)
    assert [item["kind"] for item in stored["events"]] == \
        [event.kind for event in result.events]
    # UTF-8, non-escaped Cyrillic on disk.
    raw_text = cache_path.read_text(encoding="utf-8")
    assert "ха" in raw_text and "WOW!" in raw_text

    # The metadata sidecar makes the WAV reusable on the next run.
    assert (work / "audio.wav").is_file()
    assert read_json(aa._wav_meta_path(work / "audio.wav"))[
        "source_fingerprint"] == result.source_fingerprint

    # A second call loads the cache: no model, no ffmpeg, same data.
    second = aa.analyze_audio(source, work, cfg)
    assert len(model.calls) == 1
    assert calls["extract"] == 1
    assert second.full_text == result.full_text
    assert event_dicts(second.events) == event_dicts(result.events)
    assert [(w.start, w.end, w.word) for w in second.words] == \
        [(w.start, w.end, w.word) for w in result.words]


@pytest.mark.parametrize("failure", ["transcribe", "iterate"])
def test_transcription_failure_raises_analysis_error(tmp_path, monkeypatch,
                                                     failure):
    source = make_source(tmp_path)
    work = tmp_path / "work"
    calls = {"extract": 0}

    def fake_extract(source_path, out_wav):
        calls["extract"] += 1
        write_wav(out_wav, duration=2.0)

    monkeypatch.setattr(aa, "_extract_wav", fake_extract)
    if failure == "transcribe":
        model = FakeModel([], error=RuntimeError("cuda out of memory"))
    else:
        model = FakeModel([], iteration_error=RuntimeError("decoder crashed"))
    monkeypatch.setattr(aa, "_get_whisper", lambda *a, **k: model)

    with pytest.raises(AnalysisError, match="Whisper transcription failed"):
        aa.analyze_audio(source, work, make_cfg())
    # A failed run must never leave a valid cache behind.
    assert not (work / "transcript.json").exists()
