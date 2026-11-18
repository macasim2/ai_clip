"""Audio analysis with faster-whisper + signal heuristics."""
import json
import subprocess
import wave
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .logging_setup import get_logger
from .models import (AudioAnalysisResult, AudioEvent, TranscriptSegment,
                     TranscriptWord)

log = get_logger(__name__)

_MODEL_CACHE: Dict[str, Any] = {}


def _get_whisper(model_name: str, device: str, compute: str):
    key = f"{model_name}|{device}|{compute}"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    from faster_whisper import WhisperModel
    log.info("Loading Whisper model %s (%s/%s)", model_name, device, compute)
    m = WhisperModel(model_name, device=device, compute_type=compute)
    _MODEL_CACHE[key] = m
    return m


def _extract_wav(source: Path, out_wav: Path) -> None:
    if out_wav.exists() and out_wav.stat().st_size > 1024:
        return
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", str(source),
         "-ac", "1", "-ar", "16000", "-f", "wav", str(out_wav)],
        capture_output=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg audio extraction failed: {r.stderr[-300:]!r}")


def _read_wav(path: Path):
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        data = w.readframes(n)
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    return sr, samples


def _detect_loudness(samples: np.ndarray, sr: int, window_s: float = 0.5) -> List[AudioEvent]:
    win = int(window_s * sr)
    if win <= 0 or len(samples) < win * 3:
        return []
    n = len(samples) // win
    rms = np.array([
        float(np.sqrt(np.mean(samples[i * win:(i + 1) * win] ** 2) + 1e-9))
        for i in range(n)
    ])
    if rms.size < 4:
        return []
    med = float(np.median(rms))
    p95 = float(np.percentile(rms, 95))
    threshold = max(med * 4.0, p95 * 1.2)
    events: List[AudioEvent] = []
    i = 0
    while i < rms.size:
        if rms[i] >= threshold:
            j = i
            while j < rms.size and rms[j] >= med * 2.0:
                j += 1
            events.append(AudioEvent(
                kind="loud", start=i * window_s,
                end=min(len(samples) / sr, j * window_s),
                intensity=float(rms[i] / (med + 1e-9)),
            ))
            i = j
        else:
            i += 1
    return events


def _detect_pauses(words: List[TranscriptWord], min_gap: float = 1.5) -> List[AudioEvent]:
    return [
        AudioEvent(kind="pause", start=a.end, end=b.start,
                   intensity=b.start - a.end)
        for a, b in zip(words, words[1:]) if (b.start - a.end) >= min_gap
    ]


_LAUGH = ("ха-ха", "хаха", "ха ха", "haha", "lol", "лол", "хех")
_SCREAM = ("!", "!!", "!!!")


def _detect_from_text(words: List[TranscriptWord]) -> List[AudioEvent]:
    if not words:
        return []
    events: List[AudioEvent] = []
    current: List[TranscriptWord] = []
    t0 = words[0].start
    for w in words:
        if w.end - t0 > 3.0 and current:
            _classify(current, events)
            current = []
            t0 = w.start
        current.append(w)
    if current:
        _classify(current, events)
    return events


def _classify(words: List[TranscriptWord], out: List[AudioEvent]):
    text = " ".join(w.word for w in words).lower()
    s, e = words[0].start, words[-1].end
    if any(p in text for p in _LAUGH):
        out.append(AudioEvent("laughter", s, e, 1.0))
    caps = sum(1 for w in words if any(h in w.word for h in _SCREAM))
    if caps >= 2:
        out.append(AudioEvent("scream", s, e, float(caps)))


def analyze_audio(source: Path, work_dir: Path, cfg) -> AudioAnalysisResult:
    work_dir.mkdir(parents=True, exist_ok=True)
    wav = work_dir / "audio.wav"
    _extract_wav(source, wav)
    model = _get_whisper(cfg.whisper_model, cfg.whisper_device, cfg.whisper_compute)

    segments_iter, info = model.transcribe(
        str(wav), word_timestamps=True, vad_filter=True, beam_size=5,
    )
    segments: List[TranscriptSegment] = []
    words: List[TranscriptWord] = []
    parts: List[str] = []
    for seg in segments_iter:
        segments.append(TranscriptSegment(seg.start, seg.end, seg.text.strip()))
        parts.append(seg.text.strip())
        for w in (seg.words or []):
            ww = (w.word or "").strip()
            if ww:
                words.append(TranscriptWord(float(w.start), float(w.end), ww))

    language = getattr(info, "language", "unknown") if info else "unknown"
    duration = getattr(info, "duration", 0.0) if info else 0.0
    if not duration and words:
        duration = words[-1].end

    sr, samples = _read_wav(wav)
    events = (_detect_loudness(samples, sr) + _detect_pauses(words)
              + _detect_from_text(words))
    events.sort(key=lambda e: e.start)

    result = AudioAnalysisResult(
        language=language, duration=duration,
        segments=segments, words=words, events=events,
        full_text=" ".join(parts).strip(),
    )
    (work_dir / "transcript.json").write_text(
        json.dumps({
            "language": language, "duration": duration,
            "segments": [s.__dict__ for s in segments],
            "words": [w.__dict__ for w in words],
            "events": [e.__dict__ for e in events],
            "full_text": result.full_text,
        }, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    log.info("Audio: %.1fs, %d words, %d events, lang=%s",
             duration, len(words), len(events), language)
    return result
