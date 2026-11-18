"""Audio analysis: faster-whisper transcription + numpy acoustic heuristics.

The public entry point is :func:`analyze_audio`.  It transcribes a clip with
faster-whisper (word timestamps + VAD), then derives acoustic events (loud
peaks, onsets, pauses, reactions) and text events (laughter, screams, vocal
reactions) from the 16 kHz mono WAV.

Results are cached per clip in ``<work_dir>/transcript.json``; the extracted
WAV lives in ``<work_dir>/audio.wav`` and is reused only when its sidecar
metadata proves it was extracted from the same source file version.
"""
from __future__ import annotations

import math
import re
import subprocess
import threading
import wave
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .logging_setup import get_logger
from .models import (AnalysisError, AudioAnalysisResult, AudioEvent,
                     TranscriptSegment, TranscriptWord)
from .utils import (ToolNotFound, file_fingerprint, find_tool, read_json,
                    write_json)

log = get_logger(__name__)

# Cache / file layout -------------------------------------------------------
CACHE_VERSION = 2
TRANSCRIPT_FILE = "transcript.json"
WAV_FILE = "audio.wav"
WAV_MIN_BYTES = 1024
SAMPLE_RATE = 16000

# Acoustic detection --------------------------------------------------------
WINDOW_S = 0.25                 # RMS analysis window
BASELINE_HALF_WINDOWS = 8       # local median baseline: +/- 2 s
LOUD_RATIO = 3.5                # peak must be this much above the baseline
LOUD_BRIDGE_RATIO = 2.0         # allow one short dip inside a loud burst
LOUD_MIN_RMS = 0.02             # absolute floor: never call digital noise loud
ONSET_RATIO = 2.5               # sharp rise versus the previous window
PAUSE_RATIO = 0.15              # silence versus the clip's speech level
PAUSE_MIN_RMS = 0.0005
PAUSE_MIN_S = 0.7               # shorter dips are normal word gaps
MERGE_GAP_S = 0.3               # merge same-kind events separated by <= gap
MIN_EVENT_S = 0.05              # drop degenerate events after clamping
EMPHASIS_PAD_S = 0.5            # speech considered "near" a loud event
INTENSITY_MAX = 100.0           # keep ratios usable in prompts/logs

# Text classification -------------------------------------------------------
TEXT_GROUP_MAX_SPAN = 4.0
TEXT_GROUP_MAX_GAP = 0.9
SCREAM_INTENSITY_MAX = 3.0

_LAUGH_STRONG_RE = re.compile(
    r"^(?:"
    r"а?ха(?:-?ха)+х*|"       # ха-ха, хаха, ахахаха
    r"хе(?:-?хе)+|"           # хехе, хе-хе
    r"хи(?:-?хи)+|"           # хихи
    r"хех+х*|"                # хех
    r"h[ae](?:h[ae])+|"       # haha, hahaha, hehe
    r"а?хах+х*|"              # ахах
    r"лол+|lol+|кек+|kek+|"
    r"lmao|rofl|xd+"
    r")$"
)
_LAUGH_WEAK_RE = re.compile(r"^(?:ха|хе|хи|hah|heh)$")

_TEXT_KINDS = frozenset({"laughter", "scream", "reaction"})
_REACTIVE_KINDS = frozenset({"reaction", "scream"})

_MODEL_CACHE: Dict[str, Any] = {}
_MODEL_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------
def _finite(value: Any, default: float = 0.0) -> float:
    """Return value as a finite float, or ``default`` when it is not one."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _cache_number(value: Any) -> float:
    """Strict number parser for cache data: rejects NaN/inf and non-numbers."""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"non-finite value in cache: {value!r}")
    return number


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _overlap_seconds(first: Sequence[float],
                     second: Sequence[float]) -> float:
    first_start, first_end = sorted((_finite(first[0]), _finite(first[1])))
    second_start, second_end = sorted((_finite(second[0]), _finite(second[1])))
    return max(0.0, min(first_end, second_end) - max(first_start, second_start))


def _overlaps(interval: Sequence[float],
              intervals: Iterable[Sequence[float]]) -> bool:
    return any(_overlap_seconds(interval, other) > 0.0 for other in intervals)


def _speech_intervals(words: Sequence[TranscriptWord]
                      ) -> List[Tuple[float, float]]:
    intervals: List[Tuple[float, float]] = []
    for word in words:
        start = _finite(getattr(word, "start", 0.0))
        end = _finite(getattr(word, "end", 0.0))
        if end > start:
            intervals.append((start, end))
    return intervals


# ---------------------------------------------------------------------------
# Whisper model cache (one load per model/device/compute, thread safe)
# ---------------------------------------------------------------------------
def _get_whisper(model_name: str, device: str, compute: str):
    """Load (once) and return a faster-whisper model for the given settings."""
    key = f"{model_name}|{device}|{compute}"
    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached
        log.info("Loading Whisper model %s (%s/%s)", model_name, device, compute)
        try:
            from faster_whisper import WhisperModel
            model = WhisperModel(model_name, device=device, compute_type=compute)
        except Exception as exc:  # noqa: BLE001 - surface as AnalysisError
            raise AnalysisError(
                f"Could not load Whisper model {model_name!r} "
                f"({device}/{compute}): {exc}"
            ) from exc
        _MODEL_CACHE[key] = model
        return model


# ---------------------------------------------------------------------------
# WAV extraction / reading
# ---------------------------------------------------------------------------
def _wav_meta_path(wav: Path) -> Path:
    return Path(wav).with_name(Path(wav).name + ".meta.json")


def _extract_wav(source: Path, out_wav: Path) -> None:
    """Extract 16 kHz mono PCM WAV from the source with ffmpeg."""
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    try:
        ffmpeg = find_tool("ffmpeg")
    except ToolNotFound as exc:
        raise AnalysisError(
            "ffmpeg is required to extract audio but was not found") from exc
    try:
        out_wav.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise AnalysisError(f"Cannot replace {out_wav}: {exc}") from exc
    try:
        completed = subprocess.run(
            [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
             "-i", str(source), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
             "-f", "wav", str(out_wav)],
            capture_output=True,
            timeout=600,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise AnalysisError(
            f"Could not run ffmpeg for {Path(source).name}: {exc}") from exc
    if completed.returncode != 0:
        tail = (completed.stderr or b"")[-400:]
        raise AnalysisError(
            f"ffmpeg audio extraction failed for {Path(source).name}: {tail!r}")


def _ensure_wav(source: Path, work_dir: Path, fingerprint: str) -> Path:
    """Return ``work_dir/audio.wav``, re-extracting unless provenance matches."""
    wav = Path(work_dir) / WAV_FILE
    meta = read_json(_wav_meta_path(wav), None)
    try:
        stat = wav.stat() if wav.is_file() else None
    except OSError:
        stat = None
    if (stat is not None and stat.st_size > WAV_MIN_BYTES
            and isinstance(meta, dict)
            and str(meta.get("source_fingerprint", "")) == str(fingerprint)
            and meta.get("wav_size") == stat.st_size
            and meta.get("wav_mtime_ns") == stat.st_mtime_ns):
        return wav

    log.info("Extracting audio %s -> %s", Path(source).name, wav.name)
    try:
        _extract_wav(source, wav)
    except AnalysisError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalise extractor failures
        raise AnalysisError(
            f"Audio extraction failed for {Path(source).name}: {exc}") from exc
    try:
        stat = wav.stat()
    except OSError as exc:
        raise AnalysisError(
            f"Audio extraction produced no file for {Path(source).name}: {exc}"
        ) from exc
    if stat.st_size <= WAV_MIN_BYTES:
        raise AnalysisError(
            f"Audio extraction produced an invalid WAV for {Path(source).name}")
    try:
        write_json(_wav_meta_path(wav), {
            "source_fingerprint": str(fingerprint),
            "wav_size": stat.st_size,
            "wav_mtime_ns": stat.st_mtime_ns,
        })
    except OSError as exc:
        raise AnalysisError(
            f"Could not save audio metadata for {wav}: {exc}") from exc
    return wav


def _read_wav(path: Path) -> Tuple[int, np.ndarray]:
    """Read a PCM WAV file as (sample_rate, float32 mono samples in [-1, 1])."""
    try:
        with wave.open(str(path), "rb") as handle:
            sample_rate = handle.getframerate()
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            frames = handle.readframes(handle.getnframes())
    except (OSError, wave.Error, EOFError) as exc:
        raise AnalysisError(f"Cannot read extracted audio {path}: {exc}") from exc
    if sample_rate <= 0:
        raise AnalysisError(f"Extracted audio has an invalid sample rate: {path}")
    if sample_width != 2:
        raise AnalysisError(
            f"Unsupported WAV sample width {sample_width} bytes in {path}")
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        usable = samples.size - samples.size % channels
        samples = samples[:usable].reshape(-1, channels).mean(axis=1)
    return int(sample_rate), samples


# ---------------------------------------------------------------------------
# RMS / baseline primitives (pure numpy, no extra dependencies)
# ---------------------------------------------------------------------------
def _frame_rms(samples: np.ndarray, sample_rate: int,
               window_s: float = WINDOW_S) -> np.ndarray:
    """Compute short-window RMS energy for the samples."""
    signal = np.asarray(samples, dtype=np.float64).ravel()
    window = max(1, int(round(float(window_s) * float(sample_rate))))
    if signal.size < window:
        return np.zeros(0, dtype=np.float64)
    count = signal.size // window
    frames = signal[: count * window].reshape(count, window)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    return np.nan_to_num(rms, nan=0.0, posinf=0.0, neginf=0.0)


def _local_baseline(energy: np.ndarray,
                    half_windows: int = BASELINE_HALF_WINDOWS) -> np.ndarray:
    """Local noise baseline: median of the surrounding RMS windows."""
    values = np.asarray(energy, dtype=np.float64).ravel()
    count = values.size
    if count == 0:
        return np.zeros(0, dtype=np.float64)
    baseline = np.empty(count, dtype=np.float64)
    for index in range(count):
        lo = max(0, index - half_windows)
        hi = min(count, index + half_windows + 1)
        baseline[index] = float(np.median(values[lo:hi]))
    return np.maximum(baseline, 1e-6)


# ---------------------------------------------------------------------------
# Acoustic event detectors (pure functions over the RMS array)
# ---------------------------------------------------------------------------
def _detect_loud(energy: np.ndarray, baseline: np.ndarray, window_s: float,
                 duration: float,
                 speech_intervals: Sequence[Tuple[float, float]]
                 ) -> List[AudioEvent]:
    """Group windows that are clearly above the local noise baseline.

    Without overlapping speech the burst is background ``noise``; when speech
    words overlap it stays ``loud`` and may later be promoted to a reaction.
    """
    threshold = np.maximum(baseline * LOUD_RATIO, LOUD_MIN_RMS)
    bridge = np.maximum(baseline * LOUD_BRIDGE_RATIO, LOUD_MIN_RMS * 0.5)
    count = energy.size
    events: List[AudioEvent] = []
    index = 0
    while index < count:
        if energy[index] < threshold[index]:
            index += 1
            continue
        start_index = index
        end_index = index
        while end_index < count:
            if energy[end_index] >= threshold[end_index]:
                end_index += 1
                continue
            if (end_index + 1 < count
                    and energy[end_index] >= bridge[end_index]
                    and energy[end_index + 1] >= threshold[end_index + 1]):
                end_index += 1
                continue
            break
        peak_index = start_index + int(
            np.argmax(energy[start_index:end_index]))
        ratio = float(energy[peak_index] / max(float(baseline[peak_index]), 1e-9))
        start = start_index * window_s
        end = min(duration, end_index * window_s)
        speech = _overlaps((start, end), speech_intervals)
        events.append(AudioEvent(
            kind="loud" if speech else "noise",
            start=start,
            end=end,
            intensity=_clamp(ratio, 1.0, INTENSITY_MAX),
            speech_overlap=speech,
        ))
        index = end_index
    return events


def _detect_onsets(energy: np.ndarray, baseline: np.ndarray, window_s: float,
                   duration: float,
                   speech_intervals: Sequence[Tuple[float, float]]
                   ) -> List[AudioEvent]:
    """Detect sharp energy rises (a loud window after a much quieter one)."""
    threshold = np.maximum(baseline * LOUD_RATIO, LOUD_MIN_RMS)
    previous = np.zeros_like(energy)
    if energy.size > 1:
        previous[1:] = energy[:-1]
    rise = energy >= np.maximum(previous * ONSET_RATIO, 1e-6)
    mask = (energy >= threshold) & rise
    count = energy.size
    events: List[AudioEvent] = []
    index = 0
    while index < count:
        if not mask[index]:
            index += 1
            continue
        end_index = index
        while end_index + 1 < count and mask[end_index + 1]:
            end_index += 1
        peak = float(np.max(energy[index:end_index + 1]))
        before = max(float(previous[index]), 1e-6)
        start = index * window_s
        end = min(duration, (end_index + 1) * window_s)
        events.append(AudioEvent(
            kind="onset",
            start=start,
            end=end,
            intensity=_clamp(peak / before, 1.0, INTENSITY_MAX),
            speech_overlap=_overlaps((start, end), speech_intervals),
        ))
        index = end_index + 1
    return events


def _detect_pauses(energy: np.ndarray, window_s: float, duration: float,
                   speech_intervals: Sequence[Tuple[float, float]]
                   ) -> List[AudioEvent]:
    """Detect silence gaps relative to the clip's own loud/speech level."""
    if energy.size == 0:
        return []
    speech_level = float(np.percentile(energy, 90))
    threshold = max(speech_level * PAUSE_RATIO, PAUSE_MIN_RMS)
    quiet = energy <= threshold
    # A window covered by a word timestamp is not a real gap: trim it instead
    # of discarding the whole silence run (word timings can be a bit loose).
    for index in range(energy.size):
        start = index * window_s
        if _overlaps((start, start + window_s), speech_intervals):
            quiet[index] = False
    count = energy.size
    events: List[AudioEvent] = []
    index = 0
    while index < count:
        if not quiet[index]:
            index += 1
            continue
        end_index = index
        while end_index + 1 < count and quiet[end_index + 1]:
            end_index += 1
        start = index * window_s
        end = min(duration, (end_index + 1) * window_s)
        if end - start >= PAUSE_MIN_S:
            events.append(AudioEvent(
                kind="pause",
                start=start,
                end=end,
                intensity=round(end - start, 3),
                speech_overlap=False,
            ))
        index = end_index + 1
    return events


def _detect_acoustic_events(
        energy: np.ndarray,
        window_s: float = WINDOW_S,
        duration: Optional[float] = None,
        speech_intervals: Sequence[Tuple[float, float]] = (),
) -> List[AudioEvent]:
    """Classify an RMS array into loud/noise/onset/pause events.

    Pure function: takes per-window RMS values and speech intervals, returns
    clamped and sorted :class:`AudioEvent` objects (no file I/O).
    """
    values = np.asarray(energy, dtype=np.float64).ravel()
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.maximum(values, 0.0)
    if values.size == 0:
        return []
    step = _finite(window_s, WINDOW_S)
    if step <= 0:
        step = WINDOW_S
    total = _finite(duration, values.size * step)
    if total <= 0:
        total = values.size * step
    if total <= 0:
        return []
    baseline = _local_baseline(values)
    intervals = list(speech_intervals or ())
    events: List[AudioEvent] = []
    events.extend(_detect_loud(values, baseline, step, total, intervals))
    events.extend(_detect_onsets(values, baseline, step, total, intervals))
    events.extend(_detect_pauses(values, step, total, intervals))
    events.sort(key=lambda event: (event.start, event.end, event.kind))
    return events


# ---------------------------------------------------------------------------
# Text classification (laughter / scream / reaction)
# ---------------------------------------------------------------------------
def _normalize_token(word: str) -> str:
    token = str(word or "").strip().lower().replace("ё", "е")
    token = re.sub(r"\s+", "", token)
    return re.sub(r"[^\w\-]", "", token)


def _laughter_strength(word: str) -> int:
    """0 = not laughter, 1 = weak syllable ('ха'), 2 = unambiguous laughter."""
    token = _normalize_token(word)
    if not token:
        return 0
    if _LAUGH_STRONG_RE.match(token):
        return 2
    if _LAUGH_WEAK_RE.match(token):
        return 1
    return 0


def _is_caps_word(word: str) -> bool:
    raw = str(word or "").strip()
    letters = [char for char in raw if char.isalpha()]
    if len(letters) < 3:
        return False
    if _laughter_strength(raw):
        return False
    return all(char.isupper() for char in letters)


def _is_elongated_scream(word: str) -> bool:
    token = _normalize_token(word)
    letters = re.sub(r"[^a-zа-я]", "", token)
    return len(letters) >= 4 and re.search(r"(.)\1{2,}", letters) is not None


def _group_words(words: Sequence[TranscriptWord],
                 max_span: float = TEXT_GROUP_MAX_SPAN,
                 max_gap: float = TEXT_GROUP_MAX_GAP
                 ) -> List[List[TranscriptWord]]:
    groups: List[List[TranscriptWord]] = []
    current: List[TranscriptWord] = []
    for word in words:
        start = _finite(getattr(word, "start", 0.0))
        end = _finite(getattr(word, "end", start))
        if current:
            gap = start - _finite(getattr(current[-1], "end", start))
            span = end - _finite(getattr(current[0], "start", start))
            if gap > max_gap or span > max_span:
                groups.append(current)
                current = []
        current.append(word)
    if current:
        groups.append(current)
    return groups


def _classify_text_group(words: Sequence[TranscriptWord]) -> List[AudioEvent]:
    """Classify one time-adjacent word group into laughter/scream/reaction."""
    if not words:
        return []
    valid = [
        word for word in words
        if _finite(getattr(word, "end", 0.0))
        > _finite(getattr(word, "start", 0.0))
    ]
    if not valid:
        return []
    text = " ".join(str(getattr(word, "word", "") or "") for word in valid)
    strengths = [_laughter_strength(getattr(word, "word", ""))
                 for word in valid]
    laugh_indexes = [i for i, strength in enumerate(strengths) if strength]
    strong = [i for i in laugh_indexes if strengths[i] >= 2]
    events: List[AudioEvent] = []

    if strong or len(laugh_indexes) >= 2:
        chosen = strong if strong else laugh_indexes
        start = _finite(getattr(valid[chosen[0]], "start", 0.0))
        end = _finite(getattr(valid[chosen[-1]], "end", start))
        events.append(AudioEvent(
            kind="laughter",
            start=start,
            end=end,
            intensity=min(2.0, 1.0 + 0.25 * (len(chosen) - 1)),
            speech_overlap=True,
        ))

    exclamations = text.count("!")
    questions = text.count("?")
    caps_words = [word for word in valid
                  if _is_caps_word(getattr(word, "word", ""))]
    caps = len(caps_words)
    elongated = [word for word in valid
                 if _is_elongated_scream(getattr(word, "word", ""))]
    emphasized = [
        word for word in valid
        if "!" in str(getattr(word, "word", "") or "")
        or _is_caps_word(getattr(word, "word", ""))
        or _is_elongated_scream(getattr(word, "word", ""))
    ]

    scream = (exclamations >= 2
              or (exclamations >= 1 and caps >= 1)
              or caps >= 2
              or bool(elongated))
    reaction = False
    if not scream:
        reaction = (exclamations == 1 or questions >= 1 or caps >= 1)
    if scream or reaction:
        chosen = emphasized or valid
        start = min(_finite(getattr(word, "start", 0.0)) for word in chosen)
        end = max(_finite(getattr(word, "end", start)) for word in chosen)
        intensity = min(
            SCREAM_INTENSITY_MAX,
            1.0 + 0.5 * exclamations + 0.5 * caps + 0.25 * questions,
        )
        events.append(AudioEvent(
            kind="scream" if scream else "reaction",
            start=start,
            end=end,
            intensity=max(1.0, intensity),
            speech_overlap=True,
        ))
    return events


def _detect_text_events(words: Sequence[TranscriptWord]) -> List[AudioEvent]:
    events: List[AudioEvent] = []
    for group in _group_words(words):
        events.extend(_classify_text_group(group))
    return events


def _apply_text_emphasis(events: Sequence[AudioEvent], words: Sequence[TranscriptWord]
                         ) -> List[AudioEvent]:
    """Promote speech-overlapping loud events to reaction/scream."""
    if not words:
        return list(events)
    result: List[AudioEvent] = []
    for event in events:
        if event.kind != "loud" or not event.speech_overlap:
            result.append(event)
            continue
        window = (event.start - EMPHASIS_PAD_S, event.end + EMPHASIS_PAD_S)
        nearby = [
            word for word in words
            if _overlap_seconds(
                (_finite(getattr(word, "start", 0.0)),
                 _finite(getattr(word, "end", 0.0))),
                window,
            ) > 0.0
        ]
        text_events = _classify_text_group(nearby)
        scream = next((item for item in text_events
                       if item.kind == "scream"), None)
        reaction = next((item for item in text_events
                         if item.kind in ("reaction", "laughter")), None)
        marker = scream or reaction
        if marker is None:
            result.append(event)
            continue
        result.append(AudioEvent(
            kind="scream" if scream is not None else "reaction",
            start=event.start,
            end=event.end,
            intensity=max(event.intensity, marker.intensity),
            speech_overlap=True,
        ))
    return result


# ---------------------------------------------------------------------------
# Event normalisation / merging
# ---------------------------------------------------------------------------
def _normalize_events(events: Iterable[AudioEvent], duration: float,
                      speech_intervals: Sequence[Tuple[float, float]] = ()
                      ) -> List[AudioEvent]:
    """Clamp, sanitise, de-duplicate and merge events into a sorted timeline."""
    total = max(0.0, _finite(duration, 0.0))
    cleaned: List[AudioEvent] = []
    for event in events:
        kind = str(getattr(event, "kind", "") or "loud").strip().lower()
        if not kind:
            kind = "loud"
        start_value = _finite(getattr(event, "start", 0.0))
        end_value = _finite(getattr(event, "end", 0.0))
        start, end = sorted((start_value, end_value))
        start = _clamp(start, 0.0, total)
        end = _clamp(end, 0.0, total)
        if end - start < MIN_EVENT_S:
            continue
        intensity = _finite(getattr(event, "intensity", 1.0), 1.0)
        if intensity <= 0.0:
            intensity = 1.0
        speech = bool(getattr(event, "speech_overlap", False))
        if kind == "pause":
            speech = False
        elif kind in _TEXT_KINDS:
            speech = True
        else:
            speech = speech or _overlaps((start, end), speech_intervals)
        cleaned.append(AudioEvent(kind=kind, start=start, end=end,
                                  intensity=intensity, speech_overlap=speech))

    # Prefer reaction/scream over the same acoustic burst labelled loud.
    reactive = [item for item in cleaned if item.kind in _REACTIVE_KINDS]
    if reactive:
        def covered_by_reaction(item: AudioEvent) -> bool:
            length = item.end - item.start
            return any(
                _overlap_seconds((item.start, item.end),
                                 (other.start, other.end)) >= 0.5 * length
                for other in reactive
            )
        cleaned = [item for item in cleaned
                   if not (item.kind == "loud" and covered_by_reaction(item))]

    cleaned.sort(key=lambda item: (item.start, item.end, item.kind))
    merged: List[AudioEvent] = []
    for item in cleaned:
        if merged:
            last = merged[-1]
            if last.kind == item.kind and item.start <= last.end + MERGE_GAP_S:
                last.end = min(total, max(last.end, item.end))
                last.intensity = max(last.intensity, item.intensity)
                last.speech_overlap = (last.speech_overlap
                                       or item.speech_overlap)
                continue
        merged.append(AudioEvent(
            kind=item.kind, start=item.start, end=item.end,
            intensity=item.intensity, speech_overlap=item.speech_overlap,
        ))
    return merged


def _build_events(samples: np.ndarray, sample_rate: int, duration: float,
                  words: Sequence[TranscriptWord]) -> List[AudioEvent]:
    intervals = _speech_intervals(words)
    energy = _frame_rms(samples, sample_rate)
    events = _detect_acoustic_events(energy, WINDOW_S, duration, intervals)
    events = _apply_text_emphasis(events, words)
    events.extend(_detect_text_events(words))
    return _normalize_events(events, duration, intervals)


# ---------------------------------------------------------------------------
# Transcript cache
# ---------------------------------------------------------------------------
def _validate_cached_result(result: AudioAnalysisResult) -> None:
    if not math.isfinite(result.duration) or result.duration < 0:
        raise ValueError("invalid duration")
    for item in list(result.segments) + list(result.words):
        start, end = _cache_number(item.start), _cache_number(item.end)
        if start < 0 or end < start:
            raise ValueError("invalid segment/word timestamps")
    for event in result.events:
        start, end = _cache_number(event.start), _cache_number(event.end)
        intensity = _cache_number(event.intensity)
        if not event.kind or not str(event.kind).strip():
            raise ValueError("empty event kind")
        if start < 0 or end < start or end > result.duration + 1e-6:
            raise ValueError("event outside of the clip")
        if not math.isfinite(intensity) or intensity < 0:
            raise ValueError("invalid event intensity")


def _load_cached_transcript(path: Path, fingerprint: str,
                            cfg) -> Optional[AudioAnalysisResult]:
    """Return a cached result only when it is complete and still valid."""
    path = Path(path)
    if not path.is_file():
        return None
    raw = read_json(path, None)
    if not isinstance(raw, dict):
        log.warning("Audio cache %s is unreadable; recomputing", path)
        return None
    if raw.get("version") != CACHE_VERSION:
        log.warning(
            "Audio cache %s has version %r (expected %r); recomputing",
            path, raw.get("version"), CACHE_VERSION)
        return None
    if str(raw.get("source_fingerprint", "")) != str(fingerprint):
        log.warning("Audio cache %s is stale (source file changed); recomputing",
                    path)
        return None
    expected_model = str(getattr(cfg, "whisper_model", ""))
    if str(raw.get("whisper_model", "")) != expected_model:
        log.warning(
            "Audio cache %s was made with model %r, expected %r; recomputing",
            path, raw.get("whisper_model"), expected_model)
        return None
    try:
        if not isinstance(raw.get("segments"), list) \
                or not isinstance(raw.get("words"), list) \
                or not isinstance(raw.get("events"), list):
            raise ValueError("segments/words/events must be lists")
        if not isinstance(raw.get("full_text"), str):
            raise ValueError("full_text must be a string")
        if not isinstance(raw.get("language"), str):
            raise ValueError("language must be a string")
        result = AudioAnalysisResult(
            language=raw.get("language") or "unknown",
            duration=_cache_number(raw["duration"]),
            segments=[
                TranscriptSegment(_cache_number(item["start"]),
                                  _cache_number(item["end"]),
                                  str(item.get("text", "")))
                for item in raw["segments"]
            ],
            words=[
                TranscriptWord(_cache_number(item["start"]),
                               _cache_number(item["end"]),
                               str(item.get("word", "")))
                for item in raw["words"]
            ],
            events=[
                AudioEvent(
                    kind=str(item.get("kind", "loud")),
                    start=_cache_number(item["start"]),
                    end=_cache_number(item["end"]),
                    intensity=_cache_number(item.get("intensity", 1.0)),
                    speech_overlap=bool(item.get("speech_overlap", False)),
                )
                for item in raw["events"]
            ],
            full_text=raw.get("full_text", ""),
            model=str(raw.get("model", expected_model)),
            source_fingerprint=str(fingerprint),
        )
        _validate_cached_result(result)
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        log.warning("Audio cache %s is incomplete or corrupt (%s); recomputing",
                    path, exc)
        return None
    return result


def _cache_payload(result: AudioAnalysisResult, cfg,
                   fingerprint: str) -> Dict[str, Any]:
    return {
        "version": CACHE_VERSION,
        "whisper_model": str(getattr(cfg, "whisper_model", "")),
        "whisper_device": str(getattr(cfg, "whisper_device", "")),
        "whisper_compute": str(getattr(cfg, "whisper_compute", "")),
        "source_fingerprint": str(fingerprint),
        "model": str(result.model),
        "language": str(result.language),
        "duration": float(result.duration),
        "full_text": str(result.full_text),
        "segments": [
            {"start": float(item.start), "end": float(item.end),
             "text": str(item.text)}
            for item in result.segments
        ],
        "words": [
            {"start": float(item.start), "end": float(item.end),
             "word": str(item.word)}
            for item in result.words
        ],
        "events": [
            {"kind": str(item.kind), "start": float(item.start),
             "end": float(item.end), "intensity": float(item.intensity),
             "speech_overlap": bool(item.speech_overlap)}
            for item in result.events
        ],
    }


def _write_cache(path: Path, result: AudioAnalysisResult, cfg,
                 fingerprint: str) -> None:
    try:
        write_json(Path(path), _cache_payload(result, cfg, fingerprint))
    except OSError as exc:
        raise AnalysisError(f"Could not save audio cache {path}: {exc}") from exc


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def analyze_audio(source: Path, work_dir: Path, cfg) -> AudioAnalysisResult:
    """Transcribe and acoustically analyse ``source`` inside ``work_dir``.

    Reuses ``transcript.json`` when the source fingerprint and Whisper model
    match, otherwise re-extracts the WAV and re-runs faster-whisper.
    Raises :class:`AnalysisError` when extraction, model loading or
    transcription fails; an empty transcript is still a valid result.
    """
    source = Path(source)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    if not source.is_file():
        raise AnalysisError(f"Audio source does not exist: {source}")
    try:
        fingerprint = file_fingerprint(source)
    except OSError as exc:
        raise AnalysisError(
            f"Cannot fingerprint audio source {source}: {exc}") from exc

    transcript_path = work_dir / TRANSCRIPT_FILE
    cached = _load_cached_transcript(transcript_path, fingerprint, cfg)
    if cached is not None:
        log.info("Audio cache hit: %.1fs, %d words, %d events, lang=%s",
                 cached.duration, len(cached.words), len(cached.events),
                 cached.language)
        return cached

    wav = _ensure_wav(source, work_dir, fingerprint)
    model_name = str(getattr(cfg, "whisper_model", ""))
    model = _get_whisper(
        model_name,
        str(getattr(cfg, "whisper_device", "cpu")),
        str(getattr(cfg, "whisper_compute", "int8")),
    )
    beam_size = int(getattr(cfg, "whisper_beam_size", 5) or 5)

    segments: List[TranscriptSegment] = []
    words: List[TranscriptWord] = []
    parts: List[str] = []
    language = "unknown"
    duration = 0.0
    try:
        segments_iter, info = model.transcribe(
            str(wav),
            word_timestamps=True,
            vad_filter=True,
            beam_size=beam_size,
        )
        for segment in segments_iter:
            segment_start = _finite(getattr(segment, "start", 0.0))
            segment_end = _finite(getattr(segment, "end", segment_start))
            text = str(getattr(segment, "text", "") or "").strip()
            segments.append(TranscriptSegment(
                float(segment_start), float(segment_end), text))
            if text:
                parts.append(text)
            for word in (getattr(segment, "words", None) or []):
                token = str(getattr(word, "word", "") or "").strip()
                if not token:
                    continue
                word_start = _finite(getattr(word, "start", segment_start))
                word_end = _finite(getattr(word, "end", word_start))
                words.append(TranscriptWord(
                    float(word_start), float(word_end), token))
        if info is not None:
            language = str(getattr(info, "language", "") or "unknown")
            duration = _finite(getattr(info, "duration", 0.0))
    except AnalysisError:
        raise
    except Exception as exc:  # noqa: BLE001 - surface as AnalysisError
        raise AnalysisError(
            f"Whisper transcription failed for {source.name}: {exc}") from exc

    full_text = " ".join(parts).strip()

    try:
        sample_rate, samples = _read_wav(wav)
    except AnalysisError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise AnalysisError(
            f"Could not read extracted audio for {source.name}: {exc}") from exc

    if duration <= 0:
        fallback = (samples.size / sample_rate) if sample_rate > 0 else 0.0
        if segments:
            fallback = max(fallback,
                           max(float(item.end) for item in segments))
        if words:
            fallback = max(fallback, max(float(item.end) for item in words))
        duration = fallback
    duration = max(0.0, float(duration))

    events = _build_events(samples, sample_rate, duration, words)
    result = AudioAnalysisResult(
        language=language,
        duration=duration,
        segments=segments,
        words=words,
        events=events,
        full_text=full_text,
        model=model_name,
        source_fingerprint=fingerprint,
    )
    _write_cache(transcript_path, result, cfg, fingerprint)

    counts: Dict[str, int] = {}
    for event in events:
        counts[event.kind] = counts.get(event.kind, 0) + 1
    log.info("Audio: %.1fs, %d words, %d events %s, lang=%s",
             duration, len(words), len(events), counts, language)
    return result
