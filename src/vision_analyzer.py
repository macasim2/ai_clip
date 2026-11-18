"""Multi-frame vision analysis with Ollama.

The stage probes the clip duration, plans a uniform timestamp grid that covers
the whole clip (plus sharp scene changes), extracts JPEG frames with FFmpeg,
drops near-identical frames with a dHash computed by FFmpeg, describes each
kept frame with an Ollama vision model and finally asks a text model for a
short causal narrative. Results are cached in ``work_dir/vision.json``.

No OpenCV: every image operation is delegated to FFmpeg.
"""
from __future__ import annotations

import base64
import hashlib
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import (Any, Dict, Iterable, List, Mapping, Optional, Sequence,
                    Tuple)

import requests

from .logging_setup import get_logger
from .models import AnalysisError, FrameDescription, VisionAnalysisResult
from .utils import (file_fingerprint, find_tool, probe_media, read_json,
                    write_json)

log = get_logger(__name__)

CACHE_VERSION = 2
CACHE_NAME = "vision.json"
FRAMES_DIRNAME = "frames"

#: A dHash of a 9x8 grayscale frame yields (9 - 1) * 8 = 64 bits.
DHASH_WIDTH = 9
DHASH_HEIGHT = 8
DHASH_LENGTH = DHASH_WIDTH * DHASH_HEIGHT
DEFAULT_DEDUPE_HAMMING = 5
SCENE_MERGE_GAP_SEC = 0.25
RETRY_ATTEMPTS = 2
RETRY_DELAYS = (1.0, 2.0)
FFMPEG_TIMEOUT_SEC = 120.0

FRAME_PROMPT = (
    "You are analysing one frame from a video-game clip. In 1-2 sentences, "
    "describe what is happening: the action, the characters, any danger or "
    "threat, the score/health or other UI state, and any readable on-screen "
    "text. Describe the scene itself, not the image or the frame. "
    "Answer in English."
)

BATCH_PROMPT = (
    "You are analysing {count} frames from a video-game clip, given in "
    "chronological order. For every frame write exactly one numbered line "
    "(1-{count}) with a 1-2 sentence description of the action, characters, "
    "danger, score/health/UI and readable on-screen text. Answer in English "
    "with no extra commentary."
)

SUMMARY_PROMPT = (
    "Below are descriptions of frames sampled from a {duration:.1f}-second "
    "video-game clip, in chronological order:\n\n{timeline}\n\n"
    "Write a 2-4 sentence narrative of what actually happened in the gameplay "
    "as a causal story: the sequence of events, the turning points and the "
    "outcome. Do not mention frames, images or descriptions, and do not "
    "invent events the timeline does not support. Answer in English."
)

_SHOWINFO_PTS_RE = re.compile(r"pts_time:\s*(-?\d+(?:\.\d+)?)")
_BATCH_LINE_RE = re.compile(
    r"^\s*(?:frame\s*)?(\d+)\s*[.):\-]\s*(.+)$", re.IGNORECASE)


@dataclass(frozen=True)
class SamplingPoint:
    """A planned frame timestamp, flagged when it is a scene change."""

    timestamp: float
    scene: bool = False


# --------------------------------------------------------------------- sampling
def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def plan_grid_times(duration: float, interval: float, min_frames: int,
                    max_frames: int) -> List[float]:
    """Return a uniform timestamp grid covering the whole clip.

    The requested interval is only the starting point: the step grows when the
    clip would otherwise exceed ``max_frames`` (so the last sample stays near
    the end) and shrinks when a short clip would not reach ``min_frames``.
    Samples are segment centres, so the first frame is half a step in and the
    last is half a step before the end.
    """
    try:
        duration = float(duration)
    except (TypeError, ValueError):
        return []
    if duration <= 0:
        return []
    max_frames = max(1, _as_int(max_frames, 1))
    min_frames = max(1, min(_as_int(min_frames, 1), max_frames))
    try:
        interval = float(interval)
    except (TypeError, ValueError):
        interval = 1.0
    if interval <= 0:
        interval = 1.0

    count = max(min_frames, int(round(duration / interval)))
    count = max(1, min(count, max_frames))
    step = duration / count
    limit = max(0.0, duration - 0.001)
    times: List[float] = []
    for index in range(count):
        timestamp = _clamp((index + 0.5) * step, 0.0, limit)
        times.append(round(timestamp, 3))
    return times


def _normalize_times(values: Iterable[Any], duration: float) -> List[float]:
    """Coerce, clamp and sort candidate timestamps, dropping invalid ones."""
    duration = max(0.0, float(duration or 0.0))
    normalized: List[float] = []
    for value in values or []:
        try:
            timestamp = float(value)
        except (TypeError, ValueError):
            continue
        if timestamp < 0:
            continue
        if duration > 0:
            timestamp = min(timestamp, max(0.0, duration - 0.05))
        normalized.append(round(timestamp, 3))
    return sorted(set(normalized))


def _evenly_pick(items: Sequence[Any], count: int) -> List[Any]:
    """Pick ``count`` items spread evenly across ``items``."""
    values = list(items)
    if count <= 0 or not values:
        return []
    if count >= len(values):
        return values
    if count == 1:
        return [values[len(values) // 2]]
    indices: List[int] = []
    for position in range(count):
        index = int(round(position * (len(values) - 1) / (count - 1)))
        if index not in indices:
            indices.append(index)
    return [values[index] for index in indices]


def merge_sampling_plan(grid_times: Iterable[Any], scene_times: Iterable[Any],
                        duration: float, max_frames: int,
                        min_frames: int) -> List[SamplingPoint]:
    """Merge the uniform grid with detected scene changes.

    Scene changes have priority: they are kept (up to ``max_frames``), grid
    samples too close to one are dropped, the total never exceeds
    ``max_frames`` and at least ``min_frames`` samples are requested whenever
    the clip is long enough to provide them.
    """
    duration = max(0.0, float(duration or 0.0))
    max_frames = max(1, _as_int(max_frames, 1))
    min_frames = max(1, min(_as_int(min_frames, 1), max_frames))

    scenes: List[float] = []
    for timestamp in _normalize_times(scene_times, duration):
        if all(abs(timestamp - other) > SCENE_MERGE_GAP_SEC
               for other in scenes):
            scenes.append(timestamp)
    if len(scenes) > max_frames:
        scenes = _evenly_pick(scenes, max_frames)

    points = [SamplingPoint(timestamp, scene=True) for timestamp in scenes]
    remaining = max_frames - len(points)

    grid = [timestamp for timestamp in _normalize_times(grid_times, duration)
            if all(abs(timestamp - point.timestamp) > SCENE_MERGE_GAP_SEC
                   for point in points)]
    if remaining > 0:
        picked = (grid if len(grid) <= remaining
                  else _evenly_pick(grid, remaining))
        points.extend(SamplingPoint(timestamp) for timestamp in picked)

    # Degenerate clips may leave fewer than min_frames candidates above; fill
    # from the grid even if a sample is close to a scene change.
    if len(points) < min_frames:
        needed = min_frames - len(points)
        extras = [timestamp for timestamp in _evenly_pick(grid, max_frames)
                  if all(abs(timestamp - point.timestamp) > 0.001
                         for point in points)]
        points.extend(SamplingPoint(timestamp) for timestamp in extras[:needed])

    points.sort(key=lambda point: point.timestamp)
    merged: List[SamplingPoint] = []
    for point in points:
        if merged and (abs(point.timestamp - merged[-1].timestamp)
                       <= SCENE_MERGE_GAP_SEC):
            if point.scene and not merged[-1].scene:
                merged[-1] = point
            continue
        merged.append(point)
    return merged


# -------------------------------------------------------------------- extraction
def parse_showinfo_pts(output: str) -> List[float]:
    """Extract unique, sorted ``pts_time`` values from FFmpeg showinfo output."""
    if not output:
        return []
    timestamps = []
    for match in _SHOWINFO_PTS_RE.finditer(output):
        try:
            value = float(match.group(1))
        except ValueError:
            continue
        if value >= 0:
            timestamps.append(round(value, 3))
    return sorted(set(timestamps))


def detect_scene_changes(source: Path, threshold: float,
                         duration: float = 0.0) -> List[float]:
    """Detect sharp cuts with FFmpeg's scene score filter.

    Returns an empty list and logs a warning when FFmpeg fails or the
    threshold is unusable; callers must continue with the uniform grid.
    """
    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        threshold = 0.0
    if not 0.0 < threshold <= 1.0:
        log.warning("Scene detection disabled: invalid threshold %r", threshold)
        return []
    try:
        ffmpeg = find_tool("ffmpeg")
    except Exception as exc:  # noqa: BLE001 - optional enhancement only
        log.warning("Scene detection skipped: %s", exc)
        return []

    timeout = max(60.0, min(900.0, float(duration or 0.0) * 4.0))
    command = [
        ffmpeg, "-hide_banner", "-nostats", "-i", str(source),
        "-filter:v", f"select='gt(scene,{threshold:.4f})',showinfo",
        "-an", "-f", "null", "-",
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("Scene detection failed for %s: %s", source, exc)
        return []
    if completed.returncode != 0:
        tail = (completed.stderr or "").strip()[-300:]
        log.warning("Scene detection failed for %s: %s", source, tail)
        return []

    scene_times = parse_showinfo_pts(completed.stderr or "")
    if duration and duration > 0:
        scene_times = [timestamp for timestamp in scene_times
                       if timestamp <= duration + 0.25]
    log.info("Scene detection: %d change(s) above %.2f in %s",
             len(scene_times), threshold, source.name)
    return scene_times


def extract_frame(source: Path, out_path: Path, timestamp: float,
                  width: int) -> bool:
    """Extract a single JPEG frame at ``timestamp``; return False on failure."""
    out_path = Path(out_path)
    try:
        ffmpeg = find_tool("ffmpeg")
    except Exception as exc:  # noqa: BLE001 - reported to the caller
        log.warning("Frame extraction skipped: %s", exc)
        return False
    try:
        timestamp = max(0.0, float(timestamp))
    except (TypeError, ValueError):
        return False
    width = max(16, _as_int(width, 768))
    if width % 2:
        width -= 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{timestamp:.3f}", "-i", str(source),
        "-frames:v", "1", "-an", "-vf", f"scale={width}:-2",
        "-q:v", "3", str(out_path),
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=FFMPEG_TIMEOUT_SEC,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("Frame extraction at %.2fs failed: %s", timestamp, exc)
        return False
    if (completed.returncode != 0 or not out_path.is_file()
            or out_path.stat().st_size == 0):
        tail = (completed.stderr or "").strip()[-200:]
        log.warning("Frame extraction at %.2fs failed: %s", timestamp, tail)
        try:
            out_path.unlink(missing_ok=True)
        except OSError:
            pass
        return False
    return True


def extract_frames(source: Path, frames_dir: Path,
                   plan: Sequence[SamplingPoint],
                   width: int) -> Dict[Path, float]:
    """Extract every planned frame; return an ordered mapping path -> timestamp.

    The timestamp is stored next to the path, so later filtering stages never
    have to recompute it from a frame index.
    """
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_times: Dict[Path, float] = {}
    for index, point in enumerate(plan):
        kind = "scene" if point.scene else "grid"
        name = (f"{kind}_{index:03d}_"
                f"{int(round(point.timestamp * 1000)):07d}.jpg")
        path = frames_dir / name
        if extract_frame(source, path, point.timestamp, width):
            frame_times[path] = float(point.timestamp)
    return frame_times


def _clean_frames_dir(frames_dir: Path) -> None:
    """Remove stale JPEGs from a previous run before re-extracting frames."""
    frames_dir = Path(frames_dir)
    if not frames_dir.is_dir():
        return
    for path in frames_dir.glob("*.jpg"):
        try:
            path.unlink()
        except OSError as exc:
            log.warning("Could not remove stale frame %s: %s", path, exc)


# ------------------------------------------------------------------ deduplication
def dhash_from_gray_bytes(data: bytes, width: int = DHASH_WIDTH,
                          height: int = DHASH_HEIGHT) -> str:
    """64-bit difference hash from a ``width x height`` 8-bit gray frame."""
    needed = width * height
    if len(data) < needed:
        raise ValueError(
            f"dHash needs {needed} grayscale bytes, got {len(data)}")
    bits = 0
    for row in range(height):
        offset = row * width
        for column in range(width - 1):
            left = data[offset + column]
            right = data[offset + column + 1]
            bits = (bits << 1) | (1 if left > right else 0)
    return f"{bits:016x}"


def dhash_hamming(first: str, second: str) -> int:
    """Hamming distance between two hex-encoded dHashes."""
    try:
        return bin(int(first, 16) ^ int(second, 16)).count("1")
    except (TypeError, ValueError):
        return 64


def _fallback_hash(path: Path) -> str:
    try:
        return hashlib.sha1(path.read_bytes()).hexdigest()
    except OSError:
        return hashlib.sha1(str(path).encode("utf-8")).hexdigest()


def frame_dhash(path: Path) -> str:
    """dHash a JPEG/PNG frame using FFmpeg (no OpenCV dependency)."""
    path = Path(path)
    try:
        ffmpeg = find_tool("ffmpeg")
    except Exception as exc:  # noqa: BLE001 - degrade to a unique hash
        log.warning("dHash of %s skipped: %s", path.name, exc)
        return _fallback_hash(path)
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(path),
        "-frames:v", "1",
        "-vf", f"scale={DHASH_WIDTH}:{DHASH_HEIGHT},format=gray",
        "-f", "rawvideo", "-",
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, timeout=FFMPEG_TIMEOUT_SEC)
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("dHash of %s failed, treating it as unique: %s",
                    path.name, exc)
        return _fallback_hash(path)
    if completed.returncode != 0 or len(completed.stdout) < DHASH_LENGTH:
        log.warning("dHash of %s failed, treating it as unique", path.name)
        return _fallback_hash(path)
    return dhash_from_gray_bytes(completed.stdout[:DHASH_LENGTH])


def dedupe_frames(frame_times: Mapping[Path, float],
                  hashes: Optional[Mapping[Path, str]] = None,
                  max_hamming: int = DEFAULT_DEDUPE_HAMMING,
                  always_keep: Iterable[Path] = ()) -> List[Tuple[Path, float]]:
    """Drop near-identical frames, preserving each kept frame's timestamp.

    ``frame_times`` maps every extracted frame to the timestamp it was taken
    at. The timestamp always travels with the path: it is never recomputed
    from the frame index after filtering (that was the old bug). Frames listed
    in ``always_keep`` (scene changes) are never dropped as duplicates.
    """
    forced = {Path(path) for path in always_keep}
    ordered = sorted(
        ((Path(path), float(timestamp)) for path, timestamp in frame_times.items()),
        key=lambda item: item[1],
    )
    max_hamming = max(0, _as_int(max_hamming, DEFAULT_DEDUPE_HAMMING))
    kept: List[Tuple[Path, float]] = []
    kept_hashes: List[str] = []
    for path, timestamp in ordered:
        digest: Optional[str] = None
        if hashes is not None:
            digest = hashes.get(path)
        if digest is None:
            digest = frame_dhash(path)
        if path not in forced and any(
                dhash_hamming(digest, other) <= max_hamming
                for other in kept_hashes):
            log.debug("Vision: dropping near-duplicate frame at %.2fs (%s)",
                      timestamp, path.name)
            continue
        kept.append((path, timestamp))
        kept_hashes.append(digest)
    return kept


# ----------------------------------------------------------------------- Ollama
def _response_body(response: Any) -> str:
    text = getattr(response, "text", "") or ""
    if not text:
        try:
            text = str(response.json())
        except (ValueError, TypeError):
            text = ""
    return str(text).strip()[:300]


def ollama_generate(cfg: Any, model: str, prompt: str,
                    images: Sequence[bytes] = ()) -> str:
    """POST one ``/api/generate`` request, retrying transient failures.

    Timeouts, connection errors and 5xx responses are retried up to
    ``RETRY_ATTEMPTS`` times; a 4xx or an invalid payload raises immediately.
    """
    endpoint = f"{str(cfg.ollama_url).rstrip('/')}/api/generate"
    payload: Dict[str, Any] = {
        "model": str(model),
        "prompt": prompt,
        "stream": False,
        "keep_alive": getattr(cfg, "ollama_keep_alive", "2m"),
        "options": {
            "temperature": 0.2,
            "num_ctx": _as_int(getattr(cfg, "ollama_num_ctx", 4096), 4096),
        },
    }
    if images:
        payload["images"] = [
            base64.b64encode(image).decode("ascii") for image in images
        ]
    timeout = float(getattr(cfg, "ollama_timeout", 240.0))
    last_error = "unknown error"
    for attempt in range(RETRY_ATTEMPTS + 1):
        try:
            response = requests.post(endpoint, json=payload, timeout=timeout)
        except requests.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        else:
            status = _as_int(getattr(response, "status_code", 0), 0)
            if status == 200:
                try:
                    data = response.json()
                except (ValueError, TypeError):
                    raise AnalysisError(
                        f"Ollama returned invalid JSON from {endpoint}"
                    ) from None
                text = (str(data.get("response") or "").strip()
                        if isinstance(data, dict) else "")
                if text:
                    return text
                last_error = "empty response"
            elif status >= 500:
                last_error = f"HTTP {status}: {_response_body(response)}"
            else:
                raise AnalysisError(
                    f"Ollama HTTP {status} from {endpoint} for model "
                    f"{model!r}: {_response_body(response)}")
        if attempt < RETRY_ATTEMPTS:
            delay = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
            log.warning(
                "Ollama request failed (attempt %d/%d): %s; retrying in %.1fs",
                attempt + 1, RETRY_ATTEMPTS + 1, last_error, delay)
            time.sleep(delay)
    raise AnalysisError(
        f"Ollama request to {endpoint} failed after {RETRY_ATTEMPTS + 1} "
        f"attempts (model {model!r}): {last_error}")


def _describe_one(cfg: Any, model: str, path: Path) -> str:
    try:
        image = Path(path).read_bytes()
    except OSError as exc:
        raise AnalysisError(f"Vision frame is unreadable: {path}: {exc}") from exc
    return ollama_generate(cfg, model, FRAME_PROMPT, images=(image,))


def _split_batch_response(text: str, expected: int) -> Optional[List[str]]:
    """Split a multi-image answer into one description per image, if possible."""
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if not lines or expected <= 0:
        return None
    numbered: List[str] = []
    for line in lines:
        match = _BATCH_LINE_RE.match(line)
        if match:
            numbered.append(match.group(2).strip())
    if len(numbered) == expected:
        return numbered
    if len(lines) == expected:
        return lines
    return None


def _describe_chunk(cfg: Any, model: str,
                    chunk: Sequence[Tuple[Path, float]]) -> Optional[List[str]]:
    images: List[bytes] = []
    try:
        for path, _timestamp in chunk:
            images.append(Path(path).read_bytes())
    except OSError as exc:
        log.warning("Vision batch images unreadable: %s", exc)
        return None
    prompt = BATCH_PROMPT.format(count=len(images))
    try:
        text = ollama_generate(cfg, model, prompt, images=images)
    except AnalysisError as exc:
        log.warning("Vision batch request failed: %s", exc)
        return None
    return _split_batch_response(text, len(images))


def describe_frames(frames: Sequence[Tuple[Path, float]],
                    cfg: Any) -> List[FrameDescription]:
    """Describe the kept frames with the Ollama vision model.

    ``vision_batch_size == 1`` uses one image per request. Larger batches send
    several images (the answer must contain one line per image) and fall back
    to one-frame requests if the server or the answer does not support that.
    """
    entries = [(Path(path), float(timestamp)) for path, timestamp in frames]
    if not entries:
        return []
    model = str(cfg.ollama_vision_model)
    batch_size = max(1, min(_as_int(getattr(cfg, "vision_batch_size", 1), 1), 4))
    descriptions: List[FrameDescription] = []
    failed = 0
    last_error = ""

    def attempt(path: Path, timestamp: float) -> None:
        nonlocal failed, last_error
        try:
            text = _describe_one(cfg, model, path)
        except AnalysisError as exc:
            failed += 1
            last_error = str(exc)
            log.warning("Vision frame at %.2fs failed: %s", timestamp, exc)
            return
        descriptions.append(FrameDescription(timestamp, text, str(path)))

    if batch_size == 1:
        for path, timestamp in entries:
            attempt(path, timestamp)
    else:
        for start in range(0, len(entries), batch_size):
            chunk = entries[start:start + batch_size]
            texts = _describe_chunk(cfg, model, chunk)
            if texts is not None:
                for (path, timestamp), text in zip(chunk, texts):
                    if text:
                        descriptions.append(
                            FrameDescription(timestamp, text, str(path)))
                    else:
                        failed += 1
                continue
            log.warning(
                "Vision batch request unusable for %d frame(s); falling back "
                "to one-frame requests", len(chunk))
            for path, timestamp in chunk:
                attempt(path, timestamp)

    descriptions.sort(key=lambda frame: frame.timestamp)
    if not descriptions:
        raise AnalysisError(
            f"Ollama vision model {model!r} at {cfg.ollama_url} could not "
            f"describe any of {len(entries)} extracted frame(s) "
            f"(last error: {last_error or 'no response'})")
    if failed:
        log.warning("Vision: %d frame request(s) failed, %d described",
                    failed, len(descriptions))
    return descriptions


def summarize_sequence(frames: Sequence[FrameDescription], duration: float,
                       cfg: Any) -> str:
    """Ask the text model for a causal narrative of the frame timeline.

    A summary failure is logged as an error and returns an empty string: the
    frame descriptions are still usable and no narrative is invented.
    """
    if not frames:
        return ""
    timeline = "\n".join(
        f"[{frame.timestamp:.2f}s] {frame.description}" for frame in frames)
    prompt = SUMMARY_PROMPT.format(
        duration=float(duration or 0.0), timeline=timeline)
    model = str(cfg.ollama_text_model)
    try:
        return ollama_generate(cfg, model, prompt)
    except AnalysisError as exc:
        log.error(
            "Vision sequence summary failed (model=%s): %s. Returning frame "
            "descriptions without a narrative.", model, exc)
        return ""
    except Exception as exc:  # noqa: BLE001 - never fail analysis on a summary
        log.error(
            "Vision sequence summary crashed (model=%s): %s. Returning frame "
            "descriptions without a narrative.", model, exc)
        return ""


# ------------------------------------------------------------------------ cache
def load_cached_result(work_dir: Path, fingerprint: str,
                       cfg: Any) -> Optional[VisionAnalysisResult]:
    """Load ``vision.json`` when version, source fingerprint and model match."""
    path = Path(work_dir) / CACHE_NAME
    data = read_json(path)
    if not isinstance(data, dict):
        if path.is_file():
            log.warning("Vision cache %s is unreadable; recomputing", path)
        return None
    if data.get("version") != CACHE_VERSION:
        log.info("Vision cache version mismatch (%r); recomputing",
                 data.get("version"))
        return None
    if str(data.get("source_fingerprint") or "") != str(fingerprint):
        log.info("Vision cache source fingerprint changed; recomputing")
        return None
    if str(data.get("model") or "") != str(cfg.ollama_vision_model):
        log.info("Vision cache model changed (%r -> %r); recomputing",
                 data.get("model"), cfg.ollama_vision_model)
        return None
    if str(data.get("text_model") or "") != str(cfg.ollama_text_model):
        log.info("Vision cache summary model changed (%r -> %r); recomputing",
                 data.get("text_model"), cfg.ollama_text_model)
        return None
    try:
        duration = float(data.get("duration", 0.0))
        raw_frames = data.get("frames")
        if not isinstance(raw_frames, list):
            raise ValueError("frames is not a list")
        frames: List[FrameDescription] = []
        for item in raw_frames:
            if not isinstance(item, dict):
                raise ValueError("frame entry is not an object")
            description = str(item.get("description") or "").strip()
            if not description:
                raise ValueError("frame entry has no description")
            frames.append(FrameDescription(
                timestamp=float(item.get("timestamp", 0.0)),
                description=description,
                path=str(item.get("path") or ""),
            ))
        scene_changes = [round(float(value), 3)
                         for value in (data.get("scene_changes") or [])]
        summary = str(data.get("sequence_summary") or "")
    except (TypeError, ValueError) as exc:
        log.warning("Vision cache %s is corrupt (%s); recomputing", path, exc)
        return None
    if not frames:
        log.warning("Vision cache %s contains no frames; recomputing", path)
        return None
    return VisionAnalysisResult(
        duration=duration,
        frames=frames,
        sequence_summary=summary,
        model=str(cfg.ollama_vision_model),
        scene_changes=scene_changes,
        source_fingerprint=str(fingerprint),
    )


def save_cached_result(work_dir: Path, fingerprint: str,
                       result: VisionAnalysisResult,
                       text_model: str = "") -> None:
    """Atomically write ``vision.json``; a cache write never fails analysis."""
    payload = {
        "version": CACHE_VERSION,
        "model": result.model,
        "text_model": str(text_model or ""),
        "source_fingerprint": result.source_fingerprint or str(fingerprint),
        "duration": result.duration,
        "frames": [
            {
                "timestamp": frame.timestamp,
                "description": frame.description,
                "path": frame.path,
            }
            for frame in result.frames
        ],
        "scene_changes": list(result.scene_changes),
        "sequence_summary": result.sequence_summary,
    }
    try:
        write_json(Path(work_dir) / CACHE_NAME, payload)
    except OSError as exc:
        log.warning("Could not write vision cache %s: %s", CACHE_NAME, exc)


# ------------------------------------------------------------------- entry point
def probe_duration(source: Path) -> float:
    """Return the clip duration in seconds; raise AnalysisError on failure."""
    try:
        info = probe_media(Path(source))
    except Exception as exc:  # noqa: BLE001 - MediaError, ToolNotFound, OSError
        raise AnalysisError(
            f"Vision: could not probe {source}: {exc}") from exc
    if info.duration <= 0:
        raise AnalysisError(
            f"Vision: could not determine duration of {source}")
    return float(info.duration)


def analyze_vision(source: Path, work_dir: Path, cfg: Any) -> VisionAnalysisResult:
    """Analyse a clip's frames with Ollama and cache the result.

    Covers the whole clip with a uniform grid plus scene changes, dedupes the
    frames, describes them, asks for a narrative and writes
    ``work_dir/vision.json``. A valid cache is returned without touching FFmpeg
    or Ollama.
    """
    source = Path(source)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    try:
        fingerprint = file_fingerprint(source)
    except OSError as exc:
        raise AnalysisError(
            f"Vision: cannot read source {source}: {exc}") from exc

    cached = load_cached_result(work_dir, fingerprint, cfg)
    if cached is not None:
        log.info("Vision: reusing cached analysis (%d frame(s), model=%s)",
                 len(cached.frames), cached.model)
        return cached

    duration = probe_duration(source)
    frames_dir = work_dir / FRAMES_DIRNAME
    _clean_frames_dir(frames_dir)

    grid_times = plan_grid_times(
        duration,
        getattr(cfg, "frame_interval_sec", 1.5),
        getattr(cfg, "min_frames", 4),
        getattr(cfg, "max_frames", 20),
    )
    scene_times = detect_scene_changes(
        source, getattr(cfg, "scene_threshold", 0.3), duration)
    plan = merge_sampling_plan(
        grid_times, scene_times, duration,
        getattr(cfg, "max_frames", 20),
        getattr(cfg, "min_frames", 4),
    )
    log.info("Vision: duration=%.1fs, grid=%d, scene_changes=%d, planned=%d",
             duration, len(grid_times), len(scene_times), len(plan))

    frame_times = extract_frames(
        source, frames_dir, plan, getattr(cfg, "vision_frame_width", 768))
    if not frame_times:
        raise AnalysisError(
            f"Vision: FFmpeg could not extract any frame from {source} "
            f"(duration={duration:.1f}s, planned={len(plan)})")

    scene_timestamps = {round(point.timestamp, 3)
                        for point in plan if point.scene}
    hashes = {path: frame_dhash(path) for path in frame_times}
    forced = [path for path, timestamp in frame_times.items()
              if round(timestamp, 3) in scene_timestamps]
    kept = dedupe_frames(frame_times, hashes, always_keep=forced)
    log.info("Vision: extracted %d frame(s), kept %d after dedupe",
             len(frame_times), len(kept))

    descriptions = describe_frames(kept, cfg)
    sequence_summary = summarize_sequence(descriptions, duration, cfg)

    result = VisionAnalysisResult(
        duration=duration,
        frames=descriptions,
        sequence_summary=sequence_summary,
        model=str(cfg.ollama_vision_model),
        scene_changes=list(scene_times),
        source_fingerprint=fingerprint,
    )
    save_cached_result(work_dir, fingerprint, result,
                       getattr(cfg, "ollama_text_model", ""))
    log.info("Vision: %d frame(s) described, narrative=%s",
             len(descriptions), "yes" if sequence_summary else "no")
    return result
