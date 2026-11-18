"""Multi-frame vision analysis with Ollama."""
import base64
import hashlib
import json
import subprocess
from pathlib import Path
from typing import List

import requests

from .logging_setup import get_logger
from .models import FrameDescription, VisionAnalysisResult

log = get_logger(__name__)


def _probe_duration(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {r.stderr[:200]}")
    return float(r.stdout.strip())


def _extract_frames(source: Path, out_dir: Path, duration: float,
                    interval: float, max_frames: int) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if duration <= 0:
        return []
    n = min(max_frames, max(1, int(duration // interval)))
    frames: List[Path] = []
    for i in range(n):
        t = min(duration - 0.1, i * interval + interval / 2)
        f = out_dir / f"frame_{i:03d}.jpg"
        if not f.exists():
            r = subprocess.run(
                ["ffmpeg", "-y", "-ss", f"{t:.2f}", "-i", str(source),
                 "-frames:v", "1", "-q:v", "3", "-vf", "scale=640:-1", str(f)],
                capture_output=True,
            )
            if r.returncode != 0:
                continue
        frames.append(f)
    return frames


def _phash(path: Path) -> str:
    try:
        import cv2
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return hashlib.sha1(path.read_bytes()).hexdigest()
        small = cv2.resize(img, (8, 8), interpolation=cv2.INTER_AREA)
        avg = small.mean()
        bits = "".join("1" if v > avg else "0" for v in small.flatten())
        return f"{int(bits, 2):016x}"
    except Exception:
        return hashlib.sha1(path.read_bytes()).hexdigest()


def _dedupe(frames: List[Path], max_dist: int = 4) -> List[Path]:
    kept: List[Path] = []
    seen: List[str] = []
    for f in frames:
        h = _phash(f)
        hb = bin(int(h, 16))[2:].zfill(64)
        dup = False
        for s in seen:
            sb = bin(int(s, 16))[2:].zfill(64)
            if sum(a != b for a, b in zip(hb, sb)) <= max_dist:
                dup = True
                break
        if not dup:
            kept.append(f)
            seen.append(h)
    return kept


def _ollama(model: str, url: str, prompt: str, timeout: float,
            image: Path = None) -> str:
    body = {"model": model, "prompt": prompt, "stream": False}
    if image is not None:
        body["images"] = [base64.b64encode(image.read_bytes()).decode()]
    r = requests.post(f"{url.rstrip('/')}/api/generate", json=body, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"Ollama HTTP {r.status_code}: {r.text[:200]}")
    return (r.json().get("response") or "").strip()


def analyze_vision(source: Path, work_dir: Path, cfg) -> VisionAnalysisResult:
    work_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = work_dir / "frames"
    duration = _probe_duration(source)
    raw = _extract_frames(source, frames_dir, duration,
                          cfg.frame_interval_sec, cfg.max_frames)
    frames = _dedupe(raw)
    log.info("Vision: extracted %d, deduped %d (duration=%.1fs)",
             len(raw), len(frames), duration)

    frame_prompt = (
        "One frame from a video game clip. Describe briefly (1-2 sentences) "
        "what is happening: characters, action, danger, score, UI. "
        "Answer in the language of the game UI if readable, else in English."
    )
    descs: List[FrameDescription] = []
    for i, f in enumerate(frames):
        try:
            d = _ollama(cfg.ollama_vision_model, cfg.ollama_url,
                        frame_prompt, cfg.ollama_timeout, image=f)
        except Exception as e:
            log.warning("Vision frame %d failed: %s", i, e)
            continue
        if d:
            ts = min(duration, i * cfg.frame_interval_sec + cfg.frame_interval_sec / 2)
            descs.append(FrameDescription(ts, d))

    sequence = ""
    if descs:
        timeline = "\n".join(f"[{x.timestamp:.1f}s] {x.description}" for x in descs)
        prompt = (
            "Below are descriptions of frames from a gaming clip in "
            f"chronological order (total {duration:.1f}s):\n\n{timeline}\n\n"
            "Write a short narrative (2-4 sentences) describing what actually "
            "happened: the sequence of events, turning points, and reactions. "
            "Do not describe 'images'. Describe the gameplay as a story. "
            "Answer in the language of the descriptions."
        )
        try:
            sequence = _ollama(cfg.ollama_text_model, cfg.ollama_url,
                               prompt, cfg.ollama_timeout)
        except Exception as e:
            log.warning("Sequence summarization failed: %s", e)

    result = VisionAnalysisResult(duration, descs, sequence)
    (work_dir / "vision.json").write_text(
        json.dumps({
            "duration": duration,
            "frames": [{"timestamp": d.timestamp, "description": d.description}
                       for d in descs],
            "sequence_summary": sequence,
        }, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return result
