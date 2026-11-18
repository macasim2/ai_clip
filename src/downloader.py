"""Twitch clip downloader: explicit clip id, verified output, retries."""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

from .logging_setup import get_logger
from .models import DownloadResult
from .utils import (MediaError, file_fingerprint, free_space_mb, probe_media,
                    validate_clip_id)

log = get_logger(__name__)


class DownloadError(RuntimeError):
    pass


def canonical_path(raw_dir: Path, clip_id: str) -> Path:
    return Path(raw_dir) / f"{validate_clip_id(clip_id)}_source.mp4"


def download_clip(clip: Any, raw_dir: Path, attempts: int = 3,
                  min_free_mb: int = 1024) -> Path:
    """Backward-compatible wrapper used by the existing pipeline."""
    return download(clip, raw_dir, attempts=attempts,
                    min_free_mb=min_free_mb).path


def download(clip: Any, raw_dir: Path, attempts: int = 3,
             min_free_mb: int = 1024) -> DownloadResult:
    """Download a clip by its Twitch id and URL, returning a verified file."""
    if isinstance(clip, dict):
        get = clip.get
    else:
        def get(name, default=None):
            return getattr(clip, name, default)

    clip_id = str(get("id", "") or "").strip()
    validate_clip_id(clip_id)
    url = str(get("url", "") or "").strip()
    if not url:
        raise DownloadError(f"Clip {clip_id} has no URL in the database")

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    target = canonical_path(raw_dir, clip_id)

    existing = _existing_valid(get("local_path"), target)
    if existing is not None:
        info = probe_media(existing)
        log.info("Clip %s already downloaded: %s (%.1fs)",
                 clip_id, existing.name, info.duration)
        return DownloadResult(path=existing, duration=info.duration,
                              size=existing.stat().st_size, skipped=True)

    if importlib.util.find_spec("yt_dlp") is None:
        raise DownloadError(
            "yt-dlp Python module is not installed. Run: "
            "python -m pip install -r requirements.txt"
        )
    if free_space_mb(raw_dir) < max(0, min_free_mb):
        raise DownloadError(
            f"Not enough free disk space in {raw_dir}: "
            f"{free_space_mb(raw_dir):.0f} MB free, required {min_free_mb} MB"
        )

    attempts = max(1, int(attempts or 1))
    last_error = "unknown error"
    tmp_root = raw_dir / ".tmp"
    for attempt in range(1, attempts + 1):
        # Another process may have completed the same clip while we waited.
        existing = _existing_valid(None, target)
        if existing is not None:
            info = probe_media(existing)
            return DownloadResult(path=existing, duration=info.duration,
                                  size=existing.stat().st_size, skipped=True)

        tmp_dir = tmp_root / clip_id
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        template = tmp_dir / "download.%(ext)s"
        log.info("Downloading clip %s (attempt %d/%d)", clip_id, attempt, attempts)
        try:
            result = _run_ytdlp(url, template, timeout=420)
        except FileNotFoundError as exc:
            raise DownloadError(
                "yt-dlp executable/module not found. Run: "
                "python -m pip install -r requirements.txt"
            ) from exc
        except subprocess.TimeoutExpired:
            last_error = "yt-dlp timeout"
            shutil.rmtree(tmp_dir, ignore_errors=True)
            _backoff(attempt)
            continue

        if result.returncode != 0:
            stderr = (result.stderr or "").strip()[-800:]
            last_error = f"yt-dlp exited with {result.returncode}: {stderr}"
            log.warning("Clip %s download failed: %s", clip_id, last_error)
            shutil.rmtree(tmp_dir, ignore_errors=True)
            _backoff(attempt)
            continue

        downloaded = _find_download(tmp_dir)
        if downloaded is None:
            last_error = "yt-dlp reported success but produced no file"
            shutil.rmtree(tmp_dir, ignore_errors=True)
            _backoff(attempt)
            continue

        size = downloaded.stat().st_size
        if size < 1024:
            last_error = f"downloaded file is too small ({size} bytes)"
            shutil.rmtree(tmp_dir, ignore_errors=True)
            _backoff(attempt)
            continue
        try:
            info = probe_media(downloaded)
        except MediaError as exc:
            last_error = f"downloaded file is not a readable video: {exc}"
            shutil.rmtree(tmp_dir, ignore_errors=True)
            _backoff(attempt)
            continue
        if not info.is_valid_video:
            last_error = "downloaded file has no readable video stream"
            shutil.rmtree(tmp_dir, ignore_errors=True)
            _backoff(attempt)
            continue

        if target.exists():
            try:
                target.unlink()
            except OSError:
                pass
        downloaded.replace(target)
        shutil.rmtree(tmp_dir, ignore_errors=True)
        log.info("Saved clip %s -> %s (%.1fs, %d bytes, fingerprint=%s)",
                 clip_id, target.name, info.duration, size,
                 file_fingerprint(target))
        return DownloadResult(path=target, duration=info.duration,
                              size=size, skipped=False)

    shutil.rmtree(tmp_root / clip_id, ignore_errors=True)
    raise DownloadError(f"Failed to download clip {clip_id}: {last_error}")


def _run_ytdlp(url: str, out_template: Path, timeout: int = 420):
    """Run yt-dlp from the active Python environment (venv-safe)."""
    command = [
        sys.executable, "-m", "yt_dlp",
        "--no-playlist",
        "--no-progress",
        "--newline",
        "--retries", "5",
        "--fragment-retries", "5",
        "--socket-timeout", "30",
        "-f", "mp4/best",
        "--merge-output-format", "mp4",
        "-o", str(out_template),
        url,
    ]
    return subprocess.run(
        command, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=timeout,
    )


def _find_download(tmp_dir: Path) -> Optional[Path]:
    candidates = [
        path for path in tmp_dir.glob("download.*")
        if path.is_file() and not path.name.endswith(".part")
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_size)


def _existing_valid(local_path: Any, canonical: Path) -> Optional[Path]:
    candidates = []
    if local_path:
        candidates.append(Path(str(local_path)))
    candidates.append(canonical)
    for path in candidates:
        try:
            if path.is_file() and path.stat().st_size > 1024:
                info = probe_media(path)
                if info.is_valid_video:
                    return path
        except (OSError, MediaError):
            continue
    return None


def _backoff(attempt: int) -> None:
    time.sleep(min(2 ** attempt, 10))
