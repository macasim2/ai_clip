"""Safe cleanup of intermediate files. Sources and exports are never touched."""
from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .database import Database
from .logging_setup import get_logger

log = get_logger(__name__)

PROTECTED_NAMES = {
    "transcript.json", "vision.json", "analysis.json", "decision.json",
    "subtitles.ass", "subtitles.srt",
}


def _remove(path: Path, dry_run: bool) -> int:
    if not path.exists():
        return 0
    size = 0
    if path.is_dir():
        size = sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
    else:
        size = path.stat().st_size
    if not dry_run:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)
    return size


def cleanup(cfg, db: Database, days: Optional[int] = None,
            dry_run: bool = False, orphans: bool = False) -> Dict[str, Any]:
    """Delete old frames/audio/temp files for finished clips.

    Keeps source clips, exports, transcripts, vision/analysis/decision JSON.
    """
    days = cfg.cleanup_days if days is None else max(0, int(days))
    cutoff = time.time() - days * 86400
    removed_files = 0
    removed_bytes = 0
    kept = 0
    clip_ids = set()

    for row in db.list_all(limit=100000):
        clip_id = row["id"]
        clip_ids.add(clip_id)
        work = cfg.processed / clip_id
        if not work.is_dir():
            continue
        if row["status"] not in ("READY", "REJECTED"):
            kept += 1
            continue
        candidates = [work / "frames", work / "audio.wav",
                      work / "audio.wav.meta.json", work / "vertical.mp4",
                      work / "temp", work / ".tmp"]
        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                if candidate.stat().st_mtime > cutoff:
                    continue
            except OSError:
                continue
            removed_bytes += _remove(candidate, dry_run)
            removed_files += 1
        for pattern in ("*.part", "*.tmp"):
            for candidate in work.glob(pattern):
                try:
                    if candidate.stat().st_mtime > cutoff:
                        continue
                except OSError:
                    continue
                removed_bytes += _remove(candidate, dry_run)
                removed_files += 1

    if orphans:
        if cfg.processed.is_dir():
            for child in cfg.processed.iterdir():
                if not child.is_dir() or child.name in clip_ids:
                    continue
                try:
                    if child.stat().st_mtime > cutoff:
                        continue
                except OSError:
                    continue
                # Orphan directories are safe to remove only with --orphans.
                if any((child / name).exists() for name in PROTECTED_NAMES):
                    kept += 1
                    continue
                removed_bytes += _remove(child, dry_run)
                removed_files += 1

    tmp_root = cfg.raw_clips / ".tmp"
    if tmp_root.is_dir():
        for child in tmp_root.iterdir():
            try:
                if child.stat().st_mtime > cutoff:
                    continue
            except OSError:
                continue
            removed_bytes += _remove(child, dry_run)
            removed_files += 1

    log.info("Cleanup: days=%s files=%d bytes=%d dry_run=%s orphans=%s",
             days, removed_files, removed_bytes, dry_run, orphans)
    return {
        "days": days,
        "removed_files": removed_files,
        "removed_bytes": removed_bytes,
        "kept_pending": kept,
        "dry_run": dry_run,
    }
