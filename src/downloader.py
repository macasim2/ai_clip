"""Download Twitch clips via yt-dlp with verification and retries."""
import shutil
import subprocess
import time
import sys
from pathlib import Path

from .logging_setup import get_logger
from .utils import ffprobe_duration, safe_filename

log = get_logger(__name__)


class DownloadError(RuntimeError):
    pass


def download_clip(clip: dict, raw_dir: Path, attempts: int = 3) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    cid = clip["id"]
    title = safe_filename(clip.get("title") or "clip")
    target = raw_dir / f"{cid}_{title}.mp4"

    if target.exists() and target.stat().st_size > 1024:
        try:
            dur = ffprobe_duration(target)
            log.info("Clip %s already present (%.1fs)", cid, dur)
            return target
        except Exception:
            log.warning("Existing file unreadable, re-downloading: %s", target)
            target.unlink(missing_ok=True)

    url = clip["url"]
    last_err = None
    for attempt in range(1, attempts + 1):
        tmp = raw_dir / f"{cid}.part.mp4"
        tmp.unlink(missing_ok=True)
        cmd = [
            sys.executable, "-m", "yt_dlp", "--no-playlist",
            "--retries", "5", "--fragment-retries", "5",
            "--no-progress",
            "-f", "mp4/bestvideo+bestaudio/best",
            "--merge-output-format", "mp4",
            "-o", str(tmp), url,
        ]
        log.info("Download clip %s (attempt %d/%d)", cid, attempt, attempts)
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            last_err = "yt-dlp timeout"
            tmp.unlink(missing_ok=True)
            continue
        except FileNotFoundError:
            raise DownloadError(
                "yt-dlp not found in PATH. Install: pip install yt-dlp"
            )

        if res.returncode != 0:
            last_err = res.stderr.strip()[-500:]
            log.warning("yt-dlp failed: %s", last_err)
            tmp.unlink(missing_ok=True)
            time.sleep(2 ** attempt)
            continue

        if not tmp.exists() or tmp.stat().st_size < 1024:
            last_err = "downloaded file missing or too small"
            tmp.unlink(missing_ok=True)
            continue

        try:
            dur = ffprobe_duration(tmp)
        except Exception as e:
            last_err = f"ffprobe: {e}"
            tmp.unlink(missing_ok=True)
            continue

        tmp.replace(target)
        log.info("Saved %s (%.1fs)", target.name, dur)
        return target

    raise DownloadError(f"Failed to download clip {cid}: {last_err}")
