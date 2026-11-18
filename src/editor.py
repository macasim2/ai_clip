"""Vertical 9:16 renderer with explicit clip_id and timecodes."""
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from .logging_setup import get_logger

log = get_logger(__name__)


class EditorError(RuntimeError):
    pass


def _verify_ffmpeg():
    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            raise EditorError(f"{tool} not found in PATH")


def _subtitle_path(path: Path) -> str:
    # escape for ffmpeg subtitles filter on Windows
    p = str(path).replace("\\", "/")
    p = p.replace(":", "\\:").replace("'", "\\'")
    return p


def render_vertical(clip_id: str, source: Path, out_path: Path,
                    start: float, end: float, cfg,
                    subtitles_path: Optional[Path] = None) -> Path:
    _verify_ffmpeg()
    if not source.exists():
        raise EditorError(f"Source missing: {source}")
    if end <= start:
        raise EditorError(f"Invalid time range: {start}..{end}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    W, H = cfg.video_width, cfg.video_height
    duration = end - start

    if cfg.edit_mode == "crop":
        base = (
            f"[0:v]crop='min(iw,ih*{W}/{H})':'min(ih,iw*{H}/{W})',"
            f"scale={W}:{H}:flags=lanczos,setsar=1[vout]"
        )
    else:
        base = (
            f"[0:v]split=2[bg][fg];"
            f"[bg]scale={W}:{H}:force_original_aspect_ratio=increase,"
            f"crop={W}:{H},boxblur=20:5[bgb];"
            f"[fg]scale={W}:{H}:force_original_aspect_ratio=decrease[fgv];"
            f"[bgb][fgv]overlay=(W-w)/2:(H-h)/2,setsar=1[vout]"
        )

    if subtitles_path and subtitles_path.exists():
        sp = _subtitle_path(subtitles_path)
        vf = (f"{base};[vout]subtitles='{sp}'"
              f":force_style='FontSize={cfg.subtitle_font_size},"
              f"Outline=2,Shadow=1,MarginV=80'[vfinal]")
        map_v = "[vfinal]"
    else:
        vf = base
        map_v = "[vout]"

    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(source),
        "-filter_complex", vf,
        "-map", map_v, "-map", "0:a?",
        "-c:v", "libx264", "-preset", "medium",
        "-b:v", cfg.video_bitrate, "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart", "-shortest",
        str(out_path),
    ]
    log.info("Render %s -> %s (%.2f..%.2fs)", clip_id, out_path.name, start, end)
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise EditorError(f"ffmpeg failed: {res.stderr[-800:]}")
    if not out_path.exists() or out_path.stat().st_size < 4096:
        raise EditorError("Output missing or empty")
    return out_path
