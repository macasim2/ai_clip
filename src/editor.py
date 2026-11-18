"""Vertical 9:16 renderer with explicit clip_id and timecodes.

Every call either produces a *verified* 1080x1920 MP4 that matches the
requested time window or raises :class:`EditorError`.  A render is written to
a temporary file first, probed, and only then atomically moved to the final
path, so downstream stages never see a half-written export.
"""
from __future__ import annotations

import math
import re
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from .logging_setup import get_logger
from .utils import MediaError, MediaInfo, ToolNotFound, find_tool, probe_media

log = get_logger(__name__)

# --- verification thresholds -------------------------------------------------
MIN_OUTPUT_BYTES = 4096
DURATION_TOLERANCE_SEC = 0.75
DURATION_TOLERANCE_RATIO = 0.03
# A clip often ends a fraction of a second after ffprobe's container duration.
SOURCE_END_TOLERANCE_SEC = 0.5

DEFAULT_WIDTH = 1080
DEFAULT_HEIGHT = 1920
DEFAULT_CRF = 20
DEFAULT_PRESET = "medium"

_VALID_PRESETS = frozenset({
    "ultrafast", "superfast", "veryfast", "faster", "fast",
    "medium", "slow", "slower", "veryslow",
})
_VALID_EDIT_MODES = frozenset({"blur", "crop"})

# Optional CRF bitrate cap: only sane, positive values are applied.
_MAXRATE_MIN_BPS = 100_000
_MAXRATE_MAX_BPS = 100_000_000
_BITRATE_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*([kKmMgG]?)$")
_BITRATE_MULTIPLIERS = {"": 1, "k": 1_000, "m": 1_000_000, "g": 1_000_000_000}

# A file name safe to embed unquoted in an FFmpeg filter option.
_SAFE_FILTER_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")

_CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_RENDER_TIMEOUT_BASE = 120.0
_RENDER_TIMEOUT_FACTOR = 20.0
_RENDER_TIMEOUT_MAX = 3600.0


class EditorError(RuntimeError):
    """Raised when a render cannot be produced or verified."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _require_tool(name: str) -> str:
    try:
        return find_tool(name)
    except ToolNotFound as exc:
        raise EditorError(f"{name} not found in PATH") from exc


def _verify_ffmpeg() -> str:
    """Fail fast when FFmpeg/FFprobe are missing; return the ffmpeg binary."""
    _require_tool("ffprobe")
    return _require_tool("ffmpeg")


def _remove_temp(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError as exc:  # pragma: no cover - filesystem edge case
        log.warning("Could not remove temporary render %s: %s", path, exc)


def _as_time(value, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise EditorError(f"Invalid {name} time: {value!r}") from exc
    if not math.isfinite(number):
        raise EditorError(f"Invalid {name} time: {value!r}")
    return number


def _cfg_int(cfg, name: str, default: int) -> int:
    try:
        return int(getattr(cfg, name, default))
    except (TypeError, ValueError):
        return default


def _output_size(cfg) -> Tuple[int, int]:
    width = _cfg_int(cfg, "video_width", DEFAULT_WIDTH)
    height = _cfg_int(cfg, "video_height", DEFAULT_HEIGHT)
    if width < 16 or height < 16:
        raise EditorError(f"Invalid output size: {width}x{height}")
    if width % 2 or height % 2:
        raise EditorError(
            f"Output size must be even for yuv420p: {width}x{height}")
    return width, height


def _edit_mode(cfg) -> str:
    mode = str(getattr(cfg, "edit_mode", "blur") or "blur").strip().lower()
    if mode not in _VALID_EDIT_MODES:
        log.warning("Unknown edit_mode %r, falling back to 'blur'", mode)
        return "blur"
    return mode


def _crf(cfg) -> int:
    crf = _cfg_int(cfg, "video_crf", DEFAULT_CRF)
    if not 0 <= crf <= 51:
        log.warning("video_crf=%r is out of range, using %d", crf, DEFAULT_CRF)
        return DEFAULT_CRF
    return crf


def _preset(cfg) -> str:
    raw = getattr(cfg, "video_preset", None)
    preset = str(raw or "").strip().lower()
    if preset in _VALID_PRESETS:
        return preset
    if preset:
        log.warning("video_preset=%r is not supported, using %r",
                    raw, DEFAULT_PRESET)
    return DEFAULT_PRESET


def _sane_bitrate(value) -> Optional[int]:
    """Parse VIDEO_BITRATE ("6M", "6000k", "6000000") into bits/s.

    Returns ``None`` for missing, malformed or absurd values so that a typo in
    .env can never break a render.
    """
    if value is None:
        return None
    match = _BITRATE_RE.match(str(value).strip())
    if not match:
        return None
    number, unit = match.groups()
    try:
        bits = int(float(number) * _BITRATE_MULTIPLIERS[unit.lower()])
    except (KeyError, ValueError):
        return None
    if not _MAXRATE_MIN_BPS <= bits <= _MAXRATE_MAX_BPS:
        return None
    return bits


def _format_bitrate(bits: int) -> str:
    if bits % 1_000_000 == 0:
        return f"{bits // 1_000_000}M"
    if bits % 1_000 == 0:
        return f"{bits // 1_000}k"
    return str(bits)


def _quality_args(cfg) -> List[str]:
    """x264 settings: CRF by default, optional bitrate cap from config."""
    args = ["-c:v", "libx264", "-preset", _preset(cfg), "-crf", str(_crf(cfg))]
    maxrate = _sane_bitrate(getattr(cfg, "video_bitrate", None))
    if maxrate is not None:
        bufsize = min(maxrate * 2, _MAXRATE_MAX_BPS)
        args += ["-maxrate", _format_bitrate(maxrate),
                 "-bufsize", _format_bitrate(bufsize)]
    else:
        log.debug("VIDEO_BITRATE %r is not usable as a CRF cap; omitted",
                  getattr(cfg, "video_bitrate", None))
    return args


def _escape_filter_value(value: str) -> str:
    """Escape one value for use inside a ``-filter_complex`` string.

    Only relative paths/file names are passed here (the subtitle file is opened
    relative to its own directory via ``cwd``), so this is a plain quoting
    helper, not the old absolute-Windows-path colon-escaping hack.
    """
    if _SAFE_FILTER_NAME_RE.match(value):
        return value
    escaped = value.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def _subtitle_style(cfg) -> str:
    font_size = _cfg_int(cfg, "subtitle_font_size", 64)
    if not 12 <= font_size <= 160:
        font_size = 64
    margin_v = _cfg_int(cfg, "subtitle_margin_v", 80)
    if not 0 <= margin_v <= 900:
        margin_v = 80
    return (f"FontSize={font_size},Outline=2,Shadow=1,"
            f"MarginV={margin_v},Alignment=2")


def _srt_style(cfg) -> str:
    """Scale pixel values from the 1080x1920 target to the SRT 384x288 canvas."""
    font_size = _cfg_int(cfg, "subtitle_font_size", 64)
    if not 12 <= font_size <= 160:
        font_size = 64
    margin_v = _cfg_int(cfg, "subtitle_margin_v", 80)
    if not 0 <= margin_v <= 900:
        margin_v = 80
    video_height = max(1, _cfg_int(cfg, "video_height", 1920))
    scale = 288.0 / video_height
    return (f"FontSize={max(8, round(font_size * scale))},Outline=1,"
            f"Shadow=1,MarginV={max(0, round(margin_v * scale))},Alignment=2")


def _video_graph(width: int, height: int, mode: str) -> str:
    """Video filter chain ending right before the ``[vout]`` label."""
    if mode == "crop":
        # Center-crop to the exact 9:16 window, then scale without stretching.
        # force_original_aspect_ratio=increase + crop absorbs the sub-pixel
        # rounding of the crop expressions, so the final scale never distorts.
        return (
            "[0:v]"
            f"crop=w='min(iw,ih*{width}/{height})':"
            f"h='min(ih,iw*{height}/{width})',"
            f"scale={width}:{height}:"
            "force_original_aspect_ratio=increase:force_divisible_by=2,"
            f"crop={width}:{height},setsar=1"
        )
    # Blur mode (default): blurred cover background + full foreground band.
    # The foreground uses force_original_aspect_ratio=decrease (fit inside),
    # never crop, so game HUD/minimap at the frame edges stays visible.
    return (
        "[0:v]split=2[bg][fg];"
        f"[bg]scale={width}:{height}:"
        "force_original_aspect_ratio=increase:force_divisible_by=2,"
        f"crop={width}:{height},setsar=1,boxblur=20:5[bgb];"
        f"[fg]scale={width}:{height}:"
        "force_original_aspect_ratio=decrease:force_divisible_by=2,"
        "setsar=1[fgv];"
        "[bgb][fgv]overlay=(W-w)/2:(H-h)/2,setsar=1"
    )


def _build_filter_complex(
    cfg, width: int, height: int, mode: str,
    subtitles_path: Optional[Path],
) -> Tuple[str, str, Optional[Path], bool]:
    """Return (filter_complex, video map label, subtitle cwd, burned_subs)."""
    graph = _video_graph(width, height, mode)
    subtitle_cwd: Optional[Path] = None
    subtitle_filter = ""

    if subtitles_path is not None:
        subtitles = Path(subtitles_path)
        if subtitles.is_file():
            # Windows-safe subtitles burn-in: FFmpeg runs with cwd set to the
            # subtitles folder and receives only the bare file name.  An
            # absolute "F:\..." path never enters the filter graph, so drive
            # colons and backslashes cannot break FFmpeg's escaping rules.
            resolved = subtitles.resolve()
            subtitle_cwd = resolved.parent
            name = _escape_filter_value(resolved.name)
            suffix = resolved.suffix.lower()
            subtitle_filter = f"subtitles=filename={name}"
            if suffix in {".ass", ".ssa"}:
                # The ASS file already carries PlayRes 1080x1920, size,
                # margins and outline. force_style here would override them
                # (and could move text out of the frame).
                pass
            elif suffix == ".srt":
                # SRT has an implicit 384x288 script canvas; scale the config
                # values down so they land at the intended pixel size.
                subtitle_filter += f":force_style='{_srt_style(cfg)}'"
            else:
                log.warning(
                    "Unusual subtitle extension %r; attempting burn-in: %s",
                    suffix, resolved.name)
                subtitle_filter += f":force_style='{_subtitle_style(cfg)}'"
            log.info("Burning subtitles %s (cwd=%s)",
                     resolved.name, subtitle_cwd)
        else:
            log.warning(
                "Subtitles file does not exist, rendering without subtitles: %s",
                subtitles)

    if subtitle_filter:
        return (f"{graph}[vout];[vout]{subtitle_filter}[vfinal]",
                "[vfinal]", subtitle_cwd, True)
    return f"{graph}[vout]", "[vout]", None, False


# ---------------------------------------------------------------------------
# Output verification
# ---------------------------------------------------------------------------
def _verify_output(path: Path, expected_duration: float,
                   source_info: MediaInfo, width: int, height: int,
                   label: str) -> MediaInfo:
    """Probe the rendered file and raise EditorError on any mismatch."""
    try:
        info = probe_media(path)
    except MediaError as exc:
        raise EditorError(
            f"Output verification failed for {label}: {exc}") from exc

    try:
        size = path.stat().st_size
    except OSError as exc:  # pragma: no cover - rendered file existed
        raise EditorError(
            f"Output verification failed for {label}: {exc}") from exc

    problems: List[str] = []
    if not info.has_video:
        problems.append("no video stream")
    if info.width != width or info.height != height:
        problems.append(
            f"dimensions {info.width}x{info.height}, expected {width}x{height}")
    if size <= MIN_OUTPUT_BYTES:
        problems.append(
            f"file size {size} bytes is not > {MIN_OUTPUT_BYTES} bytes")
    tolerance = max(DURATION_TOLERANCE_SEC,
                    DURATION_TOLERANCE_RATIO * expected_duration)
    if abs(info.duration - expected_duration) > tolerance:
        problems.append(
            f"duration {info.duration:.3f}s differs from expected "
            f"{expected_duration:.3f}s by more than {tolerance:.2f}s")
    if info.has_audio != source_info.has_audio:
        problems.append(
            f"audio present={info.has_audio}, source audio="
            f"{source_info.has_audio}")
    if problems:
        raise EditorError(
            f"Output verification failed for {label}: " + "; ".join(problems))
    return info


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def render_vertical(clip_id: str, source: Path, out_path: Path,
                    start: float, end: float, cfg,
                    subtitles_path: Optional[Path] = None) -> Path:
    """Render ``source[start:end]`` as a verified 1080x1920 vertical MP4.

    Raises :class:`EditorError` on invalid input, FFmpeg failure or when the
    produced file fails verification.  The final file only appears at
    ``out_path`` after it passed every check.
    """
    ffmpeg = _verify_ffmpeg()

    source = Path(source)
    out_path = Path(out_path)
    if not source.is_file():
        raise EditorError(f"Source clip not found: {source}")

    start = _as_time(start, "start")
    end = _as_time(end, "end")
    if start < 0:
        raise EditorError(
            f"Invalid time range: start={start:.3f}s must be >= 0")
    if end <= start:
        raise EditorError(
            f"Invalid time range: end={end:.3f}s must be after "
            f"start={start:.3f}s")
    expected_duration = end - start

    source_abs = source.resolve()
    try:
        source_info = probe_media(source_abs)
    except MediaError as exc:
        raise EditorError(
            f"Could not probe source {source.name}: {exc}") from exc
    if not source_info.has_video:
        raise EditorError(f"Source has no video stream: {source.name}")
    if source_info.duration > 0:
        if start >= source_info.duration:
            raise EditorError(
                f"Invalid time range: start={start:.3f}s is at/after source "
                f"duration {source_info.duration:.3f}s ({source.name})")
        limit = source_info.duration + SOURCE_END_TOLERANCE_SEC
        if end > limit:
            raise EditorError(
                f"Invalid time range: end={end:.3f}s exceeds source duration "
                f"{source_info.duration:.3f}s "
                f"(+{SOURCE_END_TOLERANCE_SEC:.1f}s tolerance) "
                f"for {source.name}")
    else:
        log.warning(
            "Source duration is unknown for %s; skipping end-time bound check",
            source.name)

    width, height = _output_size(cfg)
    mode = _edit_mode(cfg)
    filter_complex, map_video, subtitle_cwd, burned_subs = _build_filter_complex(
        cfg, width, height, mode, subtitles_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_abs = out_path.resolve()
    tmp_abs = out_path.with_suffix(".tmp.mp4").resolve()
    _remove_temp(tmp_abs)

    cmd = [
        ffmpeg, "-y", "-hide_banner", "-nostdin",
        "-loglevel", "warning",
        # Input-level trim: exact source file + exact window, audio included.
        "-ss", f"{start:.3f}", "-t", f"{expected_duration:.3f}",
        "-i", str(source_abs),
        "-filter_complex", filter_complex,
        "-map", map_video, "-map", "0:a?",
        *_quality_args(cfg),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k",
        "-movflags", "+faststart",
        str(tmp_abs),
    ]

    log.info(
        "Render %s: %s %.3f..%.3fs -> %s (mode=%s, %dx%d, subs=%s)",
        clip_id, source.name, start, end, out_path.name, mode,
        width, height, "burned" if burned_subs else "none")
    log.debug("ffmpeg filter_complex: %s", filter_complex)
    log.debug("ffmpeg command: %s", subprocess.list2cmdline(cmd))

    timeout = max(_RENDER_TIMEOUT_BASE,
                  min(_RENDER_TIMEOUT_MAX,
                      expected_duration * _RENDER_TIMEOUT_FACTOR + 60.0))
    run_kwargs = {
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "timeout": timeout,
    }
    if subtitle_cwd is not None:
        run_kwargs["cwd"] = str(subtitle_cwd)
    if _CREATE_NO_WINDOW:  # hide the console flash when started from a GUI
        run_kwargs["creationflags"] = _CREATE_NO_WINDOW

    try:
        result = subprocess.run(cmd, **run_kwargs)
    except subprocess.TimeoutExpired as exc:
        _remove_temp(tmp_abs)
        log.error("ffmpeg render of %s timed out after %.0fs",
                  clip_id, timeout)
        raise EditorError(
            f"FFmpeg timed out after {timeout:.0f}s rendering "
            f"{out_path.name}") from exc
    except OSError as exc:
        _remove_temp(tmp_abs)
        raise EditorError(f"Could not start ffmpeg: {exc}") from exc

    stderr_tail = (result.stderr or "").strip()[-1200:]
    if result.returncode != 0:
        _remove_temp(tmp_abs)
        log.error("ffmpeg render failed for %s (rc=%s): %s",
                  clip_id, result.returncode, stderr_tail or "<no stderr>")
        raise EditorError(
            f"FFmpeg failed for {out_path.name} (rc={result.returncode}): "
            f"{stderr_tail or 'no stderr output'}")

    try:
        info = _verify_output(tmp_abs, expected_duration, source_info,
                              width, height, out_path.name)
    except EditorError:
        _remove_temp(tmp_abs)
        raise

    try:
        tmp_abs.replace(out_abs)
    except OSError as exc:  # pragma: no cover - permission / locked file
        _remove_temp(tmp_abs)
        raise EditorError(
            f"Could not finalize output {out_path}: {exc}") from exc

    log.info(
        "Render %s verified: %s %dx%d %.2fs audio=%s",
        clip_id, out_path.name, info.width, info.height,
        info.duration, info.has_audio)
    return out_path


__all__ = ["EditorError", "render_vertical"]
