"""Shared helpers: names, media probing, fingerprints, filesystem safety."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_GAME_ALIASES = {
    "cs2": "Counter-Strike 2",
    "cs 2": "Counter-Strike 2",
    "csgo": "Counter-Strike 2",
    "counter strike 2": "Counter-Strike 2",
    "counter-strike 2": "Counter-Strike 2",
    "counter strike": "Counter-Strike",
    "counter-strike": "Counter-Strike",
    "dota 2": "Dota 2",
    "dota": "Dota 2",
    "league of legends": "League of Legends",
    "lol": "League of Legends",
    "valorant": "VALORANT",
    "fortnite": "Fortnite",
    "minecraft": "Minecraft",
    "gta v": "Grand Theft Auto V",
    "gta5": "Grand Theft Auto V",
    "apex": "Apex Legends",
    "apex legends": "Apex Legends",
    "pubg": "PUBG: BATTLEGROUNDS",
    "overwatch 2": "Overwatch 2",
    "overwatch": "Overwatch 2",
    "r6": "Rainbow Six Siege",
    "rainbow six siege": "Rainbow Six Siege",
    "rust": "Rust",
    "wow": "World of Warcraft",
    "world of warcraft": "World of Warcraft",
}

_WINDOWS_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}

_CLIP_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class MediaError(RuntimeError):
    """Raised when FFmpeg/ffprobe cannot read a media file."""


class ToolNotFound(RuntimeError):
    """Raised when a required external tool is absent."""


@dataclass
class MediaInfo:
    duration: float
    width: int = 0
    height: int = 0
    has_video: bool = False
    has_audio: bool = False
    video_codec: str = ""
    audio_codec: str = ""

    @property
    def is_valid_video(self) -> bool:
        return self.has_video and self.duration > 0 and self.width > 0 and self.height > 0


def normalize_game_name(raw: str) -> str:
    if not raw:
        return ""
    key = raw.strip().lower()
    return _GAME_ALIASES.get(key, raw.strip())


def validate_clip_id(clip_id: str) -> str:
    """Validate a Twitch clip id before using it in a filesystem path."""
    value = str(clip_id or "").strip()
    if not _CLIP_ID_RE.match(value):
        raise ValueError(f"Unsafe clip id: {clip_id!r}")
    return value


def safe_filename(name: str, max_len: int = 60) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", name or "")
    name = name.strip().rstrip(". ")
    name = re.sub(r"\s+", "_", name)
    if not name:
        return "clip"
    if name.upper().split(".")[0] in _WINDOWS_RESERVED:
        name = f"_{name}"
    return name[:max_len].rstrip(". ") or "clip"


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def find_tool(name: str) -> str:
    """Find ffmpeg/ffprobe in PATH, then next to the project / next to ffmpeg."""
    found = shutil.which(name)
    if found:
        return found
    local = PROJECT_ROOT / f"{name}.exe"
    if local.is_file():
        return str(local)
    ffmpeg = shutil.which("ffmpeg") or (
        str(PROJECT_ROOT / "ffmpeg.exe")
        if (PROJECT_ROOT / "ffmpeg.exe").is_file() else ""
    )
    if ffmpeg:
        sibling = Path(ffmpeg).with_name(f"{name}.exe")
        if sibling.is_file():
            return str(sibling)
    raise ToolNotFound(f"{name} not found in PATH")


def _run_probe(path: Path) -> Dict[str, Any]:
    ffprobe = find_tool("ffprobe")
    cmd = [
        ffprobe, "-v", "error",
        "-show_entries",
        "format=duration:stream=codec_type,codec_name,width,height,duration",
        "-of", "json", str(path),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=60,
        )
    except FileNotFoundError as exc:
        raise ToolNotFound("ffprobe not found in PATH") from exc
    except subprocess.TimeoutExpired as exc:
        raise MediaError(f"ffprobe timed out for {path}") from exc
    if result.returncode != 0:
        tail = (result.stderr or "").strip()[-400:]
        raise MediaError(f"ffprobe failed for {path.name}: {tail}")
    try:
        return json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise MediaError(f"ffprobe returned invalid JSON for {path.name}") from exc


def probe_media(path: Path) -> MediaInfo:
    """Return duration, dimensions and stream presence for a media file."""
    path = Path(path)
    if not path.is_file():
        raise MediaError(f"Media file does not exist: {path}")
    data = _run_probe(path)
    streams = data.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    duration = 0.0
    try:
        duration = float((data.get("format") or {}).get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    if duration <= 0:
        for stream in (video, audio):
            try:
                duration = max(duration, float((stream or {}).get("duration") or 0.0))
            except (TypeError, ValueError):
                pass
    return MediaInfo(
        duration=duration,
        width=int((video or {}).get("width") or 0),
        height=int((video or {}).get("height") or 0),
        has_video=video is not None,
        has_audio=audio is not None,
        video_codec=str((video or {}).get("codec_name") or ""),
        audio_codec=str((audio or {}).get("codec_name") or ""),
    )


def ffprobe_duration(path: Path) -> float:
    """Backward-compatible duration helper with a clear error message."""
    info = probe_media(path)
    if info.duration <= 0:
        raise MediaError(f"Could not determine duration: {path}")
    return info.duration


def file_sha1(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def file_fingerprint(path: Path) -> str:
    """Cheap identity of a file: size + mtime. Use sha1 when content matters."""
    path = Path(path)
    stat = path.stat()
    return f"{stat.st_size}:{stat.st_mtime_ns}"


def free_space_mb(path: Path) -> float:
    target = Path(path)
    while not target.exists() and target.parent != target:
        target = target.parent
    return shutil.disk_usage(target).free / (1024 * 1024)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_replace(source: Path, target: Path) -> Path:
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    source.replace(target)
    return target


def write_json(path: Path, data: Any, indent: int = 2) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=indent), encoding="utf-8")
    tmp.replace(path)
    return path


def read_json(path: Path, default: Optional[Any] = None) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def human_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{sec:02d}"
    return f"{minutes}:{sec:02d}"


def human_size(num_bytes: float) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def monotonic() -> float:
    return time.perf_counter()
