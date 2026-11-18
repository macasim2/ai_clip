"""Central configuration loaded from environment."""
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _get(name, default=None):
    v = os.getenv(name)
    return v if v not in (None, "") else default


def _getbool(name, default=False):
    v = os.getenv(name)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "on")


def _getint(name, default=0):
    v = os.getenv(name)
    try:
        return int(v) if v is not None else default
    except ValueError:
        return default


def _getfloat(name, default=0.0):
    v = os.getenv(name)
    try:
        return float(v) if v is not None else default
    except ValueError:
        return default


@dataclass
class Config:
    twitch_provider: str = "graphql"
    twitch_client_id: Optional[str] = None
    twitch_client_secret: Optional[str] = None
    twitch_web_client_id: str = "kimne78kx3ncx6brgo4mv6wki5h1ko"
    twitch_timeout: float = 20.0

    ollama_url: str = "http://localhost:11434"
    ollama_text_model: str = "qwen2.5:7b"
    ollama_vision_model: str = "moondream:latest"
    ollama_timeout: float = 180.0

    whisper_model: str = "small"
    whisper_device: str = "cpu"
    whisper_compute: str = "int8"

    raw_clips: Path = ROOT / "raw_clips"
    processed: Path = ROOT / "processed"
    exports: Path = ROOT / "exports"
    logs: Path = ROOT / "logs"
    data: Path = ROOT / "data"
    db_path: Path = ROOT / "data" / "clips.db"

    frame_interval_sec: float = 1.5
    max_frames: int = 20
    analyze_vision: bool = True

    min_score_candidate: int = 60
    min_score_review: int = 40
    min_clip_duration: float = 8.0
    max_clip_duration: float = 45.0

    video_width: int = 1080
    video_height: int = 1920
    video_bitrate: str = "6M"
    edit_mode: str = "blur"

    subtitles_enabled: bool = True
    subtitle_font_size: int = 64
    subtitle_style: str = "normal"
    subtitle_max_words: int = 4
    subtitle_max_chars: int = 20

    max_parallel: int = 1
    retry_attempts: int = 3


def load_config() -> Config:
    c = Config()
    c.twitch_provider = _get("TWITCH_PROVIDER", c.twitch_provider)
    c.twitch_client_id = _get("TWITCH_CLIENT_ID")
    c.twitch_client_secret = _get("TWITCH_CLIENT_SECRET")
    c.twitch_web_client_id = _get("TWITCH_WEB_CLIENT_ID", c.twitch_web_client_id)
    c.twitch_timeout = _getfloat("TWITCH_TIMEOUT", c.twitch_timeout)

    c.ollama_url = _get("OLLAMA_URL", c.ollama_url)
    c.ollama_text_model = _get("OLLAMA_TEXT_MODEL", c.ollama_text_model)
    c.ollama_vision_model = _get("OLLAMA_VISION_MODEL", c.ollama_vision_model)
    c.ollama_timeout = _getfloat("OLLAMA_TIMEOUT", c.ollama_timeout)

    c.whisper_model = _get("WHISPER_MODEL", c.whisper_model)
    c.whisper_device = _get("WHISPER_DEVICE", c.whisper_device)
    c.whisper_compute = _get("WHISPER_COMPUTE", c.whisper_compute)

    c.frame_interval_sec = _getfloat("FRAME_INTERVAL_SEC", c.frame_interval_sec)
    c.max_frames = _getint("MAX_FRAMES", c.max_frames)
    c.analyze_vision = _getbool("ANALYZE_VISION", c.analyze_vision)

    c.min_score_candidate = _getint("MIN_SCORE_CANDIDATE", c.min_score_candidate)
    c.min_score_review = _getint("MIN_SCORE_REVIEW", c.min_score_review)
    c.min_clip_duration = _getfloat("MIN_CLIP_DURATION", c.min_clip_duration)
    c.max_clip_duration = _getfloat("MAX_CLIP_DURATION", c.max_clip_duration)

    c.video_width = _getint("VIDEO_WIDTH", c.video_width)
    c.video_height = _getint("VIDEO_HEIGHT", c.video_height)
    c.video_bitrate = _get("VIDEO_BITRATE", c.video_bitrate)
    c.edit_mode = _get("EDIT_MODE", c.edit_mode)

    c.subtitles_enabled = _getbool("SUBTITLES_ENABLED", c.subtitles_enabled)
    c.subtitle_font_size = _getint("SUBTITLE_FONT_SIZE", c.subtitle_font_size)
    c.subtitle_style = _get("SUBTITLE_STYLE", c.subtitle_style)
    c.subtitle_max_words = _getint("SUBTITLE_MAX_WORDS", c.subtitle_max_words)
    c.subtitle_max_chars = _getint("SUBTITLE_MAX_CHARS", c.subtitle_max_chars)

    c.max_parallel = _getint("MAX_PARALLEL", c.max_parallel)
    c.retry_attempts = _getint("RETRY_ATTEMPTS", c.retry_attempts)

    for p in (c.raw_clips, c.processed, c.exports, c.logs, c.data):
        p.mkdir(parents=True, exist_ok=True)
    return c
