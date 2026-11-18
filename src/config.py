"""Central configuration loaded from environment with validation."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

VALID_PROVIDERS = {"graphql", "helix"}
VALID_EDIT_MODES = {"blur", "crop"}
VALID_SUBTITLE_STYLES = {"normal", "accent"}
VALID_PRESETS = {"ultrafast", "superfast", "veryfast", "faster", "fast",
                 "medium", "slow", "slower", "veryslow"}
VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
VALID_WHISPER_DEVICES = {"cpu", "cuda", "auto"}


class ConfigError(ValueError):
    """Invalid or inconsistent configuration."""


def _get(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    return value if value not in (None, "") else default


def _getstr(name: str, default: str) -> str:
    return str(_get(name, default)).strip()


def _getbool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{name} must be a boolean (true/false), got {value!r}")


def _getint(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {value!r}") from exc


def _getfloat(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {value!r}") from exc


def _resolve_path(value: str, default: Path) -> Path:
    if not value:
        return default
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path


@dataclass
class Config:
    # Twitch
    twitch_provider: str = "graphql"
    twitch_client_id: Optional[str] = None
    twitch_client_secret: Optional[str] = None
    twitch_web_client_id: str = "kimne78kx3ncx6brgo4mv6wki5h1ko"
    twitch_timeout: float = 20.0
    twitch_max_pages: int = 20

    # Local LLM
    ollama_url: str = "http://localhost:11434"
    ollama_text_model: str = "llama3.2:latest"
    ollama_vision_model: str = "llava:latest"
    ollama_fast_model: str = ""
    ollama_timeout: float = 240.0
    ollama_keep_alive: str = "2m"
    ollama_num_ctx: int = 4096

    # Whisper
    whisper_model: str = "small"
    whisper_device: str = "cpu"
    whisper_compute: str = "int8"
    whisper_beam_size: int = 5

    # Storage
    raw_clips: Path = ROOT / "raw_clips"
    processed: Path = ROOT / "processed"
    exports: Path = ROOT / "exports"
    logs: Path = ROOT / "logs"
    data: Path = ROOT / "data"
    db_path: Path = ROOT / "data" / "clips.db"
    db_backups: Path = ROOT / "data" / "backups"

    # Analysis
    frame_interval_sec: float = 1.5
    min_frames: int = 4
    max_frames: int = 10
    scene_threshold: float = 0.3
    analyze_vision: bool = True
    vision_batch_size: int = 1
    vision_frame_width: int = 768

    # Gating
    min_score_candidate: int = 60
    min_score_review: int = 40
    min_clip_duration: float = 8.0
    max_clip_duration: float = 45.0
    min_confidence: float = 0.35
    curator_temperature: float = 0.2

    # Editing
    video_width: int = 1080
    video_height: int = 1920
    video_bitrate: str = "6M"
    video_crf: int = 20
    video_preset: str = "medium"
    edit_mode: str = "blur"

    # Subtitles
    subtitles_enabled: bool = True
    subtitle_font: str = "Arial"
    subtitle_font_size: int = 64
    subtitle_style: str = "normal"
    subtitle_max_words: int = 4
    subtitle_max_chars: int = 24
    subtitle_max_duration: float = 4.0
    subtitle_margin_v: int = 140

    # Queue / resilience
    max_parallel: int = 1
    retry_attempts: int = 3
    max_attempts: int = 3
    stale_lock_minutes: int = 30
    min_free_disk_mb: int = 1024
    cleanup_days: int = 14
    db_backup: bool = True

    # Logging
    log_level: str = "INFO"

    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ helpers
    def ensure_dirs(self) -> "Config":
        for path in (self.raw_clips, self.processed, self.exports,
                     self.logs, self.data, self.db_path.parent, self.db_backups):
            Path(path).mkdir(parents=True, exist_ok=True)
        return self

    def as_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if key == "extra":
                continue
            result[key] = str(value) if isinstance(value, Path) else value
        return result

    def validate(self) -> "Config":
        errors: List[str] = []

        if self.twitch_provider not in VALID_PROVIDERS:
            errors.append(
                f"TWITCH_PROVIDER must be one of {sorted(VALID_PROVIDERS)}, "
                f"got {self.twitch_provider!r}"
            )
        if self.twitch_provider == "helix" and not (
                self.twitch_client_id and self.twitch_client_secret):
            errors.append(
                "TWITCH_PROVIDER=helix requires TWITCH_CLIENT_ID and "
                "TWITCH_CLIENT_SECRET. Use TWITCH_PROVIDER=graphql without "
                "own credentials."
            )
        if self.twitch_provider == "graphql" and not self.twitch_web_client_id:
            errors.append("TWITCH_WEB_CLIENT_ID is required for the graphql provider")

        if not (0 < self.twitch_timeout <= 120):
            errors.append("TWITCH_TIMEOUT must be in 0..120 seconds")
        if not (1 <= self.twitch_max_pages <= 100):
            errors.append("TWITCH_MAX_PAGES must be in 1..100")

        if not self.ollama_url.startswith(("http://", "https://")):
            errors.append("OLLAMA_URL must start with http:// or https://")
        if not (1 <= self.ollama_timeout <= 1800):
            errors.append("OLLAMA_TIMEOUT must be in 1..1800 seconds")
        if not self.ollama_text_model:
            errors.append("OLLAMA_TEXT_MODEL is required")
        if not (512 <= self.ollama_num_ctx <= 131072):
            errors.append("OLLAMA_NUM_CTX must be in 512..131072")

        if self.whisper_device not in VALID_WHISPER_DEVICES:
            errors.append(f"WHISPER_DEVICE must be one of {sorted(VALID_WHISPER_DEVICES)}")
        if not self.whisper_model:
            errors.append("WHISPER_MODEL is required")
        if not (1 <= self.whisper_beam_size <= 10):
            errors.append("WHISPER_BEAM_SIZE must be in 1..10")

        if not (0.2 <= self.frame_interval_sec <= 30):
            errors.append("FRAME_INTERVAL_SEC must be in 0.2..30")
        if not (1 <= self.min_frames <= 100):
            errors.append("MIN_FRAMES must be in 1..100")
        if not (self.min_frames <= self.max_frames <= 100):
            errors.append("MAX_FRAMES must be in MIN_FRAMES..100")
        if not (0.0 <= self.scene_threshold <= 1.0):
            errors.append("SCENE_THRESHOLD must be in 0..1")
        if not (1 <= self.vision_batch_size <= 4):
            errors.append("VISION_BATCH_SIZE must be in 1..4")
        if not (320 <= self.vision_frame_width <= 1920):
            errors.append("VISION_FRAME_WIDTH must be in 320..1920")

        if not (0 <= self.min_score_review <= 100):
            errors.append("MIN_SCORE_REVIEW must be in 0..100")
        if not (0 <= self.min_score_candidate <= 100):
            errors.append("MIN_SCORE_CANDIDATE must be in 0..100")
        if self.min_score_review > self.min_score_candidate:
            errors.append("MIN_SCORE_REVIEW cannot exceed MIN_SCORE_CANDIDATE")
        if not (0.0 <= self.min_confidence <= 1.0):
            errors.append("MIN_CONFIDENCE must be in 0..1")
        if not (0.0 <= self.curator_temperature <= 1.5):
            errors.append("CURATOR_TEMPERATURE must be in 0..1.5")
        if not (1 <= self.min_clip_duration <= 300):
            errors.append("MIN_CLIP_DURATION must be in 1..300 seconds")
        if not (self.min_clip_duration <= self.max_clip_duration <= 300):
            errors.append("MAX_CLIP_DURATION must be in MIN_CLIP_DURATION..300")

        if self.video_width < 16 or self.video_height < 16:
            errors.append("VIDEO_WIDTH/VIDEO_HEIGHT are too small")
        if self.video_width % 2 or self.video_height % 2:
            errors.append("VIDEO_WIDTH/VIDEO_HEIGHT must be even for yuv420p")
        if self.video_width > self.video_height:
            errors.append("VIDEO_WIDTH must be smaller than VIDEO_HEIGHT for 9:16")
        if not (0 <= self.video_crf <= 51):
            errors.append("VIDEO_CRF must be in 0..51")
        if self.video_preset not in VALID_PRESETS:
            errors.append(f"VIDEO_PRESET must be one of {sorted(VALID_PRESETS)}")
        if self.edit_mode not in VALID_EDIT_MODES:
            errors.append(f"EDIT_MODE must be one of {sorted(VALID_EDIT_MODES)}")
        if not self.video_bitrate:
            errors.append("VIDEO_BITRATE is required")

        if self.subtitle_style not in VALID_SUBTITLE_STYLES:
            errors.append(f"SUBTITLE_STYLE must be one of {sorted(VALID_SUBTITLE_STYLES)}")
        if not (12 <= self.subtitle_font_size <= 160):
            errors.append("SUBTITLE_FONT_SIZE must be in 12..160")
        if not (1 <= self.subtitle_max_words <= 12):
            errors.append("SUBTITLE_MAX_WORDS must be in 1..12")
        if not (5 <= self.subtitle_max_chars <= 120):
            errors.append("SUBTITLE_MAX_CHARS must be in 5..120")
        if not (0.5 <= self.subtitle_max_duration <= 15):
            errors.append("SUBTITLE_MAX_DURATION must be in 0.5..15")
        if not (0 <= self.subtitle_margin_v <= 900):
            errors.append("SUBTITLE_MARGIN_V must be in 0..900")

        if not (1 <= self.max_parallel <= 8):
            errors.append("MAX_PARALLEL must be in 1..8")
        if not (1 <= self.retry_attempts <= 10):
            errors.append("RETRY_ATTEMPTS must be in 1..10")
        if not (1 <= self.max_attempts <= 10):
            errors.append("MAX_ATTEMPTS must be in 1..10")
        if not (1 <= self.stale_lock_minutes <= 1440):
            errors.append("STALE_LOCK_MINUTES must be in 1..1440")
        if self.min_free_disk_mb < 0:
            errors.append("MIN_FREE_DISK_MB cannot be negative")
        if self.cleanup_days < 0:
            errors.append("CLEANUP_DAYS cannot be negative")
        if self.log_level.upper() not in VALID_LOG_LEVELS:
            errors.append(f"LOG_LEVEL must be one of {sorted(VALID_LOG_LEVELS)}")

        self.log_level = self.log_level.upper()
        if errors:
            joined = "\n - ".join(errors)
            raise ConfigError(f"Configuration errors:\n - {joined}")
        return self


def load_config(read_env: bool = True,
                env_file: Optional[Path] = None) -> Config:
    """Load configuration from .env and process environment, then validate."""
    if read_env:
        path = Path(env_file) if env_file is not None else ROOT / ".env"
        if path.is_file():
            load_dotenv(path, override=False)

    cfg = Config()
    cfg.twitch_provider = _getstr("TWITCH_PROVIDER", cfg.twitch_provider).lower()
    cfg.twitch_client_id = _get("TWITCH_CLIENT_ID")
    cfg.twitch_client_secret = _get("TWITCH_CLIENT_SECRET")
    cfg.twitch_web_client_id = _getstr("TWITCH_WEB_CLIENT_ID",
                                       cfg.twitch_web_client_id)
    cfg.twitch_timeout = _getfloat("TWITCH_TIMEOUT", cfg.twitch_timeout)
    cfg.twitch_max_pages = _getint("TWITCH_MAX_PAGES", cfg.twitch_max_pages)

    cfg.ollama_url = _getstr("OLLAMA_URL", cfg.ollama_url).rstrip("/")
    cfg.ollama_text_model = _getstr("OLLAMA_TEXT_MODEL", cfg.ollama_text_model)
    cfg.ollama_vision_model = _getstr("OLLAMA_VISION_MODEL",
                                      cfg.ollama_vision_model)
    cfg.ollama_fast_model = _getstr("OLLAMA_FAST_MODEL", cfg.ollama_fast_model)
    cfg.ollama_timeout = _getfloat("OLLAMA_TIMEOUT", cfg.ollama_timeout)
    cfg.ollama_keep_alive = _getstr("OLLAMA_KEEP_ALIVE", cfg.ollama_keep_alive)
    cfg.ollama_num_ctx = _getint("OLLAMA_NUM_CTX", cfg.ollama_num_ctx)

    cfg.whisper_model = _getstr("WHISPER_MODEL", cfg.whisper_model)
    cfg.whisper_device = _getstr("WHISPER_DEVICE", cfg.whisper_device).lower()
    cfg.whisper_compute = _getstr("WHISPER_COMPUTE", cfg.whisper_compute)
    cfg.whisper_beam_size = _getint("WHISPER_BEAM_SIZE", cfg.whisper_beam_size)

    cfg.raw_clips = _resolve_path(_get("RAW_CLIPS_DIR", "") or "",
                                  cfg.raw_clips)
    cfg.processed = _resolve_path(_get("PROCESSED_DIR", "") or "",
                                  cfg.processed)
    cfg.exports = _resolve_path(_get("EXPORTS_DIR", "") or "", cfg.exports)
    cfg.logs = _resolve_path(_get("LOGS_DIR", "") or "", cfg.logs)
    cfg.data = _resolve_path(_get("DATA_DIR", "") or "", cfg.data)
    cfg.db_path = _resolve_path(_get("DB_PATH", "") or "", cfg.db_path)
    cfg.db_backups = _resolve_path(_get("DB_BACKUPS_DIR", "") or "",
                                   cfg.db_backups)

    cfg.frame_interval_sec = _getfloat("FRAME_INTERVAL_SEC",
                                       cfg.frame_interval_sec)
    cfg.min_frames = _getint("MIN_FRAMES", cfg.min_frames)
    cfg.max_frames = _getint("MAX_FRAMES", cfg.max_frames)
    cfg.scene_threshold = _getfloat("SCENE_THRESHOLD", cfg.scene_threshold)
    cfg.analyze_vision = _getbool("ANALYZE_VISION", cfg.analyze_vision)
    cfg.vision_batch_size = _getint("VISION_BATCH_SIZE", cfg.vision_batch_size)
    cfg.vision_frame_width = _getint("VISION_FRAME_WIDTH",
                                     cfg.vision_frame_width)

    cfg.min_score_candidate = _getint("MIN_SCORE_CANDIDATE",
                                      cfg.min_score_candidate)
    cfg.min_score_review = _getint("MIN_SCORE_REVIEW", cfg.min_score_review)
    cfg.min_confidence = _getfloat("MIN_CONFIDENCE", cfg.min_confidence)
    cfg.curator_temperature = _getfloat("CURATOR_TEMPERATURE",
                                        cfg.curator_temperature)
    cfg.min_clip_duration = _getfloat("MIN_CLIP_DURATION",
                                      cfg.min_clip_duration)
    cfg.max_clip_duration = _getfloat("MAX_CLIP_DURATION",
                                      cfg.max_clip_duration)

    cfg.video_width = _getint("VIDEO_WIDTH", cfg.video_width)
    cfg.video_height = _getint("VIDEO_HEIGHT", cfg.video_height)
    cfg.video_bitrate = _getstr("VIDEO_BITRATE", cfg.video_bitrate)
    cfg.video_crf = _getint("VIDEO_CRF", cfg.video_crf)
    cfg.video_preset = _getstr("VIDEO_PRESET", cfg.video_preset).lower()
    cfg.edit_mode = _getstr("EDIT_MODE", cfg.edit_mode).lower()

    cfg.subtitles_enabled = _getbool("SUBTITLES_ENABLED", cfg.subtitles_enabled)
    cfg.subtitle_font = _getstr("SUBTITLE_FONT", cfg.subtitle_font)
    cfg.subtitle_font_size = _getint("SUBTITLE_FONT_SIZE",
                                     cfg.subtitle_font_size)
    cfg.subtitle_style = _getstr("SUBTITLE_STYLE", cfg.subtitle_style).lower()
    cfg.subtitle_max_words = _getint("SUBTITLE_MAX_WORDS",
                                     cfg.subtitle_max_words)
    cfg.subtitle_max_chars = _getint("SUBTITLE_MAX_CHARS",
                                     cfg.subtitle_max_chars)
    cfg.subtitle_max_duration = _getfloat("SUBTITLE_MAX_DURATION",
                                          cfg.subtitle_max_duration)
    cfg.subtitle_margin_v = _getint("SUBTITLE_MARGIN_V", cfg.subtitle_margin_v)

    cfg.max_parallel = _getint("MAX_PARALLEL", cfg.max_parallel)
    cfg.retry_attempts = _getint("RETRY_ATTEMPTS", cfg.retry_attempts)
    cfg.max_attempts = _getint("MAX_ATTEMPTS", cfg.max_attempts)
    cfg.stale_lock_minutes = _getint("STALE_LOCK_MINUTES",
                                     cfg.stale_lock_minutes)
    cfg.min_free_disk_mb = _getint("MIN_FREE_DISK_MB", cfg.min_free_disk_mb)
    cfg.cleanup_days = _getint("CLEANUP_DAYS", cfg.cleanup_days)
    cfg.db_backup = _getbool("DB_BACKUP", cfg.db_backup)
    cfg.log_level = _getstr("LOG_LEVEL", cfg.log_level).upper()

    cfg.validate()
    return cfg
