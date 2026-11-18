import os

import pytest

from src.config import Config, ConfigError, load_config


ENV_PREFIXES = (
    "TWITCH_", "OLLAMA_", "WHISPER_", "FRAME_", "MIN_", "MAX_", "SCENE_",
    "ANALYZE_", "VISION_", "VIDEO_", "EDIT_", "SUBTITLE_", "RETRY_",
    "STALE_", "CLEANUP_", "DB_", "LOG_", "DATA_", "RAW_", "PROCESSED_",
    "EXPORTS_", "LOGS_",
)


def _clear_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith(ENV_PREFIXES):
            monkeypatch.delenv(key, raising=False)


def test_defaults(monkeypatch):
    _clear_env(monkeypatch)
    cfg = load_config(read_env=False)
    assert cfg.twitch_provider == "graphql"
    assert cfg.ollama_text_model == "llama3.2:latest"
    assert cfg.ollama_vision_model == "llava:latest"
    assert cfg.video_width == 1080
    assert cfg.video_height == 1920
    assert cfg.max_parallel == 1
    assert cfg.min_clip_duration <= cfg.max_clip_duration


def test_env_overrides(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("TWITCH_PROVIDER", "helix")
    monkeypatch.setenv("TWITCH_CLIENT_ID", "id")
    monkeypatch.setenv("TWITCH_CLIENT_SECRET", "secret")
    monkeypatch.setenv("OLLAMA_TEXT_MODEL", "qwen2.5:7b-instruct")
    monkeypatch.setenv("VIDEO_CRF", "18")
    cfg = load_config(read_env=False)
    assert cfg.twitch_provider == "helix"
    assert cfg.ollama_text_model == "qwen2.5:7b-instruct"
    assert cfg.video_crf == 18


def test_unknown_provider(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("TWITCH_PROVIDER", "nope")
    with pytest.raises(ConfigError, match="TWITCH_PROVIDER"):
        load_config(read_env=False)


def test_helix_requires_credentials(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("TWITCH_PROVIDER", "helix")
    monkeypatch.delenv("TWITCH_CLIENT_ID", raising=False)
    monkeypatch.delenv("TWITCH_CLIENT_SECRET", raising=False)
    with pytest.raises(ConfigError, match="TWITCH_CLIENT_ID"):
        load_config(read_env=False)


def test_bad_integer(monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("TWITCH_TIMEOUT", "not-a-number")
    with pytest.raises(ConfigError, match="TWITCH_TIMEOUT"):
        load_config(read_env=False)


def test_bad_duration_order():
    with pytest.raises(ConfigError, match="MAX_CLIP_DURATION"):
        Config(min_clip_duration=20.0, max_clip_duration=10.0).validate()


def test_bad_score_order():
    with pytest.raises(ConfigError, match="MIN_SCORE_REVIEW"):
        Config(min_score_candidate=30, min_score_review=50).validate()


def test_ensure_dirs(tmp_path):
    cfg = Config(
        raw_clips=tmp_path / "raw",
        processed=tmp_path / "processed",
        exports=tmp_path / "exports",
        logs=tmp_path / "logs",
        data=tmp_path / "data",
        db_path=tmp_path / "data" / "clips.db",
        db_backups=tmp_path / "data" / "backups",
    ).validate()
    cfg.ensure_dirs()
    for path in (cfg.raw_clips, cfg.processed, cfg.exports, cfg.logs,
                 cfg.data, cfg.db_backups):
        assert path.is_dir()
