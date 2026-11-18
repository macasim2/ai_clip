import json
import shutil
import subprocess

import pytest

from src.config import Config
from src.database import Database
from src.models import (AudioAnalysisResult, CuratorDecision, DownloadResult,
                        TranscriptWord, TwitchClip, VisionAnalysisResult)
from src.pipeline import Pipeline
from src.utils import probe_media


@pytest.fixture
def sample_video(tmp_path):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg is not available")
    path = tmp_path / "sample.mp4"
    result = subprocess.run(
        [
            shutil.which("ffmpeg"), "-y", "-f", "lavfi",
            "-i", "testsrc=size=640x360:rate=10", "-f", "lavfi",
            "-i", "sine=frequency=440:sample_rate=16000", "-t", "3",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            "-shortest", str(path),
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        pytest.skip("cannot generate sample video")
    return path


def _cfg(tmp_path):
    cfg = Config(
        raw_clips=tmp_path / "raw",
        processed=tmp_path / "processed",
        exports=tmp_path / "exports",
        logs=tmp_path / "logs",
        data=tmp_path / "data",
        db_path=tmp_path / "data" / "clips.db",
        db_backups=tmp_path / "data" / "backups",
        min_clip_duration=2.0,
        max_clip_duration=10.0,
        subtitles_enabled=True,
        analyze_vision=True,
        video_preset="ultrafast",
        video_crf=30,
    ).validate()
    cfg.ensure_dirs()
    return cfg


def test_full_pipeline_with_real_render(tmp_path, sample_video, monkeypatch):
    cfg = _cfg(tmp_path)
    db = Database(cfg.db_path)
    db.upsert_clip(TwitchClip(
        id="int1", title="integration", game_name="Test",
        broadcaster_name="b", url="u", created_at="", duration=3,
        view_count=1,
    ))

    monkeypatch.setattr(
        "src.pipeline.download",
        lambda *a, **k: DownloadResult(path=sample_video, duration=3.0,
                                       size=sample_video.stat().st_size),
    )
    audio = AudioAnalysisResult(
        language="ru", duration=3.0,
        segments=[],
        words=[TranscriptWord(0.3, 0.8, "привет"),
               TranscriptWord(0.9, 1.4, "мир!")],
        events=[], full_text="привет мир!",
    )
    monkeypatch.setattr("src.pipeline.analyze_audio", lambda *a, **k: audio)
    monkeypatch.setattr(
        "src.pipeline.analyze_vision",
        lambda *a, **k: VisionAnalysisResult(3.0, [], "gameplay"),
    )
    decision = CuratorDecision(
        category="Экшен", event_description="test",
        unexpectedness=0.9, emotionality=0.8, payoff_strength=0.9,
        event_clarity=0.9, visual_interest=0.8, hook_strength=0.8,
        context_independence=0.8, overall_score=88, confidence=0.9,
        decision="candidate", best_moment_start=0.2, best_moment_end=2.6,
        reason="integration",
    )
    monkeypatch.setattr("src.pipeline.analyze_clip", lambda *a, **k: decision)

    pipeline = Pipeline(cfg, db, twitch=object())
    result = pipeline.process_one("int1")
    assert result["ok"] is True, result
    row = db.get_clip("int1")
    assert row["status"] == "READY"
    output = row["export_path"]
    info = probe_media(output)
    assert (info.width, info.height) == (1080, 1920)
    assert info.has_audio
    assert 1.5 <= info.duration <= 3.0
    assert (cfg.processed / "int1" / "subtitles.ass").is_file()
    assert (cfg.processed / "int1" / "subtitles.srt").is_file()
    assert (cfg.processed / "int1" / "analysis.json").is_file()
    assert (cfg.processed / "int1" / "vertical.mp4").is_file()
