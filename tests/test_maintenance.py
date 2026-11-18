import os
import time

from src.config import Config
from src.database import Database
from src.maintenance import cleanup
from src.models import TwitchClip


def _cfg(tmp_path):
    cfg = Config(
        raw_clips=tmp_path / "raw",
        processed=tmp_path / "processed",
        exports=tmp_path / "exports",
        logs=tmp_path / "logs",
        data=tmp_path / "data",
        db_path=tmp_path / "data" / "clips.db",
        db_backups=tmp_path / "data" / "backups",
        cleanup_days=1,
    ).validate()
    cfg.ensure_dirs()
    return cfg


def _clip(clip_id):
    return TwitchClip(id=clip_id, title="t", game_name="G", broadcaster_name="b",
                      url="u", created_at="", duration=2, view_count=1)


def _make_ready(db, clip_id, tmp_path):
    db.upsert_clip(_clip(clip_id))
    db.update_status(clip_id, "DOWNLOADING")
    db.update_status(clip_id, "DOWNLOADED")
    db.update_status(clip_id, "ANALYZING")
    db.update_status(clip_id, "ANALYZED")
    db.save_decision(clip_id, {"category": "Экшен", "overall_score": 80,
                              "best_moment_start": 0, "best_moment_end": 1,
                              "decision": "candidate", "reason": "ok"},
                     "APPROVED")
    db.update_status(clip_id, "EDITING")
    output = tmp_path / f"{clip_id}.mp4"
    output.write_bytes(b"x" * 4096)
    db.set_export_path(clip_id, output)


def _old(path):
    stamp = time.time() - 10 * 86400
    os.utime(path, (stamp, stamp))


def test_cleanup_removes_old_frames_but_keeps_transcript(tmp_path):
    cfg = _cfg(tmp_path)
    db = Database(cfg.db_path)
    _make_ready(db, "c1", tmp_path)
    work = cfg.processed / "c1"
    (work / "frames").mkdir(parents=True)
    (work / "frames" / "f.jpg").write_bytes(b"x")
    (work / "audio.wav").write_bytes(b"x")
    (work / "transcript.json").write_text("{}", encoding="utf-8")
    _old(work / "frames")
    _old(work / "audio.wav")
    _old(work / "transcript.json")

    result = cleanup(cfg, db, days=1)
    assert result["removed_files"] >= 2
    assert not (work / "frames").exists()
    assert not (work / "audio.wav").exists()
    assert (work / "transcript.json").exists()
    assert (tmp_path / "c1.mp4").exists()  # export is never touched


def test_cleanup_keeps_pending_clip(tmp_path):
    cfg = _cfg(tmp_path)
    db = Database(cfg.db_path)
    db.upsert_clip(_clip("pending"))
    work = cfg.processed / "pending"
    (work / "frames").mkdir(parents=True)
    (work / "frames" / "f.jpg").write_bytes(b"x")
    _old(work / "frames")
    result = cleanup(cfg, db, days=1)
    assert result["kept_pending"] == 1
    assert (work / "frames").exists()


def test_cleanup_dry_run(tmp_path):
    cfg = _cfg(tmp_path)
    db = Database(cfg.db_path)
    _make_ready(db, "c1", tmp_path)
    work = cfg.processed / "c1"
    (work / "frames").mkdir(parents=True)
    (work / "frames" / "f.jpg").write_bytes(b"x")
    _old(work / "frames")
    cleanup(cfg, db, days=1, dry_run=True)
    assert (work / "frames").exists()


def test_cleanup_orphans_only_with_flag(tmp_path):
    cfg = _cfg(tmp_path)
    db = Database(cfg.db_path)
    orphan = cfg.processed / "orphan"
    orphan.mkdir(parents=True)
    (orphan / "audio.wav").write_bytes(b"x")
    _old(orphan)
    cleanup(cfg, db, days=1)
    assert orphan.exists()
    cleanup(cfg, db, days=1, orphans=True)
    assert not orphan.exists()
