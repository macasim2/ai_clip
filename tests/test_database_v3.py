import json
import sqlite3
from pathlib import Path

import pytest

from src.database import Database, DatabaseError, InvalidTransition
from src.models import TwitchClip


def make_clip(clip_id="c1", **overrides):
    values = dict(
        id=clip_id, title="title", game_name="CS2",
        broadcaster_name="streamer", url="https://example/clip",
        created_at="2026-01-01T00:00:00Z", duration=10.0, view_count=100,
    )
    values.update(overrides)
    return TwitchClip(**values)


@pytest.fixture
def db(tmp_path):
    return Database(tmp_path / "clips.db")


def drive_to_editing(db, clip_id="c1"):
    db.update_status(clip_id, "DOWNLOADING", stage="download")
    db.update_status(clip_id, "DOWNLOADED", stage="download")
    db.update_status(clip_id, "ANALYZING", stage="analysis")
    db.update_status(clip_id, "ANALYZED", stage="curator")
    db.save_decision(clip_id, {
        "category": "Экшен", "overall_score": 80,
        "best_moment_start": 1.0, "best_moment_end": 5.0,
        "reason": "test",
    }, "APPROVED")
    db.update_status(clip_id, "EDITING", stage="edit")


def test_upsert_and_metadata_refresh(db):
    assert db.upsert_clip(make_clip()) is True
    assert db.upsert_clip(make_clip(title="new title", view_count=999)) is False
    row = db.get_clip("c1")
    assert row["title"] == "new title"
    assert row["view_count"] == 999
    assert row["status"] == "DISCOVERED"


def test_invalid_transition(db):
    db.upsert_clip(make_clip())
    with pytest.raises(InvalidTransition):
        db.update_status("c1", "READY")


def test_history_is_recorded(db):
    db.upsert_clip(make_clip())
    drive_to_editing(db)
    history = db.history("c1")
    pairs = [(item["from_status"], item["to_status"]) for item in history]
    assert ("DISCOVERED", "DOWNLOADING") in pairs
    assert ("APPROVED", "EDITING") in pairs
    assert pairs[0] == ("APPROVED", "EDITING")


def test_claim_release_and_lock(db):
    for clip_id in ("a", "b", "c"):
        db.upsert_clip(make_clip(clip_id))
    claimed = db.claim_next(worker_id="worker-1")
    assert claimed is not None
    assert claimed["locked_by"] == "worker-1"
    other = db.claim_next(worker_id="worker-2")
    assert other is not None and other["id"] != claimed["id"]
    assert db.release(claimed["id"], "worker-1") is True
    again = db.claim_next(worker_id="worker-3")
    assert again is not None
    assert db.release(again["id"], "wrong-worker") is False


def test_stale_downloading_is_recovered(db, tmp_path):
    db.upsert_clip(make_clip())
    db.update_status("c1", "DOWNLOADING", stage="download")
    # Simulate a crash: old lock, no file.
    with db.conn() as connection:
        connection.execute(
            "UPDATE clips SET locked_by='dead', locked_at='2000-01-01T00:00:00+00:00' "
            "WHERE id='c1'"
        )
    changes = db.recover_stale(stale_minutes=30)
    assert changes == [{"clip_id": "c1", "from": "DOWNLOADING",
                        "to": "DISCOVERED"}]
    row = db.get_clip("c1")
    assert row["status"] == "DISCOVERED"
    assert row["locked_by"] is None


def test_stale_analyzing_with_analysis_becomes_analyzed(db):
    db.upsert_clip(make_clip())
    db.update_status("c1", "DOWNLOADING")
    db.update_status("c1", "DOWNLOADED")
    db.update_status("c1", "ANALYZING")
    db.save_analysis("c1", "{}", "{}")
    with db.conn() as connection:
        connection.execute(
            "UPDATE clips SET locked_by='dead', locked_at='2000-01-01T00:00:00+00:00' "
            "WHERE id='c1'"
        )
    db.recover_stale(30)
    assert db.get_clip("c1")["status"] == "ANALYZED"


def test_ready_requires_a_real_file(db, tmp_path):
    db.upsert_clip(make_clip())
    drive_to_editing(db)
    missing = tmp_path / "missing.mp4"
    with pytest.raises(DatabaseError, match="Refusing to mark READY"):
        db.set_export_path("c1", missing)
    assert db.get_clip("c1")["status"] == "EDITING"

    output = tmp_path / "out.mp4"
    output.write_bytes(b"x" * 4096)
    db.set_export_path("c1", output)
    row = db.get_clip("c1")
    assert row["status"] == "READY"
    assert row["export_path"] == str(output)


def test_reconcile_ready_downgrades_missing_export(db, tmp_path):
    db.upsert_clip(make_clip())
    drive_to_editing(db)
    output = tmp_path / "out.mp4"
    output.write_bytes(b"x" * 4096)
    db.set_export_path("c1", output)
    output.unlink()
    broken = db.reconcile_ready()
    assert broken == [{"clip_id": "c1", "from": "READY", "to": "FAILED"}]
    assert db.get_clip("c1")["status"] == "FAILED"


def test_save_decision_review_and_reject(db):
    db.upsert_clip(make_clip())
    db.update_status("c1", "DOWNLOADING")
    db.update_status("c1", "DOWNLOADED")
    db.update_status("c1", "ANALYZING")
    db.update_status("c1", "ANALYZED")
    decision = {"category": "Скучное", "overall_score": 12,
                "best_moment_start": 0, "best_moment_end": 0,
                "reason": "no event"}
    db.save_decision("c1", decision, "REVIEW")
    row = db.get_clip("c1")
    assert row["status"] == "REVIEW"
    assert row["review_reason"] == "no event"
    db.save_decision("c1", decision, "REJECTED")
    row = db.get_clip("c1")
    assert row["status"] == "REJECTED"
    assert row["rejection_reason"] == "no event"


def test_analysis_error_is_distinct(db):
    db.upsert_clip(make_clip())
    db.update_status("c1", "DOWNLOADING")
    db.update_status("c1", "DOWNLOADED")
    db.update_status("c1", "ANALYZING")
    db.mark_analysis_error("c1", "bad json", "curator")
    row = db.get_clip("c1")
    assert row["status"] == "ANALYSIS_ERROR"
    assert row["attempts"] == 1
    assert row["error_message"] == "bad json"


def test_reset_for_retry_from_edit(db, tmp_path):
    db.upsert_clip(make_clip())
    drive_to_editing(db)
    output = tmp_path / "out.mp4"
    output.write_bytes(b"x" * 4096)
    db.set_export_path("c1", output)
    reset = db.reset_for_retry("c1", "edit")
    assert reset["status"] == "APPROVED"
    assert reset["export_path"] is None
    assert reset["attempts"] == 0


def test_stats_and_indexes(db, tmp_path):
    db.upsert_clip(make_clip())
    stats = db.stats()
    assert stats["DISCOVERED"] == 1
    with sqlite3.connect(tmp_path / "clips.db") as connection:
        indexes = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        )}
    assert "idx_clips_status_discovered" in indexes
    assert "idx_history_clip" in indexes


def _create_v2_database(path: Path):
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
        INSERT INTO meta VALUES ('schema_version', '2');
        CREATE TABLE clips (
            id TEXT PRIMARY KEY, title TEXT, game_name TEXT, game_id TEXT,
            broadcaster_name TEXT, url TEXT, created_at TEXT, duration REAL,
            view_count INTEGER, thumbnail_url TEXT, discovered_at TEXT,
            downloaded_at TEXT, local_path TEXT, analysis_json TEXT,
            score INTEGER, best_start REAL, best_end REAL,
            status TEXT NOT NULL DEFAULT 'DISCOVERED', category TEXT,
            rejection_reason TEXT, export_path TEXT, error_message TEXT,
            attempts INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE clip_analysis (
            clip_id TEXT PRIMARY KEY, audio_json TEXT, vision_json TEXT,
            decision_json TEXT, score INTEGER, created_at TEXT
        );
        INSERT INTO clips (id, title, status, discovered_at)
        VALUES ('old', 'old clip', 'DISCOVERED', '2026-01-01T00:00:00+00:00');
        """
    )
    connection.commit()
    connection.close()


def test_migration_from_v2_preserves_data(tmp_path):
    path = tmp_path / "clips.db"
    _create_v2_database(path)
    db = Database(path)
    with sqlite3.connect(path) as connection:
        version = connection.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()[0]
        columns = {row[1] for row in connection.execute(
            "PRAGMA table_info(clips)"
        )}
        analysis_columns = {row[1] for row in connection.execute(
            "PRAGMA table_info(clip_analysis)"
        )}
    assert int(version) == 3
    assert {"locked_by", "stage", "updated_at", "review_reason"} <= columns
    assert "updated_at" in analysis_columns
    assert db.get_clip("old")["title"] == "old clip"
    backups = list((tmp_path / "backups").glob("*.db"))
    assert len(backups) == 1
