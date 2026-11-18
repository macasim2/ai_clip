"""SQLite persistence."""
import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .utils import normalize_game_name

STATUSES = [
    "DISCOVERED", "DOWNLOADING", "DOWNLOADED", "ANALYZING",
    "ANALYZED", "REJECTED", "APPROVED", "EDITING", "READY", "FAILED",
]

SCHEMA_VERSION = 2


class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    @contextmanager
    def conn(self):
        with self._lock:
            c = sqlite3.connect(self.path, timeout=30)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("PRAGMA foreign_keys=ON")
            try:
                yield c
                c.commit()
            except Exception:
                c.rollback()
                raise
            finally:
                c.close()

    def _init_db(self):
        with self.conn() as c:
            c.executescript("""
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
            CREATE TABLE IF NOT EXISTS clips (
                id TEXT PRIMARY KEY,
                title TEXT,
                game_name TEXT,
                game_id TEXT,
                broadcaster_name TEXT,
                url TEXT,
                created_at TEXT,
                duration REAL,
                view_count INTEGER,
                thumbnail_url TEXT,
                discovered_at TEXT,
                downloaded_at TEXT,
                local_path TEXT,
                analysis_json TEXT,
                score INTEGER,
                best_start REAL,
                best_end REAL,
                status TEXT NOT NULL DEFAULT 'DISCOVERED',
                category TEXT,
                rejection_reason TEXT,
                export_path TEXT,
                error_message TEXT,
                attempts INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS clip_analysis (
                clip_id TEXT PRIMARY KEY REFERENCES clips(id) ON DELETE CASCADE,
                audio_json TEXT,
                vision_json TEXT,
                decision_json TEXT,
                score INTEGER,
                created_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_clips_status ON clips(status);
            CREATE INDEX IF NOT EXISTS idx_clips_game ON clips(game_name);
            CREATE INDEX IF NOT EXISTS idx_clips_score ON clips(score);
            """)
            row = c.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
            current = int(row["value"]) if row else 1
            if current < SCHEMA_VERSION:
                self._migrate(c, current)
                c.execute(
                    "INSERT OR REPLACE INTO meta(key,value) VALUES('schema_version',?)",
                    (str(SCHEMA_VERSION),),
                )

    def _migrate(self, c, _from):
        cols = {r["name"] for r in c.execute("PRAGMA table_info(clips)").fetchall()}
        for name, typ in {
            "game_id": "TEXT", "discovered_at": "TEXT", "downloaded_at": "TEXT",
            "local_path": "TEXT", "analysis_json": "TEXT", "best_start": "REAL",
            "best_end": "REAL", "rejection_reason": "TEXT", "export_path": "TEXT",
            "error_message": "TEXT", "attempts": "INTEGER NOT NULL DEFAULT 0",
        }.items():
            if name not in cols:
                c.execute(f"ALTER TABLE clips ADD COLUMN {name} {typ}")

    def upsert_clip(self, clip) -> bool:
        with self.conn() as c:
            if c.execute("SELECT id FROM clips WHERE id=?", (clip.id,)).fetchone():
                return False
            c.execute(
                """INSERT INTO clips(id,title,game_name,broadcaster_name,url,
                    created_at,duration,view_count,thumbnail_url,discovered_at,status)
                   VALUES(?,?,?,?,?,?,?,?,?,?, 'DISCOVERED')""",
                (clip.id, clip.title, normalize_game_name(clip.game_name),
                 clip.broadcaster_name, clip.url, clip.created_at,
                 float(clip.duration or 0), int(clip.view_count or 0),
                 clip.thumbnail_url or "",
                 datetime.now(timezone.utc).isoformat()),
            )
            return True

    def get_clip(self, clip_id: str) -> Optional[Dict[str, Any]]:
        with self.conn() as c:
            row = c.execute("SELECT * FROM clips WHERE id=?", (clip_id,)).fetchone()
            return dict(row) if row else None

    def update_status(self, clip_id: str, status: str, error: Optional[str] = None):
        if status not in STATUSES:
            raise ValueError(f"Unknown status: {status}")
        with self.conn() as c:
            c.execute("UPDATE clips SET status=?, error_message=? WHERE id=?",
                      (status, error, clip_id))

    def mark_failed(self, clip_id: str, error: str):
        with self.conn() as c:
            c.execute(
                "UPDATE clips SET status='FAILED', error_message=?, "
                "attempts=attempts+1 WHERE id=?",
                (str(error)[:2000], clip_id),
            )

    def set_local_path(self, clip_id: str, path):
        with self.conn() as c:
            c.execute(
                "UPDATE clips SET local_path=?, downloaded_at=? WHERE id=?",
                (str(path), datetime.now(timezone.utc).isoformat(), clip_id),
            )

    def save_analysis(self, clip_id: str, audio_json: str, vision_json: str):
        with self.conn() as c:
            c.execute(
                """INSERT INTO clip_analysis(clip_id,audio_json,vision_json,created_at)
                   VALUES(?,?,?,?)
                   ON CONFLICT(clip_id) DO UPDATE SET
                     audio_json=excluded.audio_json,
                     vision_json=excluded.vision_json,
                     created_at=excluded.created_at""",
                (clip_id, audio_json, vision_json,
                 datetime.now(timezone.utc).isoformat()),
            )

    def save_decision(self, clip_id: str, decision: dict, status: str):
        with self.conn() as c:
            c.execute(
                """INSERT INTO clip_analysis(clip_id,decision_json,score,created_at)
                   VALUES(?,?,?,?)
                   ON CONFLICT(clip_id) DO UPDATE SET
                     decision_json=excluded.decision_json,
                     score=excluded.score,
                     created_at=excluded.created_at""",
                (clip_id, json.dumps(decision, ensure_ascii=False),
                 int(decision.get("overall_score", 0)),
                 datetime.now(timezone.utc).isoformat()),
            )
            c.execute(
                """UPDATE clips SET score=?, best_start=?, best_end=?,
                   category=?, rejection_reason=?, status=? WHERE id=?""",
                (int(decision.get("overall_score", 0)),
                 float(decision.get("best_moment_start", 0)),
                 float(decision.get("best_moment_end", 0)),
                 decision.get("category"),
                 decision.get("reason") if status == "REJECTED" else None,
                 status, clip_id),
            )

    def set_export_path(self, clip_id: str, path):
        with self.conn() as c:
            c.execute(
                "UPDATE clips SET export_path=?, status='READY' WHERE id=?",
                (str(path), clip_id),
            )

    def get_analysis(self, clip_id: str) -> Optional[Dict[str, Any]]:
        with self.conn() as c:
            row = c.execute(
                "SELECT * FROM clip_analysis WHERE clip_id=?", (clip_id,)
            ).fetchone()
            return dict(row) if row else None

    def list_pending(self) -> List[Dict[str, Any]]:
        with self.conn() as c:
            rows = c.execute(
                "SELECT * FROM clips WHERE status NOT IN ('READY','REJECTED') "
                "ORDER BY discovered_at"
            ).fetchall()
            return [dict(r) for r in rows]
