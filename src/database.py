"""SQLite persistence: queue state, statuses, history and migrations.

The database is the single source of truth for a clip's lifecycle.  All
state changes go through short transactions with an explicit status
transition table, so a crash between stages can be recovered instead of
producing a fake READY row.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .logging_setup import get_logger
from .utils import normalize_game_name, utc_now_iso

log = get_logger(__name__)

SCHEMA_VERSION = 3

REQUIRED_STATUSES = [
    "DISCOVERED", "DOWNLOADING", "DOWNLOADED", "ANALYZING",
    "ANALYZED", "REJECTED", "APPROVED", "EDITING", "READY", "FAILED",
]
EXTRA_STATUSES = ["REVIEW", "ANALYSIS_ERROR"]
STATUSES = REQUIRED_STATUSES + EXTRA_STATUSES

TERMINAL_STATUSES = {"READY", "REJECTED"}
IN_PROGRESS_STATUSES = {"DOWNLOADING", "ANALYZING", "EDITING"}
FAILURE_STATUSES = {"FAILED", "ANALYSIS_ERROR"}
# Statuses that the worker may pick automatically. REVIEW deliberately is not
# here: it needs a human decision or a stronger second-pass model.
AUTO_QUEUE_STATUSES = ("DISCOVERED", "DOWNLOADED", "ANALYZED", "APPROVED")
QUEUE_ORDER = {
    "DISCOVERED": 0,
    "DOWNLOADED": 1,
    "ANALYZED": 2,
    "APPROVED": 3,
}

ALLOWED_TRANSITIONS: Dict[str, set] = {
    "DISCOVERED": {"DOWNLOADING", "DOWNLOADED", "FAILED", "ANALYSIS_ERROR",
                   "DISCOVERED"},
    "DOWNLOADING": {"DOWNLOADED", "DISCOVERED", "FAILED", "ANALYSIS_ERROR"},
    "DOWNLOADED": {"ANALYZING", "DOWNLOADING", "DISCOVERED", "FAILED",
                   "ANALYSIS_ERROR"},
    "ANALYZING": {"ANALYZED", "ANALYSIS_ERROR", "DOWNLOADED", "DOWNLOADING",
                  "DISCOVERED", "FAILED"},
    "ANALYZED": {"APPROVED", "REVIEW", "REJECTED", "ANALYSIS_ERROR",
                 "ANALYZING", "DOWNLOADING"},
    "REVIEW": {"APPROVED", "REJECTED", "ANALYZING", "ANALYSIS_ERROR"},
    "APPROVED": {"EDITING", "REVIEW", "REJECTED", "ANALYSIS_ERROR", "FAILED",
                 "DOWNLOADED", "DOWNLOADING", "ANALYZING"},
    "EDITING": {"READY", "APPROVED", "FAILED", "ANALYSIS_ERROR", "REVIEW",
                "REJECTED", "DOWNLOADING", "ANALYZING"},
    "READY": {"APPROVED", "EDITING", "DISCOVERED"},
    "REJECTED": {"DISCOVERED", "ANALYZED", "REVIEW"},
    "FAILED": {"DISCOVERED", "DOWNLOADING", "ANALYZED", "APPROVED",
               "EDITING"},
    "ANALYSIS_ERROR": {"DISCOVERED", "ANALYZING", "REVIEW", "APPROVED",
                       "REJECTED", "EDITING"},
}

_COLUMN_MIGRATIONS = {
    "game_id": "TEXT",
    "source_size": "INTEGER",
    "source_fingerprint": "TEXT",
    "stage": "TEXT",
    "stage_started_at": "TEXT",
    "locked_by": "TEXT",
    "locked_at": "TEXT",
    "finished_at": "TEXT",
    "analysis_version": "TEXT",
    "updated_at": "TEXT",
    "review_reason": "TEXT",
}


class DatabaseError(RuntimeError):
    pass


class InvalidTransition(DatabaseError):
    pass


def _now() -> str:
    return utc_now_iso()


def _cutoff(minutes: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        if result != result or result in (float("inf"), float("-inf")):
            return default
        return result
    except (TypeError, ValueError):
        return default


def _maybe_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _clip_value(clip: Any, name: str, default: Any = None) -> Any:
    if isinstance(clip, dict):
        return clip.get(name, default)
    return getattr(clip, name, default)


class Database:
    def __init__(self, path: Path, backup: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._backup_enabled = backup
        self._db_existed = self.path.is_file() and self.path.stat().st_size > 0
        self._lock = threading.RLock()
        self._init_db()

    # ------------------------------------------------------------- connections
    @contextmanager
    def conn(self):
        with self._lock:
            connection = sqlite3.connect(self.path, timeout=30.0)
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA busy_timeout=30000")
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA foreign_keys=ON")
            connection.execute("PRAGMA synchronous=NORMAL")
            try:
                yield connection
                connection.commit()
            except Exception:
                connection.rollback()
                raise
            finally:
                connection.close()

    # -------------------------------------------------------------- migrations
    def _create_meta(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)"
        )

    def _has_table(self, connection: sqlite3.Connection, name: str) -> bool:
        row = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (name,),
        ).fetchone()
        return row is not None

    def _read_version(self, connection: sqlite3.Connection) -> int:
        if not self._has_table(connection, "meta"):
            return 0
        row = connection.execute(
            "SELECT value FROM meta WHERE key='schema_version'"
        ).fetchone()
        if row is not None:
            try:
                return int(row["value"])
            except (TypeError, ValueError):
                return 0
        return 1 if self._has_table(connection, "clips") else 0

    def _set_version(self, connection: sqlite3.Connection, version: int) -> None:
        connection.execute(
            "INSERT OR REPLACE INTO meta(key,value) VALUES('schema_version',?)",
            (str(version),),
        )

    def _create_tables(self, connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
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
                source_size INTEGER,
                source_fingerprint TEXT,
                analysis_json TEXT,
                score INTEGER,
                best_start REAL,
                best_end REAL,
                status TEXT NOT NULL DEFAULT 'DISCOVERED',
                category TEXT,
                rejection_reason TEXT,
                review_reason TEXT,
                export_path TEXT,
                error_message TEXT,
                attempts INTEGER NOT NULL DEFAULT 0,
                stage TEXT,
                stage_started_at TEXT,
                locked_by TEXT,
                locked_at TEXT,
                finished_at TEXT,
                analysis_version TEXT,
                updated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS clip_analysis (
                clip_id TEXT PRIMARY KEY REFERENCES clips(id) ON DELETE CASCADE,
                audio_json TEXT,
                vision_json TEXT,
                decision_json TEXT,
                score INTEGER,
                created_at TEXT,
                updated_at TEXT
            );
            CREATE TABLE IF NOT EXISTS status_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                clip_id TEXT NOT NULL REFERENCES clips(id) ON DELETE CASCADE,
                from_status TEXT,
                to_status TEXT NOT NULL,
                stage TEXT,
                error TEXT,
                created_at TEXT NOT NULL
            );
            """
        )

    def _create_indexes(self, connection: sqlite3.Connection) -> None:
        connection.executescript(
            """
            CREATE INDEX IF NOT EXISTS idx_clips_status_discovered
                ON clips(status, discovered_at);
            CREATE INDEX IF NOT EXISTS idx_clips_game_status
                ON clips(game_name, status);
            CREATE INDEX IF NOT EXISTS idx_clips_game_id_status
                ON clips(game_id, status);
            CREATE INDEX IF NOT EXISTS idx_clips_locked
                ON clips(locked_at);
            CREATE INDEX IF NOT EXISTS idx_clips_updated
                ON clips(updated_at);
            CREATE INDEX IF NOT EXISTS idx_history_clip
                ON status_history(clip_id, created_at);
            CREATE INDEX IF NOT EXISTS idx_analysis_score
                ON clip_analysis(score);
            """
        )

    def _backup_before_migration(self, connection: sqlite3.Connection) -> Optional[Path]:
        if not self._backup_enabled or not self._db_existed:
            return None
        backup_dir = self.path.parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        target = backup_dir / f"{self.path.stem}-{stamp}.db"
        try:
            destination = sqlite3.connect(target)
            try:
                connection.backup(destination)
            finally:
                destination.close()
        except Exception as exc:
            if target.exists():
                target.unlink(missing_ok=True)
            raise DatabaseError(
                f"Cannot back up database before migration: {exc}"
            ) from exc
        log.info("Database backup created: %s", target)
        return target

    def _migrate(self, connection: sqlite3.Connection, from_version: int) -> None:
        columns = {
            row["name"] for row in
            connection.execute("PRAGMA table_info(clips)").fetchall()
        }
        for name, column_type in _COLUMN_MIGRATIONS.items():
            if name not in columns:
                connection.execute(
                    f"ALTER TABLE clips ADD COLUMN {name} {column_type}"
                )
        analysis_columns = {
            row["name"] for row in
            connection.execute("PRAGMA table_info(clip_analysis)").fetchall()
        }
        if "updated_at" not in analysis_columns:
            connection.execute(
                "ALTER TABLE clip_analysis ADD COLUMN updated_at TEXT"
            )
        connection.execute(
            "UPDATE clips SET updated_at=COALESCE(updated_at, discovered_at, "
            "created_at, ?) WHERE updated_at IS NULL",
            (_now(),),
        )
        log.info("Database migrated from schema v%s to v%s",
                 from_version, SCHEMA_VERSION)

    def _init_db(self) -> None:
        with self.conn() as connection:
            self._create_meta(connection)
            version = self._read_version(connection)
            if version > SCHEMA_VERSION:
                raise DatabaseError(
                    f"Database schema v{version} is newer than supported "
                    f"v{SCHEMA_VERSION}. Update the application."
                )
            if version == 0:
                self._create_tables(connection)
                self._create_indexes(connection)
                self._set_version(connection, SCHEMA_VERSION)
                return
            if version < SCHEMA_VERSION:
                self._backup_before_migration(connection)
            self._create_tables(connection)
            if version < SCHEMA_VERSION:
                self._migrate(connection, version)
                self._set_version(connection, SCHEMA_VERSION)
            self._create_indexes(connection)

    # ------------------------------------------------------------------ helpers
    def _history(self, connection: sqlite3.Connection, clip_id: str,
                 from_status: Optional[str], to_status: str,
                 stage: Optional[str] = None,
                 error: Optional[str] = None) -> None:
        connection.execute(
            """INSERT INTO status_history
               (clip_id, from_status, to_status, stage, error, created_at)
               VALUES (?,?,?,?,?,?)""",
            (clip_id, from_status, to_status, stage,
             (str(error)[:4000] if error else None), _now()),
        )

    def _clip_row(self, connection: sqlite3.Connection,
                  clip_id: str) -> Optional[sqlite3.Row]:
        return connection.execute(
            "SELECT * FROM clips WHERE id=?", (clip_id,)
        ).fetchone()

    def _transition_allowed(self, old: str, new: str) -> bool:
        return new in ALLOWED_TRANSITIONS.get(old, set())

    # -------------------------------------------------------------------- CRUD
    def upsert_clip(self, clip: Any) -> bool:
        """Insert a new clip or refresh its metadata. Never resets progress."""
        clip_id = str(_clip_value(clip, "id", "") or "").strip()
        if not clip_id:
            raise DatabaseError("Cannot insert clip without id")
        title = str(_clip_value(clip, "title", "") or "")
        game_name = normalize_game_name(
            str(_clip_value(clip, "game_name", "") or "")
        )
        game_id = str(_clip_value(clip, "game_id", "") or "")
        broadcaster = str(_clip_value(clip, "broadcaster_name", "") or "")
        url = str(_clip_value(clip, "url", "") or "")
        created_at = str(_clip_value(clip, "created_at", "") or "")
        duration = _safe_float(_clip_value(clip, "duration", 0))
        views = int(_safe_float(_clip_value(clip, "view_count", 0)))
        thumbnail = str(_clip_value(clip, "thumbnail_url", "") or "")
        now = _now()

        with self.conn() as connection:
            existing = self._clip_row(connection, clip_id)
            if existing is None:
                connection.execute(
                    """INSERT INTO clips
                       (id,title,game_name,game_id,broadcaster_name,url,
                        created_at,duration,view_count,thumbnail_url,
                        discovered_at,status,updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,'DISCOVERED',?)""",
                    (clip_id, title, game_name, game_id, broadcaster, url,
                     created_at, duration, views, thumbnail, now, now),
                )
                self._history(connection, clip_id, None, "DISCOVERED",
                              stage="discovered")
                return True

            connection.execute(
                """UPDATE clips SET title=?, game_name=?, game_id=?,
                   broadcaster_name=?, url=?, created_at=?, duration=?,
                   view_count=?, thumbnail_url=?, updated_at=?
                   WHERE id=?""",
                (title, game_name, game_id, broadcaster, url, created_at,
                 duration, views, thumbnail, now, clip_id),
            )
            return False

    def get_clip(self, clip_id: str) -> Optional[Dict[str, Any]]:
        with self.conn() as connection:
            row = self._clip_row(connection, clip_id)
            return dict(row) if row else None

    def get_clips(self, clip_ids: Sequence[str]) -> List[Dict[str, Any]]:
        if not clip_ids:
            return []
        placeholders = ",".join("?" for _ in clip_ids)
        with self.conn() as connection:
            rows = connection.execute(
                f"SELECT * FROM clips WHERE id IN ({placeholders})",
                tuple(clip_ids),
            ).fetchall()
            return [dict(row) for row in rows]

    def update_status(self, clip_id: str, status: str, error: Optional[str] = None,
                      stage: Optional[str] = None, force: bool = False,
                      expect: Optional[Iterable[str]] = None) -> bool:
        if status not in STATUSES:
            raise ValueError(f"Unknown status: {status}")
        with self.conn() as connection:
            row = self._clip_row(connection, clip_id)
            if row is None:
                raise KeyError(f"Clip not found: {clip_id}")
            old = row["status"]
            if expect is not None and old not in set(expect):
                return False
            if old != status and not force and not self._transition_allowed(old, status):
                raise InvalidTransition(
                    f"{clip_id}: {old} -> {status} is not allowed"
                )
            now = _now()
            finished = now if status in TERMINAL_STATUSES else None
            clear_lock = status in TERMINAL_STATUSES or status in FAILURE_STATUSES \
                or status in {"REVIEW", "REJECTED"}
            fields = ["status=?", "updated_at=?", "finished_at=?"]
            params: List[Any] = [status, now, finished]
            if stage is not None:
                fields.extend(["stage=?", "stage_started_at=?"])
                params.extend([stage, now])
            if error is not None:
                fields.append("error_message=?")
                params.append(str(error)[:4000])
            elif status not in FAILURE_STATUSES:
                fields.append("error_message=NULL")
            if clear_lock:
                fields.append("locked_by=NULL")
                fields.append("locked_at=NULL")
            params.append(clip_id)
            connection.execute(
                f"UPDATE clips SET {', '.join(fields)} WHERE id=?", params
            )
            if old != status:
                self._history(connection, clip_id, old, status, stage, error)
            return True

    def _mark_failure(self, clip_id: str, status: str, error: str,
                      stage: Optional[str]) -> bool:
        if status not in FAILURE_STATUSES:
            raise ValueError(f"Not a failure status: {status}")
        with self.conn() as connection:
            row = self._clip_row(connection, clip_id)
            if row is None:
                raise KeyError(f"Clip not found: {clip_id}")
            old = row["status"]
            if old in TERMINAL_STATUSES:
                raise InvalidTransition(
                    f"{clip_id}: cannot mark {old} as {status}"
                )
            now = _now()
            connection.execute(
                """UPDATE clips SET status=?, error_message=?, attempts=attempts+1,
                   stage=?, stage_started_at=?, finished_at=?, updated_at=?,
                   locked_by=NULL, locked_at=NULL WHERE id=?""",
                (status, str(error)[:4000], stage, now, now, now, clip_id),
            )
            self._history(connection, clip_id, old, status, stage, error)
            return True

    def mark_failed(self, clip_id: str, error: str,
                    stage: Optional[str] = None) -> bool:
        return self._mark_failure(clip_id, "FAILED", error, stage)

    def mark_analysis_error(self, clip_id: str, error: str,
                            stage: str = "curator") -> bool:
        return self._mark_failure(clip_id, "ANALYSIS_ERROR", error, stage)

    def set_local_path(self, clip_id: str, path: Path,
                       size: Optional[int] = None,
                       fingerprint: Optional[str] = None) -> None:
        path = Path(path)
        size = int(size if size is not None else (
            path.stat().st_size if path.is_file() else 0
        ))
        with self.conn() as connection:
            connection.execute(
                """UPDATE clips SET local_path=?, source_size=?,
                   source_fingerprint=COALESCE(?, source_fingerprint),
                   downloaded_at=COALESCE(downloaded_at, ?), updated_at=?
                   WHERE id=?""",
                (str(path), size, fingerprint, _now(), _now(), clip_id),
            )

    def save_analysis(self, clip_id: str, audio_json: str, vision_json: str,
                      version: Optional[str] = None) -> None:
        audio_value = _maybe_json(audio_json)
        vision_value = _maybe_json(vision_json)
        combined = json.dumps(
            {"audio": audio_value, "vision": vision_value},
            ensure_ascii=False,
        )
        now = _now()
        with self.conn() as connection:
            if self._clip_row(connection, clip_id) is None:
                raise KeyError(f"Clip not found: {clip_id}")
            connection.execute(
                """INSERT INTO clip_analysis
                   (clip_id,audio_json,vision_json,created_at,updated_at)
                   VALUES (?,?,?,?,?)
                   ON CONFLICT(clip_id) DO UPDATE SET
                     audio_json=excluded.audio_json,
                     vision_json=excluded.vision_json,
                     updated_at=excluded.updated_at""",
                (clip_id, json.dumps(audio_value, ensure_ascii=False),
                 json.dumps(vision_value, ensure_ascii=False), now, now),
            )
            connection.execute(
                """UPDATE clips SET analysis_json=?,
                   analysis_version=COALESCE(?, analysis_version),
                   updated_at=? WHERE id=?""",
                (combined, version, now, clip_id),
            )

    def save_decision(self, clip_id: str, decision: Any, status: str) -> None:
        if status not in {"APPROVED", "REVIEW", "REJECTED"}:
            raise ValueError(f"Decision status must be APPROVED/REVIEW/REJECTED, "
                             f"got {status}")
        if hasattr(decision, "__dict__") and not isinstance(decision, dict):
            decision = dict(decision.__dict__)
        if not isinstance(decision, dict):
            raise TypeError("decision must be a dict-like object")
        score = int(_safe_float(decision.get("overall_score", 0)))
        start = _safe_float(decision.get("best_moment_start", 0))
        end = _safe_float(decision.get("best_moment_end", 0))
        category = str(decision.get("category") or "")
        reason = str(decision.get("reason") or "")
        decision_json = json.dumps(decision, ensure_ascii=False)
        now = _now()

        with self.conn() as connection:
            row = self._clip_row(connection, clip_id)
            if row is None:
                raise KeyError(f"Clip not found: {clip_id}")
            old = row["status"]
            if old != status and not self._transition_allowed(old, status):
                raise InvalidTransition(
                    f"{clip_id}: {old} -> {status} is not allowed"
                )
            connection.execute(
                """INSERT INTO clip_analysis
                   (clip_id,decision_json,score,created_at,updated_at)
                   VALUES (?,?,?,?,?)
                   ON CONFLICT(clip_id) DO UPDATE SET
                     decision_json=excluded.decision_json,
                     score=excluded.score,
                     updated_at=excluded.updated_at""",
                (clip_id, decision_json, score, now, now),
            )
            connection.execute(
                """UPDATE clips SET score=?, best_start=?, best_end=?,
                   category=?, rejection_reason=?, review_reason=?,
                   status=?, finished_at=?, updated_at=?, error_message=NULL,
                   locked_by=NULL, locked_at=NULL WHERE id=?""",
                (score, start, end, category,
                 reason if status == "REJECTED" else None,
                 reason if status == "REVIEW" else None,
                 status, now if status == "REJECTED" else None, now, clip_id),
            )
            if old != status:
                self._history(connection, clip_id, old, status, "curator", reason)

    def set_export_path(self, clip_id: str, path: Path) -> None:
        path = Path(path)
        if not path.is_file() or path.stat().st_size < 1024:
            raise DatabaseError(
                f"Refusing to mark READY: export file is missing or empty: {path}"
            )
        with self.conn() as connection:
            row = self._clip_row(connection, clip_id)
            if row is None:
                raise KeyError(f"Clip not found: {clip_id}")
            old = row["status"]
            if old != "READY" and not self._transition_allowed(old, "READY"):
                raise InvalidTransition(
                    f"{clip_id}: {old} -> READY is not allowed"
                )
            now = _now()
            connection.execute(
                """UPDATE clips SET export_path=?, status='READY',
                   finished_at=?, updated_at=?, error_message=NULL,
                   locked_by=NULL, locked_at=NULL WHERE id=?""",
                (str(path), now, now, clip_id),
            )
            if old != "READY":
                self._history(connection, clip_id, old, "READY", "export", None)

    def get_analysis(self, clip_id: str) -> Optional[Dict[str, Any]]:
        with self.conn() as connection:
            row = connection.execute(
                "SELECT * FROM clip_analysis WHERE clip_id=?", (clip_id,)
            ).fetchone()
            return dict(row) if row else None

    # ------------------------------------------------------------------- queue
    def list_pending(self, game: Optional[str] = None,
                     limit: Optional[int] = None) -> List[Dict[str, Any]]:
        query = (
            "SELECT * FROM clips WHERE status NOT IN ('READY','REJECTED')"
        )
        params: List[Any] = []
        if game:
            query += " AND (game_name=? OR game_id=?)"
            params.extend([normalize_game_name(game), game])
        query += " ORDER BY discovered_at, id"
        if limit:
            query += " LIMIT ?"
            params.append(int(limit))
        with self.conn() as connection:
            return [dict(row) for row in connection.execute(query, params)]

    def list_all(self, limit: int = 200) -> List[Dict[str, Any]]:
        with self.conn() as connection:
            rows = connection.execute(
                "SELECT * FROM clips ORDER BY COALESCE(updated_at, discovered_at) "
                "DESC, id LIMIT ?",
                (int(limit),),
            ).fetchall()
            return [dict(row) for row in rows]

    def list_by_status(self, status: str,
                       limit: Optional[int] = None) -> List[Dict[str, Any]]:
        if status not in STATUSES:
            raise ValueError(f"Unknown status: {status}")
        query = "SELECT * FROM clips WHERE status=? ORDER BY updated_at, id"
        params: List[Any] = [status]
        if limit:
            query += " LIMIT ?"
            params.append(int(limit))
        with self.conn() as connection:
            return [dict(row) for row in connection.execute(query, params)]

    def list_review(self) -> List[Dict[str, Any]]:
        with self.conn() as connection:
            rows = connection.execute(
                "SELECT * FROM clips WHERE status IN ('REVIEW','ANALYSIS_ERROR') "
                "ORDER BY updated_at"
            ).fetchall()
            return [dict(row) for row in rows]

    def list_failed(self) -> List[Dict[str, Any]]:
        with self.conn() as connection:
            rows = connection.execute(
                "SELECT * FROM clips WHERE status IN ('FAILED','ANALYSIS_ERROR') "
                "ORDER BY updated_at"
            ).fetchall()
            return [dict(row) for row in rows]

    def stats(self) -> Dict[str, int]:
        counts = {status: 0 for status in STATUSES}
        with self.conn() as connection:
            for row in connection.execute(
                "SELECT status, COUNT(*) AS n FROM clips GROUP BY status"
            ):
                counts[row["status"]] = int(row["n"])
        return counts

    def total(self) -> int:
        with self.conn() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM clips").fetchone()[0])

    # ------------------------------------------------------------- worker locks
    def claim_next(self, worker_id: Optional[str] = None,
                   game: Optional[str] = None,
                   statuses: Optional[Sequence[str]] = None,
                   stale_minutes: int = 30) -> Optional[Dict[str, Any]]:
        worker_id = worker_id or default_worker_id()
        statuses = tuple(statuses or AUTO_QUEUE_STATUSES)
        for status in statuses:
            if status not in STATUSES:
                raise ValueError(f"Unknown status: {status}")
        cutoff = _cutoff(stale_minutes)
        placeholders = ",".join("?" for _ in statuses)
        order = " ".join(
            f"WHEN '{status}' THEN {QUEUE_ORDER.get(status, 9)}"
            for status in statuses
        )
        base_where = (
            f"status IN ({placeholders}) "
            "AND (locked_by IS NULL OR locked_at IS NULL OR locked_at < ?)"
        )
        params: List[Any] = list(statuses) + [cutoff]
        if game:
            base_where += " AND (game_name=? OR game_id=?)"
            params.extend([normalize_game_name(game), game])
        select_sql = (
            f"SELECT id FROM clips WHERE {base_where} "
            f"ORDER BY CASE status {order} ELSE 9 END, "
            "COALESCE(discovered_at,''), id LIMIT 1"
        )
        for _ in range(5):
            with self.conn() as connection:
                row = connection.execute(select_sql, tuple(params)).fetchone()
                if row is None:
                    return None
                now = _now()
                updated = connection.execute(
                    """UPDATE clips SET locked_by=?, locked_at=?, updated_at=?
                       WHERE id=? AND (locked_by IS NULL OR locked_at IS NULL
                       OR locked_at < ?)""",
                    (worker_id, now, now, row["id"], cutoff),
                )
                if updated.rowcount == 1:
                    claimed = self._clip_row(connection, row["id"])
                    return dict(claimed) if claimed else None
        log.warning("Could not claim a clip after 5 attempts (contention)")
        return None

    def release(self, clip_id: str, worker_id: Optional[str] = None) -> bool:
        with self.conn() as connection:
            if worker_id is None:
                result = connection.execute(
                    "UPDATE clips SET locked_by=NULL, locked_at=NULL, updated_at=? "
                    "WHERE id=?",
                    (_now(), clip_id),
                )
            else:
                result = connection.execute(
                    "UPDATE clips SET locked_by=NULL, locked_at=NULL, updated_at=? "
                    "WHERE id=? AND locked_by=?",
                    (_now(), clip_id, worker_id),
                )
            return result.rowcount > 0

    def heartbeat(self, clip_id: str, worker_id: str) -> bool:
        with self.conn() as connection:
            result = connection.execute(
                "UPDATE clips SET locked_at=? WHERE id=? AND locked_by=?",
                (_now(), clip_id, worker_id),
            )
            return result.rowcount > 0

    def recover_stale(self, stale_minutes: int = 30) -> List[Dict[str, str]]:
        """Return in-progress clips to a safe status after a crash."""
        cutoff = _cutoff(stale_minutes)
        changes: List[Dict[str, str]] = []
        with self.conn() as connection:
            rows = connection.execute(
                """SELECT * FROM clips WHERE
                   (status IN ('DOWNLOADING','ANALYZING','EDITING')
                    AND (locked_at IS NULL OR locked_at < ?))
                   OR (locked_by IS NOT NULL AND locked_at < ?)""",
                (cutoff, cutoff),
            ).fetchall()
            for row in rows:
                old = row["status"]
                target = None
                stage = "recovery"
                if old == "DOWNLOADING":
                    local = Path(row["local_path"]) if row["local_path"] else None
                    target = "DOWNLOADED" if local and local.is_file() and \
                        local.stat().st_size > 1024 else "DISCOVERED"
                elif old == "ANALYZING":
                    analysis = connection.execute(
                        "SELECT 1 FROM clip_analysis WHERE clip_id=?",
                        (row["id"],),
                    ).fetchone()
                    local = Path(row["local_path"]) if row["local_path"] else None
                    if analysis:
                        target = "ANALYZED"
                    elif local and local.is_file() and local.stat().st_size > 1024:
                        target = "DOWNLOADED"
                    else:
                        target = "DISCOVERED"
                elif old == "EDITING":
                    target = "APPROVED"
                now = _now()
                if target:
                    connection.execute(
                        """UPDATE clips SET status=?, error_message=?,
                           stage=?, stage_started_at=?, finished_at=NULL,
                           updated_at=?, locked_by=NULL, locked_at=NULL
                           WHERE id=?""",
                        (target, "Recovered after interrupted run", stage,
                         now, now, row["id"]),
                    )
                    self._history(connection, row["id"], old, target, stage,
                                  "Recovered after interrupted run")
                    changes.append({"clip_id": row["id"], "from": old,
                                    "to": target})
                else:
                    connection.execute(
                        "UPDATE clips SET locked_by=NULL, locked_at=NULL, "
                        "updated_at=? WHERE id=?",
                        (now, row["id"]),
                    )
                    changes.append({"clip_id": row["id"], "from": old,
                                    "to": old})
        if changes:
            log.warning("Recovered %d stale clip(s)", len(changes))
        return changes

    def reconcile_ready(self) -> List[Dict[str, str]]:
        """READY must never point to a missing file."""
        broken: List[Dict[str, str]] = []
        with self.conn() as connection:
            rows = connection.execute(
                "SELECT id, export_path FROM clips WHERE status='READY'"
            ).fetchall()
            for row in rows:
                path = Path(row["export_path"]) if row["export_path"] else None
                if path and path.is_file() and path.stat().st_size >= 1024:
                    continue
                now = _now()
                error = "Export file is missing; status downgraded from READY"
                connection.execute(
                    """UPDATE clips SET status='FAILED', error_message=?,
                       attempts=attempts+1, finished_at=?, updated_at=?,
                       locked_by=NULL, locked_at=NULL WHERE id=?""",
                    (error, now, now, row["id"]),
                )
                self._history(connection, row["id"], "READY", "FAILED",
                              "reconcile", error)
                broken.append({"clip_id": row["id"], "from": "READY",
                               "to": "FAILED"})
        if broken:
            log.warning("Downgraded %d READY clip(s) with missing exports",
                        len(broken))
        return broken

    # ------------------------------------------------------------------ retry
    def reset_for_retry(self, clip_id: str, from_stage: str = "download",
                        clear_analysis: bool = True) -> Dict[str, Any]:
        """Manually re-queue a clip. Never used automatically."""
        if from_stage not in {"download", "analysis", "edit"}:
            raise ValueError("from_stage must be download/analysis/edit")
        with self.conn() as connection:
            row = self._clip_row(connection, clip_id)
            if row is None:
                raise KeyError(f"Clip not found: {clip_id}")
            old = row["status"]
            if from_stage == "download":
                target = "DISCOVERED"
            elif from_stage == "analysis":
                local = Path(row["local_path"]) if row["local_path"] else None
                target = ("DOWNLOADED" if local and local.is_file()
                          and local.stat().st_size > 1024 else "DISCOVERED")
            else:
                target = "APPROVED"
            now = _now()
            fields = [
                "status=?", "attempts=0", "error_message=NULL", "stage=NULL",
                "stage_started_at=NULL", "finished_at=NULL", "updated_at=?",
                "locked_by=NULL", "locked_at=NULL",
            ]
            params: List[Any] = [target, now]
            if from_stage in {"download", "analysis"}:
                fields.extend([
                    "score=NULL", "best_start=NULL", "best_end=NULL",
                    "category=NULL", "rejection_reason=NULL",
                    "review_reason=NULL", "analysis_json=NULL",
                    "analysis_version=NULL",
                ])
                if from_stage == "download":
                    fields.append("export_path=NULL")
                if clear_analysis:
                    connection.execute(
                        "DELETE FROM clip_analysis WHERE clip_id=?", (clip_id,)
                    )
            elif from_stage == "edit":
                fields.append("export_path=NULL")
            params.append(clip_id)
            connection.execute(
                f"UPDATE clips SET {', '.join(fields)} WHERE id=?", params
            )
            self._history(connection, clip_id, old, target, "manual-retry")
            updated = self._clip_row(connection, clip_id)
            return dict(updated) if updated else {}

    # ---------------------------------------------------------------- history
    def history(self, clip_id: Optional[str] = None,
                limit: int = 100) -> List[Dict[str, Any]]:
        with self.conn() as connection:
            if clip_id:
                rows = connection.execute(
                    "SELECT * FROM status_history WHERE clip_id=? "
                    "ORDER BY id DESC LIMIT ?",
                    (clip_id, int(limit)),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM status_history ORDER BY id DESC LIMIT ?",
                    (int(limit),),
                ).fetchall()
            return [dict(row) for row in rows]

    # ------------------------------------------------------------ maintenance
    def backup(self, target: Optional[Path] = None) -> Path:
        target = Path(target) if target else (
            self.path.parent / "backups" /
            f"{self.path.stem}-manual-{datetime.now().strftime('%Y%m%d-%H%M%S')}.db"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        with self.conn() as connection:
            destination = sqlite3.connect(target)
            try:
                connection.backup(destination)
            finally:
                destination.close()
        return target

    def vacuum(self) -> None:
        with self.conn() as connection:
            connection.isolation_level = None
            connection.execute("VACUUM")

    def close(self) -> None:
        # Connections are short-lived; kept for a uniform lifecycle API.
        return None


def default_worker_id() -> str:
    return f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
