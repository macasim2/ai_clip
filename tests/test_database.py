import tempfile
from pathlib import Path
from src.database import Database
from src.models import TwitchClip


def _clip(cid="abc"):
    return TwitchClip(id=cid, title="x", game_name="CS2",
                      broadcaster_name="y", url="u",
                      created_at="", duration=10, view_count=100)


def test_upsert_dedup():
    with tempfile.TemporaryDirectory() as d:
        db = Database(Path(d) / "t.db")
        assert db.upsert_clip(_clip()) is True
        assert db.upsert_clip(_clip()) is False
        assert db.get_clip("abc")["status"] == "DISCOVERED"


def test_status_flow():
    with tempfile.TemporaryDirectory() as d:
        db = Database(Path(d) / "t.db")
        db.upsert_clip(_clip())
        db.update_status("abc", "DOWNLOADED")
        assert db.get_clip("abc")["status"] == "DOWNLOADED"
        db.mark_failed("abc", "boom")
        row = db.get_clip("abc")
        assert row["status"] == "FAILED"
        assert row["attempts"] == 1