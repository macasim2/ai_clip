"""Shared helpers."""
import hashlib
import re
import shutil
import subprocess
from pathlib import Path

_GAME_ALIASES = {
    "cs2": "Counter-Strike 2",
    "csgo": "Counter-Strike 2",
    "counter strike 2": "Counter-Strike 2",
    "counter-strike 2": "Counter-Strike 2",
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


def normalize_game_name(raw: str) -> str:
    if not raw:
        return ""
    key = raw.strip().lower()
    return _GAME_ALIASES.get(key, raw.strip())


def safe_filename(name: str, max_len: int = 60) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "", name or "")
    name = re.sub(r"\s+", "_", name.strip())
    return (name or "clip")[:max_len]


def ffprobe_duration(path: Path) -> float:
    if not shutil.which("ffprobe"):
        raise RuntimeError("ffprobe not found in PATH")
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {r.stderr.strip()[:200]}")
    return float(r.stdout.strip())


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))
