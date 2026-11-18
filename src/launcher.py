"""Windows-friendly launcher with dependency checks."""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _check():
    missing = []
    if sys.version_info < (3, 10):
        missing.append("Python >= 3.10")
    for tool in ("ffmpeg", "ffprobe", "yt-dlp"):
        if not shutil.which(tool):
            missing.append(tool)
    if missing:
        print("Missing dependencies:", ", ".join(missing))
        print("Install ffmpeg and yt-dlp and re-run.")
        sys.exit(1)


def _prompt_args():
    print("=== AI Clip Studio ===")
    game = input("Game (e.g. Counter-Strike 2): ").strip() or "Counter-Strike 2"
    n = input("How many clips [5]: ").strip() or "5"
    period = input("Period day/week/month/all [week]: ").strip() or "week"
    return ["run", "--game", game, "--count", n, "--period", period]


def main():
    _check()
    args = sys.argv[1:] or _prompt_args()
    sys.exit(subprocess.call(
        [sys.executable, "-m", "src.main", *args], cwd=str(ROOT),
    ))


if __name__ == "__main__":
    main()
