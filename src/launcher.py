"""Windows-friendly launcher with real environment checks."""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import requests

ROOT = Path(__file__).resolve().parent.parent


def _check_python() -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    version = sys.version_info
    if version < (3, 10):
        errors.append(f"Python 3.10+ is required, found {version.major}.{version.minor}")
    elif version >= (3, 15):
        warnings.append(
            f"Python {version.major}.{version.minor} is newer than tested; "
            "Python 3.12 is recommended if a wheel fails to install"
        )
    return errors, warnings


def _check_venv() -> Tuple[List[str], List[str]]:
    if sys.prefix == sys.base_prefix:
        return [], ["Not running inside the project .venv (start.bat activates it)"]
    return [], []


def _check_tools() -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool):
            continue
        local = ROOT / f"{tool}.exe"
        if local.is_file():
            continue
        errors.append(
            f"{tool} not found in PATH. Install FFmpeg and add its bin folder "
            "to PATH, or put ffmpeg.exe/ffprobe.exe in the project root."
        )
    return errors, []


def _check_modules() -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    required = {
        "requests": "requests",
        "dotenv": "python-dotenv",
        "numpy": "numpy",
        "faster_whisper": "faster-whisper",
        "yt_dlp": "yt-dlp",
    }
    for module, package in required.items():
        if importlib.util.find_spec(module) is None:
            errors.append(
                f"Python package {package!r} is missing. Run: "
                "python -m pip install -r requirements.txt"
            )
    return errors, []


def fetch_ollama_models(url: str, timeout: float = 5.0) -> List[str]:
    response = requests.get(f"{url.rstrip('/')}/api/tags", timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    return [str(item.get("name", "")) for item in payload.get("models", [])]


def model_installed(installed: List[str], wanted: str) -> bool:
    wanted = (wanted or "").strip().lower()
    if not wanted:
        return True
    base = wanted.split(":")[0]
    for name in installed:
        normalized = name.strip().lower()
        if normalized == wanted or normalized.split(":")[0] == base:
            return True
    return False


def _check_ollama(cfg, require_ai: bool) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []
    try:
        installed = fetch_ollama_models(cfg.ollama_url)
    except Exception as exc:  # noqa: BLE001
        message = (
            f"Ollama is not reachable at {cfg.ollama_url}: {exc}. "
            "Start Ollama (ollama serve) or fix OLLAMA_URL."
        )
        if require_ai:
            errors.append(message)
        else:
            warnings.append(message)
        return errors, warnings

    for role, model in (("text", cfg.ollama_text_model),
                        ("vision", cfg.ollama_vision_model)):
        if model and not model_installed(installed, model):
            errors.append(
                f"Ollama model for {role} is not installed: {model}. "
                f"Run: ollama pull {model}"
            )
    if installed:
        warnings.append("Ollama models: " + ", ".join(sorted(installed)))
    return errors, warnings


def check_environment(cfg, require_ai: bool = True) -> int:
    print("=== AI Clip Studio: environment check ===")
    checks = [_check_python(), _check_venv(), _check_tools(), _check_modules(),
              _check_ollama(cfg, require_ai)]
    errors: List[str] = []
    warnings: List[str] = []
    for errs, warns in checks:
        errors.extend(errs)
        warnings.extend(warns)

    print(f"Python: {sys.version.split()[0]}  ({sys.executable})")
    print(f"Project: {ROOT}")
    print(f"Twitch provider: {cfg.twitch_provider}")
    print(f"Ollama: {cfg.ollama_url}")
    for warning in warnings:
        print(f"[WARN] {warning}")
    for error in errors:
        print(f"[ERROR] {error}")
    if errors:
        print("Environment check failed. Fix the errors above and run start.bat again.")
        return 1
    print("[OK] Environment looks usable.")
    return 0


def _prompt_args() -> List[str]:
    print("=== AI Clip Studio ===")
    game = input('Game (e.g. "Counter-Strike 2"): ').strip() or "Counter-Strike 2"
    raw_count = input("How many clips [5]: ").strip() or "5"
    try:
        count = max(1, int(raw_count))
    except ValueError:
        print(f"Invalid count {raw_count!r}, using 5")
        count = 5
    period = input("Period day/week/month/all [week]: ").strip().lower() or "week"
    if period not in {"day", "week", "month", "all"}:
        print(f"Invalid period {period!r}, using week")
        period = "week"
    return ["run", "--game", game, "--count", str(count), "--period", period]


def main() -> int:
    from .config import ConfigError, load_config
    from .logging_setup import setup_logging

    args = list(sys.argv[1:])
    skip_checks = "--skip-checks" in args
    args = [arg for arg in args if arg != "--skip-checks"]

    try:
        cfg = load_config()
    except ConfigError as exc:
        print(f"[ERROR] {exc}")
        return 2

    cfg.ensure_dirs()
    setup_logging(cfg.logs, cfg.log_level)

    command = args[0] if args else "run"
    require_ai = command in {"run", "resume", "retry", "review"}
    if not skip_checks and check_environment(cfg, require_ai=require_ai) != 0:
        return 1

    if not args:
        args = _prompt_args()
    print(f"Running: python -m src.main {' '.join(args)}")
    env = dict(os.environ)
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return subprocess.call(
        [sys.executable, "-m", "src.main", *args],
        cwd=str(ROOT), env=env,
    )


if __name__ == "__main__":
    sys.exit(main())
