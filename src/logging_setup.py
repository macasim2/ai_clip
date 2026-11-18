"""Structured logging."""
import logging
import sys
from pathlib import Path

_configured = False


def setup_logging(log_dir: Path, level=logging.INFO):
    global _configured
    if _configured:
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)-24s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)
    fh = logging.FileHandler(log_dir / "app.log", encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
