"""Structured logging with per-clip context and stage timers."""
from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from contextvars import ContextVar, Token
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Iterator, Optional

_clip_var: ContextVar[str] = ContextVar("clip_id", default="-")
_configured = False


class ClipContextFilter(logging.Filter):
    """Inject the current clip id into every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.clip_id = _clip_var.get()
        return True


_FORMAT = (
    "%(asctime)s | %(levelname)-7s | %(name)-22s | "
    "clip=%(clip_id)-12s | %(message)s"
)


def setup_logging(log_dir: Path, level: str = "INFO", force: bool = False) -> None:
    """Configure console + rotating file logging once."""
    global _configured
    if _configured and not force:
        return

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(getattr(logging, str(level).upper(), logging.INFO))

    if force:
        for handler in list(root.handlers):
            root.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass

    formatter = logging.Formatter(_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
    clip_filter = ClipContextFilter()

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    console.addFilter(clip_filter)
    root.addHandler(console)

    file_handler = RotatingFileHandler(
        log_dir / "app.log", maxBytes=5 * 1024 * 1024, backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(clip_filter)
    root.addHandler(file_handler)

    # Third-party libraries are too chatty at INFO in normal runs.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("faster_whisper").setLevel(logging.INFO)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def set_clip(clip_id: Optional[str]) -> Token:
    return _clip_var.set(str(clip_id) if clip_id else "-")


def reset_clip(token: Token) -> None:
    _clip_var.reset(token)


@contextmanager
def clip_context(clip_id: str) -> Iterator[None]:
    token = set_clip(clip_id)
    try:
        yield
    finally:
        reset_clip(token)


@contextmanager
def stage_timer(logger: logging.Logger, stage: str,
                clip_id: Optional[str] = None, **fields) -> Iterator[None]:
    """Log start/finish/failure and elapsed time for a pipeline stage."""
    token = set_clip(clip_id) if clip_id else None
    started = time.perf_counter()
    suffix = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.info("stage=%s start%s", stage, f" {suffix}" if suffix else "")
    try:
        yield
    except Exception as exc:
        elapsed = time.perf_counter() - started
        logger.error("stage=%s failed in %.2fs: %s", stage, elapsed, exc)
        raise
    else:
        elapsed = time.perf_counter() - started
        logger.info("stage=%s done in %.2fs%s", stage, elapsed,
                    f" {suffix}" if suffix else "")
    finally:
        if token is not None:
            reset_clip(token)
