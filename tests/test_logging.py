import logging

from src.logging_setup import (clip_context, get_logger, setup_logging,
                               stage_timer)


def _cleanup():
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass


def test_clip_context_in_file(tmp_path):
    try:
        setup_logging(tmp_path, force=True)
        log = get_logger("test.logging")
        with clip_context("clip-123"):
            log.info("hello from clip")
        text = (tmp_path / "app.log").read_text(encoding="utf-8")
        assert "clip=clip-123" in text
        assert "hello from clip" in text
    finally:
        _cleanup()


def test_stage_timer_logs_result(tmp_path):
    try:
        setup_logging(tmp_path, force=True)
        log = get_logger("test.stage")
        with clip_context("c1"):
            with stage_timer(log, "download", clip_id="c1"):
                pass
        text = (tmp_path / "app.log").read_text(encoding="utf-8")
        assert "stage=download start" in text
        assert "stage=download done" in text
    finally:
        _cleanup()


def test_setup_is_idempotent(tmp_path):
    try:
        setup_logging(tmp_path, force=True)
        count = len(logging.getLogger().handlers)
        setup_logging(tmp_path)
        assert len(logging.getLogger().handlers) == count
    finally:
        _cleanup()
