from src.config import Config
from src.pipeline import Pipeline


def _pipeline():
    cfg = Config().validate()
    return Pipeline(cfg, db=None, twitch=object())


def test_short_moment_is_expanded_to_minimum():
    pipeline = _pipeline()
    start, end = pipeline._clamp_times(9.0, 9.5, 10.0)
    assert end - start >= pipeline.cfg.min_clip_duration - 0.001
    assert start >= 0 and end <= 10.0


def test_long_moment_is_capped():
    pipeline = _pipeline()
    pipeline.cfg.max_clip_duration = 10.0
    start, end = pipeline._clamp_times(0.0, 40.0, 45.0)
    assert end - start <= 10.0 + 0.001


def test_invalid_window_uses_whole_clip():
    pipeline = _pipeline()
    start, end = pipeline._clamp_times(8.0, 2.0, 5.0)
    assert (start, end) == (0.0, 5.0)


def test_clamp_never_exceeds_video():
    pipeline = _pipeline()
    start, end = pipeline._clamp_times(-5.0, 100.0, 12.0)
    assert 0.0 <= start < end <= 12.0
