from types import SimpleNamespace

import pytest

from src.models import AudioEvent, TranscriptWord
from src.moment import select_moment


CFG = SimpleNamespace(min_clip_duration=8.0, max_clip_duration=45.0)


def words(*pairs):
    return [TranscriptWord(start, end, word) for start, end, word in pairs]


def test_keeps_reasonable_window():
    moment = select_moment(10.0, 20.0, 30.0, [], [], CFG)
    assert moment.start <= 10.0
    assert moment.end >= 20.0
    assert 8.0 <= moment.duration <= 45.0


def test_expands_short_window_to_minimum():
    moment = select_moment(9.0, 9.5, 10.0, [], [], CFG)
    assert moment.duration >= 8.0 - 0.001
    assert moment.start >= 0.0 and moment.end <= 10.0


def test_caps_maximum_duration():
    cfg = SimpleNamespace(min_clip_duration=8.0, max_clip_duration=10.0)
    moment = select_moment(0.0, 40.0, 45.0, [], [], cfg)
    assert moment.duration <= 10.0 + 0.001


def test_does_not_start_in_middle_of_word():
    moment = select_moment(8.5, 9.4, 20.0,
                           words((8.0, 9.0, "привет")), [], CFG)
    assert moment.start <= 8.0


def test_does_not_cut_word_at_end():
    moment = select_moment(8.0, 8.5, 20.0,
                           words((8.0, 9.0, "привет")), [], CFG)
    assert moment.end >= 9.0


def test_reaction_after_event_is_kept():
    events = [AudioEvent("reaction", 5.0, 5.5, 1.0)]
    moment = select_moment(5.0, 5.2, 20.0, [], events, CFG)
    assert moment.end >= 5.5


def test_invalid_window_uses_whole_clip():
    moment = select_moment(8.0, 2.0, 5.0, [], [], CFG)
    assert moment.start == 0.0
    assert moment.end == 5.0


def test_clip_shorter_than_minimum_uses_whole_clip():
    moment = select_moment(0.5, 1.0, 3.0, [], [], CFG)
    assert (moment.start, moment.end) == (0.0, 3.0)


def test_zero_duration_raises():
    with pytest.raises(ValueError):
        select_moment(0, 1, 0)


def test_nan_is_sanitized():
    moment = select_moment(float("nan"), float("inf"), 10.0, [], [], CFG)
    assert 0.0 <= moment.start < moment.end <= 10.0


def test_never_exceeds_video():
    moment = select_moment(-5, 100, 12.0, [], [], CFG)
    assert 0.0 <= moment.start < moment.end <= 12.0
