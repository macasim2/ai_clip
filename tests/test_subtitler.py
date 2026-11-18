from types import SimpleNamespace
from src.models import TranscriptWord
from src.subtitler import build_srt


def _cfg():
    return SimpleNamespace(subtitle_max_words=4, subtitle_max_chars=20,
                           subtitle_style="normal")


def test_srt_basic():
    words = [TranscriptWord(0.1, 0.5, "hello"),
             TranscriptWord(0.6, 1.0, "world")]
    srt = build_srt(words, 0.0, 2.0, _cfg())
    assert "hello world" in srt and "-->" in srt


def test_respects_range():
    words = [TranscriptWord(0.1, 0.5, "early"),
             TranscriptWord(5.0, 5.5, "inside")]
    srt = build_srt(words, 4.0, 6.0, _cfg())
    assert "inside" in srt
    assert "early" not in srt