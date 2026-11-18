"""Tests for the ASS subtitle generator (1080x1920 burn-in) and review SRT."""
import re
from types import SimpleNamespace

import pytest

from src.models import TranscriptWord
from src.subtitler import build_ass, build_srt, generate

MIN_CUE_DURATION = 0.35

_TS_RE = re.compile(r"^(\d+):(\d{2}):(\d{2})\.(\d{2})$")


# ------------------------------------------------------------------ helpers
def _cfg(**overrides):
    values = dict(
        subtitles_enabled=True,
        subtitle_font="Arial",
        subtitle_font_size=64,
        subtitle_style="normal",
        subtitle_max_words=4,
        subtitle_max_chars=24,
        subtitle_max_duration=4.0,
        subtitle_margin_v=140,
        video_width=1080,
        video_height=1920,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _words(*triples):
    return [TranscriptWord(start, end, word) for start, end, word in triples]


def _parse_ts(value):
    match = _TS_RE.match(value.strip())
    assert match is not None, f"invalid ASS timestamp: {value!r}"
    hours, minutes, seconds, centi = (int(part) for part in match.groups())
    return hours * 3600 + minutes * 60 + seconds + centi / 100.0


def _cues(ass_text):
    """[(start_seconds, end_seconds, dialogue_text), ...] in file order."""
    cues = []
    for line in ass_text.splitlines():
        if not line.startswith("Dialogue:"):
            continue
        fields = line.split(",", 9)
        assert len(fields) == 10, f"malformed Dialogue line: {line!r}"
        cues.append((_parse_ts(fields[1]), _parse_ts(fields[2]), fields[9]))
    return cues


def _style_fields(ass_text):
    for line in ass_text.splitlines():
        if line.startswith("Style: Default,"):
            return line[len("Style: "):].split(",")
    raise AssertionError("ASS has no Default style line")


# ------------------------------------------------------------------ layout
def test_ass_script_info_geometry_and_default_style():
    ass = build_ass(_words((0.2, 0.5, "hello")), 0.0, 2.0, _cfg())

    assert "[Script Info]" in ass
    assert "ScriptType: v4.00+" in ass
    assert "PlayResX: 1080" in ass
    assert "PlayResY: 1920" in ass
    assert "WrapStyle: 2" in ass
    assert "ScaledBorderAndShadow: yes" in ass
    assert "[V4+ Styles]" in ass
    assert "[Events]" in ass

    fields = _style_fields(ass)
    # Name, Fontname, Fontsize, PrimaryColour, ...
    assert fields[0] == "Default"
    assert fields[1] == "Arial"
    assert fields[2] == "64"
    assert fields[3] == "&H00FFFFFF"  # white primary colour
    assert fields[7] == "1"           # bold
    assert fields[15] == "1"          # BorderStyle: outline
    assert fields[16] == "3"          # outline width
    assert fields[17] == "1"          # shadow depth
    assert fields[18] == "2"          # Alignment: bottom centre
    assert fields[19] == "60"         # MarginL
    assert fields[20] == "60"         # MarginR
    assert fields[21] == "140"        # MarginV


def test_ass_respects_configured_style_values():
    cfg = _cfg(subtitle_font="Roboto", subtitle_font_size=72,
               subtitle_margin_v=200)
    fields = _style_fields(build_ass(
        _words((0.2, 0.5, "hello")), 0.0, 2.0, cfg))
    assert fields[1] == "Roboto"
    assert fields[2] == "72"
    assert fields[21] == "200"


def test_ass_style_fallbacks_for_missing_config():
    fields = _style_fields(build_ass(
        _words((0.2, 0.5, "hello")), 0.0, 2.0, SimpleNamespace()))
    assert fields[1] == "Arial"
    assert fields[2] == "64"
    assert fields[21] == "140"


# ------------------------------------------------------------------- timing
def test_cues_stay_inside_window_and_do_not_overlap():
    start_off, end_off = 10.0, 16.0
    words = _words(
        (8.0, 8.5, "outside"),      # before the window
        (9.5, 10.2, "before"),      # partially overlapping window start
        (10.4, 10.8, "hello"),
        (10.82, 11.2, "world"),
        (12.0, 12.4, "again"),
        (13.5, 14.2, "and"),
        (14.2, 14.6, "more"),
        (15.7, 16.4, "tail"),       # crosses the window end
        (16.5, 17.0, "after"),      # after the window
    )
    ass = build_ass(words, start_off, end_off, _cfg())
    cues = _cues(ass)

    assert cues, "expected cues inside the window"
    assert "outside" not in ass and "after" not in ass
    assert "before" in ass.split("[Events]")[1]  # intersects -> kept

    duration = end_off - start_off
    for start, end, _text in cues:
        assert 0.0 <= start < end <= duration + 1e-9
    for (_s1, end1, _t1), (start2, _end2, _t2) in zip(cues, cues[1:]):
        assert start2 >= end1 - 1e-9, "cues must not overlap"

    # First cue includes the word that started before the window: clamped to 0.
    assert cues[0][0] == pytest.approx(0.0)
    # Last cue ends exactly at the window end because its word is clipped.
    assert cues[-1][1] == pytest.approx(duration)


def test_partially_overlapping_start_word_is_clamped_to_zero():
    words = _words((4.5, 5.2, "hello"), (5.3, 5.7, "there"),
                   (8.0, 8.5, "outside"))
    cues = _cues(build_ass(words, 5.0, 7.0, _cfg()))
    assert len(cues) >= 1
    assert cues[0][0] == pytest.approx(0.0)
    assert "hello there" in cues[0][2]
    assert "outside" not in cues[0][2]


def test_minimum_cue_display_time_is_enforced():
    # Very short words in a roomy window: each cue is padded to >= 0.35 s.
    words = _words((0.1, 0.2, "a"), (1.0, 1.1, "b"), (2.0, 2.1, "c"))
    cues = _cues(build_ass(words, 0.0, 4.0, _cfg()))
    assert len(cues) == 3
    for start, end, _text in cues:
        assert end - start >= MIN_CUE_DURATION - 1e-9
        assert 0.0 <= start < end <= 4.0 + 1e-9


# ------------------------------------------------------------------ wrapping
def test_long_text_wraps_to_at_most_two_lines():
    cfg = _cfg(subtitle_max_words=10, subtitle_max_chars=12,
               subtitle_max_duration=10.0)
    words = _words((0.2, 0.5, "hello"), (0.5, 0.8, "world"),
                   (0.8, 1.1, "foo"), (1.1, 1.4, "bar"))
    ass = build_ass(words, 0.0, 3.0, cfg)
    cues = _cues(ass)

    assert len(cues) == 1
    assert "\\N" in cues[0][2], "expected a wrapped two-line cue"
    lines = cues[0][2].split("\\N")
    assert len(lines) <= 2
    for line in lines:
        assert len(line) <= 12


def test_very_long_single_word_is_not_split():
    long_word = "supercalifragilisticexpialidocious"
    cfg = _cfg(subtitle_max_words=10, subtitle_max_chars=12,
               subtitle_max_duration=10.0)
    words = _words((0.2, 0.5, long_word), (1.0, 1.3, "and"),
                   (1.3, 1.6, "more"))
    ass = build_ass(words, 0.0, 3.0, cfg)

    assert long_word in ass.split("[Events]")[1]
    for _start, _end, text in _cues(ass):
        for line in text.split("\\N"):
            if len(line) > 12:
                # Only a single unbreakable word may exceed the limit.
                assert " " not in line
                assert line == long_word


def test_no_line_exceeds_configured_max_chars():
    cfg = _cfg(subtitle_max_words=10, subtitle_max_chars=14,
               subtitle_max_duration=10.0)
    words = _words((0.2, 0.5, "readable"), (0.5, 0.8, "short"),
                   (0.8, 1.1, "lines"), (1.1, 1.4, "here"))
    ass = build_ass(words, 0.0, 3.0, cfg)
    for _start, _end, text in _cues(ass):
        for line in text.split("\\N"):
            assert len(line) <= 14


# ---------------------------------------------------------------- splitting
@pytest.mark.parametrize("punct", [".", "!", "?", "…"])
def test_sentence_punctuation_splits_cues(punct):
    words = _words((0.1, 0.4, f"Word{punct}"), (0.5, 0.7, "next"),
                   (0.75, 0.95, "words"))
    cues = _cues(build_ass(words, 0.0, 2.0, _cfg()))
    assert len(cues) == 2
    assert cues[0][2] == f"Word{punct}"


def test_silence_gap_splits_cues():
    words = _words((0.1, 0.4, "before"), (0.45, 0.7, "pause"),
                   (1.8, 2.1, "after"), (2.15, 2.4, "gap"))
    cues = _cues(build_ass(words, 0.0, 3.0, _cfg()))
    assert len(cues) == 2
    assert cues[0][2] == "before pause"
    assert cues[1][2] == "after gap"


def test_max_words_splits_cues():
    cfg = _cfg(subtitle_max_words=2, subtitle_max_chars=100,
               subtitle_max_duration=10.0)
    words = _words((0.1, 0.3, "one"), (0.3, 0.5, "two"),
                   (0.5, 0.7, "three"), (0.7, 0.9, "four"),
                   (0.9, 1.1, "five"))
    cues = _cues(build_ass(words, 0.0, 2.0, cfg))
    assert [text for _s, _e, text in cues] == [
        "one two", "three four", "five"]


def test_max_cue_duration_splits_cues():
    cfg = _cfg(subtitle_max_words=10, subtitle_max_chars=100,
               subtitle_max_duration=1.0)
    words = _words((0.1, 0.4, "alpha"), (0.4, 0.7, "beta"),
                   (0.7, 1.0, "gamma"), (1.0, 1.4, "delta"))
    cues = _cues(build_ass(words, 0.0, 2.0, cfg))
    assert len(cues) == 2
    assert cues[0][2] == "alpha beta gamma"
    assert cues[1][2] == "delta"


# ------------------------------------------------------------------- accent
def test_accent_mode_uses_inline_colour_override():
    cfg = _cfg(subtitle_style="accent")
    words = _words((0.1, 0.4, "this"), (0.45, 0.7, "is"),
                   (0.75, 1.1, "AMAZING"))
    ass = build_ass(words, 0.0, 2.0, cfg)
    cues = _cues(ass)

    assert "{\\c&H00D7FF&}" in ass
    assert "{\\c&HFFFFFF&}" in ass
    assert "{\\b1}" not in ass
    assert ass.count("{\\c&H00D7FF&}") == 1
    assert cues[0][2] == "this is {\\c&H00D7FF&}AMAZING{\\c&HFFFFFF&}"


def test_accent_prefers_marker_word_and_falls_back_to_longest():
    accent = _cfg(subtitle_style="accent")

    numbered = build_ass(
        _words((0.1, 0.3, "wow"), (0.35, 0.6, "42"),
               (0.65, 0.9, "yeah")), 0.0, 2.0, accent)
    assert "{\\c&H00D7FF&}42{\\c&HFFFFFF&}" in numbered

    longest = build_ass(
        _words((0.1, 0.3, "cat"), (0.35, 0.6, "elephant"),
               (0.65, 0.9, "dog")), 0.0, 2.0, accent)
    assert "{\\c&H00D7FF&}elephant{\\c&HFFFFFF&}" in longest


def test_normal_mode_has_no_accent_or_srt_style_tags():
    cfg = _cfg(subtitle_style="normal")
    words = _words((0.1, 0.4, "plain"), (0.45, 0.7, "white"))
    ass = build_ass(words, 0.0, 2.0, cfg)
    assert "\\c&H" not in ass
    assert "{\\b1}" not in ass
    assert "{\\b1}" not in build_srt(words, 0.0, 2.0, cfg)


# ----------------------------------------------------------------- generate
def test_generate_writes_ass_and_srt_and_returns_ass(tmp_path):
    words = _words((0.2, 0.6, "hello"), (0.7, 1.1, "world"))
    path = generate("clip123", words, tmp_path, 0.0, 3.0, _cfg())

    assert path == tmp_path / "subtitles.ass"
    assert path.suffix == ".ass"
    assert path.is_file()
    srt_path = tmp_path / "subtitles.srt"
    assert srt_path.is_file()

    ass = path.read_text(encoding="utf-8")
    srt = srt_path.read_text(encoding="utf-8")
    assert "[Script Info]" in ass and "Dialogue:" in ass
    assert "-->" in srt
    assert "hello world" in srt
    assert "\\N" not in srt


def test_generate_empty_window_writes_valid_empty_files(tmp_path):
    words = _words((5.0, 5.5, "outside"))
    path = generate("clip-empty", words, tmp_path, 0.0, 2.0, _cfg())

    assert path == tmp_path / "subtitles.ass"
    assert path.is_file()
    ass = path.read_text(encoding="utf-8")
    assert "[Script Info]" in ass
    assert "PlayResX: 1080" in ass
    assert "Dialogue:" not in ass

    srt_path = tmp_path / "subtitles.srt"
    assert srt_path.is_file()
    assert srt_path.read_text(encoding="utf-8") == ""

    assert build_srt([], 0.0, 2.0, _cfg()) == ""
    assert "Dialogue:" not in build_ass([], 0.0, 2.0, _cfg())


def test_russian_and_english_words_are_utf8(tmp_path):
    words = _words((0.2, 0.55, "Привет"), (0.6, 0.95, "world"),
                   (1.0, 1.35, "Ёжик"))
    cfg = _cfg()
    ass = build_ass(words, 0.0, 2.0, cfg)
    srt = build_srt(words, 0.0, 2.0, cfg)

    for text in ("Привет", "world", "Ёжик"):
        assert text in ass
        assert text in srt

    path = generate("clip-ru", words, tmp_path, 0.0, 2.0, cfg)
    assert "Привет" in path.read_text(encoding="utf-8")
    assert "Ёжик" in (tmp_path / "subtitles.srt").read_text(encoding="utf-8")
