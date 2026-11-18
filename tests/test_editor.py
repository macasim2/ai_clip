"""Tests for the production-safe vertical renderer (src.editor).

The tests generate tiny synthetic clips with FFmpeg instead of relying on
checked-in media.  If FFmpeg/FFprobe (or the libass ``subtitles`` filter) is
missing, the affected tests are skipped instead of failing.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.editor import (
    EditorError,
    _crf,
    _edit_mode,
    _preset,
    _sane_bitrate,
    render_vertical,
)
from src.utils import MediaInfo, ToolNotFound, find_tool, probe_media


def _tool(name: str):
    try:
        return find_tool(name)
    except ToolNotFound:
        return None


FFMPEG = _tool("ffmpeg")
FFPROBE = _tool("ffprobe")

pytestmark = pytest.mark.skipif(
    FFMPEG is None or FFPROBE is None,
    reason="ffmpeg/ffprobe are not available",
)


def _has_filter(name: str) -> bool:
    if FFMPEG is None:
        return False
    try:
        result = subprocess.run(
            [FFMPEG, "-hide_banner", "-filters"],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return re.search(rf"\b{re.escape(name)}\b", result.stdout or "") is not None


HAS_SUBTITLES_FILTER = _has_filter("subtitles")
requires_subtitles = pytest.mark.skipif(
    not HAS_SUBTITLES_FILTER, reason="ffmpeg has no libass 'subtitles' filter")


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
def _make_video(path: Path, with_audio: bool) -> Path:
    """Create a 2 s 1280x720 testsrc2 clip (optionally with sine audio)."""
    cmd = [
        FFMPEG, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc2=size=1280x720:rate=30:duration=2",
    ]
    if with_audio:
        cmd += ["-f", "lavfi", "-i",
                "sine=frequency=440:sample_rate=48000:duration=2"]
    cmd += ["-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    if with_audio:
        cmd += ["-c:a", "aac", "-b:a", "128k", "-shortest"]
    else:
        cmd += ["-an"]
    cmd.append(str(path))
    subprocess.run(cmd, check=True, capture_output=True, text=True,
                   encoding="utf-8", errors="replace", timeout=120)
    assert path.is_file() and path.stat().st_size > 0
    return path


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory) -> Path:
    # Directory and file names with spaces also exercise argv path handling.
    media_dir = tmp_path_factory.mktemp("media dir")
    return _make_video(media_dir / "source clip.mp4", with_audio=True)


@pytest.fixture(scope="session")
def silent_video(tmp_path_factory) -> Path:
    media_dir = tmp_path_factory.mktemp("silent dir")
    return _make_video(media_dir / "silent clip.mp4", with_audio=False)


def _cfg(**overrides) -> SimpleNamespace:
    values = dict(
        video_width=1080,
        video_height=1920,
        video_crf=28,          # fast but valid encode for tests
        video_preset="ultrafast",
        video_bitrate="4M",
        edit_mode="blur",
        subtitle_font_size=48,
        subtitle_margin_v=80,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


MINIMAL_ASS = """\
[Script Info]
; Minimal ASS used by tests/test_editor.py
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,64,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,2,1,2,20,20,80,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.10,0:00:01.20,Default,,0,0,0,,Hello from ASS subtitles
"""

MINIMAL_SRT = """\
1
00:00:00,100 --> 00:00:01,200
Hello from SRT subtitles
"""


def _assert_verified_output(path: Path, *, has_audio: bool) -> None:
    info = probe_media(path)
    assert (info.width, info.height) == (1080, 1920)
    assert info.has_audio is has_audio
    assert path.stat().st_size > 4096


# ---------------------------------------------------------------------------
# Required behaviour
# ---------------------------------------------------------------------------
def test_blur_mode_renders_verified_vertical_with_audio(sample_video, tmp_path):
    out = tmp_path / "exports" / "blur.mp4"
    result = render_vertical("clip_blur", sample_video, out, 0.2, 1.4,
                             _cfg(edit_mode="blur"))
    assert result == out
    assert out.is_file()
    info = probe_media(out)
    assert (info.width, info.height) == (1080, 1920)
    assert info.has_audio is True
    assert 1.2 <= info.duration <= 2.0
    # Atomic output: the temporary file must not survive a successful render.
    assert not out.with_suffix(".tmp.mp4").exists()


def test_crop_mode_renders_verified_vertical_with_audio(sample_video, tmp_path):
    out = tmp_path / "crop.mp4"
    render_vertical("clip_crop", sample_video, out, 0.0, 1.2,
                    _cfg(edit_mode="crop"))
    assert out.is_file()
    _assert_verified_output(out, has_audio=True)


def test_render_without_subtitles_path_still_renders(sample_video, tmp_path):
    out = tmp_path / "no_subs.mp4"
    render_vertical("clip_no_subs", sample_video, out, 0.1, 1.1, _cfg(),
                    subtitles_path=None)
    assert out.is_file()
    _assert_verified_output(out, has_audio=True)


@requires_subtitles
def test_burn_ass_subtitles_from_directory_with_spaces(sample_video, tmp_path):
    # The space in "work dir" is what used to break FFmpeg filter escaping.
    work_dir = tmp_path / "work dir"
    work_dir.mkdir()
    subtitles = work_dir / "subtitles.ass"
    subtitles.write_text(MINIMAL_ASS, encoding="utf-8")

    out_dir = tmp_path / "render out"
    out = out_dir / "with_ass.mp4"
    render_vertical("clip_ass", sample_video, out, 0.0, 1.2, _cfg(),
                    subtitles_path=subtitles)
    assert out.is_file()
    _assert_verified_output(out, has_audio=True)


@requires_subtitles
def test_burn_srt_subtitles(sample_video, tmp_path):
    work_dir = tmp_path / "srt dir"
    work_dir.mkdir()
    subtitles = work_dir / "subtitles.srt"
    subtitles.write_text(MINIMAL_SRT, encoding="utf-8")

    out = tmp_path / "with_srt.mp4"
    render_vertical("clip_srt", sample_video, out, 0.0, 1.0, _cfg(),
                    subtitles_path=subtitles)
    assert out.is_file()
    _assert_verified_output(out, has_audio=True)


def test_missing_source_raises_editor_error(tmp_path):
    out = tmp_path / "never.mp4"
    with pytest.raises(EditorError, match="Source clip not found"):
        render_vertical("clip_missing", tmp_path / "nope.mp4", out,
                        0.0, 1.0, _cfg())
    assert not out.exists()


@pytest.mark.parametrize("start,end", [
    (-1.0, 1.0),   # negative start
    (1.0, 1.0),    # empty window
    (1.5, 1.0),    # reversed window
    (0.0, 9.0),    # end beyond source duration (source is 2 s)
])
def test_invalid_time_range_raises_editor_error(sample_video, tmp_path, start,
                                                end):
    out = tmp_path / "bad_window.mp4"
    with pytest.raises(EditorError):
        render_vertical("clip_bad_time", sample_video, out, start, end, _cfg())
    assert not out.exists()
    assert not out.with_suffix(".tmp.mp4").exists()


def test_video_without_audio_renders_and_verifies_without_audio(silent_video,
                                                                tmp_path):
    out = tmp_path / "silent.mp4"
    render_vertical("clip_silent", silent_video, out, 0.1, 1.1,
                    _cfg(edit_mode="crop"))
    assert out.is_file()
    _assert_verified_output(out, has_audio=False)


def test_failed_verification_raises_and_cleans_up_temp(sample_video, tmp_path,
                                                       monkeypatch):
    """A corrupt/incorrect render must never be reported as success."""
    import src.editor as editor

    real_probe = editor.probe_media
    calls = {"count": 0}

    def fake_probe(path):
        calls["count"] += 1
        info = real_probe(path)
        if calls["count"] >= 2:  # second call verifies the rendered output
            return MediaInfo(duration=info.duration, width=1, height=1,
                             has_video=True, has_audio=info.has_audio)
        return info

    monkeypatch.setattr(editor, "probe_media", fake_probe)
    out = tmp_path / "bad_verify.mp4"
    with pytest.raises(EditorError, match="verification failed"):
        render_vertical("clip_bad_verify", sample_video, out, 0.0, 0.6, _cfg())
    assert not out.exists()
    assert not out.with_suffix(".tmp.mp4").exists()


def test_ffmpeg_failure_raises_and_cleans_up_temp(sample_video, tmp_path,
                                                  monkeypatch):
    """A broken filter graph must surface EditorError and remove the temp."""
    import src.editor as editor

    monkeypatch.setattr(
        editor, "_build_filter_complex",
        lambda cfg, width, height, mode, subtitles_path:
            ("[0:v]this_filter_does_not_exist[vout]", "[vout]", None, False),
    )
    out = tmp_path / "broken.mp4"
    with pytest.raises(EditorError, match="FFmpeg failed"):
        render_vertical("clip_broken", sample_video, out, 0.0, 0.6, _cfg())
    assert not out.exists()
    assert not out.with_suffix(".tmp.mp4").exists()


# ---------------------------------------------------------------------------
# Cheap unit checks for configuration fallbacks (no FFmpeg run)
# ---------------------------------------------------------------------------
def test_quality_settings_fallback_to_defaults():
    assert _crf(SimpleNamespace()) == 20
    assert _preset(SimpleNamespace()) == "medium"
    assert _crf(SimpleNamespace(video_crf=999)) == 20
    assert _preset(SimpleNamespace(video_preset="bogus")) == "medium"
    assert _sane_bitrate(None) is None
    assert _sane_bitrate("not-a-bitrate") is None
    assert _sane_bitrate("0M") is None
    assert _sane_bitrate("9999G") is None
    assert _sane_bitrate("6M") == 6_000_000
    assert _sane_bitrate("4500k") == 4_500_000


def test_missing_or_unknown_edit_mode_defaults_to_blur():
    assert _edit_mode(SimpleNamespace()) == "blur"
    assert _edit_mode(SimpleNamespace(edit_mode="weird-mode")) == "blur"
    assert _edit_mode(SimpleNamespace(edit_mode="CROP")) == "crop"
