import shutil
import subprocess

import pytest

from src.utils import (MediaError, file_fingerprint, free_space_mb,
                       normalize_game_name, probe_media, read_json,
                       safe_filename, validate_clip_id, write_json)


def test_normalize():
    assert normalize_game_name("csgo") == "Counter-Strike 2"
    assert normalize_game_name("CS2") == "Counter-Strike 2"
    assert normalize_game_name("counter-strike") == "Counter-Strike"
    assert normalize_game_name("Some Game") == "Some Game"
    assert normalize_game_name("") == ""


def test_safe():
    assert safe_filename("a/b:c") == "abc"
    assert safe_filename("") == "clip"
    assert safe_filename("x" * 200).__len__() == 60
    assert safe_filename("CON").startswith("_")
    assert not safe_filename("name. ").endswith((".", " "))


def test_clamp():
    from src.utils import clamp
    assert clamp(5, 0, 3) == 3
    assert clamp(-1, 0, 3) == 0


def test_clip_id_validation():
    assert validate_clip_id("2021748262") == "2021748262"
    assert validate_clip_id("abc_DEF-1") == "abc_DEF-1"
    for bad in ("", "../x", "a b", "x" * 65, "-bad"):
        with pytest.raises(ValueError):
            validate_clip_id(bad)


def test_fingerprint_changes(tmp_path):
    path = tmp_path / "f.txt"
    path.write_text("a")
    first = file_fingerprint(path)
    path.write_text("ab")
    second = file_fingerprint(path)
    assert first != second


def test_json_roundtrip(tmp_path):
    path = tmp_path / "data.json"
    write_json(path, {"ключ": "значение"})
    assert read_json(path) == {"ключ": "значение"}
    assert read_json(tmp_path / "missing.json", default={}) == {}


def test_free_space(tmp_path):
    assert free_space_mb(tmp_path) > 0


@pytest.fixture(scope="module")
def sample_video(tmp_path_factory):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg is not available")
    path = tmp_path_factory.mktemp("media") / "sample.mp4"
    result = subprocess.run(
        [
            shutil.which("ffmpeg"), "-y", "-f", "lavfi",
            "-i", "testsrc=size=320x240:rate=10", "-f", "lavfi",
            "-i", "sine=frequency=440:sample_rate=16000", "-t", "1",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            "-shortest", str(path),
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        pytest.skip("cannot generate sample video")
    return path


def test_probe_media(sample_video):
    info = probe_media(sample_video)
    assert info.has_video and info.has_audio
    assert info.width == 320 and info.height == 240
    assert 0.5 < info.duration < 2.0
    assert info.is_valid_video


def test_probe_missing_file(tmp_path):
    with pytest.raises(MediaError):
        probe_media(tmp_path / "missing.mp4")
