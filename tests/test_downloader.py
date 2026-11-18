import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.downloader import DownloadError, download


def _ffmpeg():
    return shutil.which("ffmpeg")


@pytest.fixture(scope="module")
def sample_video(tmp_path_factory):
    if not _ffmpeg():
        pytest.skip("ffmpeg is not available")
    path = tmp_path_factory.mktemp("media") / "sample.mp4"
    result = subprocess.run(
        [
            _ffmpeg(), "-y",
            "-f", "lavfi", "-i", "testsrc=size=320x240:rate=10",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000",
            "-t", "1", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(path),
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        pytest.skip(f"cannot generate sample video: {result.stderr[-200:]!r}")
    return path


def _clip(clip_id="12345", url="https://example/clip", local_path=None):
    return {"id": clip_id, "url": url, "local_path": local_path}


def _success_factory(sample_video):
    def fake(url, template, timeout=420):
        target = Path(str(template).replace("%(ext)s", "mp4"))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sample_video, target)
        return SimpleNamespace(returncode=0, stderr="", stdout="")
    return fake


def test_skips_verified_existing_file(tmp_path, sample_video, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    target = raw / "12345_source.mp4"
    shutil.copy2(sample_video, target)

    def fail(*args, **kwargs):
        raise AssertionError("yt-dlp must not be called for an existing file")

    monkeypatch.setattr("src.downloader._run_ytdlp", fail)
    result = download(_clip(), raw, attempts=1)
    assert result.skipped is True
    assert result.path == target


def test_respects_local_path(tmp_path, sample_video, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    custom = raw / "custom_name.mp4"
    shutil.copy2(sample_video, custom)
    monkeypatch.setattr("src.downloader._run_ytdlp",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    result = download(_clip(local_path=custom), raw, attempts=1)
    assert result.path == custom
    assert result.skipped is True


def test_downloads_and_moves(tmp_path, sample_video, monkeypatch):
    raw = tmp_path / "raw"
    monkeypatch.setattr("src.downloader._run_ytdlp",
                        _success_factory(sample_video))
    result = download(_clip(), raw, attempts=1)
    assert result.skipped is False
    assert result.path == raw / "12345_source.mp4"
    assert result.path.is_file()
    assert result.duration > 0


def test_retries_then_succeeds(tmp_path, sample_video, monkeypatch):
    raw = tmp_path / "raw"
    calls = {"n": 0}

    def flaky(url, template, timeout=420):
        calls["n"] += 1
        if calls["n"] < 3:
            return SimpleNamespace(returncode=1, stderr="temporary", stdout="")
        return _success_factory(sample_video)(url, template, timeout)

    monkeypatch.setattr("src.downloader._run_ytdlp", flaky)
    monkeypatch.setattr("src.downloader.time.sleep", lambda _: None)
    result = download(_clip(), raw, attempts=3)
    assert calls["n"] == 3
    assert result.path.is_file()


def test_failure_cleans_temp_and_raises(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    monkeypatch.setattr("src.downloader._run_ytdlp",
                        lambda *a, **k: SimpleNamespace(returncode=1,
                                                       stderr="boom", stdout=""))
    monkeypatch.setattr("src.downloader.time.sleep", lambda _: None)
    with pytest.raises(DownloadError, match="Failed to download"):
        download(_clip(), raw, attempts=2)
    assert not (raw / ".tmp" / "12345").exists()
    assert not (raw / "12345_source.mp4").exists()


def test_missing_ytdlp_module(tmp_path, monkeypatch):
    monkeypatch.setattr("src.downloader.importlib.util.find_spec",
                        lambda name: None)
    with pytest.raises(DownloadError, match="yt-dlp"):
        download(_clip(), tmp_path / "raw")


def test_low_disk_space(tmp_path, monkeypatch):
    monkeypatch.setattr("src.downloader.free_space_mb", lambda path: 1.0)
    with pytest.raises(DownloadError, match="free disk space"):
        download(_clip(), tmp_path / "raw", min_free_mb=1024)


def test_unsafe_clip_id(tmp_path):
    with pytest.raises(ValueError, match="Unsafe clip id"):
        download({"id": "../evil", "url": "u"}, tmp_path / "raw")
