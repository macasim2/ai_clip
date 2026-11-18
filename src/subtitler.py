"""Subtitle generation from Whisper word timings."""
from pathlib import Path
from typing import List

from .logging_setup import get_logger
from .models import TranscriptWord

log = get_logger(__name__)


def _ts(t: float) -> str:
    h = int(t // 3600)
    m = int((t % 3600) // 60)
    s = t % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def _group(words: List[TranscriptWord], start_off: float, end_off: float,
           max_words: int, max_chars: int, gap_split: float = 0.6):
    groups: List[List[TranscriptWord]] = []
    cur: List[TranscriptWord] = []
    cur_chars = 0
    for w in words:
        if w.end <= start_off or w.start >= end_off:
            continue
        newline = (
            len(cur) >= max_words
            or (cur_chars + len(w.word) + 1) > max_chars
            or (cur and (w.start - cur[-1].end) > gap_split)
        )
        if newline and cur:
            groups.append(cur)
            cur, cur_chars = [], 0
        cur.append(w)
        cur_chars += len(w.word) + 1
    if cur:
        groups.append(cur)
    return groups


def build_srt(words: List[TranscriptWord], start_off: float, end_off: float,
              cfg) -> str:
    groups = _group(words, start_off, end_off,
                    cfg.subtitle_max_words, cfg.subtitle_max_chars)
    out: List[str] = []
    for i, g in enumerate(groups, 1):
        s = max(start_off, g[0].start) - start_off
        e = min(end_off, g[-1].end) - start_off
        if e <= s:
            e = s + 0.3
        text = " ".join(w.word for w in g).strip()
        if cfg.subtitle_style == "accent" and g:
            target = max(range(len(g)), key=lambda j: len(g[j].word))
            parts = [f"{{\\b1}}{w.word}{{\\b0}}" if j == target else w.word
                     for j, w in enumerate(g)]
            text = " ".join(parts)
        out.append(f"{i}\n{_ts(s)} --> {_ts(e)}\n{text}\n")
    return "\n".join(out)


def generate(clip_id: str, words: List[TranscriptWord], work_dir: Path,
             start_off: float, end_off: float, cfg) -> Path:
    work_dir.mkdir(parents=True, exist_ok=True)
    srt = build_srt(words, start_off, end_off, cfg)
    out = work_dir / "subtitles.srt"
    out.write_text(srt, encoding="utf-8")
    log.info("Subtitles %s: %d cues", clip_id, srt.count("-->"))
    return out
