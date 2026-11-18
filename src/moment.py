"""Select the final edit window around the AI-identified peak.

The curator returns a rough window.  This module turns it into a usable short:
it keeps whole spoken words, adds a little context before the peak, lets the
reaction play out afterwards, and enforces the configured min/max duration.
"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

from .models import AudioEvent, Moment, TranscriptWord


def _safe(value: float, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number or number in (float("inf"), float("-inf")):
        return default
    return number


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def select_moment(start: float, end: float, duration: float,
                  words: Optional[Sequence[TranscriptWord]] = None,
                  events: Optional[Iterable[AudioEvent]] = None,
                  cfg=None) -> Moment:
    """Build a final [start, end] window from the curator's rough choice."""
    duration = _safe(duration, 0.0)
    if duration <= 0:
        raise ValueError("Cannot select a moment from a zero-length clip")
    words = list(words or [])
    events = list(events or [])
    min_duration = min(float(getattr(cfg, "min_clip_duration", 8.0)), duration)
    max_duration = min(float(getattr(cfg, "max_clip_duration", 45.0)), duration)
    if max_duration <= 0:
        max_duration = duration
    min_duration = min(min_duration, max_duration)

    start = _clamp(_safe(start, 0.0), 0.0, duration)
    end = _clamp(_safe(end, duration), 0.0, duration)
    original = (start, end)
    if end <= start:
        start, end = 0.0, duration

    peak = _find_peak(start, end, events, words)
    start, end = _expand_to_minimum(start, end, peak, duration, min_duration)
    start, end = _snap_to_words(start, end, words, duration)
    start, end = _expand_to_minimum(start, end, peak, duration, min_duration)
    start, end = _cap_maximum(start, end, peak, duration, max_duration)
    start, end = _snap_to_words(start, end, words, duration)
    # Snapping may add a little time back; keep the hard maximum.
    if end - start > max_duration:
        start, end = _cap_maximum(start, end, peak, duration, max_duration)
    if end - start < min_duration:
        start, end = _expand_to_minimum(start, end, peak, duration, min_duration)

    start = round(_clamp(start, 0.0, duration), 3)
    end = round(_clamp(end, 0.0, duration), 3)
    if end <= start:
        start, end = 0.0, round(duration, 3)
    changed = abs(start - original[0]) > 0.05 or abs(end - original[1]) > 0.05
    reason = (
        f"AI window {original[0]:.2f}-{original[1]:.2f}s adjusted to "
        f"{start:.2f}-{end:.2f}s to keep whole words, context and reaction"
        if changed else "AI window kept"
    )
    return Moment(start=start, end=end, reason=reason, expanded=changed)


def _find_peak(start: float, end: float, events: List[AudioEvent],
               words: List[TranscriptWord]) -> float:
    candidates = [event for event in events if start <= event.start <= end]
    if candidates:
        best = max(candidates, key=lambda event: float(event.intensity or 0))
        return _clamp((best.start + best.end) / 2, start, end)
    # Fall back to a word with emphasis, otherwise the middle of the window.
    emphasized = [
        word for word in words
        if start <= word.start <= end and _is_emphasized(word.word)
    ]
    if emphasized:
        return _clamp(emphasized[-1].end, start, end)
    return (start + end) / 2


def _is_emphasized(word: str) -> bool:
    text = (word or "").strip()
    return bool(text) and (
        text.endswith(("!", "?", "…")) or
        (len(text) >= 3 and text.isupper())
    )


def _expand_to_minimum(start: float, end: float, peak: float,
                       duration: float, minimum: float) -> tuple:
    if end - start >= minimum:
        return start, end
    need = minimum - (end - start)
    # Prefer context before the event, leave room for the reaction after it.
    left = need * 0.65
    right = need * 0.35
    start = max(0.0, start - left)
    end = min(duration, end + right)
    if end - start < minimum:
        end = min(duration, start + minimum)
    if end - start < minimum:
        start = max(0.0, end - minimum)
    if end - start < minimum:
        start, end = 0.0, min(duration, minimum)
    return start, end


def _snap_to_words(start: float, end: float, words: List[TranscriptWord],
                   duration: float) -> tuple:
    if not words:
        return start, end

    # Do not start in the middle of a spoken word.
    for word in words:
        if word.start < start < word.end:
            start = word.start
            break
    else:
        before = [word for word in words if word.end <= start]
        if before:
            nearest = max(before, key=lambda word: word.end)
            if start - nearest.end <= 0.6:
                start = max(0.0, nearest.start - 0.2)
        if start > 0.05:
            start = max(0.0, start - 0.2)  # a little context before speech

    # Do not cut the current word or the reaction after the last word.
    for word in words:
        if start < word.start < end < word.end:
            end = word.end
            break
    after = [word for word in words if word.start >= end]
    if after:
        nearest = min(after, key=lambda word: word.start)
        if nearest.start - end <= 0.6:
            end = min(duration, nearest.end + 0.25)
    else:
        end = min(duration, end + 0.25)

    if end <= start:
        end = min(duration, start + 0.5)
    return start, end


def _cap_maximum(start: float, end: float, peak: float, duration: float,
                 maximum: float) -> tuple:
    if end - start <= maximum:
        return start, end
    # Keep the peak slightly before the middle so the reaction stays in frame.
    desired_start = peak - maximum * 0.55
    start = _clamp(desired_start, 0.0, max(0.0, duration - maximum))
    end = min(duration, start + maximum)
    if end - start < maximum and start > 0:
        start = max(0.0, end - maximum)
    return start, end
