"""Content-based interestingness scoring and strict decision validation."""
from __future__ import annotations

import math
from typing import Any, Dict

from .logging_setup import get_logger
from .models import CuratorDecision

log = get_logger(__name__)

WEIGHTS = {
    "unexpectedness": 0.20,
    "emotionality": 0.15,
    "payoff_strength": 0.20,
    "event_clarity": 0.15,
    "visual_interest": 0.10,
    "hook_strength": 0.10,
    "context_independence": 0.10,
}

REQUIRED_FIELDS = list(WEIGHTS.keys()) + [
    "category", "event_description", "overall_score", "confidence",
    "decision", "best_moment_start", "best_moment_end", "reason",
]

VALID_CATEGORIES = {"Прикол", "Экшен", "История", "Неожиданное событие",
                    "Реакция", "Скучное"}
VALID_DECISIONS = {"candidate", "reject", "review"}


class InvalidDecision(ValueError):
    pass


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise InvalidDecision(f"{name} must be a number, got bool")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise InvalidDecision(f"{name} is not numeric: {value!r}") from exc
    if not math.isfinite(number):
        raise InvalidDecision(f"{name} must be finite, got {value!r}")
    return number


def compute_score(factors: dict) -> int:
    total = sum(float(factors.get(key, 0.0)) * weight
                for key, weight in WEIGHTS.items())
    return int(round(max(0.0, min(1.0, total)) * 100))


def validate_decision(raw: dict, video_duration: float) -> CuratorDecision:
    if not isinstance(raw, dict):
        raise InvalidDecision(f"Decision is not a dict: {type(raw).__name__}")
    missing = [field for field in REQUIRED_FIELDS if field not in raw]
    if missing:
        raise InvalidDecision(f"Missing fields: {missing}")

    factor_values: Dict[str, float] = {}
    for key in WEIGHTS:
        value = _number(raw[key], key)
        if not 0.0 <= value <= 1.0:
            raise InvalidDecision(f"{key} out of range 0..1: {value!r}")
        factor_values[key] = value

    category = str(raw["category"]).strip()
    if category not in VALID_CATEGORIES:
        raise InvalidDecision(
            f"Unknown category: {category!r}; expected one of "
            f"{sorted(VALID_CATEGORIES)}"
        )
    decision = str(raw["decision"]).strip().lower()
    if decision not in VALID_DECISIONS:
        raise InvalidDecision(
            f"Unknown decision: {decision!r}; expected one of "
            f"{sorted(VALID_DECISIONS)}"
        )

    model_score = _number(raw["overall_score"], "overall_score")
    if model_score != int(model_score) or not 0 <= model_score <= 100:
        raise InvalidDecision(
            f"overall_score must be an integer in 0..100, got {raw['overall_score']!r}"
        )
    confidence = _number(raw["confidence"], "confidence")
    if not 0.0 <= confidence <= 1.0:
        raise InvalidDecision(
            f"confidence out of range 0..1: {confidence!r}"
        )

    start = _number(raw["best_moment_start"], "best_moment_start")
    end = _number(raw["best_moment_end"], "best_moment_end")
    duration = max(0.0, float(video_duration or 0.0))
    if duration > 0:
        start = max(0.0, min(start, duration))
        end = max(0.0, min(end, duration))
    if decision in {"candidate", "review"}:
        if end <= start:
            raise InvalidDecision(
                f"best_moment_end ({end}) <= best_moment_start ({start}) "
                f"for decision {decision!r}"
            )
        if duration > 0 and (start >= duration or end > duration + 0.001):
            raise InvalidDecision(
                f"best_moment {start}..{end} is outside video duration {duration}"
            )
    else:
        # A rejection may legitimately have no moment at all (0/0).
        if end <= start:
            start = end = 0.0

    event_description = str(raw["event_description"]).strip()
    reason = str(raw["reason"]).strip()
    if not event_description:
        raise InvalidDecision("event_description is empty")
    if not reason:
        raise InvalidDecision("reason is empty")

    score = compute_score(factor_values)
    if abs(score - int(model_score)) > 20:
        log.warning("Model score %d differs from weighted %d; using weighted",
                    int(model_score), score)
    else:
        score = int(model_score)

    return CuratorDecision(
        category=category,
        event_description=event_description,
        unexpectedness=factor_values["unexpectedness"],
        emotionality=factor_values["emotionality"],
        payoff_strength=factor_values["payoff_strength"],
        event_clarity=factor_values["event_clarity"],
        visual_interest=factor_values["visual_interest"],
        hook_strength=factor_values["hook_strength"],
        context_independence=factor_values["context_independence"],
        overall_score=score,
        confidence=confidence,
        decision=decision,
        best_moment_start=round(start, 3),
        best_moment_end=round(end, 3),
        reason=reason,
        raw=raw,
    )
