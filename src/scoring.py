"""Content-based interestingness scoring."""
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


def compute_score(factors: dict) -> int:
    return int(round(sum(float(factors.get(k, 0.0)) * w
                         for k, w in WEIGHTS.items()) * 100))


def validate_decision(raw: dict, video_duration: float) -> CuratorDecision:
    if not isinstance(raw, dict):
        raise InvalidDecision(f"Decision is not a dict: {type(raw).__name__}")
    missing = [f for f in REQUIRED_FIELDS if f not in raw]
    if missing:
        raise InvalidDecision(f"Missing fields: {missing}")

    for k in WEIGHTS:
        v = raw[k]
        if not isinstance(v, (int, float)) or not 0.0 <= float(v) <= 1.0:
            raise InvalidDecision(f"{k} out of range 0..1: {v!r}")

    cat = str(raw["category"])
    if cat not in VALID_CATEGORIES:
        raise InvalidDecision(f"Unknown category: {cat!r}")
    dec = str(raw["decision"]).lower()
    if dec not in VALID_DECISIONS:
        raise InvalidDecision(f"Unknown decision: {dec!r}")

    try:
        bs = float(raw["best_moment_start"])
        be = float(raw["best_moment_end"])
    except (TypeError, ValueError):
        raise InvalidDecision("best_moment timestamps are not numeric")

    if video_duration > 0:
        bs = max(0.0, min(bs, video_duration))
        be = max(0.0, min(be, video_duration))
    if be <= bs:
        raise InvalidDecision(f"best_moment_end ({be}) <= start ({bs})")

    score = compute_score(raw)
    model_score = int(raw.get("overall_score", -1))
    if not 0 <= model_score <= 100:
        raise InvalidDecision(f"overall_score out of 0..100: {model_score}")

    if abs(score - model_score) > 20:
        log.warning("Model score %d != weighted %d, using weighted",
                    model_score, score)
    else:
        score = model_score

    return CuratorDecision(
        category=cat, event_description=str(raw["event_description"]),
        unexpectedness=float(raw["unexpectedness"]),
        emotionality=float(raw["emotionality"]),
        payoff_strength=float(raw["payoff_strength"]),
        event_clarity=float(raw["event_clarity"]),
        visual_interest=float(raw["visual_interest"]),
        hook_strength=float(raw["hook_strength"]),
        context_independence=float(raw["context_independence"]),
        overall_score=score,
        confidence=float(raw.get("confidence", 0.5)),
        decision=dec, best_moment_start=bs, best_moment_end=be,
        reason=str(raw["reason"]), raw=raw,
    )
