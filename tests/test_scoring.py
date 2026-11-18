import pytest
from src.scoring import validate_decision, InvalidDecision


def _base(**kw):
    d = dict(
        category="Экшен", event_description="win 1v3",
        unexpectedness=0.9, emotionality=0.8, payoff_strength=0.95,
        event_clarity=0.9, visual_interest=0.85, hook_strength=0.8,
        context_independence=0.85, overall_score=88, confidence=0.9,
        decision="candidate", best_moment_start=2.0, best_moment_end=10.0,
        reason="test",
    )
    d.update(kw)
    return d


def test_valid():
    assert validate_decision(_base(), 20.0).decision == "candidate"


def test_out_of_range():
    with pytest.raises(InvalidDecision):
        validate_decision(_base(unexpectedness=1.5), 20.0)


def test_bad_category():
    with pytest.raises(InvalidDecision):
        validate_decision(_base(category="X"), 20.0)


def test_bad_time():
    with pytest.raises(InvalidDecision):
        validate_decision(_base(best_moment_start=5, best_moment_end=3), 20.0)


def test_clamped_time():
    d = validate_decision(_base(best_moment_start=-5, best_moment_end=100), 20.0)
    assert d.best_moment_start == 0.0
    assert d.best_moment_end == 20.0


def test_missing_field():
    raw = _base()
    del raw["reason"]
    with pytest.raises(InvalidDecision):
        validate_decision(raw, 20.0)


def test_reject_allows_zero_moment():
    decision = validate_decision(_base(decision="reject", overall_score=5,
                                       best_moment_start=0,
                                       best_moment_end=0), 20.0)
    assert decision.decision == "reject"
    assert (decision.best_moment_start, decision.best_moment_end) == (0.0, 0.0)


def test_candidate_with_zero_moment_is_invalid():
    with pytest.raises(InvalidDecision):
        validate_decision(_base(best_moment_start=0, best_moment_end=0), 20.0)


def test_nan_factor_is_invalid():
    with pytest.raises(InvalidDecision):
        validate_decision(_base(unexpectedness=float("nan")), 20.0)


def test_infinite_factor_is_invalid():
    with pytest.raises(InvalidDecision):
        validate_decision(_base(emotionality=float("inf")), 20.0)


def test_confidence_range():
    with pytest.raises(InvalidDecision):
        validate_decision(_base(confidence=1.5), 20.0)


def test_score_must_be_integer_range():
    with pytest.raises(InvalidDecision):
        validate_decision(_base(overall_score=101), 20.0)
    with pytest.raises(InvalidDecision):
        validate_decision(_base(overall_score=88.5), 20.0)


def test_empty_reason_is_invalid():
    with pytest.raises(InvalidDecision):
        validate_decision(_base(reason="   "), 20.0)
