import json
from types import SimpleNamespace

import pytest

from src.auto_curator import CuratorError, analyze_clip
from src.models import (AudioAnalysisResult, AudioEvent, TranscriptWord,
                        VisionAnalysisResult)


def cfg(**overrides):
    values = dict(
        ollama_url="http://localhost:11434",
        ollama_text_model="llama3.2:latest",
        ollama_timeout=30,
        ollama_keep_alive="2m",
        ollama_num_ctx=4096,
        curator_temperature=0.2,
        min_confidence=0.35,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def audio():
    return AudioAnalysisResult(
        language="ru", duration=10.0,
        segments=[], words=[TranscriptWord(0.2, 0.6, "wow"),
                            TranscriptWord(0.7, 1.0, "yes!")],
        events=[AudioEvent("reaction", 0.8, 1.0, 1.0, True)],
        full_text="wow yes!",
    )


def vision():
    return VisionAnalysisResult(10.0, [], "player wins a fight")


def decision_payload(**overrides):
    payload = {
        "category": "Экшен",
        "event_description": "player wins a 1v3 fight",
        "unexpectedness": 0.9,
        "emotionality": 0.8,
        "payoff_strength": 0.9,
        "event_clarity": 0.9,
        "visual_interest": 0.8,
        "hook_strength": 0.8,
        "context_independence": 0.8,
        "overall_score": 88,
        "confidence": 0.9,
        "decision": "candidate",
        "best_moment_start": 2.0,
        "best_moment_end": 8.0,
        "reason": "clear clutch",
    }
    payload.update(overrides)
    return payload


class FakeResponse:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code = status
        self._payload = payload
        self.text = text or str(payload or "")

    def json(self):
        return self._payload


def fake_post(monkeypatch, responses):
    calls = []

    def post(url, json=None, timeout=None):
        calls.append({"url": url, "json": json, "timeout": timeout})
        return responses.pop(0)

    monkeypatch.setattr("src.auto_curator.requests.post", post)
    return calls


def test_valid_decision_and_prompt_context(monkeypatch):
    calls = fake_post(monkeypatch, [
        FakeResponse(200, {"response": json.dumps(decision_payload())})
    ])
    result = analyze_clip(audio(), vision(), 10.0, cfg(),
                          game_name="Counter-Strike 2", title="clutch")
    assert result.decision == "candidate"
    assert result.overall_score == 88
    prompt = calls[0]["json"]["prompt"]
    assert "Counter-Strike 2" in prompt
    assert "clutch" in prompt
    assert "wow yes!" in prompt
    assert calls[0]["json"]["format"]["type"] == "object"


def test_low_confidence_candidate_becomes_review(monkeypatch):
    fake_post(monkeypatch, [
        FakeResponse(200, {"response": json.dumps(
            decision_payload(confidence=0.1))})
    ])
    result = analyze_clip(audio(), vision(), 10.0, cfg())
    assert result.decision == "review"
    assert "confidence" in result.reason


def test_invalid_json_is_retried_once(monkeypatch):
    calls = fake_post(monkeypatch, [
        FakeResponse(200, {"response": "not json"}),
        FakeResponse(200, {"response": json.dumps(decision_payload())}),
    ])
    result = analyze_clip(audio(), vision(), 10.0, cfg())
    assert result.decision == "candidate"
    assert len(calls) == 2
    assert "PREVIOUS ANSWER WAS INVALID" in calls[1]["json"]["prompt"]


def test_invalid_twice_raises(monkeypatch):
    fake_post(monkeypatch, [
        FakeResponse(200, {"response": "not json"}),
        FakeResponse(200, {"response": "still not json"}),
    ])
    with pytest.raises(Exception):
        analyze_clip(audio(), vision(), 10.0, cfg())


def test_reject_with_zero_times_is_valid(monkeypatch):
    fake_post(monkeypatch, [
        FakeResponse(200, {"response": json.dumps(decision_payload(
            decision="reject", overall_score=5, confidence=0.8,
            best_moment_start=0.0, best_moment_end=0.0,
            reason="boring gameplay"))})
    ])
    result = analyze_clip(audio(), vision(), 10.0, cfg())
    assert result.decision == "reject"
    assert (result.best_moment_start, result.best_moment_end) == (0.0, 0.0)


def test_transport_failure_raises_curator_error(monkeypatch):
    fake_post(monkeypatch, [
        FakeResponse(500, {"error": "boom"}),
        FakeResponse(500, {"error": "boom"}),
    ])
    with pytest.raises(CuratorError):
        analyze_clip(audio(), vision(), 10.0, cfg())


def test_schema_rejection_falls_back_to_json_mode(monkeypatch):
    calls = fake_post(monkeypatch, [
        FakeResponse(400, {"error": "format schema unsupported"},
                     text='{"error":"format schema unsupported"}'),
        FakeResponse(200, {"response": json.dumps(decision_payload())}),
    ])
    result = analyze_clip(audio(), vision(), 10.0, cfg())
    assert result.decision == "candidate"
    assert calls[1]["json"]["format"] == "json"
