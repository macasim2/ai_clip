"""LLM curator: audio + vision -> strict, validated JSON decision."""
from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional

import requests

from .logging_setup import get_logger
from .models import AnalysisError, AudioAnalysisResult, CuratorDecision, VisionAnalysisResult
from .scoring import InvalidDecision, validate_decision

log = get_logger(__name__)


class CuratorError(AnalysisError):
    """The curator model could not be called or returned invalid output."""


SYSTEM_INSTRUCTION = """You are a strict short-form video editor for TikTok and YouTube Shorts.
You receive one gaming clip: its transcript, word timings, acoustic events and
a chronological visual timeline. Decide whether this exact clip contains a
genuinely interesting moment that works without extra context.

Hard rules:
- View count and popularity are NOT evidence of quality. Judge the content.
- Loud noise alone is not interesting. A real emotional reaction needs speech,
  a visible event, or a clear gameplay consequence.
- Normal walking, looting, menu browsing or generic shooting is boring: reject.
- Prefer a clear event with a setup, a climax and a payoff/reaction.
- Never invent an event that is not present in the transcript, events or frames.
- If evidence is weak or contradictory, use decision "review" with low confidence.
- Scores are calibrated: 0 = boring, 0.5 = average, 1 = exceptional.
- best_moment_start/end must point at the single best continuous segment.
  For a reject without any usable moment, 0.0/0.0 is allowed.
- Output exactly one JSON object matching the schema. No prose, no markdown."""

# Concrete example from a DIFFERENT clip.  It is here to teach calibration, not
# to be copied: the model must analyze the actual input.
EXAMPLE = """EXAMPLE (different clip, do not copy its values):
GAME: Dota 2, TITLE: "1v3 clutch", DURATION: 30s
Evidence: player alone vs three enemies, tense music, double kill at 22s,
scream at 23s, "no way" at 24s.
OUTPUT:
{"category":"Экшен","event_description":"Player wins a 1v3 fight and kills the last enemy at 22s","unexpectedness":0.9,"emotionality":0.8,"payoff_strength":0.95,"event_clarity":0.85,"visual_interest":0.8,"hook_strength":0.7,"context_independence":0.8,"overall_score":87,"confidence":0.9,"decision":"candidate","best_moment_start":18.0,"best_moment_end":26.0,"reason":"Clear 1v3 clutch with a decisive final kill and a strong reaction"}
"""

JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": ["Прикол", "Экшен", "История", "Неожиданное событие",
                     "Реакция", "Скучное"],
        },
        "event_description": {"type": "string"},
        "unexpectedness": {"type": "number"},
        "emotionality": {"type": "number"},
        "payoff_strength": {"type": "number"},
        "event_clarity": {"type": "number"},
        "visual_interest": {"type": "number"},
        "hook_strength": {"type": "number"},
        "context_independence": {"type": "number"},
        "overall_score": {"type": "integer"},
        "confidence": {"type": "number"},
        "decision": {"type": "string", "enum": ["candidate", "reject", "review"]},
        "best_moment_start": {"type": "number"},
        "best_moment_end": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": [
        "category", "event_description", "unexpectedness", "emotionality",
        "payoff_strength", "event_clarity", "visual_interest", "hook_strength",
        "context_independence", "overall_score", "confidence", "decision",
        "best_moment_start", "best_moment_end", "reason",
    ],
}


def _extract_json(text: str) -> Optional[dict]:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _build_prompt(audio: AudioAnalysisResult, vision: VisionAnalysisResult,
                  duration: float, game_name: str = "",
                  title: str = "") -> str:
    events = "\n".join(
        f"- {event.kind} at {event.start:.2f}-{event.end:.2f}s "
        f"(intensity={event.intensity:.2f}, speech={event.speech_overlap})"
        for event in audio.events
    ) or "(no distinctive audio events)"
    words = "\n".join(
        f"[{word.start:.2f}-{word.end:.2f}] {word.word}"
        for word in audio.words[:300]
    ) or "(no words)"
    frames = "\n".join(
        f"[{frame.timestamp:.1f}s] {frame.description}"
        for frame in vision.frames
    ) or "(no visual frames)"
    scenes = ", ".join(f"{value:.1f}s" for value in vision.scene_changes[:20]) \
        or "(none detected)"
    transcript = (audio.full_text or "(no speech)")[:6000]
    return f"""GAME: {game_name or "unknown"}
CLIP TITLE: {title or "(none)"}
DURATION: {duration:.2f} seconds
DETECTED LANGUAGE: {audio.language or "unknown"}

TRANSCRIPT:
{transcript}

WORD TIMESTAMPS (first 300):
{words}

ACOUSTIC EVENTS:
{events}

VISUAL NARRATIVE SUMMARY:
{vision.sequence_summary or "(summary unavailable)"}

FRAMES IN CHRONOLOGICAL ORDER:
{frames}

SCENE CHANGES: {scenes}

Judge the actual clip. A short reaction shot without a visible play should be
rejected or sent to review, not promoted. A loud sound without speech should
not be called emotional. If the clip contains a real gameplay event with a
payoff, identify the exact best moment and score it honestly.

{EXAMPLE}

Return ONLY the JSON object for the actual clip now.
"""


def _call_ollama(cfg, prompt: str) -> str:
    url = f"{str(cfg.ollama_url).rstrip('/')}/api/generate"
    model = getattr(cfg, "ollama_text_model", "")
    if not model:
        raise CuratorError("OLLAMA_TEXT_MODEL is not configured")
    body: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "system": SYSTEM_INSTRUCTION,
        "stream": False,
        "format": JSON_SCHEMA,
        "options": {
            "temperature": float(getattr(cfg, "curator_temperature", 0.2)),
            "num_ctx": int(getattr(cfg, "ollama_num_ctx", 4096)),
            "num_predict": 900,
        },
        "keep_alive": getattr(cfg, "ollama_keep_alive", "2m"),
    }
    timeout = float(getattr(cfg, "ollama_timeout", 240))
    last_error = ""
    for attempt in range(2):
        try:
            response = requests.post(url, json=body, timeout=timeout)
        except requests.RequestException as exc:
            last_error = f"network: {exc}"
            if attempt == 0:
                time.sleep(1.0)
                continue
            raise CuratorError(f"Ollama is unreachable: {exc}") from exc
        if response.status_code == 200:
            payload = response.json()
            text = payload.get("response")
            if not isinstance(text, str) or not text.strip():
                raise CuratorError("Ollama returned an empty response")
            return text
        if response.status_code == 400 and "format" in response.text.lower() \
                and attempt == 0:
            # Older Ollama builds may reject a full JSON Schema.  Fall back to
            # plain JSON mode and rely on validation/retry.
            log.warning("Ollama rejected the JSON Schema; falling back to json mode")
            body["format"] = "json"
            continue
        if response.status_code in (429, 500, 502, 503, 504) and attempt == 0:
            last_error = f"HTTP {response.status_code}"
            time.sleep(1.0)
            continue
        raise CuratorError(
            f"Ollama HTTP {response.status_code}: {response.text[:300]}"
        )
    raise CuratorError(f"Ollama curator failed after retries: {last_error}")


def analyze_clip(audio: AudioAnalysisResult, vision: VisionAnalysisResult,
                 clip_duration: float, cfg,
                 game_name: str = "", title: str = "") -> CuratorDecision:
    base = _build_prompt(audio, vision, clip_duration, game_name, title)
    last_error: Optional[Exception] = None
    for attempt in range(2):
        suffix = "" if attempt == 0 else (
            "\n\nYOUR PREVIOUS ANSWER WAS INVALID. Read the schema again, fill "
            "every field with concrete values for THIS clip, and return only "
            "the JSON object."
        )
        text = _call_ollama(cfg, base + suffix)
        parsed = _extract_json(text)
        if parsed is None:
            last_error = InvalidDecision(f"Unparsable JSON: {text[:300]!r}")
            log.warning("Curator JSON parse failed (attempt %d)", attempt + 1)
            continue
        try:
            decision = validate_decision(parsed, clip_duration)
        except InvalidDecision as exc:
            last_error = exc
            log.warning("Curator validation failed (attempt %d): %s",
                        attempt + 1, exc)
            continue
        decision.raw = dict(parsed, _model=model_name(cfg),
                            _attempt=attempt + 1)
        if (decision.decision == "candidate"
                and decision.confidence < float(getattr(cfg, "min_confidence", 0.35))):
            decision.decision = "review"
            decision.reason += (
                f" (confidence {decision.confidence:.2f} below "
                f"{float(getattr(cfg, 'min_confidence', 0.35)):.2f})"
            )
            log.info("Candidate downgraded to review: confidence %.2f",
                     decision.confidence)
        return decision
    raise InvalidDecision(f"Curator failed after retry: {last_error}")


def model_name(cfg) -> str:
    return str(getattr(cfg, "ollama_text_model", ""))
