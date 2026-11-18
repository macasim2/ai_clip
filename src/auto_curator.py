"""LLM curator combining audio + vision analyses, with strict validation."""
import json
import re
from typing import Optional

import requests

from .logging_setup import get_logger
from .models import AudioAnalysisResult, CuratorDecision, VisionAnalysisResult
from .scoring import InvalidDecision, validate_decision

log = get_logger(__name__)

SYSTEM_INSTRUCTION = """You are an expert short-form video editor for TikTok / YouTube Shorts.
You receive a gaming clip's transcript, audio events and visual narrative.
Judge whether the moment is genuinely interesting and return STRICT JSON only.

Schema (no markdown, no prose):
{
  "category": one of ["Прикол","Экшен","История","Неожиданное событие","Реакция","Скучное"],
  "event_description": "one sentence describing what happened",
  "unexpectedness": 0.0..1.0,
  "emotionality": 0.0..1.0,
  "payoff_strength": 0.0..1.0,
  "event_clarity": 0.0..1.0,
  "visual_interest": 0.0..1.0,
  "hook_strength": 0.0..1.0,
  "context_independence": 0.0..1.0,
  "overall_score": integer 0..100,
  "confidence": 0.0..1.0,
  "decision": "candidate" | "reject" | "review",
  "best_moment_start": float seconds,
  "best_moment_end": float seconds,
  "reason": "short justification grounded in concrete events"
}

Rules:
- Views are NOT a signal. Judge content.
- A loud noise alone is NOT interesting. Distinguish emotion from noise.
- Boring normal gameplay must be rejected.
- best_moment_start/end must point to the single most interesting segment,
  include a small context before the peak, and let the reaction play out.
"""


def _extract_json(text: str) -> Optional[dict]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


def _build_prompt(audio: AudioAnalysisResult, vision: VisionAnalysisResult,
                  duration: float) -> str:
    events = "\n".join(
        f"- {e.kind} at {e.start:.1f}-{e.end:.1f}s (intensity {e.intensity:.2f})"
        for e in audio.events
    ) or "(no distinctive audio events)"
    words = "\n".join(
        f"[{w.start:.2f}-{w.end:.2f}] {w.word}" for w in audio.words[:400]
    ) or "(no words)"
    frames = "\n".join(
        f"[{f.timestamp:.1f}s] {f.description}" for f in vision.frames
    ) or "(no visual frames)"
    return f"""CLIP DURATION: {duration:.1f}s
LANGUAGE: {audio.language}

TRANSCRIPT:
{audio.full_text or "(no speech)"}

WORD TIMESTAMPS (first 400):
{words}

AUDIO EVENTS:
{events}

VISUAL NARRATIVE SUMMARY:
{vision.sequence_summary or "(not available)"}

FRAMES IN TIME ORDER:
{frames}

Return only the JSON object now.
"""


def _call_ollama(cfg, prompt: str) -> str:
    r = requests.post(
        f"{cfg.ollama_url.rstrip('/')}/api/generate",
        json={
            "model": cfg.ollama_text_model,
            "prompt": prompt,
            "system": SYSTEM_INSTRUCTION,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.3, "num_predict": 700},
        },
        timeout=cfg.ollama_timeout,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Ollama HTTP {r.status_code}: {r.text[:200]}")
    return r.json().get("response", "")


def analyze_clip(audio: AudioAnalysisResult, vision: VisionAnalysisResult,
                 clip_duration: float, cfg) -> CuratorDecision:
    base = _build_prompt(audio, vision, clip_duration)
    last_err: Optional[Exception] = None
    for attempt in range(2):
        suffix = ("" if attempt == 0 else
                  "\n\nPREVIOUS RESPONSE WAS INVALID. Return only the JSON object.")
        text = _call_ollama(cfg, base + suffix)
        parsed = _extract_json(text)
        if parsed is None:
            last_err = InvalidDecision(f"Unparsable JSON: {text[:200]!r}")
            continue
        try:
            return validate_decision(parsed, clip_duration)
        except InvalidDecision as e:
            last_err = e
            log.warning("Decision validation failed (attempt %d): %s",
                        attempt + 1, e)
    raise InvalidDecision(f"Curator failed after retry: {last_err}")
