"""Domain models shared across modules."""
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class TwitchClip:
    id: str
    title: str
    game_name: str
    broadcaster_name: str
    url: str
    created_at: str
    duration: float
    view_count: int
    thumbnail_url: str = ""


@dataclass
class AudioEvent:
    kind: str  # loud | pause | laughter | scream
    start: float
    end: float
    intensity: float = 1.0


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptWord:
    start: float
    end: float
    word: str


@dataclass
class AudioAnalysisResult:
    language: str
    duration: float
    segments: List[TranscriptSegment]
    words: List[TranscriptWord]
    events: List[AudioEvent]
    full_text: str


@dataclass
class FrameDescription:
    timestamp: float
    description: str


@dataclass
class VisionAnalysisResult:
    duration: float
    frames: List[FrameDescription]
    sequence_summary: str


@dataclass
class CuratorDecision:
    category: str
    event_description: str
    unexpectedness: float
    emotionality: float
    payoff_strength: float
    event_clarity: float
    visual_interest: float
    hook_strength: float
    context_independence: float
    overall_score: int
    confidence: float
    decision: str
    best_moment_start: float
    best_moment_end: float
    reason: str
    raw: Dict[str, Any] = field(default_factory=dict)
