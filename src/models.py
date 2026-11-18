"""Domain models shared across pipeline stages."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TwitchClip:
    """A clip returned by a Twitch provider."""

    id: str
    title: str
    game_name: str
    broadcaster_name: str
    url: str
    created_at: str
    duration: float
    view_count: int
    thumbnail_url: str = ""
    game_id: str = ""
    slug: str = ""


@dataclass
class AudioEvent:
    kind: str  # loud | reaction | scream | laughter | pause | noise | onset
    start: float
    end: float
    intensity: float = 1.0
    speech_overlap: bool = False


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
    model: str = ""
    source_fingerprint: str = ""


@dataclass
class FrameDescription:
    timestamp: float
    description: str
    path: str = ""


@dataclass
class VisionAnalysisResult:
    duration: float
    frames: List[FrameDescription]
    sequence_summary: str
    model: str = ""
    scene_changes: List[float] = field(default_factory=list)
    source_fingerprint: str = ""


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


@dataclass
class DownloadResult:
    """Result of a download stage."""

    path: Path
    duration: float
    size: int
    skipped: bool = False


@dataclass
class Moment:
    """Final edit window selected around the AI peak."""

    start: float
    end: float
    reason: str = ""
    expanded: bool = False

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass
class RenderResult:
    """Verified result of a render."""

    path: Path
    width: int
    height: int
    duration: float
    has_audio: bool


@dataclass
class StageResult:
    """Structured result returned by a pipeline stage."""

    stage: str
    ok: bool
    data: Any = None
    error: str = ""
    elapsed: float = 0.0


@dataclass
class ProviderHealth:
    ok: bool
    provider: str
    detail: str = ""
