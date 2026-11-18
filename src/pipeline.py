"""End-to-end pipeline orchestration."""
import json
from typing import Optional

from .audio_analyzer import analyze_audio
from .auto_curator import analyze_clip
from .database import Database
from .downloader import DownloadError, download_clip
from .editor import EditorError, render_vertical
from .logging_setup import get_logger
from .models import VisionAnalysisResult
from .scoring import InvalidDecision
from .subtitler import generate as generate_subtitles
from .twitch import TwitchAPI, TwitchAPIError
from .utils import ffprobe_duration
from .vision_analyzer import analyze_vision

log = get_logger(__name__)


class Pipeline:
    def __init__(self, cfg, db: Database, twitch: Optional[TwitchAPI] = None):
        self.cfg = cfg
        self.db = db
        self.twitch = twitch or TwitchAPI.from_config(cfg)

    def _check_prereqs(self):
        if not self.twitch.health_check():
            raise TwitchAPIError(
                f"Twitch provider '{self.twitch.provider_name}' is unreachable. "
                "Check your network or switch providers in .env."
            )

    def discover(self, game: str, limit: int, period: str = "week"):
        self._check_prereqs()
        clips = self.twitch.get_clips(game, limit, period)
        if not clips:
            raise TwitchAPIError(
                f"No clips returned for '{game}'. The GraphQL schema may have "
                "changed, or the game has no recent clips. Check logs."
            )
        new = [c for c in clips if self.db.upsert_clip(c)]
        log.info("Discovered %d, new %d", len(clips), len(new))
        return new

    def _clamp_times(self, s: float, e: float, duration: float):
        s = max(0.0, s)
        e = min(duration, e)
        if e - s < self.cfg.min_clip_duration:
            extra = self.cfg.min_clip_duration - (e - s)
            s = max(0.0, s - extra / 2)
            e = min(duration, e + extra / 2)
            if e - s < self.cfg.min_clip_duration and s == 0.0:
                e = min(duration, self.cfg.min_clip_duration)
        if e - s > self.cfg.max_clip_duration:
            e = s + self.cfg.max_clip_duration
        return round(s, 3), round(e, 3)

    def process_one(self, clip_id: str) -> dict:
        clip = self.db.get_clip(clip_id)
        if not clip:
            return {"ok": False, "error": "not in DB"}
        if clip["status"] in ("READY", "REJECTED"):
            log.info("Skipping %s (status=%s)", clip_id, clip["status"])
            return {"ok": True, "skipped": True, "status": clip["status"]}

        work = self.cfg.processed / clip_id
        work.mkdir(parents=True, exist_ok=True)

        try:
            # 1. download
            self.db.update_status(clip_id, "DOWNLOADING")
            source = download_clip(clip, self.cfg.raw_clips,
                                   attempts=self.cfg.retry_attempts)
            self.db.set_local_path(clip_id, source)
            self.db.update_status(clip_id, "DOWNLOADED")

            # 2. analyze
            self.db.update_status(clip_id, "ANALYZING")
            audio = analyze_audio(source, work, self.cfg)
            if self.cfg.analyze_vision:
                vision = analyze_vision(source, work, self.cfg)
            else:
                vision = VisionAnalysisResult(audio.duration, [], "")

            self.db.save_analysis(
                clip_id,
                json.dumps({"language": audio.language,
                            "full_text": audio.full_text,
                            "events": [e.__dict__ for e in audio.events],
                            "words": [w.__dict__ for w in audio.words]},
                           ensure_ascii=False),
                json.dumps({"frames": [f.__dict__ for f in vision.frames],
                            "sequence_summary": vision.sequence_summary},
                           ensure_ascii=False),
            )
            self.db.update_status(clip_id, "ANALYZED")

            # 3. curator
            duration = ffprobe_duration(source)
            try:
                decision = analyze_clip(audio, vision, duration, self.cfg)
            except InvalidDecision as e:
                self.db.mark_failed(clip_id, f"curator: {e}")
                return {"ok": False, "error": f"curator: {e}"}

            d = {k: v for k, v in decision.__dict__.items() if k != "raw"}

            # 4. gate
            if (decision.decision == "reject"
                    or decision.overall_score < self.cfg.min_score_review):
                self.db.save_decision(clip_id, d, "REJECTED")
                return {"ok": True, "decision": "reject",
                        "score": decision.overall_score}

            if (decision.decision == "review"
                    or decision.overall_score < self.cfg.min_score_candidate):
                self.db.save_decision(clip_id, d, "ANALYZED")
                return {"ok": True, "decision": "review",
                        "score": decision.overall_score}

            # 5. edit
            self.db.save_decision(clip_id, d, "APPROVED")
            self.db.update_status(clip_id, "EDITING")

            start, end = self._clamp_times(decision.best_moment_start,
                                           decision.best_moment_end, duration)
            subs = None
            if self.cfg.subtitles_enabled and audio.words:
                subs = generate_subtitles(clip_id, audio.words, work,
                                          start, end, self.cfg)

            export_dir = self.cfg.exports / (decision.category or "Экшен")
            export_dir.mkdir(parents=True, exist_ok=True)
            out = export_dir / f"{clip_id}_{decision.category or 'clip'}.mp4"
            render_vertical(clip_id, source, out, start, end, self.cfg, subs)
            self.db.set_export_path(clip_id, out)
            return {"ok": True, "decision": "candidate",
                    "score": decision.overall_score, "output": str(out)}

        except (DownloadError, EditorError, RuntimeError) as e:
            log.exception("Clip %s failed: %s", clip_id, e)
            self.db.mark_failed(clip_id, str(e))
            return {"ok": False, "error": str(e)}

    def run(self, game: str, count: int, period: str = "week"):
        new = self.discover(game, count, period)
        results = []
        for c in new:
            r = self.process_one(c.id)
            r["clip_id"] = c.id
            results.append(r)
        return results
