"""End-to-end pipeline orchestration (foundation version).

Stage 3 will add worker locks, resumable stage caching and parallel workers.
This version already keeps every stage bound to one exact clip/source path,
isolates per-clip failures and never marks READY without a verified export.
"""
from __future__ import annotations

import json
from typing import List, Optional

from .audio_analyzer import analyze_audio
from .auto_curator import analyze_clip
from .database import AUTO_QUEUE_STATUSES, Database
from .downloader import DownloadError, download_clip
from .editor import EditorError, render_vertical
from .logging_setup import get_logger, stage_timer
from .models import VisionAnalysisResult
from .scoring import InvalidDecision
from .subtitler import generate as generate_subtitles
from .twitch import TwitchAPI, TwitchAPIError
from .utils import clamp, file_fingerprint, ffprobe_duration
from .vision_analyzer import analyze_vision

log = get_logger(__name__)

ANALYSIS_VERSION = "2"


class Pipeline:
    def __init__(self, cfg, db: Database, twitch: Optional[TwitchAPI] = None):
        self.cfg = cfg
        self.db = db
        self.twitch = twitch or TwitchAPI.from_config(cfg)

    def _check_prereqs(self) -> None:
        health = self.twitch.check_health()
        if not health.ok:
            raise TwitchAPIError(
                f"Twitch provider '{self.twitch.provider_name}' is not usable: "
                f"{health.detail}"
            )

    def discover(self, game: str, limit: int, period: str = "week"):
        self._check_prereqs()
        clips = self.twitch.get_clips(game, limit, period)
        if not clips:
            log.warning("No clips returned for %r (period=%s)", game, period)
            return []
        new = [clip for clip in clips if self.db.upsert_clip(clip)]
        log.info("Discovered %d clip(s), new %d", len(clips), len(new))
        return new

    def _clamp_times(self, start: float, end: float, duration: float):
        """Guarantee min/max duration while keeping the requested window."""
        if duration <= 0:
            return 0.0, 0.0
        start = clamp(float(start), 0.0, duration)
        end = clamp(float(end), 0.0, duration)
        if end <= start:
            start, end = 0.0, duration

        min_duration = min(self.cfg.min_clip_duration, duration)
        max_duration = min(self.cfg.max_clip_duration, duration)
        if end - start < min_duration:
            need = min_duration - (end - start)
            start = max(0.0, start - need * 0.7)
            end = min(duration, end + need * 0.3)
            if end - start < min_duration:
                end = min(duration, start + min_duration)
            if end - start < min_duration:
                start = max(0.0, end - min_duration)
        if end - start > max_duration:
            peak = (start + end) / 2
            start = max(0.0, peak - max_duration / 2)
            end = min(duration, start + max_duration)
            if end - start > max_duration:
                start = max(0.0, end - max_duration)
        return round(start, 3), round(end, 3)

    def process_one(self, clip_id: str) -> dict:
        clip = self.db.get_clip(clip_id)
        if not clip:
            return {"ok": False, "clip_id": clip_id, "error": "not in DB"}
        if clip["status"] in ("READY", "REJECTED", "REVIEW", "ANALYSIS_ERROR",
                              "FAILED"):
            log.info("Skipping %s (status=%s)", clip_id, clip["status"])
            return {"ok": True, "clip_id": clip_id, "skipped": True,
                    "status": clip["status"]}

        work = self.cfg.processed / clip_id
        work.mkdir(parents=True, exist_ok=True)
        stage = "start"

        try:
            # 1. Download exact clip by id + URL; never pick by mtime.
            stage = "download"
            self.db.update_status(clip_id, "DOWNLOADING", stage=stage)
            with stage_timer(log, stage, clip_id):
                source = download_clip(
                    clip, self.cfg.raw_clips,
                    attempts=self.cfg.retry_attempts,
                    min_free_mb=self.cfg.min_free_disk_mb,
                )
            self.db.set_local_path(clip_id, source,
                                   fingerprint=file_fingerprint(source))
            self.db.update_status(clip_id, "DOWNLOADED", stage=stage)

            # 2. Audio + vision analysis (saved before the curator call).
            stage = "analysis"
            self.db.update_status(clip_id, "ANALYZING", stage=stage)
            with stage_timer(log, stage, clip_id):
                audio = analyze_audio(source, work, self.cfg)
                if self.cfg.analyze_vision:
                    vision = analyze_vision(source, work, self.cfg)
                else:
                    vision = VisionAnalysisResult(audio.duration, [], "")
            self.db.save_analysis(
                clip_id,
                json.dumps({
                    "language": audio.language,
                    "full_text": audio.full_text,
                    "events": [event.__dict__ for event in audio.events],
                    "words": [word.__dict__ for word in audio.words],
                }, ensure_ascii=False),
                json.dumps({
                    "frames": [frame.__dict__ for frame in vision.frames],
                    "sequence_summary": vision.sequence_summary,
                    "scene_changes": vision.scene_changes,
                }, ensure_ascii=False),
                version=ANALYSIS_VERSION,
            )
            self.db.update_status(clip_id, "ANALYZED", stage="curator")

            # 3. Content-based curator. A model/JSON failure is NOT a normal
            #    rejection and gets its own status.
            stage = "curator"
            duration = ffprobe_duration(source)
            with stage_timer(log, stage, clip_id, duration=f"{duration:.1f}s"):
                try:
                    decision = analyze_clip(audio, vision, duration, self.cfg)
                except InvalidDecision as exc:
                    self.db.mark_analysis_error(clip_id, f"curator: {exc}", stage)
                    return {"ok": False, "clip_id": clip_id,
                            "status": "ANALYSIS_ERROR", "error": str(exc)}
                except Exception as exc:  # noqa: BLE001 - model/transport error
                    self.db.mark_analysis_error(
                        clip_id, f"curator: {type(exc).__name__}: {exc}", stage
                    )
                    return {"ok": False, "clip_id": clip_id,
                            "status": "ANALYSIS_ERROR", "error": str(exc)}

            payload = {key: value for key, value in decision.__dict__.items()
                       if key != "raw"}

            # 4. Gate.
            if (decision.decision == "reject"
                    or decision.overall_score < self.cfg.min_score_review):
                self.db.save_decision(clip_id, payload, "REJECTED")
                return {"ok": True, "clip_id": clip_id, "decision": "reject",
                        "score": decision.overall_score}
            if (decision.decision == "review"
                    or decision.overall_score < self.cfg.min_score_candidate):
                self.db.save_decision(clip_id, payload, "REVIEW")
                return {"ok": True, "clip_id": clip_id, "decision": "review",
                        "score": decision.overall_score}

            # 5. Edit around the AI-selected moment, exactly this source file.
            self.db.save_decision(clip_id, payload, "APPROVED")
            self.db.update_status(clip_id, "EDITING", stage="edit")
            stage = "edit"
            start, end = self._clamp_times(decision.best_moment_start,
                                           decision.best_moment_end, duration)
            subtitles = None
            with stage_timer(log, "subtitles", clip_id,
                             start=f"{start:.2f}", end=f"{end:.2f}"):
                if self.cfg.subtitles_enabled and audio.words:
                    subtitles = generate_subtitles(clip_id, audio.words, work,
                                                   start, end, self.cfg)
                export_dir = self.cfg.exports / (decision.category or "Экшен")
                export_dir.mkdir(parents=True, exist_ok=True)
                output = export_dir / f"{clip_id}_{decision.category or 'clip'}.mp4"
                render_vertical(clip_id, source, output, start, end,
                                self.cfg, subtitles)
            self.db.set_export_path(clip_id, output)
            return {"ok": True, "clip_id": clip_id, "decision": "candidate",
                    "score": decision.overall_score, "output": str(output)}

        except (DownloadError, EditorError, TwitchAPIError) as exc:
            log.exception("Clip %s failed at stage=%s: %s", clip_id, stage, exc)
            self.db.mark_failed(clip_id, f"{stage}: {exc}", stage)
            return {"ok": False, "clip_id": clip_id, "status": "FAILED",
                    "error": str(exc)}
        except Exception as exc:  # noqa: BLE001 - one clip must not kill queue
            log.exception("Clip %s failed at stage=%s: %s", clip_id, stage, exc)
            try:
                self.db.mark_failed(clip_id, f"{stage}: {type(exc).__name__}: {exc}",
                                    stage)
            except Exception:
                log.exception("Could not persist failure for %s", clip_id)
            return {"ok": False, "clip_id": clip_id, "status": "FAILED",
                    "error": f"{type(exc).__name__}: {exc}"}

    def run(self, game: str, count: int, period: str = "week") -> List[dict]:
        discovery_error: Optional[str] = None
        new = []
        try:
            new = self.discover(game, count, period)
        except TwitchAPIError as exc:
            # A temporary provider outage must not stop the local queue.
            discovery_error = str(exc)
            log.error("Discovery failed, continuing with the queue: %s", exc)
        ids: List[str] = [clip.id for clip in new]
        # Continue the existing queue: a restart must not lose DISCOVERED/
        # DOWNLOADED/ANALYZED/APPROVED clips from an interrupted run.
        for row in self.db.list_pending(game=game):
            if row["status"] in AUTO_QUEUE_STATUSES and row["id"] not in ids:
                ids.append(row["id"])
        results = []
        for clip_id in ids:
            results.append(self.process_one(clip_id))
        if discovery_error and not results:
            results.append({"ok": False, "stage": "discover",
                            "error": discovery_error})
        return results
