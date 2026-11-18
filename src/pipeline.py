"""Resumable, per-clip pipeline with explicit paths and stage caches."""
from __future__ import annotations

import json
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .audio_analyzer import analyze_audio
from .auto_curator import analyze_clip
from .database import (AUTO_QUEUE_STATUSES, Database, DatabaseError,
                       default_worker_id)
from .downloader import DownloadError, download
from .editor import EditorError, render_vertical
from .logging_setup import get_logger, stage_timer
from .models import (AnalysisError, AudioAnalysisResult, AudioEvent,
                     CuratorDecision, FrameDescription, TranscriptSegment,
                     TranscriptWord, VisionAnalysisResult)
from .moment import select_moment
from .scoring import InvalidDecision
from .subtitler import generate as generate_subtitles
from .twitch import TwitchAPI, TwitchAPIError
from .utils import (MediaError, file_fingerprint, probe_media, safe_filename,
                    write_json)
from .vision_analyzer import analyze_vision

log = get_logger(__name__)

ANALYSIS_VERSION = "2"
SKIP_STATUSES = {"READY", "REJECTED", "REVIEW", "ANALYSIS_ERROR", "FAILED"}


class Pipeline:
    def __init__(self, cfg, db: Database, twitch: Optional[TwitchAPI] = None):
        self.cfg = cfg
        self.db = db
        self.twitch = twitch or TwitchAPI.from_config(cfg)

    def _touch(self, clip_id: str, worker_id: Optional[str]) -> None:
        """Refresh the worker lock so a long render is not treated as stale."""
        if not worker_id:
            return
        try:
            if not self.db.heartbeat(clip_id, worker_id):
                log.warning("Lock for %s is no longer held by %s",
                            clip_id, worker_id)
        except Exception as exc:  # noqa: BLE001 - heartbeat must not kill a clip
            log.warning("Heartbeat failed for %s: %s", clip_id, exc)

    def _write_analysis_artifact(self, work: Path,
                                 audio: Optional[AudioAnalysisResult],
                                 vision: Optional[VisionAnalysisResult],
                                 decision: Optional[Dict[str, Any]] = None,
                                 moment=None) -> None:
        """Persist a readable audit artifact next to transcript/vision caches."""
        data: Dict[str, Any] = {}
        if audio is not None:
            data["audio"] = asdict(audio)
        if vision is not None:
            data["vision"] = asdict(vision)
        if decision is not None:
            data["decision"] = decision
        if moment is not None:
            data["moment"] = {"start": moment.start, "end": moment.end,
                              "reason": moment.reason,
                              "expanded": moment.expanded}
        try:
            write_json(work / "analysis.json", data)
        except Exception as exc:  # noqa: BLE001 - audit artifact is best effort
            log.warning("Could not write analysis.json for %s: %s",
                        work.name, exc)

    @staticmethod
    def _link_vertical(work: Path, output: Path) -> None:
        """Keep a processed/<clip_id>/vertical.mp4 audit copy without doubling I/O.

        A hardlink is used on the same volume; otherwise fall back to a copy.
        """
        target = work / "vertical.mp4"
        try:
            target.unlink(missing_ok=True)
            try:
                os.link(output, target)
            except OSError:
                shutil.copy2(output, target)
        except Exception as exc:  # noqa: BLE001 - export already verified
            log.warning("Could not create vertical.mp4 artifact: %s", exc)

    # --------------------------------------------------------------- discovery
    def _check_prereqs(self) -> None:
        health = self.twitch.check_health()
        if not health.ok:
            raise TwitchAPIError(
                f"Twitch provider '{self.twitch.provider_name}' is not usable: "
                f"{health.detail}"
            )

    def discover(self, game: str, limit: int, period: str = "week") -> list:
        self._check_prereqs()
        clips = self.twitch.get_clips(game, limit, period)
        if not clips:
            log.warning("No clips returned for %r (period=%s)", game, period)
            return []
        new = [clip for clip in clips if self.db.upsert_clip(clip)]
        log.info("Discovered %d clip(s), new %d", len(clips), len(new))
        return new

    # ------------------------------------------------------------------ paths
    def _find_source(self, clip: Dict[str, Any]) -> Optional[Path]:
        candidates: List[Path] = []
        if clip.get("local_path"):
            candidates.append(Path(clip["local_path"]))
        candidates.append(self.cfg.raw_clips / f"{clip['id']}_source.mp4")
        for path in candidates:
            try:
                if path.is_file() and path.stat().st_size > 1024:
                    info = probe_media(path)
                    if info.is_valid_video:
                        return path
            except (OSError, MediaError):
                continue
        return None

    # ------------------------------------------------------------ cache loaders
    @staticmethod
    def _audio_from_dict(data: Dict[str, Any]) -> AudioAnalysisResult:
        segments = [
            TranscriptSegment(float(item["start"]), float(item["end"]),
                              str(item.get("text", "")))
            for item in data.get("segments", [])
        ]
        words = [
            TranscriptWord(float(item["start"]), float(item["end"]),
                           str(item.get("word", "")))
            for item in data.get("words", [])
        ]
        events = []
        for item in data.get("events", []):
            events.append(AudioEvent(
                kind=str(item.get("kind", "loud")),
                start=float(item.get("start", 0)),
                end=float(item.get("end", 0)),
                intensity=float(item.get("intensity", 1.0)),
                speech_overlap=bool(item.get("speech_overlap", False)),
            ))
        return AudioAnalysisResult(
            language=str(data.get("language", "unknown")),
            duration=float(data.get("duration", 0.0)),
            segments=segments, words=words, events=events,
            full_text=str(data.get("full_text", "")),
            model=str(data.get("model", "")),
            source_fingerprint=str(data.get("source_fingerprint", "")),
        )

    @staticmethod
    def _vision_from_dict(data: Dict[str, Any]) -> VisionAnalysisResult:
        frames = [
            FrameDescription(
                timestamp=float(item.get("timestamp", 0)),
                description=str(item.get("description", "")),
                path=str(item.get("path", "")),
            )
            for item in data.get("frames", [])
        ]
        return VisionAnalysisResult(
            duration=float(data.get("duration", 0.0)),
            frames=frames,
            sequence_summary=str(data.get("sequence_summary", "")),
            model=str(data.get("model", "")),
            scene_changes=[float(x) for x in data.get("scene_changes", [])],
            source_fingerprint=str(data.get("source_fingerprint", "")),
        )

    def _load_analysis(self, clip_id: str
                       ) -> Tuple[Optional[AudioAnalysisResult],
                                  Optional[VisionAnalysisResult]]:
        row = self.db.get_analysis(clip_id)
        if not row:
            return None, None
        try:
            audio_raw = json.loads(row["audio_json"]) if row.get("audio_json") else None
            vision_raw = json.loads(row["vision_json"]) if row.get("vision_json") else None
            if not isinstance(audio_raw, dict) or not isinstance(vision_raw, dict):
                return None, None
            return (self._audio_from_dict(audio_raw),
                    self._vision_from_dict(vision_raw))
        except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
            log.warning("Cached analysis for %s is unreadable: %s", clip_id, exc)
            return None, None

    def _load_decision(self, clip_id: str) -> Optional[CuratorDecision]:
        row = self.db.get_analysis(clip_id)
        if not row or not row.get("decision_json"):
            return None
        try:
            raw = json.loads(row["decision_json"])
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
        if not isinstance(raw, dict) or "overall_score" not in raw:
            return None
        try:
            return CuratorDecision(
                category=str(raw.get("category", "")),
                event_description=str(raw.get("event_description", "")),
                unexpectedness=float(raw.get("unexpectedness", 0)),
                emotionality=float(raw.get("emotionality", 0)),
                payoff_strength=float(raw.get("payoff_strength", 0)),
                event_clarity=float(raw.get("event_clarity", 0)),
                visual_interest=float(raw.get("visual_interest", 0)),
                hook_strength=float(raw.get("hook_strength", 0)),
                context_independence=float(raw.get("context_independence", 0)),
                overall_score=int(raw.get("overall_score", 0)),
                confidence=float(raw.get("confidence", 0)),
                decision=str(raw.get("decision", "review")),
                best_moment_start=float(raw.get("best_moment_start", 0)),
                best_moment_end=float(raw.get("best_moment_end", 0)),
                reason=str(raw.get("reason", "")),
                raw=raw,
            )
        except (TypeError, ValueError):
            return None

    # --------------------------------------------------------------- one clip
    def process_one(self, clip_id: str,
                    worker_id: Optional[str] = None) -> Dict[str, Any]:
        clip = self.db.get_clip(clip_id)
        if not clip:
            return {"ok": False, "clip_id": clip_id, "error": "not in DB"}
        status = clip["status"]
        if status in SKIP_STATUSES:
            log.info("Skipping %s (status=%s)", clip_id, status)
            return {"ok": True, "clip_id": clip_id, "skipped": True,
                    "status": status}

        work = self.cfg.processed / clip_id
        work.mkdir(parents=True, exist_ok=True)
        stage = "start"
        try:
            source = self._find_source(clip)

            # 1. Download / verify exact source file.
            if source is None:
                stage = "download"
                if status not in ("DOWNLOADING",):
                    self.db.update_status(clip_id, "DOWNLOADING", stage=stage)
                with stage_timer(log, stage, clip_id):
                    result = download(
                        clip, self.cfg.raw_clips,
                        attempts=self.cfg.retry_attempts,
                        min_free_mb=self.cfg.min_free_disk_mb,
                    )
                source = result.path
                self.db.set_local_path(clip_id, source,
                                       fingerprint=file_fingerprint(source))
                self.db.update_status(clip_id, "DOWNLOADED", stage=stage)
                status = "DOWNLOADED"
                self._touch(clip_id, worker_id)
            elif status == "DISCOVERED":
                # Source survived a crash between download and DB update.
                self.db.update_status(clip_id, "DOWNLOADING", stage="download")
                self.db.set_local_path(clip_id, source,
                                       fingerprint=file_fingerprint(source))
                self.db.update_status(clip_id, "DOWNLOADED", stage="download")
                status = "DOWNLOADED"

            # 2. Audio + vision, reusing saved analysis after a restart.
            stage = "analysis"
            audio, vision = (None, None)
            if status in ("ANALYZED", "APPROVED", "EDITING", "ANALYZING",
                          "DOWNLOADED"):
                audio, vision = self._load_analysis(clip_id)
            if audio is None:
                if status != "ANALYZING":
                    self.db.update_status(clip_id, "ANALYZING", stage=stage)
                try:
                    with stage_timer(log, stage, clip_id):
                        audio = analyze_audio(source, work, self.cfg)
                        if self.cfg.analyze_vision:
                            vision = analyze_vision(source, work, self.cfg)
                        else:
                            vision = VisionAnalysisResult(audio.duration, [], "")
                except AnalysisError as exc:
                    self.db.mark_analysis_error(
                        clip_id, f"analysis: {exc}", stage)
                    return {"ok": False, "clip_id": clip_id,
                            "status": "ANALYSIS_ERROR", "error": str(exc)}
                except Exception as exc:  # noqa: BLE001
                    # Any failure while reading media / running a model is an
                    # analysis failure, not a content rejection.
                    self.db.mark_analysis_error(
                        clip_id,
                        f"analysis: {type(exc).__name__}: {exc}", stage)
                    return {"ok": False, "clip_id": clip_id,
                            "status": "ANALYSIS_ERROR",
                            "error": f"{type(exc).__name__}: {exc}"}
                self.db.save_analysis(
                    clip_id,
                    json.dumps({
                        "language": audio.language,
                        "duration": audio.duration,
                        "full_text": audio.full_text,
                        "events": [event.__dict__ for event in audio.events],
                        "words": [word.__dict__ for word in audio.words],
                        "segments": [segment.__dict__ for segment in audio.segments],
                        "model": audio.model,
                        "source_fingerprint": audio.source_fingerprint,
                    }, ensure_ascii=False),
                    json.dumps({
                        "duration": vision.duration,
                        "frames": [frame.__dict__ for frame in vision.frames],
                        "sequence_summary": vision.sequence_summary,
                        "scene_changes": vision.scene_changes,
                        "model": vision.model,
                        "source_fingerprint": vision.source_fingerprint,
                    }, ensure_ascii=False),
                    version=ANALYSIS_VERSION,
                )
                self._write_analysis_artifact(work, audio, vision)
                self._touch(clip_id, worker_id)
                self.db.update_status(clip_id, "ANALYZED", stage="curator")
                status = "ANALYZED"
            else:
                if status == "ANALYZING":
                    self.db.update_status(clip_id, "ANALYZED", stage="curator")
                    status = "ANALYZED"
                elif status == "DOWNLOADED":
                    self.db.update_status(clip_id, "ANALYZING", stage=stage)
                    self.db.update_status(clip_id, "ANALYZED", stage="curator")
                    status = "ANALYZED"

            # 3. Curator decision, reused after a restart.
            stage = "curator"
            decision = self._load_decision(clip_id)
            if decision is None:
                duration = probe_media(source).duration
                if status in ("APPROVED", "EDITING"):
                    # A missing decision while already approved is inconsistent;
                    # step back explicitly instead of inventing a decision.
                    self.db.update_status(clip_id, "ANALYZED", force=True,
                                          stage=stage)
                    status = "ANALYZED"
                with stage_timer(log, stage, clip_id,
                                 duration=f"{duration:.1f}s"):
                    try:
                        decision = analyze_clip(
                            audio, vision, duration, self.cfg,
                            game_name=str(clip.get("game_name") or ""),
                            title=str(clip.get("title") or ""),
                        )
                    except InvalidDecision as exc:
                        self.db.mark_analysis_error(
                            clip_id, f"curator: {exc}", stage)
                        return {"ok": False, "clip_id": clip_id,
                                "status": "ANALYSIS_ERROR", "error": str(exc)}
                    except Exception as exc:  # noqa: BLE001
                        self.db.mark_analysis_error(
                            clip_id,
                            f"curator: {type(exc).__name__}: {exc}", stage)
                        return {"ok": False, "clip_id": clip_id,
                                "status": "ANALYSIS_ERROR", "error": str(exc)}
            else:
                duration = probe_media(source).duration

            payload = {key: value for key, value in decision.__dict__.items()
                       if key != "raw"}
            self._write_analysis_artifact(work, audio, vision, decision=payload)
            self._touch(clip_id, worker_id)

            # Low-confidence candidates need a human or a stronger second pass.
            if (decision.decision == "candidate"
                    and decision.confidence < self.cfg.min_confidence):
                decision.decision = "review"
                decision.reason += (
                    f" (confidence {decision.confidence:.2f} below "
                    f"{self.cfg.min_confidence:.2f})"
                )
                payload["decision"] = "review"
                payload["reason"] = decision.reason

            # 4. Gate: content rejection is distinct from analysis failure.
            # A model "reject" with a high weighted score is a disagreement,
            # not a confident rejection: send it to review instead of silently
            # destroying a possible good moment.
            score = decision.overall_score
            if decision.decision == "reject":
                if score >= self.cfg.min_score_review:
                    self.db.save_decision(clip_id, payload, "REVIEW")
                    return {"ok": True, "clip_id": clip_id,
                            "decision": "review", "score": score,
                            "reason": "model rejected but weighted score is high"}
                self.db.save_decision(clip_id, payload, "REJECTED")
                return {"ok": True, "clip_id": clip_id, "decision": "reject",
                        "score": score}
            if (decision.decision == "review"
                    or score < self.cfg.min_score_candidate):
                self.db.save_decision(clip_id, payload, "REVIEW")
                return {"ok": True, "clip_id": clip_id, "decision": "review",
                        "score": score}

            # 5. Edit the AI-selected moment of this exact source.
            if status != "APPROVED":
                self.db.save_decision(clip_id, payload, "APPROVED")
                status = "APPROVED"
            self.db.update_status(clip_id, "EDITING", stage="edit")
            stage = "edit"
            moment = select_moment(
                decision.best_moment_start, decision.best_moment_end,
                duration, audio.words, audio.events, self.cfg,
            )
            self._write_analysis_artifact(work, audio, vision,
                                          decision=payload, moment=moment)
            self._touch(clip_id, worker_id)
            subtitles = None
            with stage_timer(log, "subtitles", clip_id,
                             start=f"{moment.start:.2f}",
                             end=f"{moment.end:.2f}"):
                if self.cfg.subtitles_enabled and audio.words:
                    subtitles = generate_subtitles(
                        clip_id, audio.words, work, moment.start, moment.end,
                        self.cfg,
                    )
                category = safe_filename(decision.category or "clip")
                export_dir = self.cfg.exports / category
                export_dir.mkdir(parents=True, exist_ok=True)
                output = export_dir / f"{clip_id}_{category}.mp4"
                render_vertical(clip_id, source, output, moment.start,
                                moment.end, self.cfg, subtitles)
            self._link_vertical(work, output)
            self._touch(clip_id, worker_id)
            self.db.set_export_path(clip_id, output)
            return {"ok": True, "clip_id": clip_id, "decision": "candidate",
                    "score": decision.overall_score, "output": str(output),
                    "moment": [moment.start, moment.end]}

        except (DownloadError, EditorError) as exc:
            log.exception("Clip %s failed at stage=%s: %s", clip_id, stage, exc)
            self._safe_fail(clip_id, f"{stage}: {exc}", stage)
            return {"ok": False, "clip_id": clip_id, "status": "FAILED",
                    "error": str(exc)}
        except Exception as exc:  # noqa: BLE001 - one clip must not kill queue
            log.exception("Clip %s failed at stage=%s: %s", clip_id, stage, exc)
            self._safe_fail(clip_id, f"{stage}: {type(exc).__name__}: {exc}",
                            stage)
            return {"ok": False, "clip_id": clip_id, "status": "FAILED",
                    "error": f"{type(exc).__name__}: {exc}"}
        finally:
            if worker_id:
                self.db.release(clip_id, worker_id)

    def _safe_fail(self, clip_id: str, error: str, stage: str) -> None:
        try:
            self.db.mark_failed(clip_id, error, stage)
        except Exception:
            log.exception("Could not persist failure for %s", clip_id)

    # ------------------------------------------------------------------ queue
    def run_queue(self, game: Optional[str] = None,
                  limit: Optional[int] = None) -> List[dict]:
        workers = max(1, int(getattr(self.cfg, "max_parallel", 1)))
        if workers == 1:
            return self._run_queue_worker(game, limit, "worker-1")
        results: List[dict] = []
        limit_lock = threading.Lock()
        processed = [0]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(self._run_queue_worker, game, limit,
                                f"worker-{index + 1}", limit_lock, processed)
                for index in range(workers)
            ]
            for future in as_completed(futures):
                try:
                    results.extend(future.result())
                except Exception:  # noqa: BLE001
                    log.exception("Pipeline worker crashed")
        return results

    def _run_queue_worker(self, game: Optional[str], limit: Optional[int],
                          worker_name: str, limit_lock=None,
                          processed=None) -> List[dict]:
        worker_id = f"{default_worker_id()}-{worker_name}"
        results: List[dict] = []
        while True:
            if limit is not None and limit_lock is not None:
                with limit_lock:
                    if processed[0] >= limit:
                        break
                    processed[0] += 1
            clip = self.db.claim_next(
                worker_id=worker_id, game=game,
                stale_minutes=self.cfg.stale_lock_minutes,
            )
            if clip is None:
                if limit is not None and limit_lock is not None:
                    with limit_lock:
                        processed[0] = max(0, processed[0] - 1)
                break
            results.append(self.process_one(clip["id"], worker_id=worker_id))
        return results

    def run(self, game: str, count: int, period: str = "week") -> List[dict]:
        discovery_error: Optional[str] = None
        try:
            self.discover(game, count, period)
        except TwitchAPIError as exc:
            discovery_error = str(exc)
            log.error("Discovery failed, continuing with the queue: %s", exc)
        results = self.run_queue(game=game, limit=max(1, int(count)))
        if discovery_error and not results:
            results.append({"ok": False, "stage": "discover",
                            "error": discovery_error})
        return results
