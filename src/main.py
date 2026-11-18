"""CLI entry point."""
from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from .config import ConfigError, load_config
from .database import AUTO_QUEUE_STATUSES, Database, DatabaseError
from .logging_setup import get_logger, setup_logging
from .pipeline import Pipeline
from .twitch import TwitchAPI, TwitchAPIError

log = get_logger(__name__)


def _configure_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def _print_stats(db: Database) -> None:
    stats = db.stats()
    total = sum(stats.values())
    print(f"Clips in database: {total}")
    for status, count in stats.items():
        if count:
            print(f"  {status:<14} {count}")


def _print_clip(row: dict) -> None:
    print(f"{row.get('id'):<14} {row.get('status', ''):<14} "
          f"score={row.get('score')}  game={row.get('game_name', '')!r}  "
          f"title={(row.get('title') or '')[:60]!r}")
    if row.get("stage"):
        print(f"    stage: {row['stage']}  attempts: {row.get('attempts', 0)}")
    if row.get("local_path"):
        print(f"    source: {row['local_path']}")
    if row.get("export_path"):
        print(f"    export: {row['export_path']}")
    if row.get("error_message"):
        print(f"    error: {str(row['error_message'])[:300]}")
    if row.get("review_reason"):
        print(f"    review: {str(row['review_reason'])[:300]}")
    if row.get("rejection_reason"):
        print(f"    rejection: {str(row['rejection_reason'])[:300]}")


def _cmd_status(db: Database, args) -> int:
    if args.clip:
        row = db.get_clip(args.clip)
        if not row:
            print(f"Clip not found: {args.clip}")
            return 1
        _print_clip(row)
        print("\nHistory:")
        for item in db.history(args.clip, limit=args.limit):
            print(f"  {item['created_at']}  {item.get('from_status') or '-':<12} "
                  f"-> {item['to_status']:<14} {item.get('stage') or ''} "
                  f"{(item.get('error') or '')[:120]}")
        return 0
    _print_stats(db)
    rows = db.list_all(limit=args.limit)
    if rows:
        print()
    for row in rows:
        _print_clip(row)
    return 0


def _cmd_review(db: Database) -> int:
    rows = db.list_review()
    if not rows:
        print("Nothing to review.")
        return 0
    for row in rows:
        _print_clip(row)
    return 0


def _cmd_health(cfg, db: Database) -> int:
    from .launcher import check_environment

    code = check_environment(cfg, require_ai=True)
    print()
    try:
        api = TwitchAPI.from_config(cfg)
        health = api.check_health()
        print(f"Twitch provider '{api.provider_name}': "
              f"{'OK' if health.ok else 'FAIL'} — {health.detail}")
        if not health.ok:
            code = 1
    except TwitchAPIError as exc:
        print(f"Twitch provider config error: {exc}")
        code = 1
    _print_stats(db)
    return code


def _make_pipeline(cfg, db: Database) -> Pipeline:
    return Pipeline(cfg, db)


def _prepare_processing(db: Database, cfg) -> None:
    stale = db.recover_stale(cfg.stale_lock_minutes)
    for change in stale:
        log.warning("Recovered %s: %s -> %s", change["clip_id"],
                    change["from"], change["to"])
    broken = db.reconcile_ready()
    for change in broken:
        log.warning("READY downgraded: %s", change["clip_id"])


def _process_queue(cfg, db: Database, game: Optional[str],
                   limit: Optional[int] = None) -> List[dict]:
    pipeline = _make_pipeline(cfg, db)
    rows = db.list_pending(game=game)
    results = []
    for row in rows:
        if row["status"] not in AUTO_QUEUE_STATUSES:
            continue
        results.append(pipeline.process_one(row["id"]))
        if limit and len(results) >= limit:
            break
    return results


def main(argv: Optional[List[str]] = None) -> int:
    _configure_stdout()
    parser = argparse.ArgumentParser(description="AI Clip Studio")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Discover and process clips")
    run.add_argument("--game", required=True)
    run.add_argument("--count", type=int, default=10)
    run.add_argument("--period", default="week",
                     choices=["day", "week", "month", "all"])
    run.add_argument("--no-discovery", action="store_true",
                     help="Only continue the existing queue")

    resume = sub.add_parser("resume", help="Continue the pending queue")
    resume.add_argument("--game", default=None)
    resume.add_argument("--count", type=int, default=None)

    retry = sub.add_parser("retry", help="Manually re-queue failed clips")
    retry.add_argument("--clip", default=None)
    retry.add_argument("--from-stage", default="download",
                       choices=["download", "analysis", "edit"])
    retry.add_argument("--all-failed", action="store_true")

    status = sub.add_parser("status", help="Show queue and history")
    status.add_argument("--clip", default=None)
    status.add_argument("--limit", type=int, default=50)

    sub.add_parser("review", help="Show clips that need review")
    sub.add_parser("health", help="Check Twitch, Ollama, FFmpeg and config")

    args = parser.parse_args(argv)

    try:
        cfg = load_config()
    except ConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    cfg.ensure_dirs()
    setup_logging(cfg.logs, cfg.log_level)
    try:
        db = Database(cfg.db_path, backup=cfg.db_backup)
    except DatabaseError as exc:
        print(f"Database error: {exc}", file=sys.stderr)
        return 2

    if args.cmd == "status":
        return _cmd_status(db, args)
    if args.cmd == "review":
        return _cmd_review(db)
    if args.cmd == "health":
        return _cmd_health(cfg, db)

    _prepare_processing(db, cfg)

    if args.cmd == "retry":
        try:
            if args.clip:
                row = db.get_clip(args.clip)
                if not row:
                    print(f"Clip not found: {args.clip}", file=sys.stderr)
                    return 1
                db.reset_for_retry(args.clip, from_stage=args.from_stage)
                pipeline = _make_pipeline(cfg, db)
                print(pipeline.process_one(args.clip))
                return 0
            failed = db.list_failed()
            if not failed:
                print("No failed clips to retry.")
                return 0
            pipeline = _make_pipeline(cfg, db)
            for row in failed:
                db.reset_for_retry(row["id"], from_stage=args.from_stage)
                print(pipeline.process_one(row["id"]))
            return 0
        except (TwitchAPIError, DatabaseError) as exc:
            print(f"Retry failed: {exc}", file=sys.stderr)
            return 2

    try:
        if args.cmd == "resume":
            results = _process_queue(cfg, db, args.game, args.count)
        else:
            pipeline = _make_pipeline(cfg, db)
            if args.no_discovery:
                results = _process_queue(cfg, db, args.game, args.count)
            else:
                results = pipeline.run(args.game, args.count, args.period)
    except TwitchAPIError as exc:
        print(f"Twitch error: {exc}", file=sys.stderr)
        return 2

    if not results:
        print("Nothing processed.")
        return 0
    ok = sum(1 for result in results if result.get("ok"))
    print(f"Processed {len(results)} clip(s): ok={ok}, "
          f"failed={len(results) - ok}")
    for result in results:
        print(result)
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted. Database state is consistent; run resume to continue.",
              file=sys.stderr)
        sys.exit(130)
