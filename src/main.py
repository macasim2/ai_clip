"""CLI entry point."""
import argparse
import sys

from .config import load_config
from .database import Database
from .logging_setup import setup_logging
from .pipeline import Pipeline
from .twitch import TwitchAPI, TwitchAPIError


def main():
    p = argparse.ArgumentParser(description="AI Clip Studio")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="Discover and process clips")
    r.add_argument("--game", required=True)
    r.add_argument("--count", type=int, default=10)
    r.add_argument("--period", default="week",
                   choices=["day", "week", "month", "all"])

    sub.add_parser("retry", help="Reprocess FAILED/pending clips")
    sub.add_parser("status", help="Show queue")
    sub.add_parser("health", help="Check Twitch and config")

    args = p.parse_args()
    cfg = load_config()
    setup_logging(cfg.logs)
    db = Database(cfg.db_path)

    if args.cmd == "status":
        for row in db.list_pending():
            print(f"{row['id']:<24} {row['status']:<12} "
                  f"score={row['score']}  {(row['title'] or '')[:60]}")
        return

    if args.cmd == "health":
        try:
            api = TwitchAPI.from_config(cfg)
            ok = api.health_check()
            print(f"Twitch provider '{api.provider_name}': "
                  f"{'OK' if ok else 'UNREACHABLE'}")
            print(f"Ollama URL: {cfg.ollama_url}")
            return
        except TwitchAPIError as e:
            print(f"FAIL: {e}")
            sys.exit(1)

    try:
        pipeline = Pipeline(cfg, db)
    except TwitchAPIError as e:
        print(f"Twitch config error: {e}", file=sys.stderr)
        sys.exit(2)

    if args.cmd == "retry":
        for row in db.list_pending():
            if row["status"] == "FAILED":
                db.update_status(row["id"], "DISCOVERED")
            print(row["id"], pipeline.process_one(row["id"]))
        return

    try:
        for r in pipeline.run(args.game, args.count, args.period):
            print(r)
    except TwitchAPIError as e:
        print(f"Twitch error: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
