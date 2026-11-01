#!/usr/bin/env python3
"""Submit a verified lap time to the public Cloudflare leaderboard.

Requires env LEADERBOARD_URL (Pages/Workers base, no trailing slash) and
optionally LEADERBOARD_TOKEN if the Worker is configured with a submit secret.

  export LEADERBOARD_URL=https://drone-race-hacks.<account>.pages.dev
  python race/submit_time.py --team "alice" --time 12.345 --drone TELLO-1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--team", required=True)
    ap.add_argument("--time", type=float, required=True, help="lap time seconds")
    ap.add_argument("--drone", default="")
    ap.add_argument("--notes", default="")
    ap.add_argument("--url", default=os.environ.get("LEADERBOARD_URL", ""))
    ap.add_argument("--token", default=os.environ.get("LEADERBOARD_TOKEN", ""))
    args = ap.parse_args()
    if not args.url:
        raise SystemExit("set --url or LEADERBOARD_URL")
    if args.time <= 0:
        raise SystemExit("time must be > 0")

    body = {
        "team": args.team.strip()[:64],
        "time_s": float(args.time),
        "drone": args.drone.strip()[:32],
        "notes": args.notes.strip()[:200],
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        args.url.rstrip("/") + "/api/submit",
        data=data,
        method="POST",
        headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {args.token}"} if args.token else {})},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            print(resp.read().decode())
    except urllib.error.HTTPError as e:
        print(e.read().decode(), file=sys.stderr)
        raise SystemExit(e.code)


if __name__ == "__main__":
    main()
