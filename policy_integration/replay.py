"""Read canonical JSONL observations; print decisions. Never sends drone commands."""

import argparse
import json
import sys
from pathlib import Path
from .policy import BaselinePolicy, Observation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observations", type=Path)
    args = parser.parse_args(argv)
    policy = BaselinePolicy()
    with args.observations.open() as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                decision = policy.step(Observation(**row["observation"]), row["now_s"])
            except (ValueError, TypeError, KeyError) as error:
                print(f"invalid JSONL at line {line_no}: {error}", file=sys.stderr)
                return 2
            print(json.dumps({"line": line_no, **decision.to_dict()}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
