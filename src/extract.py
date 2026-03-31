"""Extract structured LLM/tool traces from one HyperAgent trajectory JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hyperagent_replay.trace import extract_trace, load_raw_trace


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract structured LLM/tool traces from HyperAgent logs")
    parser.add_argument("input", type=Path, help="Path to HyperAgent JSON")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save extracted trace JSON",
    )
    args = parser.parse_args()

    output = extract_trace(load_raw_trace(args.input))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2))
        print(f"Saved extracted trace to {args.output}")

    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()
