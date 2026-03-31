"""Summarize source and replay metrics for HyperAgent trajectory files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hyperagent_replay.trace import load_trace_payload


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    k = (len(ordered) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(ordered) - 1)
    return ordered[f] + (k - f) * (ordered[c] - ordered[f])


def summarize_replay(replay: dict[str, Any]) -> dict[str, Any]:
    turn_metrics = replay.get("turn_metrics", [])
    latencies = [entry["request_latency_s"] for entry in turn_metrics]
    prompt_tokens = [
        entry["prompt_tokens"] for entry in turn_metrics
        if entry.get("prompt_tokens") is not None
    ]
    completion_tokens = [
        entry["completion_tokens"] for entry in turn_metrics
        if entry.get("completion_tokens") is not None
    ]

    return {
        "num_replayed_turns": len(turn_metrics),
        "wall_solve_time_s": replay.get("timing", {}).get("wall_solve_time_s",
                                                           0.0),
        "lm_only_solve_time_s":
        replay.get("timing", {}).get("lm_only_solve_time_s", 0.0),
        "synthetic_tool_sleep_s":
        replay.get("timing", {}).get("synthetic_tool_sleep_s", 0.0),
        "avg_request_latency_s": (sum(latencies) / len(latencies)
                                   if latencies else 0.0),
        "median_request_latency_s": percentile(latencies, 50),
        "p95_request_latency_s": percentile(latencies, 95),
        "p99_request_latency_s": percentile(latencies, 99),
        "total_prompt_tokens": sum(prompt_tokens),
        "total_completion_tokens": sum(completion_tokens),
        "avg_prompt_tokens_per_turn":
        (sum(prompt_tokens) / len(prompt_tokens) if prompt_tokens else 0.0),
        "avg_completion_tokens_per_turn":
        (sum(completion_tokens) / len(completion_tokens)
         if completion_tokens else 0.0),
        "avg_prompt_chars_per_turn":
        (sum(entry["request_prompt_chars"] for entry in turn_metrics) /
         len(turn_metrics) if turn_metrics else 0.0),
        "avg_response_chars_per_turn":
        (sum(entry["response_chars"] for entry in turn_metrics) /
         len(turn_metrics) if turn_metrics else 0.0),
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a HyperAgent trajectory file and optional replay results"
    )
    parser.add_argument("input",
                        type=Path,
                        help="Raw HyperAgent JSON, extracted trace JSON, or replay JSON")
    parser.add_argument("--replay",
                        type=Path,
                        default=None,
                        help="Optional replay results JSON to merge")
    parser.add_argument("--output",
                        type=Path,
                        default=None,
                        help="Optional path to save evaluation summary JSON")
    args = parser.parse_args()

    primary = load_json(args.input)
    if isinstance(primary, dict) and "turn_metrics" in primary and "timing" in primary:
        trace = {
            "instance_id": primary.get("instance_id"),
            "problem_statement": primary.get("problem_statement", ""),
            "summary": primary.get("source_summary", {}),
            "llm_turns": [],
        }
        replay = primary
    else:
        trace = load_trace_payload(args.input)
        replay = load_json(args.replay) if args.replay is not None else None

    summary: dict[str, Any] = {
        "instance_id": trace.get("instance_id"),
        "source_metrics": trace.get("summary", {}),
    }
    if replay is not None:
        summary["replay_metrics"] = summarize_replay(replay)
        summary["replay_settings"] = replay.get("settings", {})

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2))
        print(f"Saved evaluation summary to {args.output}")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
