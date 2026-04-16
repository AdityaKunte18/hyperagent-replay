"""Derive empirical per-agent request SLOs from replay JSON files."""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from hyperagent_replay.replay import percentile
from hyperagent_replay.slo_report import is_executed_request, request_metric_value


def print_table(rows: list[list[Any]], headers: list[str]) -> None:
    col_widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            col_widths[index] = max(col_widths[index], len(str(value)))
    fmt = " | ".join("{:" + str(width) + "}" for width in col_widths)
    sep = "-+-".join("-" * width for width in col_widths)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*[str(value) for value in row]))


def discover_replay_paths(pattern: str) -> list[Path]:
    matches = sorted(Path(path) for path in glob.glob(pattern, recursive=True))
    replay_paths = [path for path in matches if path.name.endswith(".replay.json")]
    if not replay_paths:
        raise SystemExit(f"No replay JSON files matched {pattern}")
    return replay_paths


def build_empirical_agent_slos(
    replay_paths: list[Path],
    *,
    request_metric: str,
) -> dict[str, Any]:
    values_by_agent: dict[str, list[float]] = defaultdict(list)
    files_by_agent: dict[str, set[str]] = defaultdict(set)
    overall_values: list[float] = []
    total_cache_hits = 0
    total_turns = 0
    instance_ids: list[str] = []

    for replay_path in replay_paths:
        replay = json.loads(replay_path.read_text())
        instance_ids.append(str(replay.get("instance_id", replay_path.stem)))
        turn_metrics = replay.get("turn_metrics", [])
        total_turns += len(turn_metrics)
        total_cache_hits += sum(1 for turn in turn_metrics if turn.get("cache_hit"))
        for turn in turn_metrics:
            if not is_executed_request(turn):
                continue
            agent = str(turn.get("agent", ""))
            value = request_metric_value(turn, request_metric)
            values_by_agent[agent].append(value)
            files_by_agent[agent].add(replay_path.name)
            overall_values.append(value)

    by_agent: list[dict[str, Any]] = []
    interactive_targets: dict[str, float] = {}
    batch_targets: dict[str, float] = {}
    for agent in sorted(values_by_agent):
        values = values_by_agent[agent]
        p50 = percentile(values, 50) if values else 0.0
        p95 = percentile(values, 95) if values else 0.0
        p99 = percentile(values, 99) if values else 0.0
        avg = (sum(values) / len(values)) if values else 0.0
        by_agent.append({
            "agent": agent,
            "num_executed_requests": len(values),
            "num_source_files": len(files_by_agent[agent]),
            "avg_latency_s": avg,
            "p50_latency_s": p50,
            "p95_latency_s": p95,
            "p99_latency_s": p99,
        })
        interactive_targets[agent] = p95
        batch_targets[agent] = p99

    overall = {
        "num_replay_files": len(replay_paths),
        "num_instances": len(instance_ids),
        "num_turns_total": total_turns,
        "num_executed_requests": len(overall_values),
        "num_cache_hits": total_cache_hits,
        "avg_latency_s": (sum(overall_values) / len(overall_values)) if overall_values else 0.0,
        "p50_latency_s": percentile(overall_values, 50) if overall_values else 0.0,
        "p95_latency_s": percentile(overall_values, 95) if overall_values else 0.0,
        "p99_latency_s": percentile(overall_values, 99) if overall_values else 0.0,
    }
    return {
        "request_metric": request_metric,
        "source_glob": None,
        "source_files": [str(path) for path in replay_paths],
        "overall": overall,
        "by_agent": by_agent,
        "recommended_targets": {
            "interactive_p95": interactive_targets,
            "batch_p99": batch_targets,
        },
    }


def print_report(report: dict[str, Any]) -> None:
    print(
        f"Empirical request SLOs from {report['overall']['num_replay_files']} replay files "
        f"using metric={report['request_metric']}"
    )
    print()
    rows: list[list[Any]] = []
    for entry in report["by_agent"]:
        rows.append([
            entry["agent"],
            entry["num_executed_requests"],
            entry["num_source_files"],
            f"{entry['avg_latency_s']:.3f}",
            f"{entry['p50_latency_s']:.3f}",
            f"{entry['p95_latency_s']:.3f}",
            f"{entry['p99_latency_s']:.3f}",
        ])
    print_table(
        rows,
        headers=["Agent", "ExecReqs", "Files", "avg", "p50", "p95", "p99"],
    )
    print()
    print("Recommended interactive targets (p95):")
    print(json.dumps(report["recommended_targets"]["interactive_p95"], indent=2))
    print()
    print("Recommended batch targets (p99):")
    print(json.dumps(report["recommended_targets"]["batch_p99"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Derive empirical per-agent request SLOs from replay JSON files"
    )
    parser.add_argument(
        "--glob",
        required=True,
        help="Glob for replay JSON files, for example 'replays-two/*.replay.json'",
    )
    parser.add_argument(
        "--request-metric",
        choices=["request_latency_s", "stage_latency_s"],
        default="request_latency_s",
        help="Metric to aggregate per executed request",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the full empirical SLO report as JSON",
    )
    parser.add_argument(
        "--interactive-targets-output",
        type=Path,
        default=None,
        help="Optional path to save the recommended interactive p95 target map as JSON",
    )
    parser.add_argument(
        "--batch-targets-output",
        type=Path,
        default=None,
        help="Optional path to save the recommended batch p99 target map as JSON",
    )
    args = parser.parse_args()

    replay_paths = discover_replay_paths(args.glob)
    report = build_empirical_agent_slos(
        replay_paths,
        request_metric=args.request_metric,
    )
    report["source_glob"] = args.glob
    print_report(report)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        print(f"Saved empirical SLO report to {args.output}")
    if args.interactive_targets_output is not None:
        args.interactive_targets_output.parent.mkdir(parents=True, exist_ok=True)
        args.interactive_targets_output.write_text(
            json.dumps(report["recommended_targets"]["interactive_p95"], indent=2))
        print(f"Saved interactive target map to {args.interactive_targets_output}")
    if args.batch_targets_output is not None:
        args.batch_targets_output.parent.mkdir(parents=True, exist_ok=True)
        args.batch_targets_output.write_text(
            json.dumps(report["recommended_targets"]["batch_p99"], indent=2))
        print(f"Saved batch target map to {args.batch_targets_output}")


if __name__ == "__main__":
    main()
