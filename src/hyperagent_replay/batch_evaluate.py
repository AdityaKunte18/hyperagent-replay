"""Batch-evaluate HyperAgent trajectory and replay outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from hyperagent_replay.evaluate import evaluate_paths, percentile

GENERATED_SUFFIXES = (
    ".eval.json",
    ".summary.json",
)


def discover_input_paths(inputs: list[Path], pattern: str,
                         recursive: bool) -> list[Path]:
    discovered: list[Path] = []
    for input_path in inputs:
        if input_path.is_file():
            discovered.append(input_path)
            continue
        if input_path.is_dir():
            iterator = (input_path.rglob(pattern)
                        if recursive else input_path.glob(pattern))
            discovered.extend(path for path in iterator if path.is_file())
            continue
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in sorted(path.resolve() for path in discovered):
        if path in seen:
            continue
        if any(str(path).endswith(suffix) for suffix in GENERATED_SUFFIXES):
            continue
        seen.add(path)
        unique.append(path)
    return unique


def select_input_paths(paths: list[Path], offset: int,
                       limit: int | None) -> list[Path]:
    if offset < 0:
        raise ValueError("--offset must be >= 0")
    if limit is not None and limit < 0:
        raise ValueError("--limit must be >= 0")

    if limit is None:
        return paths[offset:]
    return paths[offset:offset + limit]


def strip_known_json_suffixes(name: str) -> str:
    for suffix in (".replay.json", ".extracted.json", ".json"):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name


def output_name_for(path: Path) -> str:
    return f"{strip_known_json_suffixes(path.name)}.eval.json"


def replay_name_for(path: Path) -> str:
    return f"{strip_known_json_suffixes(path.name)}.replay.json"


def source_metrics_aggregate(
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    source_metrics = [summary.get("source_metrics", {}) for summary in summaries]
    response_counts: Counter[str] = Counter()
    action_languages: Counter[str] = Counter()
    top_tool_names: Counter[str] = Counter()

    for metrics in source_metrics:
        response_counts.update(metrics.get("response_counts", {}))
        action_languages.update(metrics.get("action_languages", {}))
        top_tool_names.update(metrics.get("top_tool_names", {}))

    def avg(field: str) -> float:
        values = [metrics.get(field, 0.0) for metrics in source_metrics]
        return sum(values) / len(values) if values else 0.0

    return {
        "num_instances": len(source_metrics),
        "total_num_llm_turns": sum(
            metrics.get("num_llm_turns", 0) for metrics in source_metrics),
        "total_num_turns_with_actions": sum(
            metrics.get("num_turns_with_actions", 0)
            for metrics in source_metrics),
        "total_num_final_answer_turns": sum(
            metrics.get("num_final_answer_turns", 0)
            for metrics in source_metrics),
        "avg_num_llm_turns": avg("num_llm_turns"),
        "avg_num_turns_with_actions": avg("num_turns_with_actions"),
        "avg_num_final_answer_turns": avg("num_final_answer_turns"),
        "avg_response_chars": avg("avg_response_chars"),
        "median_response_chars": avg("median_response_chars"),
        "max_response_chars": max(
            (metrics.get("max_response_chars", 0) for metrics in source_metrics),
            default=0,
        ),
        "response_counts": dict(response_counts),
        "action_languages": dict(action_languages),
        "top_tool_names": dict(top_tool_names.most_common(25)),
    }


def replay_metrics_aggregate(
    summaries: list[dict[str, Any]],
) -> dict[str, Any] | None:
    replay_metrics = [
        summary["replay_metrics"] for summary in summaries
        if "replay_metrics" in summary
    ]
    if not replay_metrics:
        return None

    num_turns = sum(metrics.get("num_replayed_turns", 0)
                    for metrics in replay_metrics)
    wall_times = [metrics.get("wall_solve_time_s", 0.0)
                  for metrics in replay_metrics]
    lm_only_times = [metrics.get("lm_only_solve_time_s", 0.0)
                     for metrics in replay_metrics]
    tool_sleep_times = [metrics.get("synthetic_tool_sleep_s", 0.0)
                        for metrics in replay_metrics]
    instance_avg_latencies = [metrics.get("avg_request_latency_s", 0.0)
                              for metrics in replay_metrics]

    total_prompt_tokens = sum(metrics.get("total_prompt_tokens", 0)
                              for metrics in replay_metrics)
    total_completion_tokens = sum(metrics.get("total_completion_tokens", 0)
                                  for metrics in replay_metrics)

    weighted_avg_request_latency = (
        sum(metrics.get("avg_request_latency_s", 0.0) *
            metrics.get("num_replayed_turns", 0) for metrics in replay_metrics) /
        num_turns if num_turns else 0.0)

    weighted_avg_prompt_chars = (
        sum(metrics.get("avg_prompt_chars_per_turn", 0.0) *
            metrics.get("num_replayed_turns", 0) for metrics in replay_metrics) /
        num_turns if num_turns else 0.0)

    weighted_avg_response_chars = (
        sum(metrics.get("avg_response_chars_per_turn", 0.0) *
            metrics.get("num_replayed_turns", 0) for metrics in replay_metrics) /
        num_turns if num_turns else 0.0)

    return {
        "num_instances_with_replay": len(replay_metrics),
        "total_num_replayed_turns": num_turns,
        "total_wall_solve_time_s": sum(wall_times),
        "avg_wall_solve_time_s": (sum(wall_times) / len(wall_times)
                                   if wall_times else 0.0),
        "median_wall_solve_time_s": percentile(wall_times, 50),
        "p95_wall_solve_time_s": percentile(wall_times, 95),
        "p99_wall_solve_time_s": percentile(wall_times, 99),
        "total_lm_only_solve_time_s": sum(lm_only_times),
        "total_synthetic_tool_sleep_s": sum(tool_sleep_times),
        "weighted_avg_request_latency_s": weighted_avg_request_latency,
        "median_instance_avg_request_latency_s":
        percentile(instance_avg_latencies, 50),
        "p95_instance_avg_request_latency_s":
        percentile(instance_avg_latencies, 95),
        "p99_instance_avg_request_latency_s":
        percentile(instance_avg_latencies, 99),
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "avg_prompt_tokens_per_turn":
        (total_prompt_tokens / num_turns if num_turns else 0.0),
        "avg_completion_tokens_per_turn":
        (total_completion_tokens / num_turns if num_turns else 0.0),
        "avg_prompt_chars_per_turn": weighted_avg_prompt_chars,
        "avg_response_chars_per_turn": weighted_avg_response_chars,
    }


def aggregate_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {
        "source_metrics": source_metrics_aggregate(summaries),
    }
    replay_metrics = replay_metrics_aggregate(summaries)
    if replay_metrics is not None:
        aggregate["replay_metrics"] = replay_metrics
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-evaluate HyperAgent trajectory and replay outputs")
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="One or more raw, extracted, or replay JSON files or directories",
    )
    parser.add_argument(
        "--pattern",
        default="*.json",
        help="Glob to use when an input is a directory",
    )
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="Do not recurse into subdirectories when discovering inputs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where evaluation JSON files will be written",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional path for the batch evaluation manifest JSON",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip outputs that already exist",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Skip the first N inputs after deterministic sorting",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate at most N inputs after applying --offset",
    )
    parser.add_argument(
        "--replay-dir",
        type=Path,
        default=None,
        help="Optional directory of replay JSON files matched by basename",
    )
    args = parser.parse_args()

    all_input_paths = discover_input_paths(
        args.inputs,
        pattern=args.pattern,
        recursive=not args.non_recursive,
    )
    input_paths = select_input_paths(
        all_input_paths,
        offset=args.offset,
        limit=args.limit,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    successful_summaries: list[dict[str, Any]] = []

    for selection_index, input_path in enumerate(input_paths, start=args.offset):
        output_path = args.output_dir / output_name_for(input_path)
        replay_path: Path | None = None
        if args.replay_dir is not None and not input_path.name.endswith(".replay.json"):
            replay_path = args.replay_dir / replay_name_for(input_path)

        if args.skip_existing and output_path.exists():
            results.append({
                "selection_index": selection_index,
                "input_path": str(input_path),
                "replay_path": (str(replay_path) if replay_path is not None else
                                None),
                "output_path": str(output_path),
                "status": "skipped_existing",
            })
            continue

        try:
            if replay_path is not None and not replay_path.exists():
                raise FileNotFoundError(f"Replay file does not exist: {replay_path}")

            summary = evaluate_paths(input_path, replay_path=replay_path)
            output_path.write_text(json.dumps(summary, indent=2))
            successful_summaries.append(summary)
            results.append({
                "selection_index": selection_index,
                "input_path": str(input_path),
                "replay_path": (str(replay_path) if replay_path is not None else
                                None),
                "output_path": str(output_path),
                "status": "ok",
                "instance_id": summary.get("instance_id"),
                "has_replay_metrics": "replay_metrics" in summary,
            })
            print(f"[ok] {input_path} -> {output_path}")
        except Exception as exc:
            results.append({
                "selection_index": selection_index,
                "input_path": str(input_path),
                "replay_path": (str(replay_path) if replay_path is not None else
                                None),
                "output_path": str(output_path),
                "status": "error",
                "error": str(exc),
            })
            print(f"[error] {input_path}: {exc}")

    manifest = {
        "phase": "eval",
        "output_dir": str(args.output_dir.resolve()),
        "ordering": "lexicographic absolute path",
        "offset": args.offset,
        "limit": args.limit,
        "replay_dir": (str(args.replay_dir.resolve())
                        if args.replay_dir is not None else None),
        "num_discovered_inputs": len(all_input_paths),
        "num_inputs": len(input_paths),
        "num_ok": sum(item["status"] == "ok" for item in results),
        "num_skipped_existing": sum(item["status"] == "skipped_existing"
                                     for item in results),
        "num_error": sum(item["status"] == "error" for item in results),
        "aggregate_summary": aggregate_summaries(successful_summaries),
        "results": results,
    }

    manifest_path = args.manifest
    if manifest_path is None:
        manifest_path = args.output_dir / "eval_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Saved evaluation manifest to {manifest_path}")


if __name__ == "__main__":
    main()
