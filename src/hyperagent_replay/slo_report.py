"""Analyze replay JSON files against request and trajectory SLO targets."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from hyperagent_replay.replay import percentile


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


def request_metric_value(turn: dict[str, Any], metric: str) -> float:
    request_latency_s = float(turn.get("request_latency_s") or 0.0)
    if metric == "request_latency_s":
        return request_latency_s
    if metric == "stage_latency_s":
        return request_latency_s + float(turn.get("synthetic_tool_sleep_s") or 0.0)
    raise ValueError(f"Unsupported request metric: {metric}")


def is_executed_request(turn: dict[str, Any]) -> bool:
    executed_on_vllm = turn.get("executed_on_vllm")
    if executed_on_vllm is not None:
        return bool(executed_on_vllm)
    return turn.get("request_arrival_time") is not None


def cache_hit_count(turn_metrics: list[dict[str, Any]], agent: str | None = None) -> int:
    return sum(
        1
        for turn in turn_metrics
        if turn.get("cache_hit")
        and (agent is None or turn.get("agent") == agent)
    )


def service_span_s(turn_metrics: list[dict[str, Any]]) -> float:
    arrivals = [
        float(turn["request_arrival_time"])
        for turn in turn_metrics
        if turn.get("request_arrival_time") is not None
    ]
    departures = [
        float(turn["request_departure_time"])
        for turn in turn_metrics
        if turn.get("request_departure_time") is not None
    ]
    if not arrivals or not departures:
        return 0.0
    return max(departures) - min(arrivals)


def target_attainment(values: list[float], target_s: float | None) -> tuple[int | None, float | None]:
    if target_s is None:
        return None, None
    met_target = sum(1 for value in values if value <= target_s)
    pct_met_target = (met_target / len(values) * 100.0) if values else 0.0
    return met_target, pct_met_target


def build_request_summary(
    *,
    turn_metrics: list[dict[str, Any]],
    request_metric: str,
    default_target_s: float | None,
    agent_targets: dict[str, float],
) -> dict[str, Any]:
    executed_turns = [turn for turn in turn_metrics if is_executed_request(turn)]
    values_by_agent: dict[str, list[float]] = defaultdict(list)
    total_turns_by_agent: dict[str, int] = defaultdict(int)

    for turn in turn_metrics:
        total_turns_by_agent[str(turn.get("agent", ""))] += 1
    for turn in executed_turns:
        values_by_agent[str(turn.get("agent", ""))].append(
            request_metric_value(turn, request_metric))

    by_agent: list[dict[str, Any]] = []
    for agent in sorted(total_turns_by_agent):
        agent_values = values_by_agent.get(agent, [])
        target_s = agent_targets.get(agent, default_target_s)
        num_met_target, pct_met_target = target_attainment(agent_values, target_s)
        by_agent.append({
            "agent": agent,
            "num_turns_total": total_turns_by_agent[agent],
            "num_executed_requests": len(agent_values),
            "num_cache_hits": cache_hit_count(turn_metrics, agent=agent),
            "target_s": target_s,
            "num_met_target": num_met_target,
            "pct_met_target": pct_met_target,
            "avg_latency_s": (sum(agent_values) / len(agent_values)) if agent_values else 0.0,
            "p50_latency_s": percentile(agent_values, 50) if agent_values else 0.0,
            "p95_latency_s": percentile(agent_values, 95) if agent_values else 0.0,
            "p99_latency_s": percentile(agent_values, 99) if agent_values else 0.0,
        })

    overall_values = [request_metric_value(turn, request_metric) for turn in executed_turns]
    overall_met_target, overall_pct_met_target = target_attainment(
        overall_values,
        default_target_s,
    )
    overall = {
        "num_turns_total": len(turn_metrics),
        "num_executed_requests": len(executed_turns),
        "num_cache_hits": cache_hit_count(turn_metrics),
        "target_s": default_target_s,
        "num_met_target": overall_met_target,
        "pct_met_target": overall_pct_met_target,
        "avg_latency_s": (sum(overall_values) / len(overall_values)) if overall_values else 0.0,
        "p50_latency_s": percentile(overall_values, 50) if overall_values else 0.0,
        "p95_latency_s": percentile(overall_values, 95) if overall_values else 0.0,
        "p99_latency_s": percentile(overall_values, 99) if overall_values else 0.0,
    }
    return {
        "metric": request_metric,
        "default_target_s": default_target_s,
        "agent_targets": agent_targets,
        "overall": overall,
        "by_agent": by_agent,
    }


def build_episode_summary(
    *,
    replay: dict[str, Any],
    episode_metric: str,
    episode_target_s: float | None,
) -> dict[str, Any]:
    turn_metrics = replay.get("turn_metrics", [])
    timing = replay.get("timing", {})
    wall_solve_time_s = float(timing.get("wall_solve_time_s") or 0.0)
    selected_value_s = (
        service_span_s(turn_metrics)
        if episode_metric == "service_span_s"
        else wall_solve_time_s
    )
    met_target = (
        selected_value_s <= episode_target_s
        if episode_target_s is not None
        else None
    )
    return {
        "metric": episode_metric,
        "target_s": episode_target_s,
        "met_target": met_target,
        "selected_value_s": selected_value_s,
        "wall_solve_time_s": wall_solve_time_s,
        "service_span_s": service_span_s(turn_metrics),
    }


def build_slo_report(
    replay: dict[str, Any],
    *,
    request_metric: str,
    default_request_target_s: float | None,
    agent_request_targets: dict[str, float],
    episode_metric: str,
    episode_target_s: float | None,
) -> dict[str, Any]:
    return {
        "instance_id": replay.get("instance_id"),
        "request_slo": build_request_summary(
            turn_metrics=replay.get("turn_metrics", []),
            request_metric=request_metric,
            default_target_s=default_request_target_s,
            agent_targets=agent_request_targets,
        ),
        "episode_slo": build_episode_summary(
            replay=replay,
            episode_metric=episode_metric,
            episode_target_s=episode_target_s,
        ),
    }


def load_agent_targets(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("--agent-request-targets must point to a JSON object")
    return {str(key): float(value) for key, value in payload.items()}


def print_report(report: dict[str, Any]) -> None:
    print(f"Instance: {report['instance_id']}")
    request_slo = report["request_slo"]
    print(
        f"Request SLO metric: {request_slo['metric']} "
        f"(default_target_s={request_slo['default_target_s']})"
    )
    rows: list[list[Any]] = []
    for entry in request_slo["by_agent"]:
        rows.append([
            entry["agent"],
            entry["num_turns_total"],
            entry["num_executed_requests"],
            entry["num_cache_hits"],
            "" if entry["target_s"] is None else f"{entry['target_s']:.2f}",
            "" if entry["num_met_target"] is None else entry["num_met_target"],
            "" if entry["pct_met_target"] is None else f"{entry['pct_met_target']:.1f}",
            f"{entry['p50_latency_s']:.3f}",
            f"{entry['p95_latency_s']:.3f}",
            f"{entry['p99_latency_s']:.3f}",
        ])
    print()
    print_table(
        rows,
        headers=[
            "Agent",
            "Turns",
            "ExecReqs",
            "CacheHits",
            "TargetS",
            "Met",
            "MetPct",
            "p50",
            "p95",
            "p99",
        ],
    )
    overall = request_slo["overall"]
    print()
    print(
        "Overall request attainment: "
        f"exec_reqs={overall['num_executed_requests']} "
        f"cache_hits={overall['num_cache_hits']} "
        f"target_s={overall['target_s']} "
        f"met={overall['num_met_target']} "
        f"met_pct={overall['pct_met_target']}"
    )
    episode = report["episode_slo"]
    print(
        "Episode SLO: "
        f"metric={episode['metric']} "
        f"value_s={episode['selected_value_s']:.3f} "
        f"target_s={episode['target_s']} "
        f"met_target={episode['met_target']} "
        f"wall_solve_time_s={episode['wall_solve_time_s']:.3f} "
        f"service_span_s={episode['service_span_s']:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report per-agent request SLO attainment and episode SLO attainment from replay JSON"
    )
    parser.add_argument("input", type=Path, help="Replay JSON file")
    parser.add_argument("--output", type=Path, default=None, help="Path to save JSON report")
    parser.add_argument(
        "--request-metric",
        choices=["request_latency_s", "stage_latency_s"],
        default="request_latency_s",
        help="Per-request metric used for SLO attainment",
    )
    parser.add_argument(
        "--request-target-s",
        type=float,
        default=None,
        help="Default request SLO target in seconds",
    )
    parser.add_argument(
        "--agent-request-targets",
        type=Path,
        default=None,
        help="JSON file mapping agent name to request SLO target in seconds",
    )
    parser.add_argument(
        "--episode-metric",
        choices=["wall_solve_time_s", "service_span_s"],
        default="wall_solve_time_s",
        help="Trajectory-level metric used for episode SLO attainment",
    )
    parser.add_argument(
        "--episode-target-s",
        type=float,
        default=None,
        help="Episode-level SLO target in seconds",
    )
    args = parser.parse_args()

    if args.request_target_s is not None and args.request_target_s <= 0:
        raise ValueError("--request-target-s must be > 0")
    if args.episode_target_s is not None and args.episode_target_s <= 0:
        raise ValueError("--episode-target-s must be > 0")

    replay = json.loads(args.input.read_text())
    agent_request_targets = load_agent_targets(args.agent_request_targets)
    report = build_slo_report(
        replay,
        request_metric=args.request_metric,
        default_request_target_s=args.request_target_s,
        agent_request_targets=agent_request_targets,
        episode_metric=args.episode_metric,
        episode_target_s=args.episode_target_s,
    )
    print_report(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        print(f"Saved SLO report to {args.output}")


if __name__ == "__main__":
    main()
