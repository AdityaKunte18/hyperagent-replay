"""Analyze baseline and reuse replay outputs and write comparison artifacts."""

from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import defaultdict
from html import escape
from pathlib import Path
from typing import Any

from hyperagent_replay.resource_groups import turn_tool_signature
from hyperagent_replay.slo_report import is_executed_request

BASELINE_REPLAY_SUFFIX = ".baseline.replay.json"
REUSE_REPLAY_SUFFIX = ".reuse.replay.json"
GENERIC_REPLAY_SUFFIX = ".replay.json"


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


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def discover_replay_paths(pattern: str, *, suffix: str) -> list[Path]:
    matches = sorted(Path(path) for path in glob.glob(pattern, recursive=True))
    replay_paths = [path for path in matches if path.name.endswith(suffix)]
    if not replay_paths:
        raise SystemExit(f"No replay JSON files ending with {suffix} matched {pattern}")
    return replay_paths


def replay_key_for_path(path: Path) -> str:
    for suffix in (BASELINE_REPLAY_SUFFIX, REUSE_REPLAY_SUFFIX, GENERIC_REPLAY_SUFFIX):
        if path.name.endswith(suffix):
            return path.name[:-len(suffix)]
    return path.stem


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def format_float(value: float) -> str:
    return f"{value:.3f}"


def int_or_zero(value: Any) -> int:
    if value is None:
        return 0
    return int(value)


def turn_token_counts(turn: dict[str, Any]) -> tuple[int, int, int]:
    prompt_tokens = int_or_zero(turn.get("prompt_tokens"))
    completion_tokens = int_or_zero(turn.get("completion_tokens"))
    total_tokens = int_or_zero(turn.get("total_tokens"))
    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens
    return prompt_tokens, completion_tokens, total_tokens


def source_turns_by_turn_index(replay: dict[str, Any]) -> dict[int, dict[str, Any]]:
    source_by_turn_index: dict[int, dict[str, Any]] = {}
    for turn in replay.get("turn_metrics", []):
        if not is_executed_request(turn):
            continue
        turn_index = turn.get("turn_index")
        if turn_index is None:
            continue
        source_by_turn_index[int(turn_index)] = turn
    return source_by_turn_index


def estimate_cache_hit_savings(
    turn: dict[str, Any],
    source_by_turn_index: dict[int, dict[str, Any]],
) -> dict[str, float | int]:
    if not turn.get("cache_hit"):
        return {
            "saved_lm_time_s": 0.0,
            "saved_stage_time_s": 0.0,
            "saved_prompt_tokens": 0,
            "saved_completion_tokens": 0,
            "saved_total_tokens": 0,
        }

    source_turn_index = turn.get("cache_source_turn_index")
    source_turn = None
    if source_turn_index is not None:
        source_turn = source_by_turn_index.get(int(source_turn_index))

    saved_prompt_tokens = (
        int_or_zero(source_turn.get("prompt_tokens")) if source_turn is not None else
        int_or_zero(turn.get("cache_source_prompt_tokens"))
    )
    saved_completion_tokens = (
        int_or_zero(source_turn.get("completion_tokens")) if source_turn is not None else
        int_or_zero(turn.get("cache_source_completion_tokens"))
    )
    saved_total_tokens = saved_prompt_tokens + saved_completion_tokens
    saved_lm_time_s = (
        float(source_turn.get("request_latency_s") or 0.0)
        if source_turn is not None else 0.0
    )
    saved_stage_time_s = (
        saved_lm_time_s + float(source_turn.get("synthetic_tool_sleep_s") or 0.0)
        if source_turn is not None else saved_lm_time_s
    )
    return {
        "saved_lm_time_s": saved_lm_time_s,
        "saved_stage_time_s": saved_stage_time_s,
        "saved_prompt_tokens": saved_prompt_tokens,
        "saved_completion_tokens": saved_completion_tokens,
        "saved_total_tokens": saved_total_tokens,
    }


def tool_signature_for_turn(turn: dict[str, Any]) -> str:
    resource_group = turn.get("resource_group")
    if isinstance(resource_group, dict):
        tool_signature = resource_group.get("tool_signature")
        if tool_signature:
            return str(tool_signature)

    action = turn.get("recorded_action")
    if action is None:
        action = turn.get("action")
    if action is not None:
        return turn_tool_signature({"action": action})
    return "LLM_ONLY"


def total_tokens_from_timing(replay: dict[str, Any]) -> int:
    timing = replay.get("timing", {})
    return int(timing.get("total_prompt_tokens", 0)) + int(
        timing.get("total_completion_tokens", 0))


def executed_request_count(replay: dict[str, Any]) -> int:
    timing = replay.get("timing", {})
    explicit = timing.get("num_vllm_requests_executed")
    if explicit is not None:
        return int(explicit)
    return sum(1 for turn in replay.get("turn_metrics", []) if is_executed_request(turn))


def cache_hit_count(replay: dict[str, Any]) -> int:
    timing = replay.get("timing", {})
    explicit = timing.get("num_cache_hits")
    if explicit is not None:
        return int(explicit)
    return sum(1 for turn in replay.get("turn_metrics", []) if turn.get("cache_hit"))


def build_pairwise_row(
    key: str,
    baseline_path: Path,
    reuse_path: Path,
    baseline_replay: dict[str, Any],
    reuse_replay: dict[str, Any],
) -> dict[str, Any]:
    baseline_timing = baseline_replay.get("timing", {})
    reuse_timing = reuse_replay.get("timing", {})
    baseline_turns = int(baseline_timing.get("num_replayed_turns",
                                             len(baseline_replay.get("turn_metrics", []))))
    reuse_turns = int(reuse_timing.get("num_replayed_turns",
                                       len(reuse_replay.get("turn_metrics", []))))
    baseline_requests = executed_request_count(baseline_replay)
    reuse_requests = executed_request_count(reuse_replay)
    avoided_requests = cache_hit_count(reuse_replay)
    baseline_wall_s = float(baseline_timing.get("wall_solve_time_s", 0.0))
    reuse_wall_s = float(reuse_timing.get("wall_solve_time_s", 0.0))
    baseline_lm_s = float(baseline_timing.get("lm_only_solve_time_s", 0.0))
    reuse_lm_s = float(reuse_timing.get("lm_only_solve_time_s", 0.0))
    baseline_tokens = total_tokens_from_timing(baseline_replay)
    reuse_tokens = total_tokens_from_timing(reuse_replay)
    baseline_prompt_tokens = int(baseline_timing.get("total_prompt_tokens", 0))
    reuse_prompt_tokens = int(reuse_timing.get("total_prompt_tokens", 0))
    baseline_completion_tokens = int(baseline_timing.get("total_completion_tokens", 0))
    reuse_completion_tokens = int(reuse_timing.get("total_completion_tokens", 0))

    return {
        "instance_id": str(
            reuse_replay.get("instance_id", baseline_replay.get("instance_id", key))),
        "match_key": key,
        "baseline_path": str(baseline_path),
        "reuse_path": str(reuse_path),
        "baseline_num_turns": baseline_turns,
        "reuse_num_turns": reuse_turns,
        "baseline_vllm_requests": baseline_requests,
        "reuse_vllm_requests_executed": reuse_requests,
        "reuse_vllm_requests_avoided": avoided_requests,
        "reuse_cache_hit_pct": safe_div(avoided_requests,
                                         avoided_requests + reuse_requests) * 100.0,
        "baseline_wall_solve_time_s": baseline_wall_s,
        "reuse_wall_solve_time_s": reuse_wall_s,
        "wall_time_delta_s": reuse_wall_s - baseline_wall_s,
        "wall_time_delta_pct": safe_div(reuse_wall_s - baseline_wall_s,
                                         baseline_wall_s) * 100.0,
        "baseline_lm_only_solve_time_s": baseline_lm_s,
        "reuse_lm_only_solve_time_s": reuse_lm_s,
        "lm_only_time_delta_s": reuse_lm_s - baseline_lm_s,
        "lm_only_time_delta_pct": safe_div(reuse_lm_s - baseline_lm_s,
                                            baseline_lm_s) * 100.0,
        "baseline_total_prompt_tokens": baseline_prompt_tokens,
        "reuse_total_prompt_tokens": reuse_prompt_tokens,
        "baseline_total_completion_tokens": baseline_completion_tokens,
        "reuse_total_completion_tokens": reuse_completion_tokens,
        "baseline_wall_tokens_per_s": safe_div(baseline_tokens, baseline_wall_s),
        "reuse_wall_tokens_per_s": safe_div(reuse_tokens, reuse_wall_s),
        "baseline_lm_tokens_per_s": safe_div(baseline_tokens, baseline_lm_s),
        "reuse_lm_tokens_per_s": safe_div(reuse_tokens, reuse_lm_s),
        "baseline_avg_request_latency_s": safe_div(baseline_lm_s, baseline_requests),
        "reuse_avg_request_latency_s": safe_div(reuse_lm_s, reuse_requests),
        "baseline_avg_prompt_tokens_per_request": safe_div(
            baseline_prompt_tokens, baseline_requests),
        "reuse_avg_prompt_tokens_per_executed_request": safe_div(
            reuse_prompt_tokens, reuse_requests),
        "baseline_avg_completion_tokens_per_request": safe_div(
            baseline_completion_tokens, baseline_requests),
        "reuse_avg_completion_tokens_per_executed_request": safe_div(
            reuse_completion_tokens, reuse_requests),
    }


def build_pairwise_summary(
    baseline_paths: list[Path],
    reuse_paths: list[Path],
) -> dict[str, Any]:
    baseline_by_key = {replay_key_for_path(path): path for path in baseline_paths}
    reuse_by_key = {replay_key_for_path(path): path for path in reuse_paths}
    matched_keys = sorted(set(baseline_by_key) & set(reuse_by_key))
    unmatched_baseline = sorted(set(baseline_by_key) - set(reuse_by_key))
    unmatched_reuse = sorted(set(reuse_by_key) - set(baseline_by_key))

    rows: list[dict[str, Any]] = []
    for key in matched_keys:
        rows.append(
            build_pairwise_row(
                key,
                baseline_by_key[key],
                reuse_by_key[key],
                load_json(baseline_by_key[key]),
                load_json(reuse_by_key[key]),
            ))

    total_baseline_wall_s = sum(row["baseline_wall_solve_time_s"] for row in rows)
    total_reuse_wall_s = sum(row["reuse_wall_solve_time_s"] for row in rows)
    total_baseline_lm_s = sum(row["baseline_lm_only_solve_time_s"] for row in rows)
    total_reuse_lm_s = sum(row["reuse_lm_only_solve_time_s"] for row in rows)
    total_baseline_requests = sum(row["baseline_vllm_requests"] for row in rows)
    total_reuse_requests = sum(row["reuse_vllm_requests_executed"] for row in rows)
    total_avoided_requests = sum(row["reuse_vllm_requests_avoided"] for row in rows)
    total_baseline_tokens = sum(
        row["baseline_total_prompt_tokens"] + row["baseline_total_completion_tokens"]
        for row in rows)
    total_reuse_tokens = sum(
        row["reuse_total_prompt_tokens"] + row["reuse_total_completion_tokens"]
        for row in rows)

    overall = {
        "num_matched_pairs": len(rows),
        "num_pairs_with_wall_time_improvement": sum(
            1 for row in rows if row["wall_time_delta_s"] < 0.0),
        "total_baseline_wall_solve_time_s": total_baseline_wall_s,
        "total_reuse_wall_solve_time_s": total_reuse_wall_s,
        "wall_time_delta_s": total_reuse_wall_s - total_baseline_wall_s,
        "wall_time_delta_pct": safe_div(total_reuse_wall_s - total_baseline_wall_s,
                                         total_baseline_wall_s) * 100.0,
        "total_baseline_lm_only_solve_time_s": total_baseline_lm_s,
        "total_reuse_lm_only_solve_time_s": total_reuse_lm_s,
        "lm_only_time_delta_s": total_reuse_lm_s - total_baseline_lm_s,
        "lm_only_time_delta_pct": safe_div(total_reuse_lm_s - total_baseline_lm_s,
                                            total_baseline_lm_s) * 100.0,
        "total_baseline_vllm_requests": total_baseline_requests,
        "total_reuse_vllm_requests_executed": total_reuse_requests,
        "total_reuse_vllm_requests_avoided": total_avoided_requests,
        "reuse_cache_hit_pct": safe_div(total_avoided_requests,
                                         total_avoided_requests + total_reuse_requests)
        * 100.0,
        "baseline_wall_tokens_per_s": safe_div(total_baseline_tokens,
                                                total_baseline_wall_s),
        "reuse_wall_tokens_per_s": safe_div(total_reuse_tokens, total_reuse_wall_s),
        "baseline_lm_tokens_per_s": safe_div(total_baseline_tokens, total_baseline_lm_s),
        "reuse_lm_tokens_per_s": safe_div(total_reuse_tokens, total_reuse_lm_s),
        "baseline_avg_request_latency_s": safe_div(total_baseline_lm_s,
                                                    total_baseline_requests),
        "reuse_avg_request_latency_s": safe_div(total_reuse_lm_s, total_reuse_requests),
    }

    return {
        "matched_pair_keys": matched_keys,
        "unmatched_baseline_keys": unmatched_baseline,
        "unmatched_reuse_keys": unmatched_reuse,
        "overall": overall,
        "rows": rows,
    }


def build_cache_hits_by_agent(reuse_replays: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stats_by_agent: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "turns_total": 0,
            "cache_hits": 0,
            "executed_requests": 0,
            "total_request_latency_s": 0.0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "estimated_saved_lm_time_s": 0.0,
            "estimated_saved_stage_time_s": 0.0,
            "estimated_saved_prompt_tokens": 0,
            "estimated_saved_completion_tokens": 0,
            "estimated_saved_total_tokens": 0,
            "instances": set(),
        })
    for replay in reuse_replays:
        instance_id = str(replay.get("instance_id", ""))
        source_by_turn_index = source_turns_by_turn_index(replay)
        for turn in replay.get("turn_metrics", []):
            agent = str(turn.get("agent", ""))
            stats = stats_by_agent[agent]
            stats["turns_total"] += 1
            stats["cache_hits"] += int(bool(turn.get("cache_hit")))
            if is_executed_request(turn):
                stats["executed_requests"] += 1
                stats["total_request_latency_s"] += float(
                    turn.get("request_latency_s") or 0.0)
                prompt_tokens, completion_tokens, total_tokens = turn_token_counts(turn)
                stats["total_prompt_tokens"] += prompt_tokens
                stats["total_completion_tokens"] += completion_tokens
                stats["total_tokens"] += total_tokens
            savings = estimate_cache_hit_savings(turn, source_by_turn_index)
            stats["estimated_saved_lm_time_s"] += float(savings["saved_lm_time_s"])
            stats["estimated_saved_stage_time_s"] += float(savings["saved_stage_time_s"])
            stats["estimated_saved_prompt_tokens"] += int(savings["saved_prompt_tokens"])
            stats["estimated_saved_completion_tokens"] += int(
                savings["saved_completion_tokens"])
            stats["estimated_saved_total_tokens"] += int(savings["saved_total_tokens"])
            stats["instances"].add(instance_id)

    rows: list[dict[str, Any]] = []
    for agent, stats in sorted(
            stats_by_agent.items(),
            key=lambda item: (-item[1]["cache_hits"], -item[1]["turns_total"], item[0])):
        rows.append({
            "agent": agent,
            "turns_total": stats["turns_total"],
            "cache_hits": stats["cache_hits"],
            "cache_hit_pct": safe_div(stats["cache_hits"], stats["turns_total"]) * 100.0,
            "executed_requests": stats["executed_requests"],
            "avg_executed_request_latency_s": safe_div(
                stats["total_request_latency_s"], stats["executed_requests"]),
            "total_prompt_tokens": stats["total_prompt_tokens"],
            "total_completion_tokens": stats["total_completion_tokens"],
            "total_tokens": stats["total_tokens"],
            "lm_tokens_per_s": safe_div(stats["total_tokens"],
                                         stats["total_request_latency_s"]),
            "avg_prompt_tokens_per_executed_request": safe_div(
                stats["total_prompt_tokens"], stats["executed_requests"]),
            "avg_completion_tokens_per_executed_request": safe_div(
                stats["total_completion_tokens"], stats["executed_requests"]),
            "estimated_saved_lm_time_s": stats["estimated_saved_lm_time_s"],
            "estimated_saved_stage_time_s": stats["estimated_saved_stage_time_s"],
            "estimated_saved_prompt_tokens": stats["estimated_saved_prompt_tokens"],
            "estimated_saved_completion_tokens": stats[
                "estimated_saved_completion_tokens"],
            "estimated_saved_total_tokens": stats["estimated_saved_total_tokens"],
            "num_instances": len(stats["instances"]),
        })
    return rows


def build_cache_hits_by_tool_signature(
    reuse_replays: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    stats_by_tool: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "turns_total": 0,
            "cache_hits": 0,
            "executed_requests": 0,
            "total_request_latency_s": 0.0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "estimated_saved_lm_time_s": 0.0,
            "estimated_saved_stage_time_s": 0.0,
            "estimated_saved_prompt_tokens": 0,
            "estimated_saved_completion_tokens": 0,
            "estimated_saved_total_tokens": 0,
            "instances": set(),
            "agents": set(),
        })
    for replay in reuse_replays:
        instance_id = str(replay.get("instance_id", ""))
        source_by_turn_index = source_turns_by_turn_index(replay)
        for turn in replay.get("turn_metrics", []):
            tool_signature = tool_signature_for_turn(turn)
            agent = str(turn.get("agent", ""))
            stats = stats_by_tool[tool_signature]
            stats["turns_total"] += 1
            stats["cache_hits"] += int(bool(turn.get("cache_hit")))
            if is_executed_request(turn):
                stats["executed_requests"] += 1
                stats["total_request_latency_s"] += float(
                    turn.get("request_latency_s") or 0.0)
                prompt_tokens, completion_tokens, total_tokens = turn_token_counts(turn)
                stats["total_prompt_tokens"] += prompt_tokens
                stats["total_completion_tokens"] += completion_tokens
                stats["total_tokens"] += total_tokens
            savings = estimate_cache_hit_savings(turn, source_by_turn_index)
            stats["estimated_saved_lm_time_s"] += float(savings["saved_lm_time_s"])
            stats["estimated_saved_stage_time_s"] += float(savings["saved_stage_time_s"])
            stats["estimated_saved_prompt_tokens"] += int(savings["saved_prompt_tokens"])
            stats["estimated_saved_completion_tokens"] += int(
                savings["saved_completion_tokens"])
            stats["estimated_saved_total_tokens"] += int(savings["saved_total_tokens"])
            stats["instances"].add(instance_id)
            stats["agents"].add(agent)

    rows: list[dict[str, Any]] = []
    for tool_signature, stats in sorted(
            stats_by_tool.items(),
            key=lambda item: (-item[1]["cache_hits"], -item[1]["turns_total"], item[0])):
        rows.append({
            "tool_signature": tool_signature,
            "turns_total": stats["turns_total"],
            "cache_hits": stats["cache_hits"],
            "cache_hit_pct": safe_div(stats["cache_hits"], stats["turns_total"]) * 100.0,
            "executed_requests": stats["executed_requests"],
            "avg_executed_request_latency_s": safe_div(
                stats["total_request_latency_s"], stats["executed_requests"]),
            "total_prompt_tokens": stats["total_prompt_tokens"],
            "total_completion_tokens": stats["total_completion_tokens"],
            "total_tokens": stats["total_tokens"],
            "lm_tokens_per_s": safe_div(stats["total_tokens"],
                                         stats["total_request_latency_s"]),
            "avg_prompt_tokens_per_executed_request": safe_div(
                stats["total_prompt_tokens"], stats["executed_requests"]),
            "avg_completion_tokens_per_executed_request": safe_div(
                stats["total_completion_tokens"], stats["executed_requests"]),
            "estimated_saved_lm_time_s": stats["estimated_saved_lm_time_s"],
            "estimated_saved_stage_time_s": stats["estimated_saved_stage_time_s"],
            "estimated_saved_prompt_tokens": stats["estimated_saved_prompt_tokens"],
            "estimated_saved_completion_tokens": stats[
                "estimated_saved_completion_tokens"],
            "estimated_saved_total_tokens": stats["estimated_saved_total_tokens"],
            "num_instances": len(stats["instances"]),
            "agents": sorted(stats["agents"]),
        })
    return rows


def build_cache_hits_by_agent_tool_signature(
    reuse_replays: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    stats_by_key: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "turns_total": 0,
            "cache_hits": 0,
            "executed_requests": 0,
            "total_request_latency_s": 0.0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "estimated_saved_lm_time_s": 0.0,
            "estimated_saved_stage_time_s": 0.0,
            "estimated_saved_prompt_tokens": 0,
            "estimated_saved_completion_tokens": 0,
            "estimated_saved_total_tokens": 0,
            "instances": set(),
        })
    for replay in reuse_replays:
        instance_id = str(replay.get("instance_id", ""))
        source_by_turn_index = source_turns_by_turn_index(replay)
        for turn in replay.get("turn_metrics", []):
            agent = str(turn.get("agent", ""))
            tool_signature = tool_signature_for_turn(turn)
            stats = stats_by_key[(agent, tool_signature)]
            stats["turns_total"] += 1
            stats["cache_hits"] += int(bool(turn.get("cache_hit")))
            if is_executed_request(turn):
                stats["executed_requests"] += 1
                stats["total_request_latency_s"] += float(
                    turn.get("request_latency_s") or 0.0)
                prompt_tokens, completion_tokens, total_tokens = turn_token_counts(turn)
                stats["total_prompt_tokens"] += prompt_tokens
                stats["total_completion_tokens"] += completion_tokens
                stats["total_tokens"] += total_tokens
            savings = estimate_cache_hit_savings(turn, source_by_turn_index)
            stats["estimated_saved_lm_time_s"] += float(savings["saved_lm_time_s"])
            stats["estimated_saved_stage_time_s"] += float(savings["saved_stage_time_s"])
            stats["estimated_saved_prompt_tokens"] += int(savings["saved_prompt_tokens"])
            stats["estimated_saved_completion_tokens"] += int(
                savings["saved_completion_tokens"])
            stats["estimated_saved_total_tokens"] += int(savings["saved_total_tokens"])
            stats["instances"].add(instance_id)

    rows: list[dict[str, Any]] = []
    for (agent, tool_signature), stats in sorted(
            stats_by_key.items(),
            key=lambda item: (-item[1]["cache_hits"], -item[1]["turns_total"], item[0][0],
                              item[0][1])):
        rows.append({
            "agent": agent,
            "tool_signature": tool_signature,
            "turns_total": stats["turns_total"],
            "cache_hits": stats["cache_hits"],
            "cache_hit_pct": safe_div(stats["cache_hits"], stats["turns_total"]) * 100.0,
            "executed_requests": stats["executed_requests"],
            "avg_executed_request_latency_s": safe_div(
                stats["total_request_latency_s"], stats["executed_requests"]),
            "total_prompt_tokens": stats["total_prompt_tokens"],
            "total_completion_tokens": stats["total_completion_tokens"],
            "total_tokens": stats["total_tokens"],
            "lm_tokens_per_s": safe_div(stats["total_tokens"],
                                         stats["total_request_latency_s"]),
            "avg_prompt_tokens_per_executed_request": safe_div(
                stats["total_prompt_tokens"], stats["executed_requests"]),
            "avg_completion_tokens_per_executed_request": safe_div(
                stats["total_completion_tokens"], stats["executed_requests"]),
            "estimated_saved_lm_time_s": stats["estimated_saved_lm_time_s"],
            "estimated_saved_stage_time_s": stats["estimated_saved_stage_time_s"],
            "estimated_saved_prompt_tokens": stats["estimated_saved_prompt_tokens"],
            "estimated_saved_completion_tokens": stats[
                "estimated_saved_completion_tokens"],
            "estimated_saved_total_tokens": stats["estimated_saved_total_tokens"],
            "num_instances": len(stats["instances"]),
        })
    return rows


def build_throughput_by_instance(
    reuse_replays: list[dict[str, Any]],
    *,
    pairwise_summary: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if pairwise_summary is not None:
        rows: list[dict[str, Any]] = []
        for row in pairwise_summary["rows"]:
            rows.append({
                "instance_id": row["instance_id"],
                "baseline_vllm_requests": row["baseline_vllm_requests"],
                "reuse_vllm_requests_executed": row["reuse_vllm_requests_executed"],
                "reuse_vllm_requests_avoided": row["reuse_vllm_requests_avoided"],
                "baseline_wall_tokens_per_s": row["baseline_wall_tokens_per_s"],
                "reuse_wall_tokens_per_s": row["reuse_wall_tokens_per_s"],
                "baseline_lm_tokens_per_s": row["baseline_lm_tokens_per_s"],
                "reuse_lm_tokens_per_s": row["reuse_lm_tokens_per_s"],
                "baseline_avg_request_latency_s": row["baseline_avg_request_latency_s"],
                "reuse_avg_request_latency_s": row["reuse_avg_request_latency_s"],
                "baseline_avg_prompt_tokens_per_request": row[
                    "baseline_avg_prompt_tokens_per_request"],
                "reuse_avg_prompt_tokens_per_executed_request": row[
                    "reuse_avg_prompt_tokens_per_executed_request"],
                "baseline_avg_completion_tokens_per_request": row[
                    "baseline_avg_completion_tokens_per_request"],
                "reuse_avg_completion_tokens_per_executed_request": row[
                    "reuse_avg_completion_tokens_per_executed_request"],
            })
        return rows

    rows = []
    for replay in sorted(reuse_replays,
                         key=lambda replay: str(replay.get("instance_id", ""))):
        timing = replay.get("timing", {})
        total_tokens = total_tokens_from_timing(replay)
        wall_solve_time_s = float(timing.get("wall_solve_time_s", 0.0))
        lm_only_solve_time_s = float(timing.get("lm_only_solve_time_s", 0.0))
        executed_requests = executed_request_count(replay)
        total_prompt_tokens = int(timing.get("total_prompt_tokens", 0))
        total_completion_tokens = int(timing.get("total_completion_tokens", 0))
        rows.append({
            "instance_id": str(replay.get("instance_id", "")),
            "reuse_vllm_requests_executed": executed_requests,
            "reuse_vllm_requests_avoided": cache_hit_count(replay),
            "reuse_wall_tokens_per_s": safe_div(total_tokens, wall_solve_time_s),
            "reuse_lm_tokens_per_s": safe_div(total_tokens, lm_only_solve_time_s),
            "reuse_avg_request_latency_s": safe_div(lm_only_solve_time_s,
                                                     executed_requests),
            "reuse_avg_prompt_tokens_per_executed_request": safe_div(
                total_prompt_tokens, executed_requests),
            "reuse_avg_completion_tokens_per_executed_request": safe_div(
                total_completion_tokens, executed_requests),
        })
    return rows


def build_top_exact_repeats(
    reuse_replays: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for replay in reuse_replays:
        instance_id = str(replay.get("instance_id", ""))
        exact_repeats = replay.get("repeated_work", {}).get("exact_repeats", [])
        for entry in exact_repeats:
            rows.append({
                "instance_id": instance_id,
                "agent": entry.get("agent", ""),
                "num_occurrences": int(entry.get("num_occurrences", 0)),
                "num_cache_hits": int(entry.get("num_cache_hits", 0)),
                "first_turn_index": entry.get("first_turn_index"),
                "turn_indices": entry.get("turn_indices", []),
                "resource_group_key": entry.get("resource_group_key", ""),
                "cache_key": entry.get("cache_key", ""),
            })
    rows.sort(
        key=lambda row: (-row["num_cache_hits"], -row["num_occurrences"], row["instance_id"],
                         row["agent"]))
    return rows


def to_csv_value(value: Any) -> str | int | float:
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    if isinstance(value, bool):
        return int(value)
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: to_csv_value(value) for key, value in row.items()})


def svg_text(x: float, y: float, text: str, *, font_size: int = 12,
             anchor: str = "start") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{font_size}" '
        f'font-family="monospace" text-anchor="{anchor}">{escape(text)}</text>')


def write_grouped_bar_svg(
    path: Path,
    *,
    title: str,
    labels: list[str],
    series_names: list[str],
    series_values: list[list[float]],
    colors: list[str],
    x_axis_label: str,
) -> None:
    width = 1200
    left_margin = 320
    right_margin = 120
    top_margin = 80
    bottom_margin = 70
    bar_height = 14
    inner_gap = 4
    group_gap = 14
    group_height = len(series_names) * bar_height + (len(series_names) - 1) * inner_gap
    height = top_margin + bottom_margin + len(labels) * (group_height + group_gap) + 20
    plot_width = width - left_margin - right_margin
    max_value = max((value for values in series_values for value in values), default=0.0)
    max_value = max(max_value, 1.0)

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(20, 30, title, font_size=20),
        svg_text(left_margin, height - 20, x_axis_label, font_size=12),
    ]

    legend_x = left_margin
    legend_y = 50
    for index, series_name in enumerate(series_names):
        y = legend_y + index * 18
        svg_lines.append(
            f'<rect x="{legend_x:.1f}" y="{y - 10:.1f}" width="12" height="12" '
            f'fill="{colors[index]}"/>')
        svg_lines.append(svg_text(legend_x + 18, y, series_name, font_size=12))

    for tick_index in range(6):
        tick_value = max_value * tick_index / 5.0
        x = left_margin + plot_width * tick_index / 5.0
        svg_lines.append(
            f'<line x1="{x:.1f}" y1="{top_margin - 10:.1f}" x2="{x:.1f}" '
            f'y2="{height - bottom_margin + 5:.1f}" stroke="#dddddd" stroke-width="1"/>')
        svg_lines.append(svg_text(x, height - bottom_margin + 24, f"{tick_value:.0f}",
                                  font_size=11, anchor="middle"))

    for row_index, label in enumerate(labels):
        base_y = top_margin + row_index * (group_height + group_gap)
        label_y = base_y + group_height / 2 + 4
        svg_lines.append(svg_text(10, label_y, label, font_size=12))
        for series_index, values in enumerate(series_values):
            value = values[row_index]
            y = base_y + series_index * (bar_height + inner_gap)
            bar_width = plot_width * value / max_value
            svg_lines.append(
                f'<rect x="{left_margin:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
                f'height="{bar_height}" fill="{colors[series_index]}"/>')
            text_x = min(left_margin + bar_width + 6, width - right_margin + 40)
            svg_lines.append(svg_text(text_x, y + bar_height - 2, f"{value:.1f}",
                                      font_size=11))

    svg_lines.append("</svg>")
    path.write_text("\n".join(svg_lines))


def write_single_bar_svg(
    path: Path,
    *,
    title: str,
    labels: list[str],
    values: list[float],
    annotations: list[str],
    color: str,
    x_axis_label: str,
) -> None:
    width = 1200
    left_margin = 320
    right_margin = 240
    top_margin = 60
    bottom_margin = 70
    bar_height = 16
    row_gap = 10
    height = top_margin + bottom_margin + len(labels) * (bar_height + row_gap) + 20
    plot_width = width - left_margin - right_margin
    max_value = max(values, default=0.0)
    max_value = max(max_value, 1.0)

    svg_lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(20, 30, title, font_size=20),
        svg_text(left_margin, height - 20, x_axis_label, font_size=12),
    ]

    for tick_index in range(6):
        tick_value = max_value * tick_index / 5.0
        x = left_margin + plot_width * tick_index / 5.0
        svg_lines.append(
            f'<line x1="{x:.1f}" y1="{top_margin - 10:.1f}" x2="{x:.1f}" '
            f'y2="{height - bottom_margin + 5:.1f}" stroke="#dddddd" stroke-width="1"/>')
        svg_lines.append(svg_text(x, height - bottom_margin + 24, f"{tick_value:.0f}",
                                  font_size=11, anchor="middle"))

    for row_index, (label, value, annotation) in enumerate(zip(labels, values, annotations,
                                                               strict=False)):
        y = top_margin + row_index * (bar_height + row_gap)
        bar_width = plot_width * value / max_value
        svg_lines.append(svg_text(10, y + bar_height - 2, label, font_size=12))
        svg_lines.append(
            f'<rect x="{left_margin:.1f}" y="{y:.1f}" width="{bar_width:.1f}" '
            f'height="{bar_height}" fill="{color}"/>')
        svg_lines.append(svg_text(left_margin + bar_width + 8, y + bar_height - 2,
                                  annotation, font_size=11))

    svg_lines.append("</svg>")
    path.write_text("\n".join(svg_lines))


def write_analysis_plots(
    output_dir: Path,
    *,
    pairwise_rows: list[dict[str, Any]],
    throughput_by_instance: list[dict[str, Any]],
    throughput_by_agent: list[dict[str, Any]],
    throughput_by_tool_signature: list[dict[str, Any]],
    cache_hits_by_agent: list[dict[str, Any]],
    cache_hits_by_tool_signature: list[dict[str, Any]],
    savings_by_agent: list[dict[str, Any]],
    savings_by_tool_signature: list[dict[str, Any]],
    top_n_tools: int,
) -> list[Path]:
    paths: list[Path] = []

    if pairwise_rows:
        labels = [row["instance_id"] for row in pairwise_rows]
        jct_path = output_dir / "jct_comparison.svg"
        write_grouped_bar_svg(
            jct_path,
            title="Baseline vs Reuse Wall Time",
            labels=labels,
            series_names=["baseline", "reuse"],
            series_values=[
                [float(row["baseline_wall_solve_time_s"]) for row in pairwise_rows],
                [float(row["reuse_wall_solve_time_s"]) for row in pairwise_rows],
            ],
            colors=["#c35a3a", "#3a7bd5"],
            x_axis_label="wall_solve_time_s",
        )
        paths.append(jct_path)

        req_path = output_dir / "request_comparison.svg"
        write_grouped_bar_svg(
            req_path,
            title="Baseline vs Reuse Executed vLLM Requests",
            labels=labels,
            series_names=["baseline", "reuse_executed"],
            series_values=[
                [float(row["baseline_vllm_requests"]) for row in pairwise_rows],
                [float(row["reuse_vllm_requests_executed"]) for row in pairwise_rows],
            ],
            colors=["#c35a3a", "#3a7bd5"],
            x_axis_label="executed_vllm_requests",
        )
        paths.append(req_path)

    if throughput_by_instance:
        wall_tp_path = output_dir / "throughput_wall_by_instance.svg"
        write_grouped_bar_svg(
            wall_tp_path,
            title="Wall Throughput by Instance",
            labels=[row["instance_id"] for row in throughput_by_instance],
            series_names=(
                ["baseline_wall", "reuse_wall"]
                if "baseline_wall_tokens_per_s" in throughput_by_instance[0]
                else ["reuse_wall"]
            ),
            series_values=(
                [
                    [
                        float(row["baseline_wall_tokens_per_s"])
                        for row in throughput_by_instance
                    ],
                    [
                        float(row["reuse_wall_tokens_per_s"])
                        for row in throughput_by_instance
                    ],
                ]
                if "baseline_wall_tokens_per_s" in throughput_by_instance[0]
                else [[float(row["reuse_wall_tokens_per_s"]) for row in throughput_by_instance]]
            ),
            colors=(["#c35a3a", "#3a7bd5"]
                    if "baseline_wall_tokens_per_s" in throughput_by_instance[0]
                    else ["#3a7bd5"]),
            x_axis_label="tokens_per_second_over_wall_time",
        )
        paths.append(wall_tp_path)

        lm_tp_path = output_dir / "throughput_lm_by_instance.svg"
        write_grouped_bar_svg(
            lm_tp_path,
            title="LM-Only Throughput by Instance",
            labels=[row["instance_id"] for row in throughput_by_instance],
            series_names=(
                ["baseline_lm", "reuse_lm"]
                if "baseline_lm_tokens_per_s" in throughput_by_instance[0]
                else ["reuse_lm"]
            ),
            series_values=(
                [
                    [float(row["baseline_lm_tokens_per_s"]) for row in throughput_by_instance],
                    [float(row["reuse_lm_tokens_per_s"]) for row in throughput_by_instance],
                ]
                if "baseline_lm_tokens_per_s" in throughput_by_instance[0]
                else [[float(row["reuse_lm_tokens_per_s"]) for row in throughput_by_instance]]
            ),
            colors=(["#c35a3a", "#3a7bd5"]
                    if "baseline_lm_tokens_per_s" in throughput_by_instance[0]
                    else ["#3a7bd5"]),
            x_axis_label="tokens_per_second_over_executed_lm_time",
        )
        paths.append(lm_tp_path)

    if throughput_by_agent:
        agent_tp_path = output_dir / "throughput_by_agent.svg"
        write_single_bar_svg(
            agent_tp_path,
            title="LM Throughput by Agent",
            labels=[row["agent"] for row in throughput_by_agent],
            values=[float(row["lm_tokens_per_s"]) for row in throughput_by_agent],
            annotations=[
                f"{int(row['executed_requests'])} executed requests"
                for row in throughput_by_agent
            ],
            color="#0b7285",
            x_axis_label="tokens_per_second_over_executed_lm_time",
        )
        paths.append(agent_tp_path)

    if throughput_by_tool_signature:
        selected = throughput_by_tool_signature[:top_n_tools]
        tool_tp_path = output_dir / "throughput_by_tool_signature.svg"
        write_single_bar_svg(
            tool_tp_path,
            title=f"Top {len(selected)} Tool Signatures by LM Throughput",
            labels=[row["tool_signature"] for row in selected],
            values=[float(row["lm_tokens_per_s"]) for row in selected],
            annotations=[
                f"{int(row['executed_requests'])} executed requests"
                for row in selected
            ],
            color="#0b7285",
            x_axis_label="tokens_per_second_over_executed_lm_time",
        )
        paths.append(tool_tp_path)

    if cache_hits_by_agent:
        agent_path = output_dir / "cache_hits_by_agent.svg"
        write_single_bar_svg(
            agent_path,
            title="Cache Hit Rate by Agent",
            labels=[row["agent"] for row in cache_hits_by_agent],
            values=[float(row["cache_hit_pct"]) for row in cache_hits_by_agent],
            annotations=[
                f"{int(row['cache_hits'])}/{int(row['turns_total'])} hits"
                for row in cache_hits_by_agent
            ],
            color="#2f9e44",
            x_axis_label="cache_hit_pct",
        )
        paths.append(agent_path)

    if cache_hits_by_tool_signature:
        selected = cache_hits_by_tool_signature[:top_n_tools]
        tool_path = output_dir / "cache_hits_by_tool_signature.svg"
        write_single_bar_svg(
            tool_path,
            title=f"Top {len(selected)} Tool Signatures by Cache Hits",
            labels=[row["tool_signature"] for row in selected],
            values=[float(row["cache_hits"]) for row in selected],
            annotations=[
                f"{float(row['cache_hit_pct']):.1f}% hit rate" for row in selected
            ],
            color="#8f5cc2",
            x_axis_label="cache_hits",
        )
        paths.append(tool_path)

    if savings_by_agent:
        agent_savings_path = output_dir / "saved_lm_time_by_agent.svg"
        write_single_bar_svg(
            agent_savings_path,
            title="Estimated Saved LM Time by Agent",
            labels=[row["agent"] for row in savings_by_agent],
            values=[float(row["estimated_saved_lm_time_s"]) for row in savings_by_agent],
            annotations=[
                f"{int(row['cache_hits'])} cache hits" for row in savings_by_agent
            ],
            color="#d97904",
            x_axis_label="estimated_saved_lm_time_s",
        )
        paths.append(agent_savings_path)

    if savings_by_tool_signature:
        selected = savings_by_tool_signature[:top_n_tools]
        tool_savings_path = output_dir / "saved_lm_time_by_tool_signature.svg"
        write_single_bar_svg(
            tool_savings_path,
            title=f"Top {len(selected)} Tool Signatures by Estimated Saved LM Time",
            labels=[row["tool_signature"] for row in selected],
            values=[float(row["estimated_saved_lm_time_s"]) for row in selected],
            annotations=[
                f"{int(row['cache_hits'])} cache hits" for row in selected
            ],
            color="#d97904",
            x_axis_label="estimated_saved_lm_time_s",
        )
        paths.append(tool_savings_path)

    return paths


def print_summary(
    *,
    pairwise_summary: dict[str, Any] | None,
    cache_hits_by_agent: list[dict[str, Any]],
    cache_hits_by_tool_signature: list[dict[str, Any]],
) -> None:
    if pairwise_summary is not None:
        overall = pairwise_summary["overall"]
        print(
            "Matched baseline/reuse pairs: "
            f"{overall['num_matched_pairs']} "
            f"(wall delta {overall['wall_time_delta_pct']:.1f}%, "
            f"requests avoided {overall['total_reuse_vllm_requests_avoided']})"
        )
        print()
        rows = []
        for row in pairwise_summary["rows"]:
            rows.append([
                row["instance_id"],
                int(row["baseline_vllm_requests"]),
                int(row["reuse_vllm_requests_executed"]),
                int(row["reuse_vllm_requests_avoided"]),
                f"{row['reuse_cache_hit_pct']:.1f}",
                f"{row['baseline_wall_solve_time_s']:.1f}",
                f"{row['reuse_wall_solve_time_s']:.1f}",
                f"{row['wall_time_delta_pct']:.1f}",
            ])
        print_table(
            rows,
            headers=[
                "Instance",
                "BaseReqs",
                "ReuseReqs",
                "Avoided",
                "HitPct",
                "WallBase",
                "WallReuse",
                "WallDeltaPct",
            ],
        )
        print()

    if cache_hits_by_agent:
        print("Cache Hits by Agent")
        print_table(
            [[
                row["agent"],
                int(row["turns_total"]),
                int(row["cache_hits"]),
                f"{row['cache_hit_pct']:.1f}",
                int(row["executed_requests"]),
                format_float(float(row["avg_executed_request_latency_s"])),
                format_float(float(row["lm_tokens_per_s"])),
                format_float(float(row["estimated_saved_lm_time_s"])),
            ] for row in cache_hits_by_agent],
            headers=[
                "Agent",
                "Turns",
                "CacheHits",
                "HitPct",
                "ExecReqs",
                "AvgExecReqLatencyS",
                "LmTokPerS",
                "SavedLmS",
            ],
        )
        print()

    if cache_hits_by_tool_signature:
        print("Top Tool Signatures by Cache Hits")
        top_rows = cache_hits_by_tool_signature[:10]
        print_table(
            [[
                row["tool_signature"],
                int(row["turns_total"]),
                int(row["cache_hits"]),
                f"{row['cache_hit_pct']:.1f}",
                int(row["executed_requests"]),
                int(row["num_instances"]),
                format_float(float(row["lm_tokens_per_s"])),
                format_float(float(row["estimated_saved_lm_time_s"])),
            ] for row in top_rows],
            headers=[
                "ToolSig",
                "Turns",
                "CacheHits",
                "HitPct",
                "ExecReqs",
                "Instances",
                "LmTokPerS",
                "SavedLmS",
            ],
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze baseline and reuse replay outputs and write comparison artifacts"
    )
    parser.add_argument(
        "--reuse-glob",
        required=True,
        help="Glob for reuse replay JSON files, for example 'replays-two/*.reuse.replay.json'",
    )
    parser.add_argument(
        "--baseline-glob",
        default=None,
        help="Optional glob for baseline replay JSON files, for example 'replays-two/*.baseline.replay.json'",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where analysis JSON, CSV, and SVG artifacts will be written",
    )
    parser.add_argument(
        "--top-n-tools",
        type=int,
        default=15,
        help="Number of tool signatures to include in the tool cache-hit plot",
    )
    parser.add_argument(
        "--top-n-exact-repeats",
        type=int,
        default=50,
        help="Number of top exact-repeat entries to keep in the JSON report and CSV export",
    )
    args = parser.parse_args()

    if args.top_n_tools <= 0:
        raise ValueError("--top-n-tools must be > 0")
    if args.top_n_exact_repeats <= 0:
        raise ValueError("--top-n-exact-repeats must be > 0")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    reuse_paths = discover_replay_paths(args.reuse_glob, suffix=REUSE_REPLAY_SUFFIX)
    reuse_replays = [load_json(path) for path in reuse_paths]

    pairwise_summary: dict[str, Any] | None = None
    baseline_paths: list[Path] = []
    if args.baseline_glob is not None:
        baseline_paths = discover_replay_paths(args.baseline_glob,
                                               suffix=BASELINE_REPLAY_SUFFIX)
        pairwise_summary = build_pairwise_summary(baseline_paths, reuse_paths)

    cache_hits_by_agent = build_cache_hits_by_agent(reuse_replays)
    cache_hits_by_tool_signature = build_cache_hits_by_tool_signature(reuse_replays)
    cache_hits_by_agent_tool_signature = build_cache_hits_by_agent_tool_signature(
        reuse_replays)
    throughput_by_instance = build_throughput_by_instance(
        reuse_replays,
        pairwise_summary=pairwise_summary,
    )
    top_exact_repeats = build_top_exact_repeats(reuse_replays)[:args.top_n_exact_repeats]
    savings_by_agent = sorted(
        cache_hits_by_agent,
        key=lambda row: (-float(row["estimated_saved_lm_time_s"]), -int(row["cache_hits"]),
                         row["agent"]),
    )
    savings_by_tool_signature = sorted(
        cache_hits_by_tool_signature,
        key=lambda row: (-float(row["estimated_saved_lm_time_s"]), -int(row["cache_hits"]),
                         row["tool_signature"]),
    )
    throughput_by_agent = sorted(
        cache_hits_by_agent,
        key=lambda row: (-float(row["lm_tokens_per_s"]), -int(row["executed_requests"]),
                         row["agent"]),
    )
    throughput_by_tool_signature = sorted(
        cache_hits_by_tool_signature,
        key=lambda row: (-float(row["lm_tokens_per_s"]), -int(row["executed_requests"]),
                         row["tool_signature"]),
    )

    report = {
        "baseline_glob": args.baseline_glob,
        "reuse_glob": args.reuse_glob,
        "num_reuse_replays": len(reuse_paths),
        "num_baseline_replays": len(baseline_paths),
        "pairwise_summary": pairwise_summary,
        "throughput_by_instance": throughput_by_instance,
        "throughput_by_agent": throughput_by_agent,
        "throughput_by_tool_signature": throughput_by_tool_signature,
        "cache_hits_by_agent": cache_hits_by_agent,
        "cache_hits_by_tool_signature": cache_hits_by_tool_signature,
        "cache_hits_by_agent_tool_signature": cache_hits_by_agent_tool_signature,
        "savings_by_agent": savings_by_agent,
        "savings_by_tool_signature": savings_by_tool_signature,
        "top_exact_repeats": top_exact_repeats,
    }

    report_path = args.output_dir / "reuse_analysis_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    if pairwise_summary is not None:
        write_csv(args.output_dir / "pairwise_comparison.csv", pairwise_summary["rows"])
    write_csv(args.output_dir / "throughput_by_instance.csv", throughput_by_instance)
    write_csv(args.output_dir / "throughput_by_agent.csv", throughput_by_agent)
    write_csv(args.output_dir / "throughput_by_tool_signature.csv",
              throughput_by_tool_signature)
    write_csv(args.output_dir / "cache_hits_by_agent.csv", cache_hits_by_agent)
    write_csv(args.output_dir / "cache_hits_by_tool_signature.csv",
              cache_hits_by_tool_signature)
    write_csv(args.output_dir / "cache_hits_by_agent_tool_signature.csv",
              cache_hits_by_agent_tool_signature)
    write_csv(args.output_dir / "savings_by_agent.csv", savings_by_agent)
    write_csv(args.output_dir / "savings_by_tool_signature.csv",
              savings_by_tool_signature)
    write_csv(args.output_dir / "top_exact_repeats.csv", top_exact_repeats)

    plot_paths = write_analysis_plots(
        args.output_dir,
        pairwise_rows=(pairwise_summary["rows"] if pairwise_summary is not None else []),
        throughput_by_instance=throughput_by_instance,
        throughput_by_agent=throughput_by_agent,
        throughput_by_tool_signature=throughput_by_tool_signature,
        cache_hits_by_agent=cache_hits_by_agent,
        cache_hits_by_tool_signature=cache_hits_by_tool_signature,
        savings_by_agent=savings_by_agent,
        savings_by_tool_signature=savings_by_tool_signature,
        top_n_tools=args.top_n_tools,
    )

    print_summary(
        pairwise_summary=pairwise_summary,
        cache_hits_by_agent=cache_hits_by_agent,
        cache_hits_by_tool_signature=cache_hits_by_tool_signature,
    )
    print()
    print(f"Saved analysis report to {report_path}")
    if plot_paths:
        print("Saved plots:")
        for path in plot_paths:
            print(f"- {path}")


if __name__ == "__main__":
    main()
