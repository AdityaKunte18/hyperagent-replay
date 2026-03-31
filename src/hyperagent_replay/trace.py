"""Utilities for parsing and summarizing HyperAgent trajectory JSON files."""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

LOG_LINE_RE = re.compile(
    r"^HyperAgent_(?P<instance>.+?) - INFO - (?P<tag>[^:]+): ?(?P<body>.*)$")
RESPONSE_TAG_RE = re.compile(r"^(?P<agent>.+?)'s Response$")


def normalize_text(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join(str(item) for item in value).strip()
    return str(value).strip()


def is_marker(line: str) -> bool:
    stripped = line.strip()
    return (bool(LOG_LINE_RE.match(line)) or stripped == "Action:"
            or stripped == "Observation"
            or stripped.startswith("Intern Name:")
            or stripped.startswith("Subgoal:"))


def parse_code_block(lines: list[str],
                     start: int) -> tuple[str | None, str, int]:
    if start >= len(lines):
        return None, "", start

    match = re.match(r"```(\w+)?", lines[start].strip())
    if not match:
        return None, "", start

    language = match.group(1)
    code_lines: list[str] = []
    i = start + 1
    while i < len(lines) and lines[i].strip() != "```":
        code_lines.append(lines[i])
        i += 1
    if i < len(lines):
        i += 1
    return language, "\n".join(code_lines).strip(), i


def derive_tool_name(language: str | None, code: str) -> str | None:
    generic_python_call_re = re.compile(
        r"(?:[A-Za-z_]\w*\s*=\s*)?"
        r"([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*\(",
    )

    for line in code.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if language == "bash":
            return stripped.split()[0]

        if language == "python":
            match = generic_python_call_re.match(stripped)
            if match:
                func_name = match.group(1)
                if func_name.endswith("_run"):
                    return func_name

    for line in code.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if language == "python":
            match = generic_python_call_re.match(stripped)
            if match:
                return "python_exec"
    return None


def collect_block(lines: list[str], start: int) -> tuple[str, int]:
    block_lines = [lines[start]]
    i = start + 1
    while i < len(lines) and not is_marker(lines[i]):
        block_lines.append(lines[i])
        i += 1
    return "\n".join(block_lines).strip(), i


def parse_events(lines: list[str]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        log_match = LOG_LINE_RE.match(line)
        if log_match:
            tag = log_match.group("tag")
            body = log_match.group("body")
            response_match = RESPONSE_TAG_RE.match(tag)
            if response_match:
                content_lines = [body] if body else []
                i += 1
                while i < len(lines) and not is_marker(lines[i]):
                    content_lines.append(lines[i])
                    i += 1
                events.append({
                    "type": "response",
                    "agent": response_match.group("agent"),
                    "content": "\n".join(content_lines).strip(),
                })
                continue

            if "->" in tag:
                events.append({
                    "type": "handoff",
                    "agent": tag,
                    "content": body.strip(),
                })
            else:
                events.append({
                    "type": "log",
                    "agent": tag,
                    "content": body.strip(),
                })
            i += 1
            continue

        if stripped.startswith("Intern Name:"):
            events.append({
                "type": "intern_name",
                "content": stripped.split(":", 1)[1].strip(),
            })
            i += 1
            continue

        if stripped.startswith("Subgoal:"):
            content, i = collect_block(lines, i)
            events.append({
                "type": "subgoal",
                "content": content.split(":", 1)[1].strip(),
            })
            continue

        if stripped == "Action:":
            language, code, i = parse_code_block(lines, i + 1)
            events.append({
                "type": "action",
                "language": language,
                "code": code,
                "tool_name": derive_tool_name(language, code),
            })
            continue

        if stripped == "Observation":
            content, i = collect_block(lines, i)
            events.append({"type": "observation", "content": content})
            continue

        i += 1
    return events


def build_turns(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    for event in events:
        if event["type"] != "response":
            continue
        turns.append({
            "turn_index": len(turns),
            "agent": event["agent"],
            "content": event["content"],
            "contains_final_answer": "Final Answer:" in event["content"],
            "action": None,
        })

    turn_idx = 0
    for event in events:
        if event["type"] == "response":
            turn_idx += 1
            continue
        if event["type"] == "action" and turn_idx > 0:
            current_turn = turns[turn_idx - 1]
            if current_turn["action"] is None:
                current_turn["action"] = {
                    "language": event["language"],
                    "tool_name": event["tool_name"],
                    "code": event["code"],
                }
    return turns


def summarize_turns(turns: list[dict[str, Any]]) -> dict[str, Any]:
    response_counts = Counter(turn["agent"] for turn in turns)
    action_langs = Counter()
    tool_names = Counter()
    response_lengths = [len(turn["content"]) for turn in turns]
    turns_with_actions = 0
    for turn in turns:
        action = turn["action"]
        if action is None:
            continue
        turns_with_actions += 1
        if action["language"]:
            action_langs[action["language"]] += 1
        if action["tool_name"]:
            tool_names[action["tool_name"]] += 1

    return {
        "num_llm_turns": len(turns),
        "num_turns_with_actions": turns_with_actions,
        "num_final_answer_turns":
        sum(turn["contains_final_answer"] for turn in turns),
        "response_counts": dict(response_counts),
        "action_languages": dict(action_langs),
        "top_tool_names": dict(tool_names.most_common(25)),
        "avg_response_chars":
        (sum(response_lengths) / len(response_lengths) if response_lengths else
         0.0),
        "median_response_chars":
        (statistics.median(response_lengths) if response_lengths else 0.0),
        "max_response_chars": max(response_lengths) if response_lengths else 0,
    }


def extract_trace(raw: dict[str, Any]) -> dict[str, Any]:
    problem_statement = normalize_text(raw.get("problem_statement", ""))
    events = parse_events(raw.get("trajectory", []))
    turns = build_turns(events)
    summary = summarize_turns(turns)
    summary["num_trajectory_entries"] = len(raw.get("trajectory", []))
    summary["num_events"] = len(events)
    summary["trajectory_restart_markers"] = sum(
        "Initialized HyperAgent instance" in line
        for line in raw.get("trajectory", []))

    note = raw.get("note") or {}
    if isinstance(note, dict) and "options" in note:
        summary["annotated_flags"] = note["options"]

    return {
        "instance_id": raw.get("instance_id"),
        "problem_statement": problem_statement,
        "summary": summary,
        "events": events,
        "llm_turns": turns,
        "source_note": note if isinstance(note, dict) else {},
    }


def load_raw_trace(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or "trajectory" not in data:
        raise ValueError(
            f"{path} does not look like a raw HyperAgent trajectory JSON")
    return data


def load_trace_payload(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "llm_turns" in data and "summary" in data:
        return data
    return extract_trace(load_raw_trace(path))
