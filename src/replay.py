"""Replay one HyperAgent trajectory against a running or launched vLLM server."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from openai import OpenAI

from hyperagent_replay.trace import load_trace_payload

DEFAULT_SYSTEM_PROMPT = (
    "You are replaying a recorded software-engineering agent trajectory. "
    "Produce the next assistant turn in the same style and intent as the "
    "recorded turn. If the recorded turn implies a tool call, prefer exactly "
    "one fenced code block in the recorded language."
)
DEFAULT_BASE_URL = "http://127.0.0.1:8000/v1"


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


def wait_for_server(base_url: str, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    models_url = base_url.rstrip("/") + "/models"
    while time.time() < deadline:
        try:
            with urlopen(models_url, timeout=2.0) as resp:
                if 200 <= resp.status < 300:
                    return
        except (OSError, URLError):
            time.sleep(1.0)
    raise TimeoutError(f"Timed out waiting for vLLM server at {models_url}")


def kill_process_tree(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name != "nt":
            os.killpg(proc.pid, 15)
        else:
            proc.terminate()
        proc.wait(timeout=10)
    except Exception:
        if proc.poll() is None:
            if os.name != "nt":
                os.killpg(proc.pid, 9)
            else:
                proc.kill()


def launch_server(model_name: str, port: int, host: str,
                  serve_args: list[str], log_path: Path | None
                  ) -> subprocess.Popen[str]:
    cmd = ["vllm", "serve", model_name, "--host", host, "--port", str(port)]
    cmd.extend(serve_args)
    stdout = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        stdout = open(log_path, "w")
    kwargs: dict[str, Any] = {
        "stdout": stdout,
        "stderr": subprocess.STDOUT,
        "text": True,
    }
    if os.name != "nt":
        kwargs["preexec_fn"] = os.setsid
    return subprocess.Popen(cmd, **kwargs)


def context_key(turn: dict[str, Any], mode: str) -> str:
    if mode == "flattened":
        return "issue"
    return turn["agent"]


def make_seed_messages(problem_statement: str,
                       agent: str | None = None) -> list[dict[str, str]]:
    issue_intro = f"Issue statement:\n{problem_statement}"
    if agent is not None:
        issue_intro += f"\n\nCurrent replay context agent: {agent}"
    return [{
        "role": "system",
        "content": DEFAULT_SYSTEM_PROMPT,
    }, {
        "role": "user",
        "content": issue_intro,
    }]


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n...[truncated]..."


def build_turn_instruction(turn: dict[str, Any], max_reference_chars: int,
                           include_agent_name: bool) -> str:
    lines = []
    if include_agent_name:
        lines.append(f"Replay the next recorded turn for agent `{turn['agent']}`.")
    else:
        lines.append("Replay the next recorded turn.")

    lines.append("Recorded turn text:")
    lines.append(truncate_text(turn["content"], max_reference_chars))

    action = turn.get("action")
    if action:
        language = action.get("language") or "text"
        lines.append("")
        lines.append(
            "This recorded turn included a tool invocation. Prefer exactly "
            f"one fenced `{language}` code block.")
        if action.get("tool_name"):
            lines.append(
                f"Preferred tool signature or command: `{action['tool_name']}`.")
        if action.get("code"):
            lines.append("Reference action code:")
            lines.append(f"```{language}\n{action['code']}\n```")

    if turn.get("contains_final_answer"):
        lines.append("")
        lines.append(
            "This is a final-answer style turn. Resolve the issue concisely.")

    return "\n".join(lines)


def tool_delay_for_turn(turn: dict[str, Any], policy: str,
                        constant_delay: float) -> float:
    if policy == "none" or not turn.get("action"):
        return 0.0
    if policy == "constant":
        return constant_delay

    action = turn["action"]
    tool_name = (action.get("tool_name") or "").lower()
    language = action.get("language")

    if language == "bash":
        if tool_name in {"pytest", "python"}:
            return 4.0
        if tool_name in {"sed", "cat", "grep", "ls"}:
            return 0.2
        return 0.5

    if language == "python":
        if tool_name.endswith("_run"):
            if any(name in tool_name
                   for name in ("open_file", "search", "symbol", "folder")):
                return 0.15
            return 0.35
        return 1.0

    return constant_delay


def build_observation(turn: dict[str, Any], delay_s: float) -> str:
    action = turn.get("action")
    if not action:
        return ""
    tool_name = action.get("tool_name") or "tool"
    language = action.get("language") or "text"
    lines = [
        f"Observation: Synthetic replay for `{tool_name}` completed in {delay_s:.3f} seconds.",
    ]
    if action.get("code"):
        lines.extend([
            "Recorded action code:",
            f"```{language}\n{action['code']}\n```",
        ])
    return "\n".join(lines)


def replay_trace(trace: dict[str, Any], client: OpenAI, model_name: str,
                 context_mode: str, max_reference_chars: int,
                 max_turns: int | None, temperature: float,
                 max_completion_tokens: int, seed: int | None,
                 delay_policy: str, constant_delay: float
                 ) -> dict[str, Any]:
    contexts: dict[str, list[dict[str, str]]] = {}
    turns = trace["llm_turns"]
    if max_turns is not None:
        turns = turns[:max_turns]

    results: list[dict[str, Any]] = []
    total_tool_sleep_s = 0.0
    solve_t0 = time.time()

    for turn in turns:
        key = context_key(turn, context_mode)
        if key not in contexts:
            agent = turn["agent"] if context_mode != "flattened" else None
            contexts[key] = make_seed_messages(trace["problem_statement"], agent)

        instruction = build_turn_instruction(
            turn,
            max_reference_chars=max_reference_chars,
            include_agent_name=context_mode == "flattened",
        )
        request_messages = [*contexts[key], {
            "role": "user",
            "content": instruction,
        }]

        request_prompt_chars = sum(len(msg["content"]) for msg in request_messages)
        t0 = time.time()
        response = client.chat.completions.create(
            model=model_name,
            messages=request_messages,
            temperature=temperature,
            max_completion_tokens=max_completion_tokens,
            seed=seed,
        )
        t1 = time.time()

        response_text = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)

        contexts[key] = [*request_messages, {
            "role": "assistant",
            "content": response_text,
        }]

        tool_sleep_s = tool_delay_for_turn(turn, delay_policy, constant_delay)
        if tool_sleep_s > 0.0:
            time.sleep(tool_sleep_s)
            total_tool_sleep_s += tool_sleep_s

        observation = build_observation(turn, tool_sleep_s)
        if observation:
            contexts[key].append({"role": "user", "content": observation})

        results.append({
            "turn_index": turn["turn_index"],
            "agent": turn["agent"],
            "context_key": key,
            "contains_final_answer": turn["contains_final_answer"],
            "recorded_action": turn.get("action"),
            "request_latency_s": t1 - t0,
            "synthetic_tool_sleep_s": tool_sleep_s,
            "request_prompt_chars": request_prompt_chars,
            "response_chars": len(response_text),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "request_messages": request_messages,
            "response_text": response_text,
        })

    solve_t1 = time.time()
    request_latencies = [item["request_latency_s"] for item in results]
    prompt_tokens = [
        item["prompt_tokens"] for item in results
        if item["prompt_tokens"] is not None
    ]
    completion_tokens = [
        item["completion_tokens"] for item in results
        if item["completion_tokens"] is not None
    ]

    return {
        "instance_id": trace["instance_id"],
        "problem_statement": trace["problem_statement"],
        "source_summary": trace["summary"],
        "settings": {
            "model_name": model_name,
            "context_mode": context_mode,
            "max_reference_chars": max_reference_chars,
            "temperature": temperature,
            "max_completion_tokens": max_completion_tokens,
            "delay_policy": delay_policy,
            "constant_delay": constant_delay,
            "seed": seed,
            "max_turns": max_turns,
        },
        "timing": {
            "wall_solve_time_s": solve_t1 - solve_t0,
            "lm_only_solve_time_s": sum(request_latencies),
            "synthetic_tool_sleep_s": total_tool_sleep_s,
            "num_replayed_turns": len(results),
            "avg_request_latency_s": (sum(request_latencies) /
                                       len(request_latencies)
                                       if request_latencies else 0.0),
            "median_request_latency_s":
            (percentile(request_latencies, 50) if request_latencies else 0.0),
            "p95_request_latency_s":
            (percentile(request_latencies, 95) if request_latencies else 0.0),
            "p99_request_latency_s":
            (percentile(request_latencies, 99) if request_latencies else 0.0),
            "total_prompt_tokens": sum(prompt_tokens),
            "total_completion_tokens": sum(completion_tokens),
        },
        "turn_metrics": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay one HyperAgent trajectory against a vLLM server")
    parser.add_argument("input",
                        type=Path,
                        help="Raw HyperAgent JSON or extracted trace JSON")
    parser.add_argument("--output",
                        type=Path,
                        default=None,
                        help="Path to save replay results JSON")
    parser.add_argument("--model",
                        required=True,
                        help="Model id to request from the vLLM server")
    parser.add_argument("--base-url",
                        default=DEFAULT_BASE_URL,
                        help="OpenAI-compatible vLLM base URL")
    parser.add_argument("--api-key",
                        default="EMPTY",
                        help="API key for the OpenAI client")
    parser.add_argument("--context-mode",
                        choices=["flattened", "per_agent"],
                        default="per_agent")
    parser.add_argument("--max-reference-chars",
                        type=int,
                        default=4000)
    parser.add_argument("--max-turns",
                        type=int,
                        default=None,
                        help="Replay only the first N turns")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-completion-tokens",
                        type=int,
                        default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--delay-policy",
                        choices=["none", "constant", "heuristic"],
                        default="heuristic")
    parser.add_argument("--constant-delay",
                        type=float,
                        default=0.2)
    parser.add_argument("--launch-server",
                        action="store_true",
                        help="Launch `vllm serve` before replaying")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--server-timeout-s",
                        type=float,
                        default=180.0)
    parser.add_argument("--server-log",
                        type=Path,
                        default=None)
    parser.add_argument(
        "--serve-arg",
        action="append",
        default=[],
        help="Additional argument to pass through to `vllm serve`",
    )
    args = parser.parse_args()

    if (args.base_url == DEFAULT_BASE_URL and
            (args.host != "127.0.0.1" or args.port != 8000)):
        args.base_url = f"http://{args.host}:{args.port}/v1"

    trace = load_trace_payload(args.input)
    output_path = args.output
    if output_path is None:
        output_path = args.input.with_suffix(".replay.json")

    server_proc = None
    if args.launch_server:
        server_proc = launch_server(
            model_name=args.model,
            port=args.port,
            host=args.host,
            serve_args=args.serve_arg,
            log_path=args.server_log,
        )

    try:
        wait_for_server(args.base_url, args.server_timeout_s)
        client = OpenAI(base_url=args.base_url,
                        api_key=args.api_key,
                        timeout=max(args.server_timeout_s, 30.0))
        results = replay_trace(
            trace=trace,
            client=client,
            model_name=args.model,
            context_mode=args.context_mode,
            max_reference_chars=args.max_reference_chars,
            max_turns=args.max_turns,
            temperature=args.temperature,
            max_completion_tokens=args.max_completion_tokens,
            seed=args.seed,
            delay_policy=args.delay_policy,
            constant_delay=args.constant_delay,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(results, indent=2))
        print(f"Saved replay results to {output_path}")
        print(json.dumps(results["timing"], indent=2))
    finally:
        if server_proc is not None:
            kill_process_tree(server_proc)


if __name__ == "__main__":
    main()
