"""Replay one HyperAgent trajectory against a running or launched vLLM server."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from collections.abc import Callable
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
DEFAULT_CONTEXT_SAFETY_MARGIN = 512
DEFAULT_MIN_REFERENCE_CHARS = 256
DEFAULT_CHARS_PER_TOKEN_ESTIMATE = 3.0


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


def wait_for_server(base_url: str,
                    timeout_s: float,
                    server_proc: subprocess.Popen[str] | None = None,
                    server_log: Path | None = None) -> None:
    deadline = time.time() + timeout_s
    models_url = base_url.rstrip("/") + "/models"
    while time.time() < deadline:
        if server_proc is not None and server_proc.poll() is not None:
            detail = f"vLLM server exited before becoming ready (exit code {server_proc.returncode})."
            if server_log is not None:
                detail += f" Check {server_log} for startup logs."
            raise RuntimeError(detail)
        try:
            with urlopen(models_url, timeout=2.0) as resp:
                if 200 <= resp.status < 300:
                    return
        except (OSError, URLError):
            time.sleep(1.0)
    detail = f"Timed out waiting for vLLM server at {models_url}"
    if server_log is not None:
        detail += f". Check {server_log} for startup logs."
    raise TimeoutError(detail)


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
            lines.append(
                f"```{language}\n{truncate_text(action['code'], max_reference_chars)}\n```"
            )

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


def estimate_text_tokens(text: str, chars_per_token: float) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / chars_per_token))


def estimate_messages_tokens(messages: list[dict[str, str]],
                             chars_per_token: float) -> int:
    token_count = 2
    for message in messages:
        token_count += 4 + estimate_text_tokens(message["content"],
                                                chars_per_token)
    return token_count


def compute_input_token_budget(max_model_len: int | None,
                               max_completion_tokens: int,
                               context_safety_margin: int) -> int | None:
    if max_model_len is None:
        return None
    return max(1, max_model_len - max_completion_tokens - context_safety_margin)


def seed_message_count(messages: list[dict[str, str]]) -> int:
    return min(2, len(messages))


def shrink_reference_chars(current_max_reference_chars: int,
                           min_reference_chars: int,
                           turn: dict[str, Any]) -> int:
    if current_max_reference_chars > 0:
        current_budget = current_max_reference_chars
    else:
        action = turn.get("action") or {}
        current_budget = max(
            len(turn.get("content", "")),
            len(action.get("code", "")),
            min_reference_chars * 2,
        )

    next_budget = max(min_reference_chars, current_budget // 2)
    if next_budget >= current_budget:
        return current_max_reference_chars
    return next_budget


def reduce_context_for_retry(context_messages: list[dict[str, str]],
                             current_max_reference_chars: int,
                             min_reference_chars: int,
                             turn: dict[str, Any]
                             ) -> tuple[list[dict[str, str]], int, int]:
    keep_prefix = seed_message_count(context_messages)
    if len(context_messages) > keep_prefix:
        return (
            [*context_messages[:keep_prefix], *context_messages[keep_prefix + 1:]],
            current_max_reference_chars,
            1,
        )

    next_reference_chars = shrink_reference_chars(
        current_max_reference_chars,
        min_reference_chars,
        turn,
    )
    if next_reference_chars == current_max_reference_chars:
        return context_messages, current_max_reference_chars, 0
    return context_messages, next_reference_chars, 0


def fit_request_to_budget(
    context_messages: list[dict[str, str]],
    turn: dict[str, Any],
    include_agent_name: bool,
    max_reference_chars: int,
    input_token_budget: int | None,
    chars_per_token_estimate: float,
    min_reference_chars: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]], int, int, int | None]:
    active_context = list(context_messages)
    active_max_reference_chars = max_reference_chars
    dropped_context_messages = 0

    while True:
        instruction = build_turn_instruction(
            turn,
            max_reference_chars=active_max_reference_chars,
            include_agent_name=include_agent_name,
        )
        request_messages = [*active_context, {
            "role": "user",
            "content": instruction,
        }]

        if input_token_budget is None:
            return (
                request_messages,
                active_context,
                active_max_reference_chars,
                dropped_context_messages,
                None,
            )

        estimated_prompt_tokens = estimate_messages_tokens(
            request_messages,
            chars_per_token_estimate,
        )
        if estimated_prompt_tokens <= input_token_budget:
            return (
                request_messages,
                active_context,
                active_max_reference_chars,
                dropped_context_messages,
                estimated_prompt_tokens,
            )

        next_context, next_reference_chars, dropped_for_retry = reduce_context_for_retry(
            active_context,
            active_max_reference_chars,
            min_reference_chars,
            turn,
        )
        if (next_context == active_context and
                next_reference_chars == active_max_reference_chars):
            return (
                request_messages,
                active_context,
                active_max_reference_chars,
                dropped_context_messages,
                estimated_prompt_tokens,
            )
        active_context = next_context
        active_max_reference_chars = next_reference_chars
        dropped_context_messages += dropped_for_retry


def is_context_length_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "maximum context length" in message
        or "reduce the length of the input prompt" in message
        or "parameter=input_tokens" in message
    )


def replay_trace(trace: dict[str, Any], client: OpenAI, model_name: str,
                 context_mode: str, max_reference_chars: int,
                 max_turns: int | None, temperature: float,
                 max_completion_tokens: int, seed: int | None,
                 delay_policy: str, constant_delay: float,
                 max_model_len: int | None,
                 context_safety_margin: int,
                 min_reference_chars: int,
                 chars_per_token_estimate: float,
                 progress_callback: Callable[[int, int, dict[str, Any]], None]
                 | None = None
                 ) -> dict[str, Any]:
    contexts: dict[str, list[dict[str, str]]] = {}
    turns = trace["llm_turns"]
    if max_turns is not None:
        turns = turns[:max_turns]

    results: list[dict[str, Any]] = []
    total_tool_sleep_s = 0.0
    solve_t0 = time.time()
    input_token_budget = compute_input_token_budget(
        max_model_len,
        max_completion_tokens,
        context_safety_margin,
    )

    total_turns = len(turns)

    for completed_turns, turn in enumerate(turns, start=1):
        key = context_key(turn, context_mode)
        if key not in contexts:
            agent = turn["agent"] if context_mode != "flattened" else None
            contexts[key] = make_seed_messages(trace["problem_statement"], agent)

        active_context = list(contexts[key])
        active_max_reference_chars = max_reference_chars
        dropped_context_messages = 0
        context_retry_count = 0
        estimated_prompt_tokens: int | None = None

        while True:
            (request_messages, active_context, active_max_reference_chars,
             dropped_for_budgeting,
             estimated_prompt_tokens) = fit_request_to_budget(
                context_messages=active_context,
                turn=turn,
                include_agent_name=context_mode == "flattened",
                max_reference_chars=active_max_reference_chars,
                input_token_budget=input_token_budget,
                chars_per_token_estimate=chars_per_token_estimate,
                min_reference_chars=min_reference_chars,
            )
            dropped_context_messages += dropped_for_budgeting
            request_prompt_chars = sum(
                len(msg["content"]) for msg in request_messages)
            t0 = time.time()
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=request_messages,
                    temperature=temperature,
                    max_completion_tokens=max_completion_tokens,
                    seed=seed,
                )
                t1 = time.time()
                break
            except Exception as exc:
                if not is_context_length_error(exc):
                    raise
                next_context, next_reference_chars, dropped_for_retry = reduce_context_for_retry(
                    active_context,
                    active_max_reference_chars,
                    min_reference_chars,
                    turn,
                )
                if (next_context == active_context and
                        next_reference_chars == active_max_reference_chars):
                    raise RuntimeError(
                        "Replay request exceeded the model context window and "
                        "context budgeting could not reduce it further."
                    ) from exc
                active_context = next_context
                active_max_reference_chars = next_reference_chars
                dropped_context_messages += dropped_for_retry
                context_retry_count += 1

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
            "estimated_prompt_tokens": estimated_prompt_tokens,
            "effective_max_reference_chars": active_max_reference_chars,
            "dropped_context_messages": dropped_context_messages,
            "context_retry_count": context_retry_count,
            "request_messages": request_messages,
            "response_text": response_text,
        })

        if progress_callback is not None:
            progress_callback(completed_turns, total_turns, turn)

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
            "max_model_len": max_model_len,
            "context_safety_margin": context_safety_margin,
            "min_reference_chars": min_reference_chars,
            "chars_per_token_estimate": chars_per_token_estimate,
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
            "num_turns_context_trimmed": sum(
                1 for item in results
                if item["dropped_context_messages"] > 0
                or item["effective_max_reference_chars"] != max_reference_chars
            ),
            "num_context_retry_events": sum(
                item["context_retry_count"] for item in results),
            "total_dropped_context_messages": sum(
                item["dropped_context_messages"] for item in results),
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
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--context-safety-margin",
                        type=int,
                        default=DEFAULT_CONTEXT_SAFETY_MARGIN)
    parser.add_argument("--min-reference-chars",
                        type=int,
                        default=DEFAULT_MIN_REFERENCE_CHARS)
    parser.add_argument("--chars-per-token-estimate",
                        type=float,
                        default=DEFAULT_CHARS_PER_TOKEN_ESTIMATE)
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

    if args.max_model_len is not None and args.max_model_len <= 0:
        raise ValueError("--max-model-len must be > 0")
    if args.context_safety_margin < 0:
        raise ValueError("--context-safety-margin must be >= 0")
    if args.min_reference_chars < 0:
        raise ValueError("--min-reference-chars must be >= 0")
    if args.chars_per_token_estimate <= 0:
        raise ValueError("--chars-per-token-estimate must be > 0")

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
        wait_for_server(
            args.base_url,
            args.server_timeout_s,
            server_proc=server_proc,
            server_log=args.server_log,
        )
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
            max_model_len=args.max_model_len,
            context_safety_margin=args.context_safety_margin,
            min_reference_chars=args.min_reference_chars,
            chars_per_token_estimate=args.chars_per_token_estimate,
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
