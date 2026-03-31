"""Batch-replay extracted HyperAgent traces against a vLLM server."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from openai import OpenAI

from hyperagent_replay.replay import (
    DEFAULT_BASE_URL,
    kill_process_tree,
    launch_server,
    replay_trace,
    wait_for_server,
)
from hyperagent_replay.trace import load_trace_payload

GENERATED_SUFFIXES = (
    ".replay.json",
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


def output_name_for(path: Path) -> str:
    name = path.name
    if name.endswith(".extracted.json"):
        name = name[:-len(".extracted.json")]
    elif name.endswith(".json"):
        name = name[:-len(".json")]
    return f"{name}.replay.json"


def format_percent(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "0.0%"
    return f"{(100.0 * numerator / denominator):.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-replay HyperAgent traces against a vLLM server")
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="One or more extracted trace JSON files or directories",
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
        help="Directory where replay result JSON files will be written",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional path for the batch summary JSON",
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
        help="Replay at most N inputs after applying --offset",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Model id to request from the vLLM server",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument(
        "--context-mode",
        choices=["flattened", "per_agent"],
        default="per_agent",
    )
    parser.add_argument("--max-reference-chars", type=int, default=4000)
    parser.add_argument("--max-turns", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-completion-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--delay-policy",
        choices=["none", "constant", "heuristic"],
        default="heuristic",
    )
    parser.add_argument("--constant-delay", type=float, default=0.2)
    parser.add_argument(
        "--launch-server",
        action="store_true",
        help="Launch `vllm serve` once for the whole batch",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--server-timeout-s", type=float, default=180.0)
    parser.add_argument("--server-log", type=Path, default=None)
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

    server_proc = None
    results: list[dict[str, Any]] = []

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
        client = OpenAI(
            base_url=args.base_url,
            api_key=args.api_key,
            timeout=max(args.server_timeout_s, 30.0),
        )

        total_inputs = len(input_paths)
        completed_inputs = 0

        for file_position, (selection_index,
                            input_path) in enumerate(
                                zip(range(args.offset, args.offset + total_inputs),
                                    input_paths),
                                start=1):
            output_path = args.output_dir / output_name_for(input_path)
            trace = load_trace_payload(input_path)
            total_turns = len(trace["llm_turns"])

            print(
                "[progress] "
                f"completed files {completed_inputs}/{total_inputs} "
                f"({format_percent(completed_inputs, total_inputs)}); "
                f"starting file {file_position}/{total_inputs}: "
                f"{input_path.name} ({total_turns} turns)",
                flush=True,
            )

            if args.skip_existing and output_path.exists():
                results.append({
                    "selection_index": selection_index,
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "status": "skipped_existing",
                })
                completed_inputs += 1
                print(
                    "[progress] "
                    f"completed files {completed_inputs}/{total_inputs} "
                    f"({format_percent(completed_inputs, total_inputs)}); "
                    f"skipped existing {input_path.name}",
                    flush=True,
                )
                continue

            try:
                replay = replay_trace(
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
                    progress_callback=lambda completed_turns, total_turns,
                    turn, *, file_position=file_position, total_inputs=total_inputs,
                    name=input_path.name: print(
                        "[progress] "
                        f"file {file_position}/{total_inputs} "
                        f"{name}: turn {completed_turns}/{total_turns} "
                        f"({format_percent(completed_turns, total_turns)})",
                        flush=True,
                    ),
                )
                output_path.write_text(json.dumps(replay, indent=2))
                completed_inputs += 1
                results.append({
                    "selection_index": selection_index,
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "status": "ok",
                    "instance_id": replay.get("instance_id"),
                    "timing": replay.get("timing", {}),
                })
                print(
                    "[ok] "
                    f"completed files {completed_inputs}/{total_inputs} "
                    f"({format_percent(completed_inputs, total_inputs)}): "
                    f"{input_path} -> {output_path}",
                    flush=True,
                )
            except Exception as exc:
                results.append({
                    "selection_index": selection_index,
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "status": "error",
                    "error": str(exc),
                })
                print(f"[error] {input_path}: {exc}", flush=True)
    finally:
        if server_proc is not None:
            kill_process_tree(server_proc)

    manifest = {
        "phase": "replay",
        "model": args.model,
        "base_url": args.base_url,
        "output_dir": str(args.output_dir.resolve()),
        "ordering": "lexicographic absolute path",
        "offset": args.offset,
        "limit": args.limit,
        "num_discovered_inputs": len(all_input_paths),
        "num_inputs": len(input_paths),
        "num_ok": sum(item["status"] == "ok" for item in results),
        "num_skipped_existing": sum(item["status"] == "skipped_existing"
                                     for item in results),
        "num_error": sum(item["status"] == "error" for item in results),
        "results": results,
    }

    manifest_path = args.manifest
    if manifest_path is None:
        manifest_path = args.output_dir / "replay_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Saved replay manifest to {manifest_path}")


if __name__ == "__main__":
    main()
