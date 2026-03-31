"""Batch-extract structured traces from HyperAgent trajectory JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from hyperagent_replay.trace import extract_trace, load_raw_trace

GENERATED_SUFFIXES = (
    ".extracted.json",
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


def output_name_for(path: Path) -> str:
    return f"{path.stem}.extracted.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-extract structured traces from HyperAgent logs")
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="One or more raw HyperAgent JSON files or directories",
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
        help="Directory where extracted trace JSON files will be written",
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
    args = parser.parse_args()

    input_paths = discover_input_paths(
        args.inputs,
        pattern=args.pattern,
        recursive=not args.non_recursive,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for input_path in input_paths:
        output_path = args.output_dir / output_name_for(input_path)
        if args.skip_existing and output_path.exists():
            results.append({
                "input_path": str(input_path),
                "output_path": str(output_path),
                "status": "skipped_existing",
            })
            continue

        try:
            trace = extract_trace(load_raw_trace(input_path))
            output_path.write_text(json.dumps(trace, indent=2))
            results.append({
                "input_path": str(input_path),
                "output_path": str(output_path),
                "status": "ok",
                "instance_id": trace.get("instance_id"),
                "summary": trace.get("summary", {}),
            })
            print(f"[ok] {input_path} -> {output_path}")
        except Exception as exc:
            results.append({
                "input_path": str(input_path),
                "output_path": str(output_path),
                "status": "error",
                "error": str(exc),
            })
            print(f"[error] {input_path}: {exc}")

    manifest = {
        "phase": "extract",
        "output_dir": str(args.output_dir.resolve()),
        "num_inputs": len(input_paths),
        "num_ok": sum(item["status"] == "ok" for item in results),
        "num_skipped_existing": sum(item["status"] == "skipped_existing"
                                     for item in results),
        "num_error": sum(item["status"] == "error" for item in results),
        "results": results,
    }

    manifest_path = args.manifest
    if manifest_path is None:
        manifest_path = args.output_dir / "extract_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Saved extraction manifest to {manifest_path}")


if __name__ == "__main__":
    main()
