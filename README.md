# hyperagent-replay

`hyperagent-replay` is a standalone Python project for:

- parsing raw HyperAgent trajectory JSON files
- extracting structured LLM turns and tool calls
- replaying those turns against a running `vllm serve` endpoint
- measuring solve-time and per-turn latency metrics
- running the workflow over one file or many files

## Python version

Use Python `3.10`, `3.11`, or `3.12`.

Recommended: Python `3.11`.

## Dependencies

This project installs:

- `openai>=1.30.0`

If you want the replay CLI to launch a server with `--launch-server`, you also need `vllm` available in the environment.

## Install

With `uv`:

```bash
cd /Users/adityakunte/Desktop/MONET/hyperagent-replay
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

With standard `venv`:

```bash
cd /Users/adityakunte/Desktop/MONET/hyperagent-replay
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

## Phase 1: Extract structured trace JSON

Single file:

```bash
ha-trace-extract /path/to/raw_trajectory.json --output /tmp/trace.extracted.json
```

Batch:

```bash
ha-trace-batch-extract /path/to/hyperagent_runs \
  --output-dir /tmp/hyperagent-extracted
```

Each extracted file contains:

- `instance_id`
- `problem_statement`
- `summary`
- `events`
- `llm_turns`

## Phase 2: Replay extracted traces against vLLM

Use a running server:

```bash
ha-trace-replay /tmp/trace.extracted.json \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --base-url http://127.0.0.1:8000/v1 \
  --output /tmp/trace.replay.json
```

Or launch `vllm serve` from the replay CLI:

```bash
ha-trace-replay /tmp/trace.extracted.json \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --launch-server \
  --port 8000 \
  --output /tmp/trace.replay.json
```

Batch replay:

```bash
ha-trace-batch-replay /tmp/hyperagent-extracted \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --base-url http://127.0.0.1:8000/v1 \
  --output-dir /tmp/hyperagent-replays
```

Or batch replay with one shared launched server:

```bash
ha-trace-batch-replay /tmp/hyperagent-extracted \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --launch-server \
  --port 8000 \
  --output-dir /tmp/hyperagent-replays
```

## Evaluate metrics

Single replay:

```bash
ha-trace-eval /path/to/raw_trajectory.json \
  --replay /tmp/trace.replay.json \
  --output /tmp/trace.eval.json
```

## What gets measured

Source metrics from the trajectory:

- number of LLM turns
- number of action turns
- number of final-answer turns
- response counts by agent
- action languages
- top tool names

Replay metrics from vLLM:

- `wall_solve_time_s`
- `lm_only_solve_time_s`
- `synthetic_tool_sleep_s`
- `num_replayed_turns`
- average, median, p95, and p99 request latency
- total prompt tokens
- total completion tokens

## Notes

- The replay is approximate. It preserves turn order and context growth, not the exact original HyperAgent outputs.
- In `per_agent` mode, each HyperAgent role gets a separate replay context.
- In `flattened` mode, all turns share one context.
