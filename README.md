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

For NCSA Delta with the `gpuA40x4` partition and `cudatoolkit/25.3_12.8`, install the CUDA 12.8 vLLM build. Current vLLM releases default to CUDA 12.9 wheels, but the project also publishes CUDA 12.8 wheels and explicitly supports selecting `cu128`.

## Install

Recommended: create a conda environment, install `uv`, then install this project with editable `pip`.

```bash
cd /Users/adityakunte/Desktop/MONET/hyperagent-replay
conda create -n ha-replay python=3.11 -y
conda activate ha-replay
python -m pip install -U pip uv
```

## NCSA Delta setup

Recommended model on a single 46 GB A40:

- `Qwen/Qwen2.5-Coder-14B-Instruct`

Requset gpu node
```bash
srun -A bewu-delta-gpu -p gpuA40x4 --gpus=1 --cpus-per-task=16 --mem=32g --time=02:30:00 --pty /bin/bash
```

```bash
module load gcc-native/13.2 cudatoolkit/25.3_12.8
cd /Users/adityakunte/Desktop/MONET/hyperagent-replay
conda activate ha-replay
python -m pip install -e .
UV_TORCH_BACKEND=cu128 uv pip install vllm
```

Notes:

- Use `UV_TORCH_BACKEND=cu128` on Delta to force the vLLM install to match the cluster CUDA 12.8 stack.
- Let the vLLM wheel provide its compatible PyTorch build; do not preinstall a different CUDA-specific PyTorch into the same environment.
- On a single A40, start with a moderate context limit such as `--max-model-len 32768` unless your traces require more.

If you want to launch the server yourself on Delta, a good starting command is:

```bash
vllm serve Qwen/Qwen2.5-Coder-14B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768
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

First 20 inputs after deterministic sorting:

```bash
ha-trace-batch-extract /path/to/hyperagent_runs \
  --output-dir /tmp/hyperagent-extracted \
  --offset 0 \
  --limit 20
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
  --model Qwen/Qwen2.5-Coder-14B-Instruct \
  --base-url http://127.0.0.1:8000/v1 \
  --output /tmp/trace.replay.json
```

Or launch `vllm serve` from the replay CLI:

```bash
ha-trace-replay /tmp/trace.extracted.json \
  --model Qwen/Qwen2.5-Coder-14B-Instruct \
  --launch-server \
  --port 8000 \
  --output /tmp/trace.replay.json
```

Batch replay:

```bash
ha-trace-batch-replay /tmp/hyperagent-extracted \
  --model Qwen/Qwen2.5-Coder-14B-Instruct \
  --base-url http://127.0.0.1:8000/v1 \
  --output-dir /tmp/hyperagent-replays
```

`ha-trace-batch-replay` also reads a repo-level skip list from `replay_skip_list.txt`. Put one extracted input basename per line to exclude it before `--offset` and `--limit` are applied.

Replay now includes context budgeting for long traces. It preserves the system prompt and issue statement, drops the oldest replay history first, and then shrinks the recorded-turn reference if a request would overflow the model context window. If the server still returns a context-length error, replay retries with a tighter budget instead of failing immediately.

First 20 extracted traces after deterministic sorting:

```bash
ha-trace-batch-replay /tmp/hyperagent-extracted \
  --model Qwen/Qwen2.5-Coder-14B-Instruct \
  --base-url http://127.0.0.1:8000/v1 \
  --output-dir /tmp/hyperagent-replays \
  --offset 0 \
  --limit 20
```

If you know the active vLLM context window, pass it to replay so budgeting can trim proactively:

```bash
ha-trace-batch-replay /tmp/hyperagent-extracted \
  --model Qwen/Qwen2.5-Coder-14B-Instruct \
  --launch-server \
  --port 8000 \
  --output-dir /tmp/hyperagent-replays \
  --max-model-len 32768 \
  --serve-arg=--max-model-len \
  --serve-arg=32768
```

Or batch replay with one shared launched server:

```bash
ha-trace-batch-replay /tmp/hyperagent-extracted \
  --model Qwen/Qwen2.5-Coder-14B-Instruct \
  --launch-server \
  --port 8000 \
  --output-dir /tmp/hyperagent-replays
```

Batch replay prints progress lines for each file and each completed turn. If you use `--launch-server`, add `--server-log /path/to/vllm.log` to keep the vLLM server logs out of the main terminal stream.

## Evaluate metrics

Single replay:

```bash
ha-trace-eval /path/to/raw_trajectory.json \
  --replay /tmp/trace.replay.json \
  --output /tmp/trace.eval.json
```

Batch evaluation from raw trajectories paired with a replay directory:

```bash
ha-trace-batch-eval /path/to/hyperagent_runs \
  --replay-dir /tmp/hyperagent-replays \
  --output-dir /tmp/hyperagent-evals
```

You can also batch-evaluate replay JSON files directly:

```bash
ha-trace-batch-eval /tmp/hyperagent-replays \
  --output-dir /tmp/hyperagent-evals
```

Batch evaluation writes one `*.eval.json` per input plus an `eval_manifest.json` that includes aggregate source and replay metrics across all successful inputs.

## SLO Derivation

You can derive proxy SLOs and resource groups directly from trajectory JSON files:

```bash
python3 derive_slos_and_resource_groups.py \
  --glob "trajectories/**/*.json" \
  --slo-class interactive
```

If you only want the human astropy files:

```bash
python3 derive_slos_and_resource_groups.py \
  --glob "trajectories/**/astropy**human.json" \
  --slo-class interactive
```

For a single file:

```bash
python3 derive_slos_and_resource_groups.py \
  --glob "trajectories/astropy__astropy-12907_human.json" \
  --slo-class interactive
```

If you want per-file reports from a batched run written to a specific directory:

```bash
python3 derive_slos_and_resource_groups.py \
  --glob "trajectories/**/astropy**human.json" \
  --slo-class interactive \
  --single-file-output-dir "trajectory_reports"
```

`--glob` controls which trajectory JSON files are read. `--single-file-output-dir` controls where the per-trajectory report files from batched runs are saved.

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
