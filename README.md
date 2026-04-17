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

Single-file replay now also writes a Continuum-compatible sidecar:

```text
/tmp/trace.scheduler_timestamps.json
```

This file stores one job history keyed by `instance_id`, with alternating
`Request_arrival_time` and `Request_departure_time` events so you can apply the
same completion-time analysis used by Continuum.

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

Batch replay also writes a merged file:

```text
/tmp/hyperagent-replays/scheduler_timestamps
```

This merged JSON can be consumed directly by Continuum-style analyzers that
expect the `scheduler_timestamps` format.

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

## Reuse-aware replay

If you want to run a single trajectory on vLLM while reusing exact repeated stages, use:

```bash
ha-trace-replay-reuse /tmp/trace.extracted.json \
  --model Qwen/Qwen2.5-Coder-14B-Instruct \
  --base-url http://127.0.0.1:8000/v1 \
  --slo-class interactive \
  --output /tmp/trace.reuse.replay.json
```

This command stays on top of `vllm serve`: it does not modify vLLM internals. It replays the trajectory turn-by-turn, and when it sees an exact repeated stage it reuses the prior assistant response instead of sending a new request to vLLM.

Reuse-aware replay prints a progress line after every completed turn, for example `Completed 37/116 turns`. These counts include cache-hit turns, so they reflect trajectory progress rather than only vLLM requests.

If you pass SLO targets, reuse-aware replay can also tighten request budgets on turns that are predicted to be at risk of missing the target. The cache key and reuse semantics stay exact-repeat only; SLOs change budgeting, not cache identity.

```bash
ha-trace-replay-reuse /tmp/trace.extracted.json \
  --model Qwen/Qwen2.5-Coder-14B-Instruct \
  --base-url http://127.0.0.1:8000/v1 \
  --slo-class interactive \
  --request-target-s 10 \
  --episode-target-s 1800 \
  --slo-policy budget \
  --output /tmp/trace.reuse.replay.json
```

The reuse-aware replay output adds:

- per-turn `cache_hit` and `executed_on_vllm`
- per-turn `cache_key` and `cache_source_turn_index`
- per-turn resource-group annotations
- per-turn SLO slack, selected budget mode, and effective request budgets
- timing counters for executed versus avoided vLLM requests

`scheduler_timestamps` only includes real vLLM requests. Cache hits are not added to the scheduler trace.

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

## SLO Attainment

You can evaluate whether replayed requests meet per-agent request SLOs and whether the full trajectory meets an episode SLO:

```bash
ha-trace-slo-report /tmp/trace.reuse.replay.json \
  --request-metric request_latency_s \
  --request-target-s 10 \
  --episode-metric wall_solve_time_s \
  --episode-target-s 1800
```

To set different request SLOs for different agents, pass a JSON file such as:

```json
{
  "Planner": 8.0,
  "Inner-Navigator-Assistant": 12.0
}
```

Then run:

```bash
ha-trace-slo-report /tmp/trace.reuse.replay.json \
  --request-metric stage_latency_s \
  --agent-request-targets /tmp/agent_targets.json \
  --episode-metric service_span_s \
  --episode-target-s 1500 \
  --output /tmp/trace.slo_report.json
```

`request_latency_s` measures only executed vLLM request time. `stage_latency_s` adds synthetic tool sleep. `wall_solve_time_s` is the end-to-end replay time, while `service_span_s` is the time from first executed request arrival to last executed request departure.

## Empirical SLOs

You can derive empirical per-agent request SLOs directly from replay JSON files. This groups executed requests by agent and computes percentiles over `request_latency_s` or `stage_latency_s`.

```bash
ha-trace-derive-empirical-slos \
  --glob "replays-two/*.replay.json" \
  --request-metric request_latency_s \
  --output /tmp/empirical_slos.json \
  --interactive-targets-output /tmp/agent_targets_interactive.json \
  --batch-targets-output /tmp/agent_targets_batch.json
```

The script recommends:

- interactive targets = per-agent `p95`
- batch targets = per-agent `p99`

## Reuse Analysis

You can analyze baseline versus reuse replay runs and write a full artifact bundle with pairwise comparisons, cache-hit breakdowns, and SVG plots:

```bash
ha-trace-analyze-reuse \
  --baseline-glob "replays-two/*.baseline.replay.json" \
  --reuse-glob "replays-two/*.reuse.replay.json" \
  --output-dir reuse-analysis
```

This writes:

- `reuse-analysis/reuse_analysis_report.json`
- `reuse-analysis/pairwise_comparison.csv`
- `reuse-analysis/throughput_by_instance.csv`
- `reuse-analysis/throughput_by_agent.csv`
- `reuse-analysis/throughput_by_tool_signature.csv`
- `reuse-analysis/cache_hits_by_agent.csv`
- `reuse-analysis/cache_hits_by_tool_signature.csv`
- `reuse-analysis/cache_hits_by_agent_tool_signature.csv`
- `reuse-analysis/savings_by_agent.csv`
- `reuse-analysis/savings_by_tool_signature.csv`
- `reuse-analysis/top_exact_repeats.csv`
- `reuse-analysis/jct_comparison.svg`
- `reuse-analysis/request_comparison.svg`
- `reuse-analysis/cache_hits_by_agent.svg`
- `reuse-analysis/cache_hits_by_tool_signature.svg`
- `reuse-analysis/saved_lm_time_by_agent.svg`
- `reuse-analysis/saved_lm_time_by_tool_signature.svg`

Use `--reuse-glob` by itself if you only want cache-hit analysis on reuse runs and do not need baseline-vs-reuse comparisons.

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

In this repo, resource groups are used on top of vLLM. They help classify replay stages, report where repeated work appears, and summarize cache hits by stage type. They do not change vLLM's internal scheduler in this single-trajectory workflow.

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
