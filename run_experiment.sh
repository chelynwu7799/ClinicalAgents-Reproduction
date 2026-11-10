#!/usr/bin/env bash
# Reproducible experiment runner for ClinicalAgents reproduction work.
#
# Usage: bash run_experiment.sh <A|B> <seed> <n_cases> [skip_count]
#   A = full system (MCTS + working memory + experience memory)  [paper: ALL]
#   B = modification: experience memory disabled                  [paper: +Orchestrator]
#   skip_count: skip the first N dataset items (for resuming a
#   budget-truncated run on the remaining cases)
#
# Requirements:
#   - .env file in repo root with DEEPSEEK_API_KEY=sk-...
#   - data/merged_cases.json (MedChain, see README)
#   - data/experience/cdc_cases.json (for config A)
# Outputs:
#   results/benchmark_results_<config>_s<seed>[_skip<N>].json
#   results/usage_summary_<config>_s<seed>[_skip<N>].json
set -euo pipefail

CONFIG="${1:-A}"
SEED="${2:-42}"
N="${3:-30}"
SKIP="${4:-0}"

# Load API key from .env without printing it.
export DEEPSEEK_API_KEY="$(grep '^DEEPSEEK_API_KEY=' .env | cut -d= -f2-)"

# Persistent third-party deps (sandbox-safe; on a normal machine: pip install openai tqdm).
export PYTHONPATH="/mnt/agents/pylibs${PYTHONPATH:+:$PYTHONPATH}"

# Config B: point the experience memory at a nonexistent file -> degrades to
# "no experience memory" (the paper's +Orchestrator ablation configuration).
if [ "$CONFIG" = "B" ]; then
  export EXPERIENCE_CDC_PATH="/nonexistent/cdc_cases.json"
  export EXPERIENCE_GUIDE_PATH="/nonexistent/guidelines.json"
fi

mkdir -p results

TAG="${CONFIG}_s${SEED}"
if [ "$SKIP" != "0" ]; then
  TAG="${TAG}_skip${SKIP}"
fi

python3 - <<PYEOF
from main import run_benchmark
run_benchmark(
    dataset_path="data/merged_cases.json",
    max_samples=${N},
    seed=${SEED},
    skip_count=${SKIP},
    sample_position="head",
)
PYEOF

cp benchmark_results.json "results/benchmark_results_${TAG}.json"
cp orchestration_usage_summary.json "results/usage_summary_${TAG}.json"
echo "Saved: results/benchmark_results_${TAG}.json"
