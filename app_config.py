"""Central configuration for ClinicalAgents benchmark runs.

NOTE: This file is MISSING from the official repository
(https://github.com/ZhuohanGe/ClinicalAgents-Code). It was reconstructed
for reproducibility work by inspecting every import site across
main.py / utils.py / med_orchestrator.py / mcts_config.py.

Reconstruction date: 2026-11-10
Sources for values:
- Paths / dataset: local environment (MedChain data from HF ljwztc/MedChain).
- LLM: paper Sec 4.1.3 uses GPT-5.2 backbone + Gemini-2.5-Pro judge;
  here both are swapped to DeepSeek (documented controlled variable):
  backbone = deepseek-v4-pro, rollout/simulation = deepseek-flash,
  judge    = deepseek-v4-pro.
- MCTS_ETA = 4: paper Sec 4.1.3 ("maximum orchestration length eta = 4").
- MCTS_ROLLOUT_CALL_CAP_* = 0: med_orchestrator.py docstring says zero is
  "the documented paper-mode default: no artificial rollout cap".
- MCTS_{K, N_SIM, D, ALPHA, BETA, GAMMA, GAMMA_D, LAMBDA_PUCT, DELTA_MAX}:
  NOT stated in the paper's released materials; reconstructed to standard
  MCTS / Eq.(4)-style values. Recorded here explicitly because they are a
  hidden-assumption risk for reproduction.
- Temperature: judge temperature = 0 per paper; backbone also pinned to 0
  to reduce run-to-run variance (stochasticity then comes from MCTS
  rollouts / API nondeterminism and is reported across seeds).
"""

import os

# ---------------------------------------------------------------- paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PROMPTS_DIR = os.path.join(PROJECT_ROOT, "prompts")
CASES_DIR = os.path.join(PROJECT_ROOT, "state_memories")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
IMG_BASE_DIR = os.path.join(DATA_DIR, "images")
DEFAULT_DATASET_PATH = os.path.join(DATA_DIR, "merged_cases.json")

EXPERIENCE_GUIDE_PATH = os.getenv(
    "EXPERIENCE_GUIDE_PATH", os.path.join(DATA_DIR, "experience", "guidelines.json")
)
EXPERIENCE_CDC_PATH = os.getenv(
    "EXPERIENCE_CDC_PATH", os.path.join(DATA_DIR, "experience", "cdc_cases.json")
)

# ---------------------------------------------------------------- runtime
DEBUG = False
BATCH_SIZE = 4                      # concurrent cases in run_benchmark
BENCHMARK_MAX_SAMPLES = None        # None -> full dataset (override via config)
BENCHMARK_SAMPLE_POSITION = "head"  # "head" | "tail" | "random"
BENCHMARK_SEED = 42
FORCE_FINALIZATION = False
MAX_RECHECK_PER_CASE = 3
DOCLENS_AGGREGATION = "f1"

# ---------------------------------------------------------------- LLM
_DEEPSEEK_KEY = os.getenv("DEEPSEEK_API_KEY", "")
_DEEPSEEK_BASE = "https://api.deepseek.com"

LLM_CONFIG = {
    "api_key": _DEEPSEEK_KEY,
    "base_url": _DEEPSEEK_BASE,
    "model": "deepseek-v4-pro",     # backbone (paper: GPT-5.2)
}
JUDGE_CONFIG = {
    "api_key": _DEEPSEEK_KEY,
    "base_url": _DEEPSEEK_BASE,
    "model": "deepseek-v4-pro",     # judge (paper: Gemini-2.5-Pro, T=0)
}
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 8192
JUDGE_MAX_TOKENS = 2048
API_TIMEOUT = 180

# MCTS planning (orchestrator policy) and rollout (simulation) endpoints.
# med_orchestrator enforces that the rollout model differs from LL_CONFIG's.
MCTS_PLANNING_CONFIG = {
    "api_key": _DEEPSEEK_KEY,
    "base_url": _DEEPSEEK_BASE,
    "model": "deepseek-v4-pro",
}
MCTS_ROLLOUT_CONFIG = {
    "api_key": _DEEPSEEK_KEY,
    "base_url": _DEEPSEEK_BASE,
    "model": "deepseek-flash",
}

# ---------------------------------------------------------------- MCTS
# eta=4 from paper Sec 4.1.3; caps=0 = paper-mode default (see docstring).
MCTS_ETA = 4
MCTS_K = 3              # candidate action set size (Top-K), reconstructed
MCTS_N_SIM = 2          # rollouts per candidate action, reconstructed
MCTS_D = 3              # rollout depth, reconstructed
MCTS_ROLLOUT_CALL_CAP_PER_SEARCH = 0
MCTS_ROLLOUT_CALL_CAP_TOTAL = 0
MCTS_WARM_START = True
# Reward shaping (Eq. 4-6 in paper); values reconstructed, see file docstring.
MCTS_ALPHA = 1.0
MCTS_BETA = 1.0
MCTS_GAMMA = 0.1
MCTS_GAMMA_D = 0.9
MCTS_LAMBDA_PUCT = 1.4
MCTS_DELTA_MAX = 0.5
