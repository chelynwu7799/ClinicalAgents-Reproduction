# ClinicalAgents — Reproduction Work (FTEC5660)

Reproduction and modification of **ClinicalAgents: Multi-Agent Orchestration for
Clinical Decision Making with Dual-Memory** (KDD 2026, arXiv:2603.26182;
[original repo](https://github.com/ZhuohanGe/ClinicalAgents-Code), which is
missing `app_config.py` and `experience_memory.py` — both reconstructed here,
see *Reconstructed files* below).

This repository contains:
- the original authors' code (unchanged except one bug fix, see commits),
- the two reconstructed missing modules,
- experiment runners and all raw results (`results/`), logs, and per-case
  decision trajectories (`state_memories*/`).

## What was reproduced

| | Paper | This repo |
|---|---|---|
| Full system average on MedChain | 0.5107 (Table 3) | 0.349 (30-case subset, DeepSeek backbone — see report §4 for comparability statement) |
| Modification (experience memory off) | +Orchestrator 0.4962 (ablation table) | see `results/final_analysis.json` (regenerate with `python3 analysis.py`) |

## Setup

```bash
pip install openai tqdm scikit-learn
```

Data (not committed, ~50 MB):
download `merged_cases.json` from the MedChain dataset
(https://huggingface.co/datasets/ljwztc/MedChain) into `data/`.
Optionally build the experience-memory pool:

```bash
python3 data/build_cdc.py   # writes data/experience/cdc_cases.json
```

Configuration: put your key in `.env` (`DEEPSEEK_API_KEY=sk-...`).
Models, temperatures and MCTS parameters are in `app_config.py`
(backbone/judge: deepseek-v4-pro, rollout: deepseek-flash, T=0, η=4).
Any OpenAI-compatible endpoint works — this is the documented controlled
variable vs the paper's GPT-5.2 + Gemini-2.5-Pro.

## Running the experiments

```bash
bash run_experiment.sh A 42 30        # config A: full system, seed 42, 30 cases
bash run_experiment.sh B 42 30        # config B: modification (experience memory off)
bash run_experiment.sh B 42 16 14     # resume a truncated run at dataset index 14
python3 analysis.py                   # regenerate results/final_analysis.json from raw runs
```

Outputs land in `results/benchmark_results_<config>_s<seed>[_skip<N>].json`.
A single case makes 8–138 LLM calls (mean ≈ 60–107 depending on config);
budget accordingly.

## Reconstructed files

- `app_config.py` — 36 configuration symbols reverse-engineered from import
  sites across the authors' code. Values from the paper are annotated; the rest
  are marked as reconstructed (MCTS reward weights are *not* stated in any
  released material).
- `experience_memory.py` — the Dual-Memory experience module. Implements the
  interface the orchestrator calls (`.freeze()`, `.available`,
  `.retrieve(state)`), TF-IDF retrieval over a historical-case DB, and graceful
  degradation to the "no experience memory" ablation configuration when the data
  files are absent. The original was never committed by the authors.
- Bug fix in `utils.py`: removed a function-local `import re` that caused
  `UnboundLocalError` on a subset of cases (see commit).

## Layout

```
main.py                  # benchmark driver (authors')
med_orchestrator.py      # MCTS orchestrator (authors')
mcts_*.py, agents/       # search + specialist agents (authors')
app_config.py            # RECONSTRUCTED configuration
experience_memory.py     # RECONSTRUCTED experience memory
run_experiment.sh        # one-command experiment runner
analysis.py              # statistics: regenerates results/final_analysis.json
data/                    # MedChain data + experience pool (git-ignored)
results/                 # all raw results, logs, final analysis
state_memories*/         # per-case decision trajectories
REPORT.md                # the full reproducibility report
```

## License / attribution

Authors' code is from ZhuohanGe/ClinicalAgents-Code (no license file was
present at clone time; treat as research-use only). MedChain data is CC-BY-NC-4.0.
Reconstructed files and all experiment results: this reproduction work.
No API keys are committed; `.env` is git-ignored.
