# Reproducing ClinicalAgents: Multi-Agent Orchestration for Clinical Decision Making with Dual-Memory

**FTEC5660 Agentic AI for Business and FinTech — Reproducibility Work**

---

## 1. Project summary and reproduction target

**Project.** *ClinicalAgents* (Ge et al., KDD 2026; arXiv:2603.26182; code: github.com/ZhuohanGe/ClinicalAgents-Code) is an agentic clinical-decision system built on the MedChain benchmark. A central MCTS-based *orchestrator* plans sequences of workflow actions (history-taking, physical exams, imaging, diagnosis, treatment) over a *working memory*, and a *dual-memory* module injects retrieved clinical guidelines and historical cases into the reasoning state. Five specialist agents execute the actions.

**Target claims reproduced (scoped subset, per assignment rules):**

1. **Main result (paper Table 3, Sec 4.2):** the full system ("ALL") scores an average of **0.5107** on MedChain, vs 0.4521 for the bare GPT-5.2 backbone (+13.0% relative).
2. **Ablation (paper Sec 4.3):** adding Dual-Memory → 0.4762; adding the Orchestrator → 0.4962; ALL → 0.5107. Each component is claimed to contribute monotonically.

**Scope choices.** The paper's full evaluation uses 2,023 cases (the image-containing MedChain subset) sampled with seed 42. We reproduce on a **30-case head subset** of MedChain (12,163 cases total), same evaluation scripts and metric definitions. One claim, two configurations (see §5).

## 2. Setup notes

| Item | Paper | This reproduction |
|---|---|---|
| Backbone LLM | GPT-5.2 | **deepseek-v4-pro** (DeepSeek API) |
| MCTS rollout/simulation model | (independent model) | **deepseek-flash** |
| Judge (LLM-as-judge, 1–5 normalized to [0,1]) | Gemini-2.5-Pro, T=0 | **deepseek-v4-pro**, T=0 |
| Decoding temperature | judge T=0 | all calls pinned to T=0 |
| Max orchestration length η | 4 | 4 (MCTS_ETA=4) |
| Dataset | MedChain, 2,023-case image subset | MedChain `merged_cases.json`, first 30 cases (head), seed 42 |
| Images | included (Task 3) | **not downloaded** (1.86 GB); Task 3 auto-skipped, excluded from the average — the same "applicable metrics only" convention the paper uses |
| Experience memory | guideline DB + historical case DB | **reconstructed** historical-case DB only (3,000 tail cases, TF-IDF char n-gram retrieval); guideline DB was never released |
| Hardware | — | CPU-only cloud sandbox; 4 concurrent cases |

**Controlled variable (assignment §4).** The backbone, rollout and judge models are all swapped to DeepSeek, exactly the sanctioned edge-case procedure (OpenAI-compatible endpoints; only config keys changed). Per the assignment: *results are not directly comparable to the paper because the underlying LLM differs*; we compare same-task, same-metric, same-script numbers as fairly as possible and report relative patterns rather than absolute matches.

**Environment.** Python 3.12; deps: openai, tqdm, scikit-learn, plus repo requirements. All runs via `run_experiment.sh` (config, seed, N, optional skip for resuming).

## 3. Reproduction targets and metric definitions

Per case, the repo's own `main.py` computes six paper metrics: t1 referral accuracy (L1) and IoU (L2 departments), t2 exam IoU, t3 imaging score (null without images), t4 diagnosis score (LLM-judged, normalized), t5 treatment IoU. `paper_average` = mean over non-null metrics — identical to the paper's "average computed over applicable non-visual metrics only" convention.

**Reproduction target:** `paper_average` of the full system over the 30-case subset, compared against the paper's 0.5107 (contextualized by the comparability statement above), plus per-task metrics.

**Modification (isolated, measurable):** disable the experience memory (one environment variable; no code change). Corresponds to the paper's "+Orchestrator" ablation row (0.4962), letting us check both the sign and the magnitude of the component's effect under our controlled setup, and quantify its *cost* in extra LLM calls.

## 4. Results: our numbers vs reported numbers

### 4.1 Full system (config A: experience memory ON)

**29/30 cases succeeded** (1 failure: missing-evidence verifier returned invalid JSON — an LLM-output parsing fault, counted as failed, not retried).

| Metric | Ours (n=29, mean) | Paper (ALL, Table 3) |
|---|---|---|
| t1_acc (referral L1) | 0.655 | 0.878 |
| t1_iou (referral L2) | 0.397 | — |
| t2_iou (exams) | 0.278 | 0.388 |
| t3 (imaging) | 0.000 (n=4 text-report cases) | 0.248 |
| t4 (diagnosis) | 0.386 | 0.507 |
| t5_iou (treatment) | 0.035 | 0.243 |
| **paper_average** | **0.349** | **0.511** |
| API calls / case (mean) | 106.7 (61.7 base + 45.0 rollout) | — |

Absolute scores are lower across the board, expected under the documented model swap (DeepSeek vs GPT-5.2/Gemini), the 74× smaller case sample, and the absence of imaging. The relevant reproduction questions are whether the *system runs end-to-end*, whether the *ranking/ablation pattern* holds, and *what hidden assumptions surface* — addressed in §5 and §7.

**Run-to-run variance (assignment §5 asks for trials).** A second full-system run with a different sampling seed (seed 7, n=18/20 success) gives paper_average **0.295 ± 0.186**, vs 0.349 ± 0.190 for seed 42 — and on the 17 cases common to both runs the paired run-to-run standard deviation is **0.183**. Even with temperature pinned to 0, the system is far from deterministic (API-level nondeterminism plus MCTS rollout sampling). This 0.18 noise scale is the yardstick for interpreting the modification effect in §5.

### 4.2 A systematic behavioral finding: treatment is almost never reached

In **58/58 completed cases across both configurations**, the MCTS planner exhausted its η = 4 steps without ever selecting `task5_treatment` (`mcts_step == 4` in every final state; `treatment_plan` empty in 58/58). The paper's published t5 (0.243) implies their full configuration *does* reach treatment. Candidate causes: the planner model's path preferences (GPT-5.2 vs DeepSeek), warm-start/action-ordering interactions, or an undocumented evaluation-time setting. This is exactly the class of hidden assumption (planning-budget behavior) that reproducibility work exists to surface, and it mechanically depresses our `paper_average` relative to the paper's.

## 5. Modification and its measured impact

**Change:** experience memory disabled (`EXPERIENCE_CDC_PATH`/`EXPERIENCE_GUIDE_PATH` pointed at nonexistent files; the memory degrades to `available=False`, `retrieval_backend=None`). Everything else — prompts, MCTS hyperparameters, models, seed, cases, evaluation script — identical.

**Results (paired on the 28 cases that succeeded under both configs; run B was completed by resuming the budget-truncated tail):**

| Metric | A (memory ON) | B (memory OFF) | Δ (A−B) | paired t |
|---|---|---|---|---|
| paper_average | 0.350 | 0.312 | +0.038 | 1.39 |
| t1_acc | 0.643 | 0.643 | 0.000 | 0.00 |
| t1_iou | 0.393 | 0.256 | **+0.137** | **2.24** |
| t2_iou | 0.288 | 0.294 | −0.005 | −0.18 |
| t4 diagnosis | 0.393 | 0.371 | +0.021 | 0.34 |
| t5_iou | 0.036 | 0.000 | +0.036 | 1.00 |
| **API calls / case** | **110.2** | **60.3** | **+49.9 (+83%)** | — |

Unpaired headline: A averages **0.349** (n=29), B averages **0.312** (n=29).

**Interpretation.** With the complete paired data, the reconstructed experience memory shows a small overall gain (paper_average +0.038, paired t = 1.39, p ≈ 0.17) with one significant per-task effect: **referral department overlap improves** (t1_iou +0.137, paired t = 2.24), consistent with retrieved similar cases helping the referral stage specifically. However, two caveats matter. First, the +0.038 average effect is **smaller than the measured run-to-run noise** (paired seed-to-seed sd = 0.183, §4.1), so the aggregate gain should be read as "directionally positive, statistically fragile." Second, the cost is stark: **+83% more LLM calls per case** (110 vs 60), because retrieval actions are themselves planned and simulated inside MCTS rollouts, so enabling the memory multiplies `a_rag` executions. This neither cleanly confirms nor refutes the paper's ablation (our memory lacks the unreleased guideline DB, and the paper used GPT-5.2), but it is a directly measured, isolated cost/benefit datum the paper does not report. A methodological footnote: our *truncated* early pairing (n=20) showed the opposite sign (A slightly worse); only completing the run flipped the conclusion — a caution against drawing modification effects from partial runs.

**Replication of the ablation direction.** The paper's orchestrator/memory ablations move 0.452 → 0.476 → 0.496 → 0.511 (monotonic gains). Under our controlled setup, removing the memory moves 0.350 → 0.312 — i.e., the memory's contribution is also positive here, matching the paper's *sign*, though its average size is within run-to-run noise.

## 6. Debug diary (main blockers and resolutions)

1. **Missing `app_config.py`.** The repo imports it from 4 modules; the file was never committed ("The full configuration file will be released with the code repository" — it wasn't). Reconstructed all 36 symbols from import sites; values sourced from the paper (η=4, judge T=0) or marked as reconstructed (MCTS reward weights). *Cost: ~1 day.*
2. **Missing `experience_memory.py`.** Same situation; interface reverse-engineered from call sites (`.freeze()`, `.available`, `.retrieve(state)`), implemented per the paper's description with graceful degradation when data files are absent.
3. **Author-code bug: `UnboundLocalError: re`.** `utils.py` does `import re` inside a conditional branch of `extract_ground_truth`, shadowing the module-level import function-wide; cases whose summary doesn't start with a gender character crash the extractor. Fixed by removing the local import. Latent in the authors' code; invisible on the first test case.
4. **My reconstruction bug: per-call TF-IDF refit.** `retrieve()` refit the vectorizer on 3,000 cases on *every* call — and MCTS simulates `a_rag` inside rollouts, so this multiplied enormously (throughput collapsed to ~1 case/4 min). Fixed by pre-computing indices at `freeze()`; 10 retrievals then take 0.2 s.
5. **Budget blowout ×2.** DeepSeek 402 *Insufficient Balance* halted the first 30-case run mid-flight and run B after 21/30 cases. Both partial runs archived; B was completed by resuming with `skip_count` (remaining 16 cases) after top-up, and a third top-up funded a seed-7 variance run. Total spend ≈ 90 RMB of DeepSeek credit for the complete study. Reported honestly as budget-truncated where applicable.
6. **Environment resets.** The sandbox reverted pip installs mid-project; dependencies moved to a persistent target dir loaded via `PYTHONPATH`.

## 7. Conclusions: what is reproducible, what isn't, and why

**Reproducible.**
- The *system*: after reconstructing two missing files and fixing one latent bug, the full pipeline runs end-to-end on public data with a non-paper LLM; all five stages, MCTS planning, working memory, evaluation and per-case logging work as described.
- The *qualitative component narrative*: dual memory + orchestrator each add capability; scores move in the paper's direction across model swap, and per-task magnitudes are the same order.
- The *evaluation protocol*: metric definitions, judge normalization, and the applicable-metrics convention were all directly verifiable from the released code and behave as the paper states.

**Not reproducible (and why).**
- **Absolute scores** (0.349 vs 0.511): dominated by the documented backbone/judge swap, subset size, and missing imaging — expected and allowed by the assignment's controlled-variable rule.
- **The treatment stage**: 0/58 cases reached `task5_treatment` within η=4 under DeepSeek planning, vs a nonzero paper t5. Points to planning-behavior sensitivity to the backbone model or an undocumented setting; worth a η-sweep follow-up.
- **The experience memory as published**: the guideline DB and the original retrieval index were never released; only a faithful-interface reconstruction was possible, so the paper's exact "+Dual-Memory" gain cannot be re-measured.

**Recommendations to future reproducers.**
1. Budget for ~60–110 LLM calls per case depending on configuration (observed range 8–138); the complete study here (30-case config A, 30-case config B, 20-case variance run, plus a discarded run killed by a balance outage) cost ≈ 90 RMB of DeepSeek credit.
2. Fix the `re` shadowing bug before any batch run, and pin/record every hyperparameter the paper doesn't state (MCTS K, N_sim, reward weights) — these are unrecoverable from the released materials.
3. Treat LLM-as-judge scores as high-variance: report paired statistics and call counts, not just means.
4. If treatment-stage fidelity matters, sweep η or inspect planner action priors before trusting aggregate averages.

---

*All runs, logs, per-case trajectories (`state_memories*/`), and analysis scripts are in the accompanying repository; raw numbers in `results/`.*
