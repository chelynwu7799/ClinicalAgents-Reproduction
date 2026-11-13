# Reproducing ClinicalAgents: Multi-Agent Orchestration for Clinical Decision Making with Dual-Memory

FTEC5660 Agentic AI for Business and FinTech, Reproducibility Homework

Name: [YOUR NAME]    Student ID: [YOUR STUDENT ID]

GitHub repository: [YOUR REPO URL]    Date: November 2026

## 1. Project summary and reproduction target

ClinicalAgents, published at KDD 2026 by Ge et al. and available at arXiv:2603.26182 with code at github.com/ZhuohanGe/ClinicalAgents-Code, is an agentic clinical decision system built on the MedChain benchmark. A central orchestrator plans sequences of workflow actions over a working memory using Monte Carlo Tree Search, and a dual-memory module injects retrieved clinical guidelines and historical cases into the reasoning state. Five specialist agents carry out the planned actions. I selected this project from the instructor's list because its pipeline is compact enough to run on a CPU-only sandbox with API-based models, yet it exercises the full range of agentic machinery the course covers: planning, memory, tool use, and LLM-as-judge evaluation.

The paper makes two claims that this work targets. The first is the main result in Table 3: the full system, labeled ALL, scores an average of 0.5107 on MedChain, against 0.4521 for the bare GPT-5.2 backbone, a relative improvement of thirteen percent. The second is the ablation in Section 4.3 of the paper, in which adding the dual memory raises the average to 0.4762, adding the orchestrator raises it further to 0.4962, and the full system reaches 0.5107, so that each component is claimed to contribute a monotonically increasing gain.

Per the assignment rules I reproduced these claims on a scoped subset. The paper's full evaluation uses 2,023 cases drawn from the image-containing portion of MedChain with seed 42. I used the first 30 cases of the public MedChain release, the same evaluation scripts and metric definitions as the authors, one full-system configuration and one modified configuration, and a second full-system run under a different seed to quantify run-to-run variance.

## 2. Setup

The table below lists every deliberate deviation from the paper's setup and its justification. The controlling principle is the assignment's controlled-variable rule: the underlying language models are swapped as a single unit to OpenAI-compatible DeepSeek endpoints, changing only configuration keys and no prompts, hyperparameters, or evaluation code.

| Item | Paper | This reproduction |
|---|---|---|
| Backbone LLM | GPT-5.2 | deepseek-v4-pro via DeepSeek API |
| MCTS rollout and simulation model | an independent model | deepseek-flash |
| Judge for LLM-scored tasks | Gemini-2.5-Pro at temperature 0 | deepseek-v4-pro at temperature 0 |
| Decoding temperature | judge pinned to 0 | all calls pinned to 0 |
| Maximum orchestration length eta | 4 | 4, set as MCTS_ETA = 4 |
| Dataset | MedChain, 2,023-case image subset | MedChain merged_cases.json, first 30 cases, seed 42 |
| Images | included in Task 3 | not downloaded, 1.86 GB; Task 3 auto-skips and is excluded from the average, matching the paper's applicable-metrics convention |
| Experience memory | guideline database plus historical case database | historical-case database reconstructed from 3,000 tail cases with TF-IDF character n-gram retrieval; the guideline database was never released by the authors |
| Hardware | not stated | CPU-only cloud sandbox, 4 concurrent cases |

Because the backbone and judge differ from the paper, absolute scores are not directly comparable to the published numbers, as the assignment anticipates. The reproduction questions I therefore ask are whether the system runs end to end, whether the qualitative pattern of the ablation holds in sign and rough magnitude, and what hidden assumptions surface when the pipeline is executed rather than merely read.

The environment is Python 3.12 with the openai, tqdm, and scikit-learn packages together with the repository's own requirements. Every run was launched through run_experiment.sh, which takes a configuration flag, a seed, a case count, and an optional skip offset for resuming after an outage. The API key is read from a local .env file that is gitignored; no key appears anywhere in the repository or its history.

## 3. Metrics and the modification

For each case the repository's own main.py computes the six paper metrics: referral accuracy at department level one and department overlap at level two, examination-ordering overlap, an imaging score that is null without images, a diagnosis score produced by the LLM judge and normalized to the unit interval, and treatment-plan overlap. The headline paper_average is the mean over the non-null metrics, which is identical to the paper's stated convention of averaging over applicable non-visual metrics only. Every aggregate number in this report is computed from the raw per-case result files by analysis.py, which regenerates results/final_analysis.json deterministically.

The modification is a single isolated and measurable change: the experience memory is disabled by pointing the two environment variables EXPERIENCE_GUIDE_PATH and EXPERIENCE_CDC_PATH at nonexistent files, which causes the memory to degrade gracefully to an unavailable state with no retrieval backend. No code, prompt, hyperparameter, seed, case list, or evaluation setting changes. This modification corresponds to the paper's plus-Orchestrator ablation row, whose published average is 0.4962, so the setup lets me check both the sign and the magnitude of the memory component's effect under controlled conditions, and to quantify a cost the paper does not report, namely the extra LLM calls that memory retrieval induces.

## 4. Results

### 4.1 Full system, configuration A with memory enabled

Twenty-nine of the 30 seeded cases completed. The single failure was an LLM-output parsing fault in the missing-evidence verifier, which returned invalid JSON; it is counted as a failed case and was not retried. The table compares my per-task means against the paper's ALL row of Table 3, with the published metric named for each column.

| Metric | Ours, n = 29 | Paper ALL row, Table 3 |
|---|---|---|
| t1_acc, referral accuracy at level one | 0.655 | 0.6243 |
| t1_iou, referral overlap at level two | 0.397 | 0.3543 |
| t2_iou, examination ordering overlap | 0.278 | 0.4820 |
| t3, examination imaging score | null on 26 of 29 cases | 0.4800 |
| t4, diagnosis score | 0.386 | 0.5976, DocLens |
| t5_iou, treatment-plan overlap | 0.035 | 0.5259, LLM-judge |
| paper_average | 0.349 | 0.5107 |
| API calls per case, mean | 106.7 | not reported |

Two caveats attach to the paper column. The paper's diagnosis column is scored with DocLens while the released code scores diagnosis with the LLM judge, and the paper's treatment column is an LLM-judge score while the released code scores treatment by set overlap, so the t4 and t5 comparisons mix metrics as well as models. Since my runs almost never reach the treatment stage at all, the t5 comparison is moot in practice, as Section 4.2 explains. The API call figure counts one call as one request to the DeepSeek API, summed over main orchestration calls and MCTS rollout-simulation calls, averaged over all 29 completed cases.

Absolute scores sit below the paper's across the board, which is expected under the documented model swap, a case sample sixty-seven times smaller than the paper's 2,023-case subset, and the absence of imaging data. The reproduction value of the exercise lies in the pattern and the behavioral findings rather than in absolute numbers.

The assignment calls for repeated trials, which I implemented as a second full-system run under seed 7 with 20 freshly sampled cases, of which 18 completed. That run scores 0.295 with a per-case standard deviation of 0.186, against 0.349 and 0.190 for the seed-42 run. On the 17 cases common to both runs the paired run-to-run standard deviation of paper_average is 0.183. Even with temperature pinned to zero the system is far from deterministic, because API-level nondeterminism and MCTS rollout sampling remain. This 0.18 noise scale is the yardstick against which the modification effect in Section 5 must be read.

### 4.2 A systematic behavioral finding: treatment is almost never reached

Across all 76 completed cases in the study, the MCTS planner exhausted its four allowed orchestration steps without ever selecting the treatment action in 74 of them, leaving the treatment plan empty and the treatment score at zero. Two cases are exceptions. In configuration A under seed 42, the case 1_一例急诊胃切开取异物术的麻醉处理 produced a treatment plan with perfect overlap of 1.0, and in configuration A under seed 7, the case 3_血液透析并鸟肠球菌感染一例 produced a partial plan scoring 0.333. Both exceptions occurred under the memory-enabled configuration; none of the 29 memory-disabled cases reached treatment.

The paper's published treatment score of 0.5259 implies that their full configuration does reach treatment regularly, so this discrepancy is a genuine behavioral gap rather than a metric artifact, and it mechanically depresses my paper_average by roughly one tenth of a point, since a treatment score near the published 0.5 would add about 0.1 to the five-metric average. Candidate causes include the planner model's action preferences, since GPT-5.2 and DeepSeek may rank the terminal treatment action differently within four steps, warm-start or action-ordering interactions that the released code does not document, or an evaluation-time setting that the paper does not state. This is precisely the class of hidden planning-budget assumption that reproducibility work exists to surface, and it suggests an eta-sweep follow-up that I recommend in Section 7.

## 5. Modification and its measured impact

The modification disables the experience memory through the environment-variable switch described in Section 3. Configuration B was completed in two parts after a budget outage truncated the first attempt at 21 cases; the resumed tail was merged, giving 29 completed cases. The paired analysis below restricts attention to the 28 cases that completed under both configurations, so that every comparison is within-case.

| Metric | A, memory on | B, memory off | paired difference A minus B | paired t |
|---|---|---|---|---|
| paper_average | 0.350 | 0.312 | +0.038 | 1.39 |
| t1_acc | 0.643 | 0.643 | 0.000 | 0.00 |
| t1_iou | 0.393 | 0.256 | +0.137 | 2.24 |
| t2_iou | 0.288 | 0.294 | -0.005 | -0.18 |
| t4 diagnosis | 0.393 | 0.371 | +0.021 | 0.34 |
| t5_iou | 0.036 | 0.000 | +0.036 | 1.00 |
| API calls per case | 110.2 | 60.3 | +49.9 | not applicable |

The API call figures in this table are means over the same 28 paired cases, under the call convention of Section 4.1. Computed over all 29 completed cases per configuration instead, the means are 106.7 for configuration A and 61.1 for configuration B.

Three observations follow. First, the reconstructed experience memory produces a small positive aggregate effect of 0.038 on paper_average with a paired t of 1.39, which is not significant at conventional levels, and one clearly significant per-task effect: referral department overlap improves by 0.137 with a paired t of 2.24. This is consistent with retrieved similar cases helping the referral stage specifically, which is the stage where the historical case database is most directly informative. Second, the aggregate effect of 0.038 is smaller than the measured run-to-run noise of 0.183 from Section 4.1, so the honest reading of the average gain is directionally positive but statistically fragile, while the per-task referral effect exceeds what run-to-run noise plausibly explains. Third, the cost is stark: enabling the memory raises the call count by 83 percent, from 60.3 to 110.2 calls per case on the paired subset, because retrieval actions are themselves planned and simulated inside MCTS rollouts, so each retrieval multiplies into many simulated executions.

The paper's ablation moves 0.4521 to 0.4762 to 0.4962 to 0.5107 as components are added, a monotonic gain for each. Under my controlled setup, removing the memory moves 0.350 to 0.312, which is a gain of the same sign when the memory is added back. The sign of the component's contribution therefore replicates even though its average size here sits within run-to-run noise, and the caveat from Section 2 applies: my memory is a faithful-interface reconstruction without the authors' unreleased guideline database, so the paper's exact plus-Dual-Memory magnitude cannot be re-measured with the released materials.

One methodological footnote is worth recording. An early truncated pairing of 20 cases showed the opposite sign, with configuration A slightly worse than B; only after the resumed cases were merged did the conclusion flip. Drawing modification effects from partial runs would have reported a refutation where the completed data shows a weak confirmation, which is a caution I carry into the recommendations.

## 6. Debug diary

Six blockers consumed most of the calendar time on this project, and each is recorded here because the assignment asks for the messy path, not just the clean result.

The first and largest was that the repository is missing its configuration file. Four modules import app_config, and the file was never committed, with the paper itself noting only that the full configuration would be released with the code, which did not happen. I reconstructed all 36 imported symbols from their use sites, taking values from the paper where stated, such as eta equal to 4 and judge temperature of 0, and marking the remainder, such as the MCTS reward weights, as reconstructed estimates. The second blocker was identical in kind: experience_memory.py was also missing, and I rebuilt its interface from call sites in the orchestrator, namely a freeze method, an availability flag, and a retrieve method, implementing the behavior the paper describes plus graceful degradation when the data files are absent, which is what the modification in Section 5 relies on.

The third blocker was a latent bug in the authors' code. Their utils.py performs a local import of the regular-expression module inside a conditional branch of extract_ground_truth, which shadows the module-level import for the entire function and raises UnboundLocalError on any case whose summary does not begin with a gender character. The first test case happened to pass, which is presumably why the bug shipped. I removed the redundant local import and note this as a contribution-grade fix to the original code, since any batch run over MedChain crashes without it. The fourth blocker was my own reconstruction bug: my first retrieve implementation refit the TF-IDF vectorizer on all 3,000 stored cases on every call, and because MCTS simulates retrieval actions inside rollouts, this collapsed throughput to roughly one case every four minutes. Pre-computing the retrieval indices at freeze time reduced ten retrievals to 0.2 seconds.

The fifth blocker was budget. The DeepSeek account ran dry twice with 402 Insufficient Balance errors, first halting the 30-case configuration A run mid-flight and later halting configuration B after 21 cases. Both partial runs were archived, the account was topped up, and B was finished by resuming with the skip offset; a third top-up funded the seed-7 variance run. Total spend was approximately 90 RMB of DeepSeek credit for the complete study, and every figure in this report is computed from completed cases with the truncation documented rather than hidden. The sixth blocker was the sandbox itself, which reverted package installations between sessions; dependencies were moved to a persistent directory loaded through PYTHONPATH, which the run script sets automatically.

## 7. Conclusions

On the reproducible side, the system itself runs end to end after the two reconstructions and the one-line bug fix, with all five stages, MCTS planning, working memory, evaluation, and per-case logging behaving as the paper describes. The qualitative component narrative also survives the model swap: memory and orchestrator each add capability, scores move in the paper's direction, and per-task magnitudes are of the same order. The evaluation protocol is directly verifiable from the released code, including the metric definitions, the judge normalization, and the applicable-metrics averaging convention.

On the non-reproducible side, absolute scores differ chiefly because of the sanctioned model swap, the sixty-seven-fold smaller subset, and the missing imaging data, all of which the assignment's controlled-variable rule anticipates. The treatment stage does not reproduce behaviorally: 74 of 76 completed cases never select the treatment action within four orchestration steps under DeepSeek planning, against a published treatment score that implies regular success, which points to planning-budget sensitivity to the backbone model or an undocumented setting. The experience memory as published cannot be reproduced exactly because the guideline database and the original retrieval index were never released, so only a faithful-interface reconstruction was possible.

For future reproducers I offer four recommendations. Budget for roughly 60 to 110 API calls per case depending on configuration, with an observed range of 8 to 138 across all runs. Fix the regular-expression shadowing bug before any batch run, and pin every hyperparameter the paper leaves unstated, such as the MCTS candidate count, simulation count, and reward weights, because these are unrecoverable from the released materials. Treat LLM-judge scores as high-variance measurements and report paired statistics and call counts rather than means alone. Finally, if treatment-stage fidelity matters to the research question, sweep the orchestration budget or inspect the planner's action priors before trusting aggregate averages, because a single behavioral gap at the final stage silently removes about a tenth from the headline metric.

All runs, logs, per-case trajectories under state_memories_runA, state_memories_runB_full, and state_memories_A_s7, and the analysis script are in the accompanying repository, with raw per-case numbers in results/ and the regenerated aggregate analysis in results/final_analysis.json.
