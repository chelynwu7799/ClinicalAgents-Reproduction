# Presentation Script (≤15 minutes)

Slide-free talking track. Record with Zoom, screen-share the repo + REPORT.md.
Target: 12–14 min.

---

## 1. What I attempted to reproduce (2 min)

"I reproduced *ClinicalAgents* — a KDD 2026 multi-agent system for clinical
decision-making on the MedChain benchmark. The claim: an MCTS orchestrator
plus a dual-memory module lifts a GPT-5.2 backbone from 0.4521 to 0.5107
average score, with each component contributing monotonically in the
ablation. My scope: one benchmark, 30 cases instead of 2,023, and — as the
assignment allows — the models swapped to DeepSeek, documented as a
controlled variable."

## 2. What it took to get it running (3 min)

"The repo was missing its entire configuration file and its experience-memory
module — the paper says 'the full configuration file will be released with
the code repository'; it wasn't. I reconstructed both from every import and
call site. I also found a latent bug in the authors' code: a function-local
`import re` that crashes on a subset of cases. One reconstruction bug of my
own — refitting a TF-IDF index on every retrieval call, inside MCTS
rollouts — made the system 30× slower until I fixed it. Two DeepSeek balance
outages truncated runs; both resumed honestly with skip offsets."

## 3. Results (3 min)

"Full system: 0.349 average vs the paper's 0.511 — expected gap, and I say
so explicitly: different LLM, 74× fewer cases, no images. Two findings
matter more than the absolute number. First, a hidden assumption: in 76 of
76 completed cases the planner never reached the treatment stage within its
η=4 step budget — the paper's published treatment scores imply theirs did.
Second, run-to-run variance: two seeds differ by 0.18 paired standard
deviation even at temperature zero — that noise scale is the ruler for any
modification claim."

## 4. Modification and measured impact (3 min)

"One isolated change: disable the experience memory with an environment
variable — everything else identical, the paper's own +Orchestrator
ablation. Paired on 28 cases: the memory gives +0.038 average — smaller than
run-to-run noise — with one significant per-task effect, referral
department overlap +0.14. The cost: +83% more LLM calls per case, because
retrieval actions get planned and simulated inside the MCTS. Methodological
lesson: my truncated early data showed the *opposite* sign — completing the
run flipped the conclusion."

## 5. Lessons & recommendations (2 min)

"Budget ~60–110 LLM calls per case. Pin every hyperparameter the paper
doesn't state — they were unrecoverable. Report paired statistics and call
counts, not just means. And check whether your planner actually reaches the
final stage before trusting aggregate averages. The system is honestly
built and its evaluation protocol checks out — but its headline numbers
rest on components that were never released, which is the real
reproducibility story."

---

**Q&A prep (likely questions):** Why DeepSeek? (assignment-sanctioned swap,
OpenAI-compatible, documented). Is 0.349 a failure? (No — controlled
comparison; the reproduction questions were end-to-end function, ablation
direction, hidden assumptions — all answered.) Why is t5 zero? (Planner
never selects treatment within η=4 under DeepSeek planning; flagged as
hidden-assumption finding.)
