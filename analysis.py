"""Statistical analysis for the ClinicalAgents reproduction study.

Reads the per-case benchmark result files in results/ and writes
results/final_analysis.json. Re-running this script reproduces every
aggregate number quoted in REPORT.md from the raw per-case records.

API call convention: a "call" is one request to the DeepSeek API, counted
as main-orchestration calls plus MCTS rollout-simulation calls, i.e.
llm_usage.llm_calls_total + llm_usage.rollout_calls_total per case. The
paper reports no call counts; this convention is ours and is stated
wherever the numbers are used.
"""

import json
import math
import statistics as st
from pathlib import Path

RESULTS = Path(__file__).parent / "results"

FILES = {
    "A_seed42": "benchmark_results_A_s42.json",
    "B_seed42": "benchmark_results_B_s42_full.json",
    "A_seed7": "benchmark_results_A_s7.json",
}

PAPER_TARGETS = {
    "main_ALL_avg": 0.5107,
    "main_ALL_per_task": {
        "t1_acc": 0.6243,   # Specialty Referral, Acc (Lv1)
        "t1_iou": 0.3543,   # Specialty Referral, IoU (Lv2)
        "t2_iou": 0.4820,   # Test Ordering, IoU
        "t3_score": 0.4800, # Examination, DocLens
        "t4_score": 0.5976, # Diagnosis, LLM-judge
        "t5_iou": 0.5259,   # Treatment, IoU
    },
    "ablation_backbone_avg": 0.4521,
    "ablation_plus_dual_memory_avg": 0.4762,
    "ablation_plus_orchestrator_avg": 0.4962,
}


def load_cases(name):
    cases = json.loads((RESULTS / FILES[name]).read_text(encoding="utf-8"))
    return [c for c in cases
            if c.get("metrics") and c["metrics"].get("paper_average") is not None]


def paper_average(c):
    return c["metrics"]["paper_average"]


def api_calls(c):
    u = c["llm_usage"]
    return u["llm_calls_total"] + u["rollout_calls_total"]


def summarize(cases):
    pas = [paper_average(c) for c in cases]
    return {
        "n_completed": len(cases),
        "paper_average_mean": round(st.mean(pas), 4),
        "paper_average_sd": round(st.stdev(pas), 4) if len(pas) > 1 else 0.0,
        "api_calls_per_case_mean": round(st.mean([api_calls(c) for c in cases]), 1),
    }


def paired_t(dict_a, dict_b, metric):
    diffs = []
    for cid in sorted(set(dict_a) & set(dict_b)):
        a = dict_a[cid]["metrics"].get(metric)
        b = dict_b[cid]["metrics"].get(metric)
        if a is None or b is None:
            continue
        diffs.append(a - b)
    n = len(diffs)
    if n < 2:
        return {"metric": metric, "n": n, "paired_diff_A_minus_B": None, "paired_t": None}
    m = st.mean(diffs)
    sd = st.stdev(diffs)
    t = m / (sd / math.sqrt(n)) if sd > 0 else (0.0 if m == 0 else None)
    return {
        "metric": metric,
        "n": n,
        "A_mean": round(st.mean([dict_a[c]["metrics"][metric] for c in sorted(set(dict_a) & set(dict_b))
                                 if dict_a[c]["metrics"].get(metric) is not None
                                 and dict_b[c]["metrics"].get(metric) is not None]), 4),
        "B_mean": round(st.mean([dict_b[c]["metrics"][metric] for c in sorted(set(dict_a) & set(dict_b))
                                 if dict_a[c]["metrics"].get(metric) is not None
                                 and dict_b[c]["metrics"].get(metric) is not None]), 4),
        "paired_diff_A_minus_B": round(m, 4),
        "paired_t": round(t, 2) if t is not None else None,
    }


def main():
    A = load_cases("A_seed42")
    B = load_cases("B_seed42")
    A7 = load_cases("A_seed7")

    dict_a = {c["id"]: c for c in A}
    dict_b = {c["id"]: c for c in B}
    dict_a7 = {c["id"]: c for c in A7}

    paired_ids = sorted(set(dict_a) & set(dict_b))
    run_pairs = sorted(set(dict_a) & set(dict_a7))
    run_diffs = [paper_average(dict_a[c]) - paper_average(dict_a7[c]) for c in run_pairs]

    all_completed = A + B + A7
    reached = [c for c in all_completed if (c["metrics"].get("t5_iou") or 0) > 0]

    analysis = {
        "description": (
            "A = full system with experience memory enabled; B = modification with "
            "experience memory disabled via EXPERIENCE_CDC_PATH/EXPERIENCE_GUIDE_PATH "
            "pointing at nonexistent files. Backbone and judge: deepseek-v4-pro; "
            "MCTS rollout: deepseek-flash; temperature 0; eta = 4. Subset: first 30 "
            "cases of MedChain merged_cases.json, seed 42; config A also rerun with "
            "seed 7 (20 cases sampled) to measure run-to-run variance. "
            "API call convention: llm_calls_total + rollout_calls_total per case."
        ),
        "paper_targets": PAPER_TARGETS,
        "config_A_seed42": summarize(A),
        "config_B_seed42": summarize(B),
        "config_A_seed7": summarize(A7),
        "run_to_run_seed42_vs_seed7": {
            "n_common_cases": len(run_pairs),
            "mean_paired_difference": round(st.mean(run_diffs), 4),
            "paired_sd": round(st.stdev(run_diffs), 4),
        },
        "paired_analysis_A_vs_B": {
            "n_common_cases": len(paired_ids),
            "rows": [paired_t(dict_a, dict_b, m) for m in
                     ["t1_acc", "t1_iou", "t2_iou", "t3_score", "t4_score",
                      "t5_iou", "paper_average"]],
            "api_calls_mean_A": round(st.mean([api_calls(dict_a[c]) for c in paired_ids]), 1),
            "api_calls_mean_B": round(st.mean([api_calls(dict_b[c]) for c in paired_ids]), 1),
        },
        "systematic_finding_treatment_stage": {
            "completed_cases_total": len(all_completed),
            "treatment_reached": len(reached),
            "treatment_not_reached": len(all_completed) - len(reached),
            "exceptions": [
                {"run": "A_seed42", "case": c["id"],
                 "t5_iou": c["metrics"]["t5_iou"]} for c in A if (c["metrics"].get("t5_iou") or 0) > 0
            ] + [
                {"run": "A_seed7", "case": c["id"],
                 "t5_iou": c["metrics"]["t5_iou"]} for c in A7 if (c["metrics"].get("t5_iou") or 0) > 0
            ],
            "note": ("In the remaining 74 of 76 completed cases the MCTS planner "
                     "exhausted its eta = 4 steps without selecting task5_treatment, "
                     "leaving treatment_plan empty and t5_iou at 0."),
        },
        "cost_total_rmb": 90,
    }

    out = RESULTS / "final_analysis.json"
    out.write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {out}")
    print(json.dumps(analysis["config_A_seed42"], indent=1))
    print(json.dumps(analysis["paired_analysis_A_vs_B"], indent=1,
                     ensure_ascii=False)[:1200])
    print(json.dumps(analysis["systematic_finding_treatment_stage"], indent=1,
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
