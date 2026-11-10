"""Build the experience-memory historical-case DB (CDC) from MedChain.

Uses the TAIL of the dataset as the retrieval pool so that head-subset
benchmark runs never retrieve their own test cases (the same held-out
protocol the paper describes in Sec 4.1.1).

Usage: python3 data/build_cdc.py [pool_size]
Output: data/experience/cdc_cases.json
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POOL = int(sys.argv[1]) if len(sys.argv) > 1 else 3000


def flatten(v):
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        return " ".join(flatten(x) for x in v.values())
    if isinstance(v, (list, tuple)):
        return " ".join(flatten(x) for x in v)
    return ""


def as_dict(x):
    return x if isinstance(x, dict) else {}


def main():
    with open(os.path.join(ROOT, "data", "merged_cases.json"), encoding="utf-8") as f:
        data = json.load(f)
    pool = list(data.items())[-POOL:]

    cdc = []
    for cid, case in pool:
        if not isinstance(case, dict):
            continue
        info = as_dict(case.get("【病案介绍】"))
        diag = as_dict(case.get("【诊治过程】"))
        text_parts = [
            flatten(info.get("主诉", [])),
            flatten(info.get("现病史", [])),
            flatten(diag.get("初步诊断", [])),
            flatten(diag.get("诊断结果", [])),
            flatten(diag.get("鉴别诊断", [])),
            flatten(case.get("【分析总结】")),
        ]
        exams = []
        physical = as_dict(as_dict(info.get("查体")).get("体格检查"))
        auxiliary = as_dict(as_dict(info.get("查体")).get("辅助检查"))
        exams += list(physical.keys()) + list(auxiliary.keys())
        cdc.append({
            "case_id": str(cid),
            "text": " ".join(p for p in text_parts if p).strip(),
            "summary": flatten(case.get("【病例摘要】"))[:500],
            "exams": exams,
        })

    out = os.path.join(ROOT, "data", "experience", "cdc_cases.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(cdc, f, ensure_ascii=False)
    print(f"wrote {len(cdc)} cases to {out}")


if __name__ == "__main__":
    main()
