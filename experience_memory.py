"""Experience memory for ClinicalAgents (Dual-Memory architecture).

NOTE: This module is MISSING from the official repository
(https://github.com/ZhuohanGe/ClinicalAgents-Code). med_orchestrator.py
imports `ExperienceMemory` from it, but the file was never committed.

This reconstruction implements the interface the orchestrator actually
uses (derived by inspecting every call site in med_orchestrator.py):

    em = ExperienceMemory(
        guide_path=..., cdc_path=...,
        held_out_case_ids=..., allow_dataset_fallback=False,
        guide_retriever=None,
    ).freeze()
    em.available                      -> bool
    em.retrieve(state) -> {
        "knowledge": [...],                      # guideline snippets
        "potential_missing_evidence": [...],      # exams seen in similar cases
        "retrieved_guidelines": [...],
        "retrieved_cdc_cases": [...],
        "evidence_importance": {...},
        "retrieval_backend": "tfidf" | None,
    }

Per the paper (Sec 3.3.2), experience memory = clinical-guideline DB +
historical case DB (CDC) with hybrid retrieval. The released code does
not include either DB, so retrieval degrades gracefully: if the files do
not exist, `available` is False and `retrieve` returns a no-op result
(retrieval_backend=None), which corresponds to the paper's
"+ Orchestrator (no experience memory)" ablation configuration.
"""

import json
import os
import threading
from typing import Any, Callable, Dict, List, Optional

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _HAS_SKLEARN = True
except Exception:  # pragma: no cover
    _HAS_SKLEARN = False


def _as_text(value: Any) -> str:
    """Flatten nested case fields into searchable text."""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(_as_text(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return " ".join(_as_text(v) for v in value)
    return ""


class ExperienceMemory:
    """Guideline + historical-case retrieval with graceful degradation."""

    def __init__(
        self,
        guide_path: str = "",
        cdc_path: str = "",
        held_out_case_ids: Optional[List[str]] = None,
        allow_dataset_fallback: bool = False,
        guide_retriever: Optional[Callable[[str, List[Dict[str, Any]], int], List[Dict[str, Any]]]] = None,
        top_k_guidelines: int = 3,
        top_k_cases: int = 3,
    ):
        self.guide_path = guide_path
        self.cdc_path = cdc_path
        self.held_out_case_ids = set(str(c) for c in (held_out_case_ids or []))
        self.allow_dataset_fallback = allow_dataset_fallback
        self.guide_retriever = guide_retriever
        self.top_k_guidelines = max(1, int(top_k_guidelines))
        self.top_k_cases = max(1, int(top_k_cases))

        self._guidelines: List[Dict[str, Any]] = []
        self._cdc_cases: List[Dict[str, Any]] = []
        self._frozen = False
        self._lock = threading.Lock()

        self._load()

    # ---------------------------------------------------------------- io
    def _load_json(self, path: str) -> Any:
        if not path or not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:  # corrupted file -> treat as unavailable
            print(f"[ExperienceMemory] failed to load {path}: {e}")
            return None

    def _load(self) -> None:
        raw_guides = self._load_json(self.guide_path)
        if isinstance(raw_guides, dict):
            raw_guides = list(raw_guides.values())
        if isinstance(raw_guides, list):
            self._guidelines = [
                g if isinstance(g, dict) else {"text": str(g)}
                for g in raw_guides
            ]

        raw_cdc = self._load_json(self.cdc_path)
        if isinstance(raw_cdc, dict):
            raw_cdc = list(raw_cdc.values())
        if isinstance(raw_cdc, list):
            for c in raw_cdc:
                if not isinstance(c, dict):
                    continue
                cid = str(c.get("case_id", c.get("id", "")))
                if cid and cid in self.held_out_case_ids:
                    continue  # never retrieve held-out (test) cases
                self._cdc_cases.append(c)

    def freeze(self) -> "ExperienceMemory":
        """Lock the memory (paper: experience memory is static at inference).

        Pre-computes the TF-IDF indices once: retrieve() then only transforms
        the (short) query, which keeps MCTS rollout retrieval cheap.
        """
        self._frozen = True
        if _HAS_SKLEARN and (self._guidelines or self._cdc_cases):
            self._g_vectorizer = self._g_vecs = None
            self._c_vectorizer = self._c_vecs = None
            if self._guidelines:
                self._g_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
                self._g_vecs = self._g_vectorizer.fit_transform(
                    [_as_text(g.get("text", g)) for g in self._guidelines]
                )
            if self._cdc_cases:
                self._c_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
                self._c_vecs = self._c_vectorizer.fit_transform(
                    [_as_text(c.get("text", c)) for c in self._cdc_cases]
                )
        return self

    # ----------------------------------------------------------- public
    @property
    def available(self) -> bool:
        return bool(self._guidelines or self._cdc_cases)

    def retrieve(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Retrieve guideline knowledge + similar historical cases."""
        empty = {
            "knowledge": [],
            "potential_missing_evidence": [],
            "retrieved_guidelines": [],
            "retrieved_cdc_cases": [],
            "evidence_importance": {},
            "retrieval_backend": None,
        }
        if not self.available:
            return empty

        query = " ".join(
            _as_text(state.get(k, ""))
            for k in ("chief_complaint", "present_illness", "hypothesis_illness")
        ).strip()

        guidelines: List[Dict[str, Any]] = []
        if self.guide_retriever is not None and self._guidelines:
            guidelines = self.guide_retriever(query, self._guidelines, self.top_k_guidelines)
        elif _HAS_SKLEARN and self._guidelines and query:
            with self._lock:
                qv = self._g_vectorizer.transform([query])
            sims = cosine_similarity(qv, self._g_vecs).ravel()
            top_idx = sims.argsort()[::-1][:self.top_k_guidelines]
            guidelines = [
                {**self._guidelines[i], "similarity": float(sims[i])}
                for i in top_idx if sims[i] > 0
            ]

        cdc_cases: List[Dict[str, Any]] = []
        if _HAS_SKLEARN and self._cdc_cases and query:
            with self._lock:
                qv = self._c_vectorizer.transform([query])
            sims = cosine_similarity(qv, self._c_vecs).ravel()
            top_idx = sims.argsort()[::-1][:self.top_k_cases]
            cdc_cases = [
                {**self._cdc_cases[i], "similarity": float(sims[i])}
                for i in top_idx if sims[i] > 0
            ]

        knowledge = [_as_text(g.get("text", g)) for g in guidelines]
        knowledge += [_as_text(c.get("summary", c.get("text", "")))[:500] for c in cdc_cases]

        missing: List[str] = []
        for c in cdc_cases:
            for exam in c.get("exams", []) or []:
                exam = str(exam)
                if exam and exam not in _as_text(state.get("physical_exams", {})) \
                        and exam not in _as_text(state.get("auxiliary_exams", {})):
                    missing.append(exam)
        missing = list(dict.fromkeys(missing))[:8]

        return {
            "knowledge": knowledge,
            "potential_missing_evidence": missing,
            "retrieved_guidelines": guidelines,
            "retrieved_cdc_cases": cdc_cases,
            "evidence_importance": {},
            "retrieval_backend": "tfidf" if _HAS_SKLEARN else None,
        }
