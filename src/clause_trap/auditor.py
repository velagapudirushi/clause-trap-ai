"""Live Contract Red-Flag Auditor & Inline HTML Highlighter for ClauseTrap AI."""

from __future__ import annotations

import html
import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from pathlib import Path
from typing import Any

import torch
import numpy as np

from clause_trap.config import LORA_CHECKPOINT_PATH, load_taxonomy_config
from clause_trap.dataset_loader import NUM_TRAP_CLASSES
from clause_trap.quote_verifier import (
    compute_contract_safety_grade,
    filter_verified_flags,
)
from clause_trap.schemas import ContractAuditResult, DetectedTrapFlag
from clause_trap.segmenter import segment_contract_into_clauses

# High-precision legal domain cues that boost recall on unambiguous statutory waiver phrasings
_LEGAL_CUE_BOOSTS: dict[int, tuple[str, ...]] = {
    0: ("liable for any", "disclaim all liability", "exceed zero dollars", "no liability", "consequential"),
    1: ("terminate or suspend", "without prior notice", "for any reason whatsoever", "forfeit any"),
    2: ("modify, amend", "revise these", "sole discretion at any time", "change any aspect"),
    3: ("remove, disable", "delete any user content", "erase or withhold"),
    4: ("accessing or browsing", "continued login constitutes", "continued use", "without creating an account"),
    5: ("governed exclusively by the laws", "governed by and construed", "excluding your local"),
    6: ("brought solely in the state", "courts located in", "exclusively in the courts"),
    7: ("binding individual arbitration", "class-action lawsuit", "trial by jury", "waive any right to participate"),
}


class ContractTrapAuditor:
    """Scans any contract or Terms of Service, verifies quotes verbatim, and highlights red flags."""

    def __init__(self, checkpoint_path: Path | None = None) -> None:
        self.cfg = load_taxonomy_config()
        ckpt_file = checkpoint_path or LORA_CHECKPOINT_PATH
        self.bundle = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        self.vec = self.bundle["tfidf_vectorizer"]
        self.clf = self.bundle["aligned_ovr_classifier"]
        self.thresholds: np.ndarray = np.asarray(
            self.bundle["calibrated_thresholds"], dtype=np.float32
        )

    def predict_clause_probabilities(self, texts: list[str]) -> np.ndarray:
        """Predict 8-class unfair trap probabilities for a list of clause strings."""
        if not texts:
            return np.zeros((0, NUM_TRAP_CLASSES), dtype=np.float32)
        X_tfidf = self.vec.transform(texts)
        base_probs = np.asarray(self.clf.predict_proba(X_tfidf), dtype=np.float32)

        for row_idx, txt in enumerate(texts):
            lower = txt.lower()
            for class_id, cues in _LEGAL_CUE_BOOSTS.items():
                hits = sum(1 for c in cues if c in lower)
                if hits > 0:
                    base_probs[row_idx, class_id] = min(
                        0.99, float(base_probs[row_idx, class_id]) + 0.22 * hits
                    )
        return np.round(base_probs, 4)

    def audit_contract(self, contract_text: str) -> ContractAuditResult:
        """Segment a contract, score every clause, verify verbatim quotes, and compute Grade A–F."""
        clauses = segment_contract_into_clauses(contract_text)
        if not clauses:
            return ContractAuditResult(
                safety_grade="A",
                risk_score_0_to_100=0.0,
                verdict_headline="Grade A — Paste a contract above to scan for red flags",
                total_clauses_scanned=0,
                flagged_clause_count=0,
                total_trap_hits=0,
                verified_quote_rate_pct=100.0,
                flags=[],
            )

        probs = self.predict_clause_probabilities([c.text for c in clauses])
        candidate_flags: list[DetectedTrapFlag] = []

        for idx, clause in enumerate(clauses):
            for label_id in range(NUM_TRAP_CLASSES):
                prob = float(probs[idx, label_id])
                th = float(self.thresholds[label_id])
                if prob >= th:
                    cat_meta: dict[str, Any] = self.cfg["trap_categories"][label_id]
                    candidate_flags.append(
                        DetectedTrapFlag(
                            clause_index=clause.clause_index,
                            label_id=label_id,
                            trap_code=cat_meta["code"],
                            trap_name=cat_meta["name"],
                            severity=int(cat_meta["severity"]),
                            badge_color=cat_meta["badge_color"],
                            confidence=round(prob, 3),
                            exact_quote=clause.text,
                            start_char=clause.start_char,
                            end_char=clause.end_char,
                            quote_verified_in_source=False,
                            plain_english_warning=cat_meta["plain_english"],
                            pushback_question=cat_meta["pushback_question"],
                        )
                    )

        verified_flags = filter_verified_flags(contract_text, candidate_flags)
        verified_flags.sort(key=lambda f: (f.severity, f.confidence), reverse=True)

        flagged_clause_indices = {f.clause_index for f in verified_flags}
        grade, risk_score, headline = compute_contract_safety_grade(
            verified_flags, total_clauses=len(clauses)
        )
        ver_rate = (
            100.0 * len(verified_flags) / len(candidate_flags)
            if candidate_flags
            else 100.0
        )

        return ContractAuditResult(
            safety_grade=grade,
            risk_score_0_to_100=risk_score,
            verdict_headline=headline,
            total_clauses_scanned=len(clauses),
            flagged_clause_count=len(flagged_clause_indices),
            total_trap_hits=len(verified_flags),
            verified_quote_rate_pct=round(ver_rate, 1),
            flags=verified_flags,
        )

    @staticmethod
    def render_highlighted_document_html(
        contract_text: str,
        audit: ContractAuditResult,
    ) -> str:
        """Render the full contract with inline red/orange highlights on verified trap clauses."""
        if not contract_text.strip():
            return ""

        # Group flags by (start_char, end_char) span
        span_map: dict[tuple[int, int], list[DetectedTrapFlag]] = {}
        for f in audit.flags:
            key = (f.start_char, f.end_char)
            span_map.setdefault(key, []).append(f)

        sorted_spans = sorted(span_map.keys(), key=lambda x: x[0])
        parts: list[str] = []
        cursor = 0

        for start, end in sorted_spans:
            if start < cursor or end > len(contract_text):
                continue
            # Safe unflagged text before this clause
            before = html.escape(contract_text[cursor:start]).replace("\n", "<br/>")
            parts.append(before)

            flags_here = span_map[(start, end)]
            top_flag = max(flags_here, key=lambda x: (x.severity, x.confidence))
            labels_str = " + ".join(dict.fromkeys(f.trap_name for f in flags_here))
            bg_rgba = (
                "rgba(239, 68, 68, 0.25)"
                if top_flag.severity >= 8
                else "rgba(249, 115, 22, 0.22)"
            )
            border_col = top_flag.badge_color
            escaped_clause = html.escape(contract_text[start:end])

            parts.append(
                f'<mark style="background:{bg_rgba}; color:#f8fafc; padding:3px 6px; '
                f'border-left:4px solid {border_col}; border-radius:4px; display:inline; line-height:1.75;">'
                f"{escaped_clause} "
                f'<span style="background:{border_col}; color:#fff; font-size:0.74rem; '
                f'font-weight:700; padding:2px 7px; border-radius:99px; margin-left:4px; '
                f'vertical-align:middle; white-space:nowrap;">🚩 {html.escape(labels_str)}</span>'
                f"</mark>"
            )
            cursor = end

        tail = html.escape(contract_text[cursor:]).replace("\n", "<br/>")
        parts.append(tail)

        body = "".join(parts)
        return (
            '<div style="background:#0f172a; border:1px solid #334155; border-radius:10px; '
            'padding:18px 22px; font-size:0.98rem; line-height:1.75; color:#e2e8f0;">'
            f"{body}</div>"
        )
