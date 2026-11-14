"""Deterministic Zero-Hallucination Quote Guardrail and Contract Safety Grader."""

from __future__ import annotations

from clause_trap.schemas import DetectedTrapFlag


def verify_quote_in_source(source_text: str, exact_quote: str, start_char: int, end_char: int) -> bool:
    """Verify mathematically that `exact_quote` exists verbatim in `source_text`."""
    if not exact_quote or not source_text:
        return False
    if 0 <= start_char < end_char <= len(source_text):
        if source_text[start_char:end_char] == exact_quote:
            return True
    return exact_quote in source_text


def filter_verified_flags(source_text: str, candidate_flags: list[DetectedTrapFlag]) -> list[DetectedTrapFlag]:
    """Reject any candidate flag whose quote cannot be verified verbatim in `source_text`."""
    verified: list[DetectedTrapFlag] = []
    for flag in candidate_flags:
        is_ok = verify_quote_in_source(
            source_text=source_text,
            exact_quote=flag.exact_quote,
            start_char=flag.start_char,
            end_char=flag.end_char,
        )
        if is_ok:
            verified.append(flag.model_copy(update={"quote_verified_in_source": True}))
    return verified


def compute_contract_safety_grade(
    flags: list[DetectedTrapFlag],
    total_clauses: int,
) -> tuple[str, float, str]:
    """Compute an overall A–F safety grade and 0–100 risk score from verified trap flags."""
    if total_clauses <= 0 or not flags:
        return "A", 0.0, "Grade A — Clean & Balanced Agreement (No Predatory Traps Detected)"

    # Weight severity by model confidence; distinct clause hits prevent a single sentence double-counting excessively
    max_sev_by_clause: dict[int, float] = {}
    for f in flags:
        weighted = float(f.severity) * (0.65 + 0.35 * float(f.confidence))
        max_sev_by_clause[f.clause_index] = max(max_sev_by_clause.get(f.clause_index, 0.0), weighted)

    total_severity_points = sum(max_sev_by_clause.values())
    has_critical_arbitration = any(f.trap_code == "arbitration" for f in flags)

    raw_risk = min(100.0, total_severity_points * 3.1 + (14.0 if has_critical_arbitration else 0.0))
    risk_score = round(float(raw_risk), 1)

    if risk_score < 12.0:
        return "A", risk_score, "Grade A — Low Risk (Minimal Boilerplate)"
    if risk_score < 28.0:
        return "B", risk_score, "Grade B — Moderate Caution (Standard One-Sided Clauses)"
    if risk_score < 48.0:
        return "C", risk_score, "Grade C — High Caution (Multiple Unilateral Rights Grabs)"
    if risk_score < 70.0:
        return "D", risk_score, "Grade D — Predatory Terms (Heavy Waivers & Unfair Lock-Ins)"
    return "F", risk_score, "Grade F — Severe Trap Alert (Forced Arbitration & Rights Waivers)"
