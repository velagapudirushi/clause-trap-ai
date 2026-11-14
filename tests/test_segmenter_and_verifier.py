"""Unit tests for atomic clause segmentation and the deterministic zero-hallucination quote guardrail."""

from clause_trap.quote_verifier import (
    compute_contract_safety_grade,
    filter_verified_flags,
    verify_quote_in_source,
)
from clause_trap.schemas import DetectedTrapFlag
from clause_trap.segmenter import segment_contract_into_clauses


def test_segmenter_preserves_exact_character_offsets() -> None:
    doc = (
        "Welcome to Acme Cloud.\n\n"
        "We reserve the right to modify or terminate the service at any time without prior notice.\n"
        "All disputes shall be resolved exclusively by binding individual arbitration."
    )
    clauses = segment_contract_into_clauses(doc)
    assert len(clauses) == 3
    for seg in clauses:
        assert doc[seg.start_char : seg.end_char] == seg.text


def test_quote_verifier_blocks_hallucinated_excerpts() -> None:
    source = "We may suspend your account without notice if you violate these terms."
    real_flag = DetectedTrapFlag(
        clause_index=0,
        label_id=1,
        trap_code="unilateral_termination",
        trap_name="Unilateral Termination",
        severity=8,
        badge_color="#ef4444",
        confidence=0.88,
        exact_quote=source,
        start_char=0,
        end_char=len(source),
        quote_verified_in_source=False,
        plain_english_warning="Account can be suspended without prior notice.",
        pushback_question="Can we add a 14-day written cure period?",
    )
    hallucinated_flag = real_flag.model_copy(
        update={
            "exact_quote": "You must pay a $5,000 penalty fee upon cancellation.",
            "start_char": 0,
            "end_char": 52,
        }
    )

    assert verify_quote_in_source(source, real_flag.exact_quote, 0, len(source)) is True
    assert verify_quote_in_source(source, hallucinated_flag.exact_quote, 0, 52) is False

    kept = filter_verified_flags(source, [real_flag, hallucinated_flag])
    assert len(kept) == 1
    assert kept[0].quote_verified_in_source is True

    grade, score, _ = compute_contract_safety_grade(kept, total_clauses=2)
    assert grade in ("B", "C", "D")
    assert score > 0.0
