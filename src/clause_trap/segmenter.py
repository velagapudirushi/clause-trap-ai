"""Atomic legal clause splitter preserving exact character spans in the source document."""

from __future__ import annotations

import re

from clause_trap.schemas import ClauseSegment

# Split on newlines or sentence-ending punctuation followed by whitespace, ignoring common abbreviations
_SPLIT_PATTERN = re.compile(r"(?:\r?\n)+|(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def segment_contract_into_clauses(raw_text: str, min_chars: int = 20) -> list[ClauseSegment]:
    """Split a contract into atomic clauses while keeping exact `[start_char, end_char]` spans."""
    if not raw_text or not raw_text.strip():
        return []

    segments: list[ClauseSegment] = []
    cursor = 0
    for match in _SPLIT_PATTERN.finditer(raw_text):
        chunk_raw = raw_text[cursor : match.start()]
        _append_trimmed_segment(raw_text, cursor, match.start(), chunk_raw, min_chars, segments)
        cursor = match.end()

    if cursor < len(raw_text):
        tail_raw = raw_text[cursor:]
        _append_trimmed_segment(raw_text, cursor, len(raw_text), tail_raw, min_chars, segments)

    return segments


def _append_trimmed_segment(
    full_text: str,
    span_start: int,
    span_end: int,
    chunk_raw: str,
    min_chars: int,
    out: list[ClauseSegment],
) -> None:
    stripped = chunk_raw.strip()
    if len(stripped) < min_chars:
        return
    lead_ws = len(chunk_raw) - len(chunk_raw.lstrip())
    trail_ws = len(chunk_raw) - len(chunk_raw.rstrip())
    exact_start = span_start + lead_ws
    exact_end = span_end - trail_ws
    if full_text[exact_start:exact_end] != stripped:  # pragma: no cover
        idx = full_text.find(stripped, span_start, span_end)
        if idx != -1:
            exact_start = idx
            exact_end = idx + len(stripped)

    out.append(
        ClauseSegment(
            clause_index=len(out),
            text=stripped,
            start_char=exact_start,
            end_char=exact_end,
        )
    )
