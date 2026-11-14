"""Pydantic v2 and Pandera data validation contracts for ClauseTrap AI."""

from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, Field

try:
    import pandera.pandas as pa
except ImportError:  # pragma: no cover
    import pandera as pa


class ClauseSegment(BaseModel):
    """A single clause extracted from a document with exact character offsets."""

    clause_index: int = Field(..., ge=0)
    text: str = Field(..., min_length=8)
    start_char: int = Field(..., ge=0)
    end_char: int = Field(..., gt=0)


class DetectedTrapFlag(BaseModel):
    """A single verified red-flag prediction emitted for a risky clause."""

    clause_index: int = Field(..., ge=0)
    label_id: int = Field(..., ge=0, le=7)
    trap_code: str = Field(..., min_length=2)
    trap_name: str = Field(..., min_length=2)
    severity: int = Field(..., ge=1, le=10)
    badge_color: str = Field(..., min_length=4)
    confidence: float = Field(..., ge=0.0, le=1.0)
    exact_quote: str = Field(..., min_length=8)
    start_char: int = Field(..., ge=0)
    end_char: int = Field(..., gt=0)
    quote_verified_in_source: bool = Field(...)
    plain_english_warning: str = Field(..., min_length=10)
    pushback_question: str = Field(..., min_length=10)


class ContractAuditResult(BaseModel):
    """Complete structured output for a scanned contract or Terms of Service."""

    safety_grade: str = Field(..., pattern=r"^[A-F]$")
    risk_score_0_to_100: float = Field(..., ge=0.0, le=100.0)
    verdict_headline: str = Field(..., min_length=4)
    total_clauses_scanned: int = Field(..., ge=0)
    flagged_clause_count: int = Field(..., ge=0)
    total_trap_hits: int = Field(..., ge=0)
    verified_quote_rate_pct: float = Field(..., ge=0.0, le=100.0)
    flags: list[DetectedTrapFlag]


CLAUSE_DF_SCHEMA = pa.DataFrameSchema(
    columns={
        "text": pa.Column(str, pa.Check.str_length(min_value=5), nullable=False),
        "is_unfair": pa.Column(int, pa.Check.isin([0, 1]), coerce=True, nullable=False),
    },
    strict=False,
    coerce=True,
)


def validate_clause_df(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a LexGLUE unfair_tos split DataFrame with Pandera."""
    checked = df.copy()
    checked["text"] = checked["text"].astype(str).str.strip()
    return CLAUSE_DF_SCHEMA.validate(checked, lazy=True)
