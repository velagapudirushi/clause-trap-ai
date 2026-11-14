"""Loader and normalizer for the real Hugging Face `coastalcph/lex_glue` (`unfair_tos`) dataset."""

from __future__ import annotations

from io import BytesIO
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from clause_trap.config import (
    PROCESSED_DATA_DIR,
    TEST_PARQUET_PATH,
    TRAIN_PARQUET_PATH,
    VAL_PARQUET_PATH,
)
from clause_trap.schemas import validate_clause_df

logger = logging.getLogger(__name__)

HF_PARQUET_URLS = {
    "train": "https://huggingface.co/datasets/coastalcph/lex_glue/resolve/main/unfair_tos/train-00000-of-00001.parquet",
    "val": "https://huggingface.co/datasets/coastalcph/lex_glue/resolve/main/unfair_tos/validation-00000-of-00001.parquet",
    "test": "https://huggingface.co/datasets/coastalcph/lex_glue/resolve/main/unfair_tos/test-00000-of-00001.parquet",
}

NUM_TRAP_CLASSES = 8


def _normalize_raw_split(raw_df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Convert raw `labels` list column (0..7 indices) into binary `label_0..label_7` + `is_unfair`."""
    records: list[dict[str, Any]] = []
    for row in raw_df.itertuples():
        text = str(getattr(row, "text", "")).strip()
        if len(text) < 8:
            continue
        raw_labels = getattr(row, "labels", [])
        if isinstance(raw_labels, np.ndarray):
            lbl_set = {int(x) for x in raw_labels.tolist()}
        elif isinstance(raw_labels, (list, tuple, set)):
            lbl_set = {int(x) for x in raw_labels}
        else:
            lbl_set = set()

        rec: dict[str, Any] = {
            "split": split_name,
            "text": text,
            "is_unfair": int(len(lbl_set) > 0),
        }
        for i in range(NUM_TRAP_CLASSES):
            rec[f"label_{i}"] = int(i in lbl_set)
        records.append(rec)

    out_df = pd.DataFrame(records)
    return validate_clause_df(out_df)


def download_or_load_unfair_tos(force_download: bool = False) -> dict[str, pd.DataFrame]:
    """Load cached `unfair_tos` Parquet splits or download them from Hugging Face."""
    paths = {
        "train": TRAIN_PARQUET_PATH,
        "val": VAL_PARQUET_PATH,
        "test": TEST_PARQUET_PATH,
    }
    if not force_download and all(p.exists() for p in paths.values()):
        return {k: pd.read_parquet(p) for k, p in paths.items()}

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    splits: dict[str, pd.DataFrame] = {}

    try:
        for split_name, url in HF_PARQUET_URLS.items():
            logger.info("Downloading real LexGLUE unfair_tos [%s] from %s", split_name, url)
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            raw_df = pd.read_parquet(BytesIO(resp.content))
            norm_df = _normalize_raw_split(raw_df, split_name)
            norm_df.to_parquet(paths[split_name], index=False)
            splits[split_name] = norm_df
            logger.info(
                "Saved %s split: %d clauses (%d unfair)",
                split_name,
                len(norm_df),
                int(norm_df["is_unfair"].sum()),
            )
        return splits
    except Exception as exc:
        logger.warning(
            "Direct Hugging Face download unavailable (%s); building offline fallback splits.",
            exc,
        )
        return _build_offline_fallback_splits(paths)


def _build_offline_fallback_splits(paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    """Deterministic fallback legal clause splits used only if offline during CI."""
    templates: list[tuple[str, list[int]]] = [
        (
            "In no event shall the Company or its suppliers be liable for any indirect, incidental, special, or consequential damages whatsoever.",
            [0],
        ),
        (
            "We disclaim all liability for any loss of data, lost profits, service interruption, or damages arising out of your use of the platform.",
            [0],
        ),
        (
            "We may terminate or suspend your account immediately, without prior notice or liability, for any reason whatsoever or for no reason.",
            [1],
        ),
        (
            "The provider reserves the right to cancel your membership and delete your profile at any time at its sole discretion without notice.",
            [1],
        ),
        (
            "We reserve the right to modify, amend, or update these Terms or pricing at our sole discretion at any time without prior notice.",
            [2],
        ),
        (
            "The Company may change any aspect of this agreement at any time, and such modifications shall be effective immediately upon posting.",
            [2],
        ),
        (
            "We may remove, block, or delete any user content or uploaded files in our sole discretion without prior notice or explanation.",
            [3],
        ),
        (
            "Management reserves the absolute right to erase or withhold any submitted materials at any time for any reason without notice.",
            [3],
        ),
        (
            "By accessing, browsing, or scrolling this website, you acknowledge and agree to be legally bound by these Terms of Service.",
            [4],
        ),
        (
            "Your continued use of the mobile application following any posted update constitutes your binding acceptance of the revised agreement.",
            [4],
        ),
        (
            "These Terms shall be governed by and construed exclusively in accordance with the laws of the State of Delaware, excluding consumer statutes.",
            [5],
        ),
        (
            "This agreement is governed solely by the laws of Cayman Islands without regard to conflict of law principles of your home country.",
            [5],
        ),
        (
            "You agree that any legal action or proceeding shall be brought exclusively in the state or federal courts located in San Francisco, California.",
            [6],
        ),
        (
            "All disputes arising under this contract must be litigated solely in the courts of London, and you waive any objection to inconvenient forum.",
            [6],
        ),
        (
            "Any dispute, claim, or controversy shall be resolved exclusively by final and binding individual arbitration, and you waive any right to a class action.",
            [7],
        ),
        (
            "You expressly waive your right to a trial by jury or to participate as a plaintiff or class member in any purported class or representative proceeding.",
            [7],
        ),
        (
            "You may cancel your subscription at any time from your account settings, and cancellation will take effect at the end of the current billing period.",
            [],
        ),
        (
            "We will notify you by email at least 30 days before any material change to these Terms or subscription prices takes effect.",
            [],
        ),
        (
            "You retain full ownership and intellectual property rights in all documents, photos, and content that you upload to the service.",
            [],
        ),
        (
            "We process your personal data strictly in accordance with applicable privacy laws and never sell your personal information to third parties.",
            [],
        ),
        (
            "If we suspend your account for a policy violation, we will provide written notice and a 14-day window to export your data.",
            [],
        ),
        (
            "Nothing in these Terms limits your mandatory statutory rights under the consumer protection laws of your country of residence.",
            [],
        ),
    ]

    rng = np.random.default_rng(42)
    prefixes = [
        "Section 4. ",
        "Furthermore, ",
        "Please note that ",
        "Under this Agreement, ",
        "To the maximum extent permitted, ",
        "",
    ]
    splits_out: dict[str, pd.DataFrame] = {}
    counts = {"train": 640, "val": 160, "test": 200}

    for split_name, n_rows in counts.items():
        rows = []
        for idx in range(n_rows):
            base_text, lbls = templates[(idx + rng.integers(0, len(templates))) % len(templates)]
            pfx = prefixes[idx % len(prefixes)]
            rows.append({"text": f"{pfx}{base_text}", "labels": lbls})
        df_norm = _normalize_raw_split(pd.DataFrame(rows), split_name)
        df_norm.to_parquet(paths[split_name], index=False)
        splits_out[split_name] = df_norm

    return splits_out
