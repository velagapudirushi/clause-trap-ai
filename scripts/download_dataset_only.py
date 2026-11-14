"""Download the 3 official Hugging Face `coastalcph/lex_glue` (`unfair_tos`) Parquet splits."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import sys

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from clause_trap.config import (
    PROCESSED_DATA_DIR,
    TEST_PARQUET_PATH,
    TRAIN_PARQUET_PATH,
    VAL_PARQUET_PATH,
)
from clause_trap.dataset_loader import HF_PARQUET_URLS, _normalize_raw_split


def main() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": TRAIN_PARQUET_PATH,
        "val": VAL_PARQUET_PATH,
        "test": TEST_PARQUET_PATH,
    }
    for split_name, url in HF_PARQUET_URLS.items():
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        raw_df = pd.read_parquet(BytesIO(resp.content))
        norm_df = _normalize_raw_split(raw_df, split_name)
        norm_df.to_parquet(paths[split_name], index=False)
        print(f"Downloaded & validated [{split_name}]: {len(norm_df)} real clauses ({int(norm_df['is_unfair'].sum())} unfair)")


if __name__ == "__main__":
    main()
