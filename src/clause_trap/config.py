"""Canonical repository paths and YAML configuration loader for ClauseTrap AI."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "trap_taxonomy.yaml"
PROCESSED_DATA_DIR = REPO_ROOT / "data" / "processed"
TRAIN_PARQUET_PATH = PROCESSED_DATA_DIR / "unfair_tos_train.parquet"
VAL_PARQUET_PATH = PROCESSED_DATA_DIR / "unfair_tos_val.parquet"
TEST_PARQUET_PATH = PROCESSED_DATA_DIR / "unfair_tos_test.parquet"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
LORA_CHECKPOINT_PATH = ARTIFACTS_DIR / "lora_clause_trap_bundle.pt"
EVAL_METRICS_PATH = ARTIFACTS_DIR / "eval_metrics.json"
ASSETS_DIR = REPO_ROOT / "assets"


def load_taxonomy_config(path: Path | None = None) -> dict[str, Any]:
    """Load the 8 LexGLUE unfair_tos trap definitions, severities, and sample contracts."""
    cfg_file = path or CONFIG_PATH
    with open(cfg_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
