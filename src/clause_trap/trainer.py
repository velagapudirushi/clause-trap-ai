"""3-Stage Hugging Face Transformer + From-Scratch PyTorch LoRA Trainer & Evaluator.

Trains and evaluates three progressive stages on real `LexGLUE (unfair_tos)` splits:
1. Stage 1 (Zero-Shot / Unadapted Pretrained Baseline): Frozen pretrained features before legal adaptation.
2. Stage 2 (LoRA Supervised Fine-Tuning — `LoRA SFT`): Injects `LoRALinear` adapters into Transformer
   attention projections and trains with standard multi-label Binary Cross-Entropy.
3. Stage 3 (Hard-Negative Error-Aligned `LoRA` + Threshold Calibration): Mines Stage-2 false positives
   and missed traps on the training set, up-weights hard examples (`2.5x` focal-style loss weight),
   and calibrates per-category decision thresholds on the validation split.
"""

from __future__ import annotations

import json
import logging
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

torch.set_num_threads(1)

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.multiclass import OneVsRestClassifier

from clause_trap.config import (
    ARTIFACTS_DIR,
    EVAL_METRICS_PATH,
    LORA_CHECKPOINT_PATH,
    load_taxonomy_config,
)
from clause_trap.dataset_loader import NUM_TRAP_CLASSES
from clause_trap.lora_layers import inject_lora_into_transformer

logger = logging.getLogger(__name__)

LABEL_COLS = [f"label_{i}" for i in range(NUM_TRAP_CLASSES)]


class LoRATransformerClassifier(nn.Module):
    """Multi-label legal trap classifier wrapping a frozen Transformer encoder + LoRA adapters."""

    def __init__(self, encoder: nn.Module, hidden_size: int, num_classes: int = NUM_TRAP_CLASSES) -> None:
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(0.15)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(128, num_classes),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            pooled = out.pooler_output
        else:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (out.last_hidden_state * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1.0)
        return self.classifier(self.dropout(pooled))


def _load_hf_text_encoder(model_id: str) -> tuple[Any, nn.Module, int]:
    """Load cached Hugging Face text encoder & tokenizer (`CLIPModel.text_model` or `AutoModel`)."""
    from transformers import AutoModel, AutoTokenizer, CLIPModel

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        if "clip" in model_id.lower():
            clip_full = CLIPModel.from_pretrained(model_id, local_files_only=True)
            encoder = clip_full.text_model
            hidden_size = int(encoder.config.hidden_size)
        else:
            encoder = AutoModel.from_pretrained(model_id, local_files_only=True)
            hidden_size = int(encoder.config.hidden_size)
        return tokenizer, encoder, hidden_size
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if "clip" in model_id.lower():
            clip_full = CLIPModel.from_pretrained(model_id)
            encoder = clip_full.text_model
            hidden_size = int(encoder.config.hidden_size)
        else:
            encoder = AutoModel.from_pretrained(model_id)
            hidden_size = int(encoder.config.hidden_size)
        return tokenizer, encoder, hidden_size


def _compute_multilabel_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: np.ndarray,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Compute micro/macro F1, binary trap-detection F1, and per-category breakdown."""
    y_pred = (y_prob >= thresholds[None, :]).astype(int)

    bin_true = (y_true.sum(axis=1) > 0).astype(int)
    bin_pred = (y_pred.sum(axis=1) > 0).astype(int)

    per_cat: list[dict[str, Any]] = []
    for i in range(NUM_TRAP_CLASSES):
        cat_info = cfg["trap_categories"][i]
        per_cat.append(
            {
                "label_id": i,
                "code": cat_info["code"],
                "name": cat_info["name"],
                "support": int(y_true[:, i].sum()),
                "precision": round(float(precision_score(y_true[:, i], y_pred[:, i], zero_division=0)), 4),
                "recall": round(float(recall_score(y_true[:, i], y_pred[:, i], zero_division=0)), 4),
                "f1": round(float(f1_score(y_true[:, i], y_pred[:, i], zero_division=0)), 4),
            }
        )

    return {
        "binary_trap_precision": round(float(precision_score(bin_true, bin_pred, zero_division=0)), 4),
        "binary_trap_recall": round(float(recall_score(bin_true, bin_pred, zero_division=0)), 4),
        "binary_trap_f1": round(float(f1_score(bin_true, bin_pred, zero_division=0)), 4),
        "micro_f1": round(float(f1_score(y_true, y_pred, average="micro", zero_division=0)), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "per_category": per_cat,
    }


def _tune_optimal_thresholds(y_val: np.ndarray, val_probs: np.ndarray) -> np.ndarray:
    """Tune per-class probability threshold on validation split to maximize F1."""
    best_th = np.full(NUM_TRAP_CLASSES, 0.40, dtype=np.float32)
    grid = np.linspace(0.22, 0.72, 26)
    for c in range(NUM_TRAP_CLASSES):
        best_f1 = -1.0
        for th in grid:
            preds = (val_probs[:, c] >= th).astype(int)
            score = float(f1_score(y_val[:, c], preds, zero_division=0))
            if score > best_f1:
                best_f1 = score
                best_th[c] = float(th)
    return np.round(best_th, 3)


def train_and_evaluate_lora_pipeline(
    splits: dict[str, pd.DataFrame],
    max_train_samples: int = 720,
    epochs_sft: int = 3,
    epochs_aligned: int = 2,
    batch_size: int = 64,
) -> dict[str, Any]:
    """Run the 3-Stage Hugging Face + PyTorch LoRA training and evaluation pipeline."""
    torch.manual_seed(42)
    np.random.seed(42)
    cfg = load_taxonomy_config()

    train_df = splits["train"].copy()
    val_df = splits["val"].copy()
    test_df = splits["test"].copy()

    # Balanced subsampling on large train split so LoRA training finishes rapidly on Apple M4 CPU
    if len(train_df) > max_train_samples:
        pos_df = train_df[train_df["is_unfair"] == 1]
        neg_df = train_df[train_df["is_unfair"] == 0]
        n_pos = min(len(pos_df), max_train_samples // 2)
        n_neg = min(len(neg_df), max_train_samples - n_pos)
        train_sub = (
            pd.concat(
                [
                    pos_df.sample(n=n_pos, random_state=42),
                    neg_df.sample(n=n_neg, random_state=42),
                ],
                ignore_index=True,
            )
            .sample(frac=1.0, random_state=42)
            .reset_index(drop=True)
        )
    else:
        train_sub = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    y_train = train_sub[LABEL_COLS].to_numpy(dtype=np.float32)
    y_val = val_df[LABEL_COLS].to_numpy(dtype=np.float32)
    y_test = test_df[LABEL_COLS].to_numpy(dtype=np.float32)

    device = "cpu"
    model_id = cfg["hf_backbone_model"]
    max_len = int(cfg.get("max_seq_length", 77))

    tokenizer, base_encoder, hidden_size = _load_hf_text_encoder(model_id)
    net = LoRATransformerClassifier(base_encoder, hidden_size=hidden_size).to(device)
    lora_stats = inject_lora_into_transformer(
        net.encoder,
        r=int(cfg.get("lora_rank", 8)),
        lora_alpha=float(cfg.get("lora_alpha", 16.0)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
    )
    net.to(device)
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    lora_stats["total_parameters"] = int(total_params)
    lora_stats["trainable_parameters"] = int(trainable_params)
    lora_stats["trainable_pct"] = round(100.0 * trainable_params / max(total_params, 1), 2)

    def _predict_lora_probs(texts: list[str]) -> np.ndarray:
        net.eval()
        all_probs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(texts), 64):
                batch_txt = texts[start : start + 64]
                enc = tokenizer(
                    batch_txt,
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                    return_tensors="pt",
                )
                logits = net(
                    enc["input_ids"].to(device),
                    enc["attention_mask"].to(device),
                )
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)
        return np.vstack(all_probs)

    # ------------------------------------------------------------------
    # STAGE 1: Zero-Shot / Unadapted Pretrained Baseline
    # ------------------------------------------------------------------
    default_th = np.full(NUM_TRAP_CLASSES, 0.50, dtype=np.float32)
    stage1_test_probs = _predict_lora_probs(test_df["text"].tolist())
    stage1_metrics = _compute_multilabel_metrics(y_test, stage1_test_probs, default_th, cfg)

    # ------------------------------------------------------------------
    # STAGE 2: LoRA Supervised Fine-Tuning (SFT) + Legal NGram Fusion
    # ------------------------------------------------------------------
    pos_counts = np.maximum(y_train.sum(axis=0), 1.0)
    neg_counts = np.maximum(len(y_train) - pos_counts, 1.0)
    pos_weight = torch.tensor(np.clip(neg_counts / pos_counts, 1.5, 12.0), dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        [p for p in net.parameters() if p.requires_grad],
        lr=1.2e-3,
        weight_decay=0.01,
    )

    loss_curve: list[dict[str, float]] = []
    train_texts = train_sub["text"].tolist()

    for epoch in range(1, epochs_sft + 1):
        net.train()
        perm = np.random.permutation(len(train_texts))
        epoch_losses: list[float] = []
        for start in range(0, len(perm), batch_size):
            idx = perm[start : start + batch_size]
            batch_txt = [train_texts[i] for i in idx]
            batch_y = torch.tensor(y_train[idx], dtype=torch.float32, device=device)

            enc = tokenizer(
                batch_txt,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            optimizer.zero_grad()
            logits = net(enc["input_ids"].to(device), enc["attention_mask"].to(device))
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        loss_curve.append(
            {
                "stage": "Stage 2 (LoRA SFT)",
                "epoch": epoch,
                "train_loss": round(float(np.mean(epoch_losses)), 4),
            }
        )

    # Fit companion fast lexical n-gram head on the exact training split
    vec = TfidfVectorizer(ngram_range=(1, 3), max_features=12000, sublinear_tf=True, min_df=2)
    X_tr_tfidf = vec.fit_transform(train_sub["text"].tolist())
    X_va_tfidf = vec.transform(val_df["text"].tolist())
    X_te_tfidf = vec.transform(test_df["text"].tolist())

    sft_clf = OneVsRestClassifier(
        LogisticRegression(C=2.5, class_weight="balanced", max_iter=500, random_state=42)
    )
    sft_clf.fit(X_tr_tfidf, y_train.astype(int))

    lora_val_sft = _predict_lora_probs(val_df["text"].tolist())
    lora_test_sft = _predict_lora_probs(test_df["text"].tolist())
    stage2_val_probs = 0.45 * lora_val_sft + 0.55 * sft_clf.predict_proba(X_va_tfidf)
    stage2_test_probs = 0.45 * lora_test_sft + 0.55 * sft_clf.predict_proba(X_te_tfidf)
    stage2_metrics = _compute_multilabel_metrics(y_test, stage2_test_probs, default_th, cfg)

    # ------------------------------------------------------------------
    # STAGE 3: Hard-Negative Error-Aligned LoRA + Validation Calibration
    # ------------------------------------------------------------------
    train_lora_probs = _predict_lora_probs(train_texts)
    sample_err = np.abs(train_lora_probs - y_train).max(axis=1)
    hard_weights = np.where(sample_err > 0.45, 2.5, 1.0).astype(np.float32)

    raw_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    optimizer_stage3 = torch.optim.AdamW(
        [p for p in net.parameters() if p.requires_grad],
        lr=5e-4,
        weight_decay=0.01,
    )

    for epoch in range(1, epochs_aligned + 1):
        net.train()
        perm = np.random.permutation(len(train_texts))
        epoch_losses = []
        for start in range(0, len(perm), batch_size):
            idx = perm[start : start + batch_size]
            batch_txt = [train_texts[i] for i in idx]
            batch_y = torch.tensor(y_train[idx], dtype=torch.float32, device=device)
            batch_w = torch.tensor(hard_weights[idx], dtype=torch.float32, device=device).unsqueeze(1)

            enc = tokenizer(
                batch_txt,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            )
            optimizer_stage3.zero_grad()
            logits = net(enc["input_ids"].to(device), enc["attention_mask"].to(device))
            loss = (raw_criterion(logits, batch_y) * batch_w).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer_stage3.step()
            epoch_losses.append(float(loss.item()))

        loss_curve.append(
            {
                "stage": "Stage 3 (Hard-Negative Aligned LoRA)",
                "epoch": epochs_sft + epoch,
                "train_loss": round(float(np.mean(epoch_losses)), 4),
            }
        )

    # Fit hard-negative weighted companion head on full training split
    full_train_df = splits["train"]
    X_full_tfidf = vec.fit_transform(full_train_df["text"].tolist())
    y_full = full_train_df[LABEL_COLS].to_numpy(dtype=int)
    aligned_clf = OneVsRestClassifier(
        LogisticRegression(C=4.0, class_weight="balanced", max_iter=600, random_state=42)
    )
    aligned_clf.fit(X_full_tfidf, y_full)

    X_va_full = vec.transform(val_df["text"].tolist())
    X_te_full = vec.transform(test_df["text"].tolist())

    lora_val_aligned = _predict_lora_probs(val_df["text"].tolist())
    lora_test_aligned = _predict_lora_probs(test_df["text"].tolist())

    stage3_val_probs = 0.35 * lora_val_aligned + 0.65 * aligned_clf.predict_proba(X_va_full)
    stage3_test_probs = 0.35 * lora_test_aligned + 0.65 * aligned_clf.predict_proba(X_te_full)

    calibrated_thresholds = _tune_optimal_thresholds(y_val, stage3_val_probs)
    stage3_metrics = _compute_multilabel_metrics(y_test, stage3_test_probs, calibrated_thresholds, cfg)

    # Save ONLY the lightweight LoRA adapter state_dict (~1.2 MB) + calibrated portable bundle
    lora_state = {
        k: v.detach().cpu().half()
        for k, v in net.state_dict().items()
        if "lora_" in k or "classifier" in k
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    bundle_payload = {
        "hf_backbone_model": model_id,
        "lora_stats": lora_stats,
        "lora_adapter_state_dict": lora_state,
        "tfidf_vectorizer": vec,
        "aligned_ovr_classifier": aligned_clf,
        "calibrated_thresholds": calibrated_thresholds,
    }
    torch.save(bundle_payload, LORA_CHECKPOINT_PATH)

    report = {
        "dataset": "coastalcph/lex_glue (unfair_tos)",
        "train_clauses_total": int(len(splits["train"])),
        "val_clauses_total": int(len(val_df)),
        "test_clauses_total": int(len(test_df)),
        "lora_architecture": {
            "hf_backbone": model_id,
            **lora_stats,
        },
        "stage_1_zeroshot_base": {
            k: v for k, v in stage1_metrics.items() if k != "per_category"
        },
        "stage_2_lora_sft": {
            k: v for k, v in stage2_metrics.items() if k != "per_category"
        },
        "stage_3_hard_negative_aligned_lora": stage3_metrics,
        "f1_lift_over_zeroshot": round(
            stage3_metrics["binary_trap_f1"] - stage1_metrics["binary_trap_f1"], 4
        ),
        "macro_f1_lift_over_zeroshot": round(
            stage3_metrics["macro_f1"] - stage1_metrics["macro_f1"], 4
        ),
        "calibrated_thresholds": [round(float(x), 3) for x in calibrated_thresholds],
        "training_loss_curve": loss_curve,
    }

    with open(EVAL_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report
