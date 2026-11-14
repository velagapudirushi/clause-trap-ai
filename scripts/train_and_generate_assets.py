"""Download real LexGLUE unfair_tos splits, train PyTorch LoRA, and generate README visual charts."""

from __future__ import annotations

import json
import logging
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from pathlib import Path
import sys

import torch
torch.set_num_threads(1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from clause_trap.config import ASSETS_DIR, EVAL_METRICS_PATH
from clause_trap.dataset_loader import download_or_load_unfair_tos
from clause_trap.trainer import train_and_evaluate_lora_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def generate_readme_charts(report: dict) -> None:
    """Generate high-resolution matplotlib charts for README.md from real evaluation metrics."""
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("dark_background")

    # ------------------------------------------------------------------
    # Chart 1: 3-Stage LoRA Ablation (Binary Trap F1 & 8-Class Macro F1 + Loss Curve)
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=180)
    fig.patch.set_facecolor("#0f172a")
    for ax in (ax1, ax2):
        ax.set_facecolor("#1e293b")

    stages = [
        "Stage 1\nZero-Shot Base",
        "Stage 2\nLoRA SFT",
        "Stage 3\nHard-Neg Aligned LoRA",
    ]
    bin_f1 = [
        report["stage_1_zeroshot_base"]["binary_trap_f1"],
        report["stage_2_lora_sft"]["binary_trap_f1"],
        report["stage_3_hard_negative_aligned_lora"]["binary_trap_f1"],
    ]
    macro_f1 = [
        report["stage_1_zeroshot_base"]["macro_f1"],
        report["stage_2_lora_sft"]["macro_f1"],
        report["stage_3_hard_negative_aligned_lora"]["macro_f1"],
    ]

    x = np.arange(len(stages))
    width = 0.34
    b1 = ax1.bar(x - width / 2, bin_f1, width, label="Binary Trap Detection F1", color="#ef4444", edgecolor="#f8fafc")
    b2 = ax1.bar(x + width / 2, macro_f1, width, label="8-Category Macro F1", color="#38bdf8", edgecolor="#f8fafc")

    ax1.set_ylim(0, 1.08)
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages, fontsize=10, fontweight="bold")
    ax1.set_ylabel("Out-of-Sample Test F1 Score", fontsize=11)
    ax1.set_title("3-Stage PyTorch LoRA Fine-Tuning Ablation (Unseen Contracts)", fontsize=12, fontweight="bold", pad=12)
    ax1.legend(loc="upper left", frameon=True, facecolor="#0f172a")
    ax1.grid(axis="y", linestyle="--", alpha=0.25)

    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                h + 0.02,
                f"{h:.3f}",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight="bold",
                color="#f8fafc",
            )

    curve = report["training_loss_curve"]
    epochs = [pt["epoch"] for pt in curve]
    losses = [pt["train_loss"] for pt in curve]
    ax2.plot(epochs, losses, marker="o", linewidth=2.8, markersize=7, color="#10b981", label="LoRA Weighted BCE Loss")
    ax2.axvline(x=3.5, color="#f59e0b", linestyle="--", linewidth=1.5, label="Stage 2 → Stage 3 Hard-Negative Shift")
    ax2.set_xlabel("Training Epoch (LoRA SFT → Hard-Negative Alignment)", fontsize=11)
    ax2.set_ylabel("Training Loss", fontsize=11)
    ax2.set_title("PyTorch LoRA Adapter Convergence on Apple M4", fontsize=12, fontweight="bold", pad=12)
    ax2.legend(loc="upper right", frameon=True, facecolor="#0f172a")
    ax2.grid(True, linestyle="--", alpha=0.25)

    plt.tight_layout()
    chart1_path = ASSETS_DIR / "lora_ablation_results.png"
    fig.savefig(chart1_path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)

    # ------------------------------------------------------------------
    # Chart 2: Per-Category F1 Breakdown Across All 8 Legal Trap Types
    # ------------------------------------------------------------------
    per_cat = report["stage_3_hard_negative_aligned_lora"]["per_category"]
    per_cat_sorted = sorted(per_cat, key=lambda d: d["f1"])
    names = [d["name"] for d in per_cat_sorted]
    f1s = [d["f1"] for d in per_cat_sorted]
    precs = [d["precision"] for d in per_cat_sorted]
    recs = [d["recall"] for d in per_cat_sorted]

    fig2, ax = plt.subplots(figsize=(11.5, 5.4), dpi=180)
    fig2.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, f1s, height=0.58, color="#f43f5e", edgecolor="#f8fafc", label="Test F1 Score")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=10.5, fontweight="bold")
    ax.set_xlim(0, 1.26)
    ax.set_xlabel("Held-Out Test F1 Score", fontsize=11)
    ax.set_title("ClauseTrap AI: Per-Category Test F1 Across 8 Unfair Contract Trap Types", fontsize=12.5, fontweight="bold", pad=12)
    ax.grid(axis="x", linestyle="--", alpha=0.25)

    for idx, bar in enumerate(bars):
        w = bar.get_width()
        ax.text(
            w + 0.015,
            bar.get_y() + bar.get_height() / 2.0,
            f"F1: {w:.3f} (P={precs[idx]:.2f}, R={recs[idx]:.2f})",
            va="center",
            fontsize=9.2,
            color="#e2e8f0",
            fontweight="bold",
        )

    plt.tight_layout()
    chart2_path = ASSETS_DIR / "per_category_trap_f1.png"
    fig2.savefig(chart2_path, dpi=180, facecolor=fig2.get_facecolor())
    plt.close(fig2)

    # ------------------------------------------------------------------
    # Chart 3: Hero Architecture & Single-Screen Highlighter Banner
    # ------------------------------------------------------------------
    fig3, ax3 = plt.subplots(figsize=(14.0, 4.8), dpi=180)
    fig3.patch.set_facecolor("#090d16")
    ax3.set_facecolor("#090d16")
    ax3.set_xlim(0, 100)
    ax3.set_ylim(0, 36)
    ax3.axis("off")

    ax3.text(
        50,
        32.5,
        "ClauseTrap AI — Hugging Face + PyTorch LoRA Contract Red-Flag Highlighter",
        ha="center",
        va="center",
        fontsize=15.5,
        fontweight="bold",
        color="#f8fafc",
    )
    ax3.text(
        50,
        29.2,
        "Trained on 9,414 Real Clauses (LexGLUE unfair_tos)  •  From-Scratch PyTorch LoRA (r=8)  •  0.0% Hallucinated Spans",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#94a3b8",
    )

    boxes = [
        (
            3,
            6,
            20,
            19,
            "#1e293b",
            "#38bdf8",
            "1. Contract Segmenter\n(segmenter.py)",
            "• Splits ToS / Lease / Offer\n  into atomic clauses\n• Preserves exact character\n  offsets [start, end]",
        ),
        (
            27,
            6,
            22,
            19,
            "#1e1b4b",
            "#a855f7",
            "2. HF + PyTorch LoRA\n(lora_layers.py)",
            "• Frozen 63M HF Transformer\n• Trainable LoRA (A, B)\n  h = W₀x + (α/r)·B·A·x\n• 8 Unfair Trap Classes",
        ),
        (
            53,
            6,
            20,
            19,
            "#064e3b",
            "#10b981",
            "3. Verbatim Quote Gate\n(quote_verifier.py)",
            "• Exact substring check:\n  source[start:end] == quote\n• Hard-rejects any span drift\n• 0.0% Hallucination Rate",
        ),
        (
            77,
            6,
            20,
            19,
            "#450a0a",
            "#ef4444",
            "4. Single-Tab Highlighter\n(streamlit_app.py)",
            "• Inline <mark> red flags\n• Contract Safety Grade (A-F)\n• Plain-English Explanation\n• Negotiation Pushback Qs",
        ),
    ]

    for bx, by, bw, bh, face_col, edge_col, title_txt, body_txt in boxes:
        rect = matplotlib.patches.FancyBboxPatch(
            (bx, by),
            bw,
            bh,
            boxstyle="round,pad=0.6,rounding_size=1.4",
            linewidth=2.2,
            edgecolor=edge_col,
            facecolor=face_col,
        )
        ax3.add_patch(rect)
        ax3.text(
            bx + bw / 2,
            by + bh - 3.3,
            title_txt,
            ha="center",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            color="#f8fafc",
        )
        ax3.text(
            bx + 1.2,
            by + bh / 2 - 2.2,
            body_txt,
            ha="left",
            va="center",
            fontsize=8.7,
            color="#e2e8f0",
            linespacing=1.4,
        )

    for arrow_x in [23.4, 49.4, 73.4]:
        ax3.annotate(
            "",
            xy=(arrow_x + 3.2, 15.5),
            xytext=(arrow_x, 15.5),
            arrowprops=dict(arrowstyle="->", lw=2.5, color="#f59e0b"),
        )

    plt.tight_layout()
    chart3_path = ASSETS_DIR / "clausetrap_hero_banner.png"
    fig3.savefig(chart3_path, dpi=180, facecolor=fig3.get_facecolor())
    plt.close(fig3)
    logger.info("Saved README charts -> %s, %s, %s", chart1_path, chart2_path, chart3_path)


def main() -> None:
    if "--charts-only" in sys.argv and EVAL_METRICS_PATH.exists():
        with open(EVAL_METRICS_PATH, "r", encoding="utf-8") as f:
            report = json.load(f)
        generate_readme_charts(report)
        return
    splits = download_or_load_unfair_tos()
    report = train_and_evaluate_lora_pipeline(splits)
    generate_readme_charts(report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
