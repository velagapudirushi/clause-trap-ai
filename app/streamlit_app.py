"""Single-Screen Contract & Terms-of-Service Red-Flag Highlighter (ClauseTrap AI)."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from clause_trap.auditor import ContractTrapAuditor
from clause_trap.config import EVAL_METRICS_PATH, LORA_CHECKPOINT_PATH, load_taxonomy_config
from clause_trap.dataset_loader import download_or_load_unfair_tos
from clause_trap.trainer import train_and_evaluate_lora_pipeline

st.set_page_config(
    page_title="ClauseTrap AI | Contract & Terms Red-Flag Highlighter",
    page_icon="🚩",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading Fine-Tuned Hugging Face + PyTorch LoRA Contract Auditor...")
def load_auditor_and_metrics() -> tuple[ContractTrapAuditor, dict]:
    if not LORA_CHECKPOINT_PATH.exists() or not EVAL_METRICS_PATH.exists():
        splits = download_or_load_unfair_tos()
        train_and_evaluate_lora_pipeline(splits)
    auditor = ContractTrapAuditor(LORA_CHECKPOINT_PATH)
    with open(EVAL_METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    return auditor, metrics


def main() -> None:
    st.title("🚩 ClauseTrap AI: Contract & Fine-Print Red-Flag Highlighter")
    st.caption(
        "Paste any App Terms of Service, Privacy Policy, Rental Lease, or Job Offer Letter below. "
        "Powered by a Hugging Face Transformer fine-tuned with from-scratch PyTorch LoRA on real "
        "`LexGLUE (unfair_tos)` contracts + 100% deterministic verbatim quote verification."
    )

    auditor, metrics = load_auditor_and_metrics()
    cfg = load_taxonomy_config()
    samples: dict[str, str] = cfg["sample_contracts"]

    with st.sidebar:
        st.header("🧠 Model & Guardrail")
        lora_arch = metrics["lora_architecture"]
        s1 = metrics["stage_1_zeroshot_base"]
        s3 = metrics["stage_3_hard_negative_aligned_lora"]
        st.markdown(
            f"- **Dataset:** `{metrics['dataset']}`\n"
            f"- **Splits:** `{metrics['train_clauses_total']:,}` train / `{metrics['test_clauses_total']:,}` test\n"
            f"- **HF Backbone:** `{lora_arch['hf_backbone']}`\n"
            f"- **LoRA Layers Injected:** `{lora_arch['lora_injected_linear_layers']}` (`r={cfg['lora_rank']}`, `α={cfg['lora_alpha']}`)\n"
            f"- **Trainable Weights:** `{lora_arch['trainable_parameters']:,}` / `{lora_arch['total_parameters']:,}` (`{lora_arch['trainable_pct']}%`)\n"
            f"- **Trap Detection F1:** `{s1['binary_trap_f1']:.3f}` (Base) → **`{s3['binary_trap_f1']:.3f}` (LoRA)**\n"
            f"- **8-Class Macro F1:** `{s1['macro_f1']:.3f}` (Base) → **`{s3['macro_f1']:.3f}` (LoRA)**\n"
            "- **Quote Hallucinations:** **`0.0%`** *(Deterministic Substring Gate)*"
        )
        st.divider()
        st.markdown("### 8 Legal Trap Categories")
        for idx in range(8):
            cat = cfg["trap_categories"][idx]
            st.markdown(f"- **{cat['name']}** *(Severity `{cat['severity']}/10`)*")

    # Single-Screen Contract Input Controls
    c_preset, c_upload = st.columns([1.8, 1.0])
    with c_preset:
        preset_options = list(samples.keys()) + ["✍️ Paste Custom Contract / Offer Letter"]
        selected_preset = st.selectbox("Choose a Sample Contract or Paste Your Own", preset_options)
    with c_upload:
        uploaded_file = st.file_uploader("Or upload a `.txt` / `.md` contract file", type=["txt", "md"])

    if uploaded_file is not None:
        default_text = uploaded_file.read().decode("utf-8", errors="ignore")
    elif selected_preset in samples:
        default_text = samples[selected_preset].strip()
    else:
        default_text = (
            "We may modify these terms or subscription fees at any time in our sole discretion without notice.\n"
            "Any dispute arising under this Agreement shall be resolved exclusively through binding individual arbitration, "
            "and you waive your right to participate in a class action or jury trial."
        )

    contract_text = st.text_area(
        "Contract / Terms of Service Text to Audit",
        value=default_text,
        height=175,
    )

    st.button("🔍 Scan Contract & Highlight Red Flags", type="primary")

    audit = auditor.audit_contract(contract_text)

    # Top KPI Strip
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Contract Safety Grade",
        f"Grade {audit.safety_grade}",
        delta=f"Risk Score: {audit.risk_score_0_to_100:.0f}/100",
        delta_color="inverse" if audit.risk_score_0_to_100 >= 28 else "normal",
    )
    k2.metric("Clauses Scanned", f"{audit.total_clauses_scanned}")
    k3.metric(
        "Red-Flag Clauses Caught",
        f"{audit.flagged_clause_count}",
        delta=f"{audit.total_trap_hits} total trap triggers",
        delta_color="inverse" if audit.flagged_clause_count > 0 else "normal",
    )
    k4.metric(
        "Verbatim Quote Verification",
        f"{audit.verified_quote_rate_pct:.0f}%",
        delta="0 Hallucinated Spans",
    )

    if audit.safety_grade in ("D", "F"):
        st.error(f"**{audit.verdict_headline}**")
    elif audit.safety_grade in ("B", "C"):
        st.warning(f"**{audit.verdict_headline}**")
    else:
        st.success(f"**{audit.verdict_headline}**")

    col_doc, col_cards = st.columns([1.15, 1.0])

    with col_doc:
        st.subheader("📄 Inline Red-Flag Highlighter")
        highlighted_html = auditor.render_highlighted_document_html(contract_text, audit)
        st.markdown(highlighted_html, unsafe_allow_html=True)

    with col_cards:
        st.subheader(f"🚩 Detected Traps & Pushback Guide ({len(audit.flags)})")
        if not audit.flags:
            st.info(
                "No predatory clauses were triggered above the calibrated thresholds. "
                "Every clause in this document passed the 8-category unfairness check."
            )
        else:
            for idx, flag in enumerate(audit.flags, start=1):
                with st.container(border=True):
                    st.markdown(
                        f"**#{idx} · {flag.trap_name}** — "
                        f"`Severity {flag.severity}/10` · `Confidence {flag.confidence * 100:.0f}%` · "
                        f"`✅ Verified chars [{flag.start_char}:{flag.end_char}]`"
                    )
                    st.markdown(f"> *\"{flag.exact_quote}\"*")
                    st.markdown(f"**Why It's a Trap:** {flag.plain_english_warning}")
                    st.markdown(f"💬 **How to Push Back:** *{flag.pushback_question}*")


if __name__ == "__main__":
    main()
