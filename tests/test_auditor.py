"""Unit tests for live ContractTrapAuditor red-flag detection and inline HTML highlighting."""

from clause_trap.auditor import ContractTrapAuditor
from clause_trap.config import LORA_CHECKPOINT_PATH, load_taxonomy_config


def test_auditor_flags_predatory_contract_and_passes_fair_contract() -> None:
    assert LORA_CHECKPOINT_PATH.exists(), "Run train_and_generate_assets.py before running auditor tests."
    auditor = ContractTrapAuditor(LORA_CHECKPOINT_PATH)
    cfg = load_taxonomy_config()

    predatory_txt = cfg["sample_contracts"]["Ride-Share & Delivery App Terms (Real Unfair ToS Excerpt)"]
    res_bad = auditor.audit_contract(predatory_txt)
    assert res_bad.safety_grade in ("D", "F")
    assert res_bad.flagged_clause_count >= 4
    assert res_bad.verified_quote_rate_pct == 100.0
    assert all(f.quote_verified_in_source for f in res_bad.flags)

    html_out = auditor.render_highlighted_document_html(predatory_txt, res_bad)
    assert "<mark" in html_out
    assert "🚩" in html_out

    fair_txt = cfg["sample_contracts"]["Clean & Fair Consumer-Friendly Service Agreement (Grade A Demo)"]
    res_good = auditor.audit_contract(fair_txt)
    assert res_good.safety_grade in ("A", "B")
    assert res_good.risk_score_0_to_100 < res_bad.risk_score_0_to_100
