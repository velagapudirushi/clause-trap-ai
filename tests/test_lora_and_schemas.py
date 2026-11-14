"""Unit tests for from-scratch PyTorch `LoRALinear` layers and Pandera/Pydantic contracts."""

import torch
import torch.nn as nn
import pandas as pd
import pytest

from clause_trap.lora_layers import LoRALinear, inject_lora_into_transformer
from clause_trap.schemas import validate_clause_df


def test_lora_linear_zero_init_preserves_pretrained_forward_pass() -> None:
    torch.manual_seed(42)
    base = nn.Linear(32, 16)
    x = torch.randn(4, 32)
    expected = base(x).detach().clone()

    lora_layer = LoRALinear(base, r=4, lora_alpha=8.0, lora_dropout=0.0)
    assert base.weight.requires_grad is False
    assert lora_layer.lora_A.requires_grad is True
    assert lora_layer.lora_B.requires_grad is True

    # Because B is initialized to zeros, step-0 output must match base_layer(x) exactly
    actual = lora_layer(x)
    assert torch.allclose(expected, actual, atol=1e-6)


def test_inject_lora_freezes_backbone_and_exposes_adapters() -> None:
    class ToyTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(64, 64)
            self.v_proj = nn.Linear(64, 64)
            self.other = nn.Linear(64, 64)
            self.classifier = nn.Linear(64, 8)

    toy = ToyTransformer()
    stats = inject_lora_into_transformer(toy, r=4, lora_alpha=8.0)
    assert stats["lora_injected_linear_layers"] == 2
    assert isinstance(toy.q_proj, LoRALinear)
    assert isinstance(toy.v_proj, LoRALinear)
    assert toy.other.weight.requires_grad is False
    assert toy.classifier.weight.requires_grad is True


def test_pandera_schema_rejects_invalid_is_unfair_flag() -> None:
    good = pd.DataFrame([{"text": "Valid legal clause text here.", "is_unfair": 1}])
    validated = validate_clause_df(good)
    assert len(validated) == 1

    bad = pd.DataFrame([{"text": "Invalid label value.", "is_unfair": 5}])
    with pytest.raises(Exception):
        validate_clause_df(bad)
