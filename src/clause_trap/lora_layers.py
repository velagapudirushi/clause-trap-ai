"""From-scratch PyTorch Low-Rank Adaptation (`LoRALinear`) for Hugging Face Transformers.

Implements the exact LoRA equation from Hu et al. (2021):
    h = W_0 x + (alpha / r) * B(A(dropout(x)))
where `W_0` is frozen, `A ~ N(0, sigma^2)` is `(r, in_features)`, and `B = 0` is `(out_features, r)`.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wraps a frozen `nn.Linear` layer with trainable low-rank matrices `lora_A` and `lora_B`."""

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / max(float(r), 1.0)
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0.0 else nn.Identity()

        # Freeze original pretrained weights & bias
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False

        # Trainable low-rank adapter matrices A (r x in) and B (out x r)
        device = base_layer.weight.device
        dtype = base_layer.weight.dtype
        self.lora_A = nn.Parameter(torch.empty((r, self.in_features), device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros((self.out_features, r), device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)
        lora_out = (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_out + lora_out


def inject_lora_into_transformer(
    model: nn.Module,
    r: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.05,
    target_suffixes: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "fc1",
        "fc2",
        "query",
        "key",
        "value",
        "dense",
        "q_lin",
        "k_lin",
        "v_lin",
    ),
) -> dict[str, int]:
    """Freeze base encoder weights and inject `LoRALinear` into matching attention/MLP projections."""
    for param in model.parameters():
        param.requires_grad = False

    replaced_layers = 0
    for module_name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and any(
                child_name.endswith(sfx) for sfx in target_suffixes
            ):
                # Do not wrap the final classification head with LoRA; unfreeze it directly below
                if "classifier" in module_name or "classifier" in child_name:
                    continue
                setattr(
                    module,
                    child_name,
                    LoRALinear(child, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout),
                )
                replaced_layers += 1

    # Always keep the task classification head trainable
    for name, param in model.named_parameters():
        if "classifier" in name or "score" in name or "lora_" in name:
            param.requires_grad = True

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "lora_injected_linear_layers": replaced_layers,
        "total_parameters": int(total_params),
        "trainable_parameters": int(trainable_params),
        "trainable_pct": round(100.0 * trainable_params / max(total_params, 1), 2),
    }
