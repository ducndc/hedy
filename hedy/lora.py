"""LoRA (Low-Rank Adaptation) from scratch.

Freeze a pretrained weight W (out x in) and learn a low-rank update:
    y = W x + (alpha / r) * B (A x),     A: r x in,  B: out x r,  r << min(in, out)
B starts at 0, so the adapted model initially equals the base model exactly.
Trainable parameters per layer: r * (in + out) instead of in * out.
"""
import math

import torch
import torch.nn as nn

DEFAULT_TARGETS = ("qkv", "proj", "fc", "gate")   # every linear layer in our blocks (not the tied head)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        assert base.bias is None, "this repo's linears are bias-free"
        self.base, self.r, self.scale = base, r, alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))   # same as nn.Linear's default
        self.drop = nn.Dropout(dropout)
        base.weight.requires_grad_(False)

    def forward(self, x):
        return self.base(x) + (self.drop(x) @ self.lora_A.T @ self.lora_B.T) * self.scale

    @torch.no_grad()
    def merged(self) -> nn.Linear:
        """Fold the update into W: W' = W + scale * B A. Same outputs, zero inference overhead."""
        lin = nn.Linear(self.base.in_features, self.base.out_features, bias=False)
        lin.weight.copy_(self.base.weight + self.scale * (self.lora_B @ self.lora_A))
        return lin


def _targets(model: nn.Module, targets):
    for name, mod in model.named_modules():
        for cname, child in mod.named_children():
            if isinstance(child, nn.Linear) and cname in targets:
                yield mod, cname, f"{name}.{cname}".lstrip(".")


def apply_lora(model: nn.Module, r: int = 8, alpha: float | None = None, dropout: float = 0.0,
               targets=DEFAULT_TARGETS) -> nn.Module:
    """Freeze every base parameter and wrap the target linears with LoRA adapters (in place).
    Call AFTER loading the pretrained weights (wrapping renames state-dict keys)."""
    for p in model.parameters():
        p.requires_grad_(False)
    for parent, cname, _ in list(_targets(model, targets)):
        setattr(parent, cname, LoRALinear(getattr(parent, cname), r, alpha if alpha is not None else 2 * r, dropout))
    return model


def merge_lora(model: nn.Module) -> nn.Module:
    """Replace every LoRALinear by its merged plain Linear (in place)."""
    for parent in list(model.modules()):
        for cname, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                setattr(parent, cname, child.merged())
    for p in model.parameters():
        p.requires_grad_(True)
    return model


def lora_state_dict(model: nn.Module) -> dict:
    """Only the adapter tensors: what you ship/store per task (kilobytes, not the whole model)."""
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if "lora_" in k}


def count_params(model: nn.Module):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, sum(p.numel() for p in model.parameters())
