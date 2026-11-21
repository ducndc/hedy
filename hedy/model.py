"""Decoder-only transformer (GPT-2 style, pre-norm)."""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.1
    pos: str = "learned"    # "learned" (GPT-2) | "rope"
    norm: str = "layernorm"  # "layernorm" | "rmsnorm"
    mlp: str = "gelu"        # "gelu" | "swiglu"


class RMSNorm(nn.Module):
    """x / rms(x) * g. LayerNorm without mean-centering and without bias."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


def make_norm(cfg: GPTConfig):
    return RMSNorm(cfg.n_embd) if cfg.norm == "rmsnorm" else nn.LayerNorm(cfg.n_embd)


def rope_cache(T: int, head_dim: int, device, base: float = 10000.0):
    """cos/sin tables of shape (T, head_dim/2): angle[t, i] = t * base^(-2i/head_dim)."""
    inv_freq = base ** (-torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    ang = torch.outer(torch.arange(T, device=device).float(), inv_freq)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    """x: (B, heads, T, d). Rotate each consecutive (even, odd) channel pair by its position angle."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = cfg.dropout
        self.rope = cfg.pos == "rope"
        assert not self.rope or (cfg.n_embd // cfg.n_head) % 2 == 0

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        if self.rope:  # position enters here, on q and k only (not v), so scores depend on relative offset
            cos, sin = rope_cache(T, C // self.n_head, x.device)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.swiglu = cfg.mlp == "swiglu"
        if self.swiglu:
            # 3 matrices instead of 2, so shrink hidden to ~(8/3)C to keep parameter count ~equal to 4C GELU
            h = (int(8 * cfg.n_embd / 3) + 7) // 8 * 8
            self.gate = nn.Linear(cfg.n_embd, h, bias=False)
        else:
            h = 4 * cfg.n_embd
        self.fc = nn.Linear(cfg.n_embd, h, bias=False)
        self.proj = nn.Linear(h, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        h = F.silu(self.gate(x)) * self.fc(x) if self.swiglu else F.gelu(self.fc(x))
        return self.drop(self.proj(h))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = make_norm(cfg)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = make_norm(cfg)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd) if cfg.pos == "learned" else None
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = make_norm(cfg)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # weight tying
        self.apply(self._init)
        for n, p in self.named_parameters():  # scaled init for residual projections
            if n.endswith("proj.weight"):
                nn.init.normal_(p, 0.0, 0.02 / (2 * cfg.n_layer) ** 0.5)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(T, device=idx.device))
        x = self.drop(x)
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.block_size:])
            logits = logits[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx
