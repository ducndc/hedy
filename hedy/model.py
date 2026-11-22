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
    n_kv_head: int = 0       # 0 = same as n_head (MHA); 1 = MQA; in between = GQA


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


def rope_cache(T: int, head_dim: int, device, base: float = 10000.0, offset: int = 0):
    """cos/sin tables of shape (T, head_dim/2): angle[t, i] = (offset+t) * base^(-2i/head_dim).
    `offset` = number of tokens already processed (non-zero when decoding with a KV-cache)."""
    inv_freq = base ** (-torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    ang = torch.outer(torch.arange(offset, offset + T, device=device).float(), inv_freq)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    """x: (B, heads, T, d). Rotate each consecutive (even, odd) channel pair by its position angle."""
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2)


class KVCache:
    """Per-layer cache of past keys/values, each (B, n_kv_head, T_past, head_dim).
    Simple version: grows by torch.cat each step (copies O(T) per token; a production cache
    preallocates a (B, H, max_len, d) buffer and writes in place)."""

    def __init__(self, n_layer: int):
        self.k: list = [None] * n_layer
        self.v: list = [None] * n_layer

    @property
    def seq_len(self) -> int:
        return 0 if self.k[0] is None else self.k[0].size(2)

    def update(self, layer: int, k, v):
        if self.k[layer] is not None:
            k, v = torch.cat([self.k[layer], k], dim=2), torch.cat([self.v[layer], v], dim=2)
        self.k[layer], self.v[layer] = k, v
        return k, v

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.k + self.v if t is not None)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head or cfg.n_head
        assert self.n_head % self.n_kv_head == 0, "n_head must be a multiple of n_kv_head"
        self.head_dim = cfg.n_embd // cfg.n_head
        # one fused projection: [ q: n_head*d | k: n_kv_head*d | v: n_kv_head*d ]
        self.qkv = nn.Linear(cfg.n_embd, (self.n_head + 2 * self.n_kv_head) * self.head_dim, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = cfg.dropout
        self.rope = cfg.pos == "rope"
        assert not self.rope or self.head_dim % 2 == 0

    def forward(self, x, cache: "KVCache | None" = None, layer: int = 0, offset: int = 0):
        B, T, C = x.shape
        d, H, Hk = self.head_dim, self.n_head, self.n_kv_head
        q, k, v = self.qkv(x).split([H * d, Hk * d, Hk * d], dim=2)
        q = q.view(B, T, H, d).transpose(1, 2)
        k = k.view(B, T, Hk, d).transpose(1, 2)
        v = v.view(B, T, Hk, d).transpose(1, 2)
        if self.rope:  # rotate q and k at their absolute positions offset..offset+T
            cos, sin = rope_cache(T, d, x.device, offset=offset)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            k, v = cache.update(layer, k, v)  # cache stores only Hk heads (the GQA saving)
        if Hk != H:  # share each kv head among H/Hk query heads
            k, v = k.repeat_interleave(H // Hk, dim=1), v.repeat_interleave(H // Hk, dim=1)
        drop = self.dropout if self.training else 0.0
        if offset == 0:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=drop)
        elif T == 1:  # decoding one token: it may see everything before it, no mask needed
            y = F.scaled_dot_product_attention(q, k, v, dropout_p=drop)
        else:  # chunk after a non-empty cache: query i (global pos offset+i) sees keys <= offset+i
            mask = torch.ones(T, offset + T, dtype=torch.bool, device=x.device).tril(diagonal=offset)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=drop)
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

    def forward(self, x, cache=None, layer=0, offset=0):
        x = x + self.attn(self.ln1(x), cache, layer, offset)
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

    def forward(self, idx, targets=None, cache: "KVCache | None" = None):
        B, T = idx.shape
        offset = cache.seq_len if cache is not None else 0  # tokens already in the cache
        assert offset + T <= self.cfg.block_size
        x = self.tok_emb(idx)
        if self.pos_emb is not None:
            x = x + self.pos_emb(torch.arange(offset, offset + T, device=idx.device))
        x = self.drop(x)
        for i, b in enumerate(self.blocks):
            x = b(x, cache, i, offset)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def new_cache(self) -> KVCache:
        return KVCache(self.cfg.n_layer)

    @staticmethod
    def _sample(logits, temperature, top_k):
        logits = logits[:, -1, :] / max(temperature, 1e-8)
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        return torch.multinomial(F.softmax(logits, dim=-1), 1)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, use_cache=True):
        """Autoregressive sampling.
        With use_cache: prefill the prompt once, then feed ONE new token per step (O(T) per step).
        Without: re-run the whole window every step (O(T^2) per step). Once the sequence reaches
        block_size we fall back to the sliding-window recompute, because cached keys/values were
        computed with a different (now dropped) left context and positions can't go past block_size."""
        self.eval()
        cache = self.new_cache() if use_cache else None
        for _ in range(max_new_tokens):
            if cache is not None and idx.size(1) < self.cfg.block_size:
                inp = idx[:, cache.seq_len:]            # prompt on the first step, then 1 new token
                logits, _ = self(inp, cache=cache)
            else:
                cache = None
                logits, _ = self(idx[:, -self.cfg.block_size:])
            idx = torch.cat([idx, self._sample(logits, temperature, top_k)], dim=1)
        return idx
