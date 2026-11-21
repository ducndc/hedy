import itertools

import pytest
import torch

from hedy.model import GPT, GPTConfig, RMSNorm, apply_rope, rope_cache

VARIANTS = list(itertools.product(["learned", "rope"], ["layernorm", "rmsnorm"], ["gelu", "swiglu"]))


def make(pos, norm, mlp):
    return GPT(GPTConfig(vocab_size=20, block_size=16, n_layer=2, n_head=2, n_embd=32,
                         dropout=0.0, pos=pos, norm=norm, mlp=mlp))


@pytest.mark.parametrize("v", VARIANTS)
def test_variant_shapes_causality_overfit(v):
    torch.manual_seed(0)
    m = make(*v).eval()
    x = torch.randint(0, 20, (2, 16))
    y = x.clone(); y[:, -1] = (y[:, -1] + 1) % 20
    assert torch.allclose(m(x)[0][:, :-1], m(y)[0][:, :-1], atol=1e-5)  # no peeking at the future
    m.train(); opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    first = m(x, x)[1].item()
    for _ in range(80):
        loss = m(x, x)[1]; opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < first * 0.5


def test_rmsnorm_unit_rms():
    y = RMSNorm(64)(torch.randn(4, 10, 64) * 7 + 3)
    assert torch.allclose(y.pow(2).mean(-1), torch.ones(4, 10), atol=1e-3)


def test_rope_scores_depend_only_on_relative_position():
    torch.manual_seed(0)
    d, T = 16, 12
    q, k = torch.randn(1, 1, 1, d), torch.randn(1, 1, 1, d)
    cos, sin = rope_cache(T, d, "cpu")

    def score(m, n):  # place q at position m, k at position n
        qm = apply_rope(q.expand(1, 1, T, d), cos, sin)[0, 0, m]
        kn = apply_rope(k.expand(1, 1, T, d), cos, sin)[0, 0, n]
        return (qm * kn).sum()

    assert torch.allclose(score(5, 2), score(9, 6), atol=1e-4)    # same offset 3
    assert not torch.allclose(score(5, 2), score(5, 3), atol=1e-4)  # different offset


def test_rope_preserves_norm():
    x = torch.randn(1, 2, 8, 16)
    cos, sin = rope_cache(8, 16, "cpu")
    assert torch.allclose(apply_rope(x, cos, sin).norm(dim=-1), x.norm(dim=-1), atol=1e-4)


def test_swiglu_param_count_close_to_gelu():
    a = sum(p.numel() for p in make("learned", "layernorm", "gelu").parameters())
    b = sum(p.numel() for p in make("learned", "layernorm", "swiglu").parameters())
    assert abs(a - b) / a < 0.03
