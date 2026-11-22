import itertools

import pytest
import torch

from hedy.model import GPT, GPTConfig

CASES = list(itertools.product(["learned", "rope"], [0, 2, 1]))  # (pos, n_kv_head): MHA / GQA / MQA


def make(pos="rope", n_kv=0, norm="rmsnorm", mlp="swiglu", n_head=4, block=32):
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=30, block_size=block, n_layer=2, n_head=n_head, n_embd=32,
                         dropout=0.0, pos=pos, norm=norm, mlp=mlp, n_kv_head=n_kv)).eval()


@pytest.mark.parametrize("pos,n_kv", CASES)
def test_incremental_logits_match_full_forward(pos, n_kv):
    m = make(pos, n_kv)
    x = torch.randint(0, 30, (2, 20))
    full, _ = m(x)
    cache = m.new_cache()
    steps = [m(x[:, :8], cache=cache)[0]]                  # prefill 8 tokens
    steps += [m(x[:, t:t + 1], cache=cache)[0] for t in range(8, 20)]  # then one at a time
    inc = torch.cat(steps, dim=1)
    assert cache.seq_len == 20
    assert torch.allclose(full, inc, atol=1e-4)


@pytest.mark.parametrize("pos,n_kv", CASES)
def test_chunked_decode_with_nonempty_cache(pos, n_kv):
    """The T>1-after-cache branch (explicit mask) must also match."""
    m = make(pos, n_kv)
    x = torch.randint(0, 30, (1, 18))
    full, _ = m(x)
    cache = m.new_cache()
    out = torch.cat([m(x[:, :6], cache=cache)[0], m(x[:, 6:13], cache=cache)[0], m(x[:, 13:], cache=cache)[0]], 1)
    assert torch.allclose(full, out, atol=1e-4)


@pytest.mark.parametrize("pos,n_kv", CASES)
def test_cached_generation_equals_uncached_greedy(pos, n_kv):
    m = make(pos, n_kv)
    prompt = torch.randint(0, 30, (1, 5))
    a = m.generate(prompt, 20, top_k=1, use_cache=True)
    b = m.generate(prompt, 20, top_k=1, use_cache=False)
    assert torch.equal(a, b)


def test_generation_past_block_size_falls_back():
    m = make("rope", 2, block=16)
    out = m.generate(torch.randint(0, 30, (1, 4)), 30, top_k=1)  # 34 tokens > block_size 16
    assert out.shape == (1, 34)


def test_gqa_shrinks_cache_and_params():
    sizes = {}
    for n_kv in (4, 2, 1):
        m = make("rope", n_kv)
        c = m.new_cache(); m(torch.randint(0, 30, (1, 16)), cache=c)
        sizes[n_kv] = c.nbytes()
    assert sizes[4] == 2 * sizes[2] == 4 * sizes[1]       # cache scales with n_kv_head
    assert sum(p.numel() for p in make("rope", 1).parameters()) < sum(p.numel() for p in make("rope", 0).parameters())


def test_invalid_group():
    with pytest.raises(AssertionError):
        make("rope", 3)
