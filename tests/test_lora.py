import copy

import torch

from hedy.lora import LoRALinear, apply_lora, count_params, lora_state_dict, merge_lora
from hedy.model import GPT, GPTConfig


def base(**kw):
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=30, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0, **kw)).eval()


X = torch.randint(0, 30, (2, 16))


def test_zero_init_is_identity():
    m = base(); ref = m(X)[0]
    apply_lora(m, r=4)
    assert torch.allclose(m(X)[0], ref, atol=1e-6)


def test_only_adapters_trainable_and_param_count():
    m = base(mlp="swiglu")
    n_base = count_params(m)[1]
    apply_lora(m, r=4)
    trainable, total = count_params(m)
    expected = sum(mod.r * (mod.base.in_features + mod.base.out_features) for mod in m.modules() if isinstance(mod, LoRALinear))
    assert trainable == expected and total == n_base + expected
    assert all("lora_" in n for n, p in m.named_parameters() if p.requires_grad)
    assert not m.tok_emb.weight.requires_grad                    # tied embedding/head stays frozen
    assert trainable < 0.2 * n_base


def test_first_step_only_B_gets_gradient():
    """With B=0, dL/dA = 0 (A only matters through B), while dL/dB != 0. A starts learning on step 2."""
    m = base(); apply_lora(m, r=4)
    m.train(); m(X, X)[1].backward()
    for mod in m.modules():
        if isinstance(mod, LoRALinear):
            assert mod.lora_A.grad.abs().max() == 0 and mod.lora_B.grad.abs().max() > 0
    assert all(p.grad is None for n, p in m.named_parameters() if "lora_" not in n)


def test_merge_equals_adapted_and_removes_overhead():
    m = base(); frozen = copy.deepcopy(m)
    apply_lora(m, r=4)
    for mod in m.modules():                                       # pretend training happened
        if isinstance(mod, LoRALinear):
            torch.nn.init.normal_(mod.lora_B, std=0.1)
    out = m(X)[0]
    merge_lora(m)
    assert not any(isinstance(mod, LoRALinear) for mod in m.modules())
    assert count_params(m) == count_params(frozen)               # back to the base architecture
    assert torch.allclose(m(X)[0], out, atol=1e-5)
    assert not torch.allclose(m(X)[0], frozen(X)[0], atol=1e-3)  # and it really differs from the base


def test_adapter_save_load_roundtrip():
    m = base(); fresh = copy.deepcopy(m)
    apply_lora(m, r=4)
    for mod in m.modules():
        if isinstance(mod, LoRALinear):
            torch.nn.init.normal_(mod.lora_B, std=0.1)
    sd = lora_state_dict(m)
    assert all("lora_" in k for k in sd) and len(sd) == 2 * sum(isinstance(x, LoRALinear) for x in m.modules())
    apply_lora(fresh, r=4)
    assert fresh.load_state_dict(sd, strict=False).unexpected_keys == []
    assert torch.allclose(fresh(X)[0], m(X)[0], atol=1e-6)


def test_attention_only_targets_and_gqa_shapes():
    m = base(n_kv_head=1, pos="rope")
    apply_lora(m, r=2, targets=("qkv", "proj"))
    # per block: attn.qkv, attn.proj and mlp.proj (the MLP's output layer is also named "proj") -> 3 x 2 blocks
    assert sum(isinstance(x, LoRALinear) for x in m.modules()) == 6
    assert m(X)[0].shape == (2, 16, 30)
