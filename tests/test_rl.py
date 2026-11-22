import argparse
import math

import numpy as np
import torch

from hedy.dpo import seq_logps
from hedy.model import GPT, GPTConfig
from hedy.rl import advantages, k3_kl, pg_loss, token_logps, train_rl
from hedy.tokenizer import BPETokenizer

TOK = BPETokenizer.train("### Instruction:\nWrite a short story.\n### Response:\nTim had a ball. Tim liked it.\n### End\n" * 30, 300)


def tiny(seed=0):
    torch.manual_seed(seed)
    return GPT(GPTConfig(vocab_size=TOK.vocab_size, block_size=128, n_layer=2, n_head=2, n_embd=32, dropout=0.0)).eval()


def test_advantages():
    r = [1, 0, 0, 1, 0, 0, 0, 0]
    g = advantages(r, "grpo")
    assert abs(g.mean()) < 1e-9 and abs(g.std() - 1) < 1e-3
    assert (advantages([0] * 8, "grpo") == 0).all() and (advantages([1] * 8, "rloo") == 0).all()   # no contrast, no signal
    rl = advantages(r, "rloo")
    assert np.allclose(rl, 8 / 7 * (np.asarray(r) - np.mean(r)))      # LOO == K/(K-1) * (r - mean)
    assert np.allclose(advantages(r, "mean"), np.asarray(r) - np.mean(r))
    assert advantages(r, "grpo")[0] > 0 > advantages(r, "grpo")[1]


def test_token_logps_matches_sequence_logps():
    m = tiny()
    insts = ["Write a short story.", "Write a short story about a character named Tim."]
    resps = [TOK.encode("Tim ran."), TOK.encode("Tim had a ball. Tim liked it.")]
    lp, mask = token_logps(m, TOK, insts, resps)
    total, n = seq_logps(m, TOK, insts, resps)
    assert torch.allclose((lp * mask).sum(1), total, atol=1e-4) and mask.sum(1).tolist() == n.tolist()


def test_k3_kl_nonnegative_zero_at_equal_and_matches_analytic():
    x = torch.randn(1000)
    assert k3_kl(x, x).abs().max() == 0 and (k3_kl(x, x + torch.randn(1000)) >= 0).all()
    # unbiased: sampling y~pi with pi=N(0,1) over 5 categories, ref=other categorical
    p = torch.tensor([0.5, 0.2, 0.15, 0.1, 0.05]); q = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2])
    ys = torch.multinomial(p, 200_000, replacement=True)
    est = k3_kl(p.log()[ys], q.log()[ys]).mean().item()
    assert abs(est - (p * (p / q).log()).sum().item()) < 0.01


def test_policy_gradient_raises_rewarded_lowers_unrewarded():
    m = tiny().train()
    inst = "Write a short story."
    good, bad = TOK.encode("Tim had a ball.\n### End"), TOK.encode("Mia flew a kite.\n### End")
    opt = torch.optim.SGD(m.parameters(), lr=0.05)
    before = [seq_logps(m.eval(), TOK, [inst], [x])[0].item() for x in (good, bad)]
    for _ in range(10):
        m.train()
        lp, mask = token_logps(m, TOK, [inst] * 2, [good, bad])
        loss, _ = pg_loss(lp, lp.detach(), lp.detach(), torch.tensor([1.0, -1.0]), mask)
        opt.zero_grad(); loss.backward(); opt.step()
    after = [seq_logps(m.eval(), TOK, [inst], [x])[0].item() for x in (good, bad)]
    assert after[0] > before[0] and after[1] < before[1]


def test_clipping_zeroes_gradient_when_ratio_left_trust_region():
    lp = torch.full((1, 3), -1.0, requires_grad=True)
    old = torch.full((1, 3), -1.0 - math.log(1.5))               # ratio = 1.5 > 1 + 0.2, positive advantage
    mask = torch.ones(1, 3)
    pg_loss(lp, old, lp.detach(), torch.tensor([1.0]), mask, clip=0.2)[0].backward()
    assert lp.grad.abs().max() == 0                               # clipped: no incentive to go further
    lp2 = torch.full((1, 3), -1.0, requires_grad=True)
    pg_loss(lp2, old, lp2.detach(), torch.tensor([-1.0]), mask, clip=0.2)[0].backward()
    assert lp2.grad.abs().max() > 0                               # negative advantage is NOT clipped on this side


def test_kl_penalty_pulls_back_toward_reference():
    lp = torch.full((1, 3), -2.0, requires_grad=True)
    ref = torch.full((1, 3), -1.0)
    pg_loss(lp, lp.detach(), ref, torch.tensor([0.0]), torch.ones(1, 3), kl_coef=1.0)[0].backward()
    # d k3 / d logp = 1 - e^(ref-logp) < 0 when the policy is below the reference, so gradient DESCENT raises logp
    assert (lp.grad < 0).all()


def test_full_loop_learns_a_toy_reward():
    """End-to-end: sampling -> advantages -> update. Dense reward = fraction of response tokens that are
    even (chance ~0.5). Every token contributes to it, so credit assignment is easy and a correct loop
    must raise it a lot. (A reward hinging on ONE of ~100 tokens would be swamped by the other 99
    tokens' noise: sequence-level rewards dilute credit, see the study note.)"""
    m = tiny()
    prompts = [{"inst": f"Write a short story {i}.", "kind": "plain", "key": ""} for i in range(4)]
    args = argparse.Namespace(lr=3e-3, kl_coef=0.0, clip=0.2, inner_epochs=1, micro=64, grad_clip=1.0, seed=0,
                              steps=25, prompts_per_step=4, k=8, adv="grpo", log_every=100)
    hist = train_rl(m, TOK, prompts, args, reward_fn=lambda ex, s: float(np.mean([t % 2 == 0 for t in s["ids"]])))
    first, last = np.mean([h["reward"] for h in hist[:3]]), np.mean([h["reward"] for h in hist[-3:]])
    assert last > first + 0.15 and last > 0.65
