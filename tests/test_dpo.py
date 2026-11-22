import copy
import math
import random

import numpy as np
import torch

from hedy.dpo import (boot_ci, build_pairs, cut_at_end, dpo_loss, passes, ref_logps, sample_k, seq_logps,
                      train_preferences)
from hedy.model import GPT, GPTConfig
from hedy.tokenizer import BPETokenizer

TEXT = ("### Instruction:\nWrite a short story about a character named Tim.\n### Response:\n"
        "Tim had a ball. Tim liked it.\n### End\nMia had a kite. Mia flew it high.\n") * 20
TOK = BPETokenizer.train(TEXT, 300)


def tiny(seed=0):
    torch.manual_seed(seed)
    return GPT(GPTConfig(vocab_size=TOK.vocab_size, block_size=128, n_layer=2, n_head=2, n_embd=32, dropout=0.0)).eval()


def test_dpo_loss_at_init_is_ln2_and_pushes_the_right_way():
    z = torch.zeros(4)
    loss, st = dpo_loss(z, z, z, z, beta=0.1)
    assert abs(loss.item() - math.log(2)) < 1e-6 and st["margin"] == 0
    pc, pr = torch.zeros(4, requires_grad=True), torch.zeros(4, requires_grad=True)
    dpo_loss(pc, pr, z, z, 0.1)[0].backward()
    assert (pc.grad < 0).all() and (pr.grad > 0).all()          # descent raises chosen, lowers rejected
    big = dpo_loss(torch.full((4,), 50.0), z, z, z, 0.1)[0]
    assert big.item() < 0.01                                     # large correct margin -> ~0 loss


def test_seq_logps_matches_manual_and_ignores_padding():
    m = tiny()
    insts = ["Write a short story.", "Write a short story about a character named Tim."]
    resps = [TOK.encode("Tim ran."), TOK.encode("Tim had a ball. Tim liked it.")]
    both, n = seq_logps(m, TOK, insts, resps)
    from hedy.sft import PROMPT
    for i in range(2):
        p = TOK.encode(PROMPT.format(inst=insts[i])); ids = p + resps[i]
        lp = torch.log_softmax(m(torch.tensor([ids[:-1]]))[0][0], -1)
        manual = sum(lp[j - 1, ids[j]].item() for j in range(len(p), len(ids)))
        assert abs(both[i].item() - manual) < 1e-4 and n[i].item() == len(resps[i])
        alone, _ = seq_logps(m, TOK, [insts[i]], [resps[i]])
        assert abs(alone.item() - both[i].item()) < 1e-4         # padding in a batch changes nothing


def test_cut_at_end_and_sample_k():
    ids = TOK.encode(" Tim ran.\n### End\n and more text")
    c = cut_at_end(TOK, ids)
    assert c["done"] and c["text"] == "Tim ran." and END_OK(TOK.decode(c["ids"]))
    assert not cut_at_end(TOK, TOK.encode(" Tim ran and ran"))["done"]
    ss = sample_k(tiny(), TOK, "Write a short story.", 3, temperature=1.0)
    assert len(ss) == 3 and all(len(s["ids"]) > 0 for s in ss)


def END_OK(t):
    return t.rstrip("\n").endswith("### End") or t.endswith("### End\n") or "### End" in t


def test_verifier_and_pairs():
    ex = {"kind": "name", "key": "Tim", "inst": "x", "samples": [
        {"ids": [1, 2], "text": "Tim ran.", "done": True},
        {"ids": [3, 4], "text": "Mia ran.", "done": True},
        {"ids": [5], "text": "Tim", "done": False},          # contains the word but never finished: not a pass
        {"ids": [6], "text": "Sue sat", "done": False}]}      # failing AND unfinished: never used as rejected
    assert [passes(ex, s) for s in ex["samples"]] == [True, False, False, False]
    pairs = build_pairs([ex], random.Random(0))
    assert len(pairs) == 1 and pairs[0]["chosen"] == [1, 2] and pairs[0]["rejected"] == [3, 4]
    assert build_pairs([{**ex, "samples": ex["samples"][:1]}], random.Random(0)) == []   # no failing sample -> no pair


def test_dpo_training_learns_the_preference_and_reference_is_frozen():
    m = tiny()
    c = TOK.encode("Tim had a ball. Tim liked it.\n### End")
    r = TOK.encode("Mia had a kite. Mia flew it high.\n### End")
    pairs = [{"inst": "Write a short story about a character named Tim.", "chosen": c, "rejected": r}] * 8
    ref0 = ref_logps(m, TOK, pairs[:1])
    assert abs(dpo_loss(*(torch.tensor([x]) for x in (ref0[0][0], ref0[0][1], ref0[0][0], ref0[0][1])), 0.1)[0].item() - math.log(2)) < 1e-6
    args = type("A", (), dict(lr=3e-3, epochs=15, batch_size=8, warmup=1, seed=0, objective="dpo", beta=0.1, sft_coef=0.0))()
    train_preferences(m, TOK, pairs, args)
    pc, _ = seq_logps(m.eval(), TOK, [pairs[0]["inst"]], [c]); pr, _ = seq_logps(m, TOK, [pairs[0]["inst"]], [r])
    assert (pc - pr).item() > (ref0[0][0] - ref0[0][1]) + 3       # preference margin grew by > 3 nats


def test_rft_objective_is_chosen_nll_and_boot_ci():
    m = tiny()
    c = TOK.encode("Tim had a ball.\n### End")
    pairs = [{"inst": "Write a short story.", "chosen": c, "rejected": c}] * 8
    nll0 = -(seq_logps(m, TOK, ["Write a short story."], [c])[0] / len(c)).item()
    args = type("A", (), dict(lr=3e-3, epochs=80, batch_size=8, warmup=1, seed=0, objective="rft", beta=0.1, sft_coef=0.0))()
    train_preferences(m, TOK, pairs, args)
    nll1 = -(seq_logps(m.eval(), TOK, ["Write a short story."], [c])[0] / len(c)).item()
    assert nll1 < nll0 * 0.5
    lo, hi = boot_ci([0, 1, 1, 0, 1, 1, 1, 0, 1, 1])
    assert lo < 0.7 < hi
