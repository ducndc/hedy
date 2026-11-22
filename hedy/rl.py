"""Online RL with a verifiable reward: GRPO / RLOO-style policy gradient.

    python -m hedy.rl --policy checkpoints/sft-full --kinds name --steps 60 --out checkpoints/rl-name

Each step:  sample K responses for each of P prompts FROM THE CURRENT POLICY (temperature 1, no top-k, so
we sample exactly the distribution whose log-probs we differentiate)  ->  reward = verifier (1 if the
response finished and contains the required word, else 0)  ->  advantage = reward relative to the
other samples of the SAME prompt (a group baseline: no value network)  ->  policy-gradient update with
an optional PPO-style clip and a KL penalty to the frozen starting policy.

    advantage  grpo : (r - mean_group) / (std_group + eps)          (DeepSeekMath's GRPO)
               rloo : r - mean of the *other* K-1 samples            (Leave-One-Out REINFORCE)
               mean : r - mean_group                                 (no std scaling; 'Dr. GRPO')
Groups whose samples all got the same reward have zero advantage: no signal, and no gradient.
"""
import argparse
import copy
import json
import math
import os
import random
import time

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from .dpo import full_eval, passes, print_eval, sample_k
from .sft import PROMPT, load_sft_model


# ----------------------------------------------------------------------------- pieces
def advantages(rewards, kind="grpo", eps=1e-4):
    r = np.asarray(rewards, dtype=np.float64)
    k = len(r)
    if kind == "rloo":                                   # baseline for sample i = mean of the other k-1
        return (r - (r.sum() - r) / (k - 1)) if k > 1 else np.zeros_like(r)
    a = r - r.mean()
    if kind == "grpo":
        a = a / (r.std() + eps) if r.std() > eps else np.zeros_like(r)   # all-equal group -> exactly 0
    return a


def token_logps(model, tok, insts, resp_ids):
    """(B, Lr) log-prob of every response token given the prompt and the previous response tokens,
    plus a (B, Lr) 0/1 mask for real (non-pad) positions."""
    prompts = [tok.encode(PROMPT.format(inst=i)) for i in insts]
    seqs = [p + r for p, r in zip(prompts, resp_ids)]
    L = max(map(len, seqs)) - 1
    x = torch.zeros(len(seqs), L, dtype=torch.long)
    for i, s in enumerate(seqs):                         # right-padded: causal attention ignores pads
        x[i, : len(s) - 1] = torch.tensor(s[:-1])
    logp = torch.log_softmax(model(x)[0], dim=-1)
    rows = [logp[i, len(p) - 1 : len(p) - 1 + len(r)].gather(1, torch.tensor(r)[:, None])[:, 0]
            for i, (p, r) in enumerate(zip(prompts, resp_ids))]
    out = pad_sequence(rows, batch_first=True)
    mask = pad_sequence([torch.ones(len(r)) for r in resp_ids], batch_first=True)
    return out, mask


def k3_kl(logp, ref_logp):
    """Schulman's k3 estimator of KL(pi || ref) from samples of pi: e^(d) - d - 1, d = log ref - log pi.
    Always >= 0, unbiased, low variance (unlike the raw log-ratio)."""
    d = ref_logp - logp
    return torch.exp(d) - d - 1


def pg_loss(logp, old_logp, ref_logp, adv, mask, clip=0.2, kl_coef=0.0):
    """Per-sequence token-mean of  -min(ratio*A, clip(ratio)*A) + kl_coef*KL ; mean over sequences.
    With one update per batch ratio == 1 and this is exactly REINFORCE with a group baseline."""
    ratio = torch.exp(logp - old_logp)
    a = adv[:, None]
    surr = torch.minimum(ratio * a, torch.clamp(ratio, 1 - clip, 1 + clip) * a)
    per_tok = -surr + kl_coef * k3_kl(logp, ref_logp)
    loss = ((per_tok * mask).sum(1) / mask.sum(1)).mean()
    stats = {"kl": ((k3_kl(logp, ref_logp) * mask).sum() / mask.sum()).item(),
             "clipfrac": (((ratio - 1).abs() > clip).float() * mask).sum().item() / mask.sum().item()}
    return loss, stats


# ----------------------------------------------------------------------------- loop
def train_rl(model, tok, prompts, args, reward_fn=passes, log=print):
    ref = copy.deepcopy(model).eval()                    # frozen starting policy for the KL penalty
    for p in ref.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=0.0)
    rng = random.Random(args.seed)
    order, pos, history = list(range(len(prompts))), 0, []
    rng.shuffle(order)
    t0 = time.perf_counter()
    for step in range(args.steps):
        batch = []
        for _ in range(args.prompts_per_step):
            if pos == len(order):
                rng.shuffle(order); pos = 0
            batch.append(prompts[order[pos]]); pos += 1

        model.eval()
        insts, ids, adv, all_rewards, groups_signal, lens, done, mentions = [], [], [], [], 0, [], [], []
        for ex in batch:
            ss = sample_k(model, tok, ex["inst"], args.k, temperature=1.0, top_k=None)   # on-policy
            r = [float(reward_fn(ex, s)) for s in ss]
            a = advantages(r, args.adv)
            all_rewards += r
            lens += [len(s["ids"]) for s in ss]; done += [s["done"] for s in ss]
            if np.abs(a).max() > 0:                      # only groups with contrast carry gradient
                groups_signal += 1
                for s, ai in zip(ss, a):
                    insts.append(ex["inst"]); ids.append(s["ids"]); adv.append(ai)
        rec = {"step": step + 1, "reward": float(np.mean(all_rewards)), "signal": groups_signal / len(batch),
               "len": float(np.mean(lens)), "done": float(np.mean(done))}

        if insts:
            model.train()
            adv_t = torch.tensor(adv, dtype=torch.float32)
            with torch.no_grad():                        # reference log-probs (frozen)
                ref_lp = [token_logps(ref, tok, insts[i : i + args.micro], ids[i : i + args.micro])[0]
                          for i in range(0, len(insts), args.micro)]
            old_store, kls, clips = {}, [], []
            for ie in range(args.inner_epochs):
                opt.zero_grad(set_to_none=True)
                for mi, i in enumerate(range(0, len(insts), args.micro)):
                    lp, mask = token_logps(model, tok, insts[i : i + args.micro], ids[i : i + args.micro])
                    if mi not in old_store:              # first pass: "old" policy == current policy
                        old_store[mi] = lp.detach()
                    loss, st = pg_loss(lp, old_store[mi], ref_lp[mi], adv_t[i : i + args.micro], mask,
                                       args.clip, args.kl_coef)
                    (loss * len(lp) / len(insts)).backward()
                    kls.append(st["kl"]); clips.append(st["clipfrac"])
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
            rec.update(kl=float(np.mean(kls)), clipfrac=float(np.mean(clips)), gnorm=float(gn))
        history.append(rec)
        if (step + 1) % args.log_every == 0:
            w = history[-args.log_every:]
            m = lambda k: np.nanmean([h.get(k, float("nan")) for h in w])
            log(f"step {step + 1}/{args.steps}: reward {m('reward'):.3f} signal-groups {m('signal') * 100:.0f}% "
                f"finished {m('done') * 100:.0f}% len {m('len'):.0f} kl {m('kl'):.4f} clip {m('clipfrac'):.3f} "
                f"({time.perf_counter() - t0:.0f}s)")
    return history


def main():
    ap = argparse.ArgumentParser()
    d = ap.add_argument
    d("--policy", required=True); d("--out", required=True)
    d("--data", default="data/sft"); d("--kinds", default="name", help="comma list of: name, kw")
    d("--steps", type=int, default=60); d("--prompts_per_step", type=int, default=32); d("--k", type=int, default=8)
    d("--adv", choices=["grpo", "rloo", "mean"], default="grpo")
    d("--lr", type=float, default=2e-5); d("--kl_coef", type=float, default=0.02)
    d("--clip", type=float, default=0.2); d("--inner_epochs", type=int, default=1)
    d("--micro", type=int, default=64); d("--grad_clip", type=float, default=1.0)
    d("--log_every", type=int, default=5); d("--seed", type=int, default=0); d("--threads", type=int, default=4)
    d("--prompts", default="data/dpo/eval_prompts.jsonl"); d("--data_dir", default="data/tinystories")
    d("--eval_k", type=int, default=4); d("--no_eval", action="store_true", help="skip the final held-out evaluation")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    model, tok = load_sft_model(a.policy, a.policy)
    kinds = a.kinds.split(",")
    train = [e for e in (json.loads(l) for l in open(os.path.join(a.data, "train.jsonl"))) if e["kind"] in kinds]
    print(f"{len(train)} training prompts ({a.kinds}), {a.steps} steps x {a.prompts_per_step} prompts x K={a.k} "
          f"| adv={a.adv} lr={a.lr} kl_coef={a.kl_coef} inner_epochs={a.inner_epochs}", flush=True)
    hist = train_rl(model, tok, train, a, log=lambda s: print(s, flush=True))
    os.makedirs(a.out, exist_ok=True)
    torch.save({"config": model.cfg.__dict__, "model": model.state_dict()}, os.path.join(a.out, "model.pt"))
    tok.save(os.path.join(a.out, "tokenizer.json"))
    if a.no_eval:
        json.dump({"history": hist, "args": vars(a)}, open(os.path.join(a.out, "results.json"), "w"))
        return
    val = [json.loads(l) for l in open(os.path.join(a.data, "val.jsonl"))]
    res = full_eval(model, tok, [json.loads(l) for l in open(a.prompts)], val, a.data_dir, a.eval_k)
    res["history"], res["args"] = hist, vars(a)
    json.dump(res, open(os.path.join(a.out, "results.json"), "w"))
    print_eval(res, a.eval_k)


if __name__ == "__main__":
    main()
