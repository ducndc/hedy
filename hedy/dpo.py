"""Preference optimization with a program as the judge.

    python -m hedy.dpo prompts --text data/tinystories-valid.txt --out data/dpo            # held-out eval prompts
    python -m hedy.dpo sample  --policy checkpoints/sft-full --data data/sft --out data/dpo --shard 0 --nshards 3
    python -m hedy.dpo pairs   --out data/dpo                                              # verifier -> (chosen, rejected)
    python -m hedy.dpo train   --policy checkpoints/sft-full --pairs data/dpo/pairs.jsonl --objective dpo \
                               --beta 0.1 --lr 5e-5 --out checkpoints/dpo-b0.1
    python -m hedy.dpo eval    --policy checkpoints/sft-full --ckpt checkpoints/dpo-b0.1 --prompts data/dpo/eval_prompts.jsonl

Pipeline: sample K responses per instruction from the SFT model (on-policy) -> label each with the
verifier from step 07 (does the response contain the required word, and did it finish?) -> pair a
passing with a failing response -> optimize.  Objectives:
    dpo : -log sigmoid( beta * [ (logp_pi(c) - logp_ref(c)) - (logp_pi(r) - logp_ref(r)) ] )
    rft : rejection-sampling fine-tuning, plain NLL on the chosen responses only (strong simple baseline)
"""
import argparse
import json
import math
import os
import random
import re
import time

import numpy as np
import torch
import torch.nn.functional as F

from .data import load_bin
from .eval import ppl_report
from .lora import apply_lora, merge_lora
from .prepare import split_docs
from .sft import (END_MARK, KINDS, PROMPT, build_dataset, doc_freq, has_word, key_nll, load_sft_model,
                  response_loss)

LN2 = math.log(2)


# ----------------------------------------------------------------------------- sampling + verifier
@torch.no_grad()
def sample_k(model, tok, inst, k, temperature=1.0, top_k=None):
    """k samples for one instruction in ONE batched generate call (identical prompts => equal lengths,
    so no padding is needed). Each sample is cut right after the token that completes '### End'.
    We keep the sampled token ids: re-encoding the decoded text could yield a different (canonical)
    tokenization than the one the policy actually produced, and DPO needs the policy's own tokens."""
    model.eval()
    prompt = tok.encode(PROMPT.format(inst=inst))
    n_new = model.cfg.block_size - len(prompt) - 1
    out = model.generate(torch.tensor([prompt] * k), n_new, temperature, top_k)[:, len(prompt):].tolist()
    return [cut_at_end(tok, ids) for ids in out]


def cut_at_end(tok, ids):
    """-> {'ids', 'text', 'done'}: ids up to and including the token that completes the End marker."""
    if END_MARK not in tok.decode(ids):
        return {"ids": ids, "text": tok.decode(ids).strip(), "done": False}
    lo, hi = 1, len(ids)                       # smallest prefix whose decoding contains the marker
    while lo < hi:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if END_MARK in tok.decode(ids[:mid]) else (mid + 1, hi)
    return {"ids": ids[:lo], "text": tok.decode(ids[:lo]).split(END_MARK)[0].strip(), "done": True}


def passes(ex, sample):
    """The verifier: finished AND contains the required word (plain prompts have no content check)."""
    return sample["done"] and (ex["kind"] == "plain" or has_word(sample["text"], ex["key"]))


def build_pairs(rows, rng, max_per_prompt=2):
    """rows: [{ex fields..., 'samples': [...]}] -> pairs (passing chosen vs FINISHED-but-failing rejected;
    requiring the rejected one to finish keeps the preference about content, not about stopping)."""
    pairs = []
    for r in rows:
        good = [s for s in r["samples"] if passes(r, s)]
        bad = [s for s in r["samples"] if s["done"] and not passes(r, s)]
        if not good or not bad:
            continue
        combos = [(g, b) for g in good for b in bad]
        rng.shuffle(combos)
        for g, b in combos[:max_per_prompt]:
            pairs.append({"inst": r["inst"], "key": r["key"], "kind": r["kind"],
                          "chosen": g["ids"], "rejected": b["ids"]})
    return pairs


# ----------------------------------------------------------------------------- objectives
def seq_logps(model, tok, insts, resp_ids):
    """Sum of log p(response token | prompt, previous response tokens) for each (instruction, response)."""
    prompts = [tok.encode(PROMPT.format(inst=i)) for i in insts]
    seqs = [p + r for p, r in zip(prompts, resp_ids)]
    L = max(map(len, seqs)) - 1
    x = torch.zeros(len(seqs), L, dtype=torch.long)
    y = torch.full((len(seqs), L), -100, dtype=torch.long)
    for i, (p, s) in enumerate(zip(prompts, seqs)):       # right-padded: causal attention ignores pads
        x[i, : len(s) - 1] = torch.tensor(s[:-1])
        y[i, len(p) - 1 : len(s) - 1] = torch.tensor(s[len(p):])   # only response tokens are scored
    lp = -F.cross_entropy(model(x)[0].transpose(1, 2), y, ignore_index=-100, reduction="none")
    return lp.sum(1), (y != -100).sum(1)


def dpo_loss(pc, pr, rc, rr, beta):
    """pc/pr: policy log-probs of chosen/rejected; rc/rr: frozen-reference log-probs."""
    margin = beta * ((pc - rc) - (pr - rr))              # implicit reward of chosen minus rejected
    return -F.logsigmoid(margin).mean(), {"acc": (margin > 0).float().mean().item(), "margin": margin.mean().item()}


@torch.no_grad()
def ref_logps(model, tok, pairs, batch=32):
    model.eval()
    out = []
    for i in range(0, len(pairs), batch):
        b = pairs[i : i + batch]
        c = seq_logps(model, tok, [p["inst"] for p in b], [p["chosen"] for p in b])[0]
        r = seq_logps(model, tok, [p["inst"] for p in b], [p["rejected"] for p in b])[0]
        out += list(zip(c.tolist(), r.tolist()))
    return out


def train_preferences(model, tok, pairs, args):
    ref = ref_logps(model, tok, pairs)                   # frozen reference = the policy BEFORE training
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    steps = args.epochs * math.ceil(len(pairs) / args.batch_size)
    rng, step, t0 = random.Random(args.seed), 0, time.perf_counter()
    model.train()
    for ep in range(args.epochs):
        order = list(range(len(pairs))); rng.shuffle(order)
        agg = []
        for b in range(0, len(order), args.batch_size):
            idx = order[b : b + args.batch_size]
            P = [pairs[i] for i in idx]
            lr = args.lr * min(1.0, (step + 1) / args.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps)))
            for g in opt.param_groups:
                g["lr"] = lr
            insts = [p["inst"] for p in P]
            pc, nc = seq_logps(model, tok, insts, [p["chosen"] for p in P])
            if args.objective == "rft":                  # NLL per token of the chosen responses only
                loss, st = -(pc / nc).mean(), {"acc": float("nan"), "margin": float("nan")}
            else:
                pr, _ = seq_logps(model, tok, insts, [p["rejected"] for p in P])
                rc = torch.tensor([ref[i][0] for i in idx]); rr = torch.tensor([ref[i][1] for i in idx])
                loss, st = dpo_loss(pc, pr, rc, rr, args.beta)
                st.update(dc=(pc - rc).mean().item(), dr=(pr - rr).mean().item())
                if args.sft_coef:                        # optional NLL anchor on the chosen (RPO-style)
                    loss = loss - args.sft_coef * (pc / nc).mean()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); step += 1
            agg.append((loss.item(), st))
        def m(k):
            v = [a[1][k] for a in agg if k in a[1] and not math.isnan(a[1][k])]
            return float(np.mean(v)) if v else float("nan")
        print(f"epoch {ep + 1}/{args.epochs}: loss {np.mean([a[0] for a in agg]):.3f} reward-acc {m('acc'):.2f} "
              f"margin {m('margin'):.2f} dlogp(chosen) {m('dc'):+.2f} dlogp(rejected) {m('dr'):+.2f} "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)
        model.train()


# ----------------------------------------------------------------------------- evaluation
def boot_ci(per_prompt, n=2000, seed=0):
    """95% bootstrap CI of the mean, resampling PROMPTS (samples for one prompt are correlated)."""
    rng = np.random.RandomState(seed)
    a = np.asarray(per_prompt, dtype=float)
    means = [a[rng.randint(0, len(a), len(a))].mean() for _ in range(n)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def distinct2(ids):
    g = [tuple(ids[i : i + 2]) for i in range(len(ids) - 1)]
    return len(set(g)) / max(len(g), 1)


def evaluate_policy(model, tok, prompts, k=4, temperature=0.7, top_k=40, seed=0):
    res = {}
    for kind in ("kw", "name"):
        ps = [p for p in prompts if p["kind"] == kind]
        pass1, passk, lens, reps, done, d2 = [], [], [], [], [], []
        for i, ex in enumerate(ps):
            torch.manual_seed(seed + i)
            ss = sample_k(model, tok, ex["inst"], k, temperature, top_k)
            ok = [has_word(s["text"], ex["key"]) for s in ss]
            pass1.append(np.mean(ok)); passk.append(float(any(ok)))
            lens += [len(s["ids"]) for s in ss]; done += [s["done"] for s in ss]
            d2 += [distinct2(s["ids"]) for s in ss]
            reps += [len(re.findall(rf"\b{re.escape(ex['key'])}\b", s["text"], flags=re.I)) for s in ss]
        res[kind] = {"n_prompts": len(ps), "pass1": float(np.mean(pass1)), "pass1_ci": boot_ci(pass1),
                     f"pass@{k}": float(np.mean(passk)), "tokens": float(np.mean(lens)), "done": float(np.mean(done)),
                     "distinct2": float(np.mean(d2)), "mean_key_count": float(np.mean(reps)),
                     "frac_key_ge3": float(np.mean([r >= 3 for r in reps]))}
    return res


def full_eval(model, tok, prompts, val_sft, data_dir, k):
    res = evaluate_policy(model, tok, prompts, k)
    val_tok = np.asarray(load_bin(os.path.join(data_dir, "val.bin")))[:60_000]
    res["pretrain_bpb"] = ppl_report(model, tok, val_tok, model.cfg.block_size, 64)[0]["bpb"]
    res["sft_val_loss"] = response_loss(model, tok, val_sft)
    kn = key_nll(model, tok, val_sft)
    res["gain"] = {k_: v[1] - v[0] for k_, v in kn.items()}
    return res


def print_eval(res, k):
    for kind in ("name", "kw"):
        r = res[kind]
        print(f"{kind}: pass@1 {r['pass1'] * 100:.1f}% (95% CI {r['pass1_ci'][0] * 100:.1f}-{r['pass1_ci'][1] * 100:.1f})  "
              f"pass@{k} {r[f'pass@{k}'] * 100:.1f}%  tokens {r['tokens']:.0f}  finished {r['done'] * 100:.0f}%  "
              f"distinct-2 {r['distinct2']:.2f}  key-count {r['mean_key_count']:.2f}  key>=3x {r['frac_key_ge3'] * 100:.1f}%")
    print(f"teacher-forced gain name {res['gain']['name']:.2f} kw {res['gain']['kw']:.2f} | "
          f"SFT val loss {res['sft_val_loss']:.3f} | pretrain bits/byte {res['pretrain_bpb']:.3f}")


# ----------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prompts"); p.add_argument("--text", required=True); p.add_argument("--out", required=True)
    p.add_argument("--base", default="checkpoints/tinystories"); p.add_argument("--n", type=int, default=450)
    s = sub.add_parser("sample"); s.add_argument("--policy", required=True); s.add_argument("--data", required=True)
    s.add_argument("--out", required=True); s.add_argument("--n_prompts", type=int, default=2000)
    s.add_argument("--k", type=int, default=8); s.add_argument("--shard", type=int, default=0)
    s.add_argument("--nshards", type=int, default=1); s.add_argument("--threads", type=int, default=4)
    pp = sub.add_parser("pairs"); pp.add_argument("--out", required=True); pp.add_argument("--max_per_prompt", type=int, default=2)
    t = sub.add_parser("train"); t.add_argument("--policy", required=True); t.add_argument("--pairs", required=True)
    t.add_argument("--objective", choices=["dpo", "rft"], default="dpo"); t.add_argument("--mode", choices=["full", "lora"], default="full")
    t.add_argument("--beta", type=float, default=0.1); t.add_argument("--lr", type=float, default=5e-5)
    t.add_argument("--epochs", type=int, default=3); t.add_argument("--batch_size", type=int, default=16)
    t.add_argument("--warmup", type=int, default=10); t.add_argument("--sft_coef", type=float, default=0.0)
    t.add_argument("--rank", type=int, default=8); t.add_argument("--seed", type=int, default=0)
    t.add_argument("--threads", type=int, default=4); t.add_argument("--out", required=True)
    t.add_argument("--prompts", default="data/dpo/eval_prompts.jsonl"); t.add_argument("--sft_data", default="data/sft")
    t.add_argument("--data_dir", default="data/tinystories"); t.add_argument("--k", type=int, default=4)
    e = sub.add_parser("eval"); e.add_argument("--policy", required=True); e.add_argument("--ckpt", default=None)
    e.add_argument("--prompts", default="data/dpo/eval_prompts.jsonl"); e.add_argument("--sft_data", default="data/sft")
    e.add_argument("--data_dir", default="data/tinystories"); e.add_argument("--k", type=int, default=4)
    e.add_argument("--threads", type=int, default=4); e.add_argument("--json", default=None)
    a = ap.parse_args()
    rl = lambda path: [json.loads(l) for l in open(path)]

    if a.cmd == "prompts":
        from .eval import load_model
        _, tok = load_model(a.base)
        train_docs, val_docs = split_docs(open(a.text, encoding="utf-8").read())
        ex = [e for e in build_dataset(val_docs, a.n, tok, doc_freq(train_docs), seed=3) if e["kind"] != "plain"]
        os.makedirs(a.out, exist_ok=True)
        open(os.path.join(a.out, "eval_prompts.jsonl"), "w").writelines(json.dumps(e) + "\n" for e in ex)
        print(len(ex), "held-out prompts"); return

    if getattr(a, "threads", 0):
        torch.set_num_threads(a.threads)
    if a.cmd == "sample":
        model, tok = load_sft_model(a.policy, a.policy)
        rows = [e for e in rl(os.path.join(a.data, "train.jsonl")) if e["kind"] != "plain"][: a.n_prompts]
        rows = rows[a.shard :: a.nshards]
        os.makedirs(a.out, exist_ok=True)
        t0 = time.perf_counter()
        with open(os.path.join(a.out, f"samples_{a.shard}.jsonl"), "w") as f:
            for i, ex in enumerate(rows):
                torch.manual_seed(1000 * a.shard + i)
                f.write(json.dumps({**ex, "samples": sample_k(model, tok, ex["inst"], a.k)}) + "\n")
                if (i + 1) % 100 == 0:
                    print(f"shard {a.shard}: {i + 1}/{len(rows)} ({time.perf_counter() - t0:.0f}s)", flush=True)
        return

    if a.cmd == "pairs":
        rows = [r for f in sorted(os.listdir(a.out)) if f.startswith("samples_") for r in rl(os.path.join(a.out, f))]
        pairs = build_pairs(rows, random.Random(0), a.max_per_prompt)
        open(os.path.join(a.out, "pairs.jsonl"), "w").writelines(json.dumps(p) + "\n" for p in pairs)
        for kind in ("name", "kw"):
            rs = [r for r in rows if r["kind"] == kind]
            ps = [p for p in pairs if p["kind"] == kind]
            n_ok = [sum(passes(r, s) for s in r["samples"]) for r in rs]
            print(f"{kind}: {len(rs)} prompts, pass-rate {np.sum(n_ok) / sum(len(r['samples']) for r in rs) * 100:.1f}%, "
                  f"prompts with >=1 pass {np.mean([n > 0 for n in n_ok]) * 100:.0f}%, pairs {len(ps)}, "
                  f"chosen len {np.mean([len(p['chosen']) for p in ps]):.1f} vs rejected {np.mean([len(p['rejected']) for p in ps]):.1f}")
        return

    prompts, val_sft = rl(a.prompts), rl(os.path.join(a.sft_data, "val.jsonl"))
    if a.cmd == "eval":
        model, tok = load_sft_model(a.policy, a.ckpt or a.policy)
        res = full_eval(model, tok, prompts, val_sft, a.data_dir, a.k)
        print_eval(res, a.k)
        if a.json:
            json.dump(res, open(a.json, "w"), indent=1)
        return

    torch.manual_seed(a.seed)
    model, tok = load_sft_model(a.policy, a.policy)
    if a.mode == "lora":
        apply_lora(model, a.rank, 2 * a.rank)
    pairs = rl(a.pairs)
    print(f"{len(pairs)} pairs | objective={a.objective} mode={a.mode} beta={a.beta} lr={a.lr} sft_coef={a.sft_coef}")
    train_preferences(model, tok, pairs, a)
    os.makedirs(a.out, exist_ok=True)
    if a.mode == "lora":
        from .lora import lora_state_dict
        torch.save({"rank": a.rank, "alpha": 2 * a.rank, "targets": "qkv,proj,fc,gate", "state": lora_state_dict(model)},
                   os.path.join(a.out, "adapter.pt"))
        merge_lora(model)
    else:
        torch.save({"config": model.cfg.__dict__, "model": model.state_dict()}, os.path.join(a.out, "model.pt"))
    tok.save(os.path.join(a.out, "tokenizer.json"))
    res = full_eval(model, tok, prompts, val_sft, a.data_dir, a.k)
    res["args"] = {k: v for k, v in vars(a).items()}
    json.dump(res, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    print_eval(res, a.k)


if __name__ == "__main__":
    main()
