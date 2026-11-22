"""Supervised fine-tuning (SFT), full or LoRA, on synthetic *verifiable* instructions built from TinyStories.

    python -m hedy.sft build --text data/tinystories-valid.txt --out data/sft
    python -m hedy.sft run --base checkpoints/tinystories --data data/sft --data_dir data/tinystories \
                           --mode lora --rank 8 --out checkpoints/sft-lora8

Three instruction types, each checkable by a program (no judge model needed):
    kw    "Write a short story that includes the word "X"."    -> X must appear
    name  "Write a short story about a character named N."     -> N must appear
    plain "Write a short story."                               -> just a story
Format (plain text, our BPE has no special tokens):
    ### Instruction:\\n{inst}\\n### Response:\\n{response}\\n### End\\n
The loss is computed on the response and the End marker only (prompt tokens are masked).
"""
import argparse
import json
import math
import os
import random
import re
import time
from collections import Counter

import numpy as np
import torch

from .eval import load_model, ppl_report, sentences, wilson
from .lora import apply_lora, count_params, lora_state_dict, merge_lora
from .data import load_bin
from .prepare import split_docs

PROMPT = "### Instruction:\n{inst}\n### Response:\n"
END = "\n### End\n"
END_MARK = "### End"
TEMPLATES = {
    "kw": ['Write a short story that includes the word "{k}".', 'Tell a story using the word "{k}".'],
    "name": ["Write a short story about a character named {k}.", "Tell a story about {k}."],
    "plain": ["Write a short story.", "Tell me a story."],
}
KINDS = ("kw", "name", "plain")


# ----------------------------------------------------------------------------- data
def doc_freq(docs):
    c = Counter()
    for d in docs:
        c.update(set(re.findall(r"[a-z]{5,}", d)))
    return c


def truncate_story(doc, tok, budget):
    """Longest prefix of whole sentences that fits in `budget` tokens (>= 2 sentences)."""
    sents, used = sentences(doc), []
    for s in sents:
        if len(tok.encode(" ".join(used + [s]))) > budget:
            break
        used.append(s)
    return " ".join(used) if len(used) >= 2 else None


def find_name(response):
    """A capitalized word that appears mid-sentence (so not just a sentence-initial word) and >= 2 times."""
    mid = Counter()
    for s in sentences(response):
        mid.update(re.findall(r"\b[A-Z][a-z]+\b", " ".join(s.split()[1:])))
    total = Counter(re.findall(r"\b[A-Z][a-z]+\b", response))
    cands = sorted(w for w in mid if total[w] >= 2)
    return cands


def make_example(doc, kind, tok, rng, df, budget=72, max_len=128):
    """One instruction/response pair, guaranteed to fit `max_len` tokens (prompt + response + End).
    If the prompt tokenizes long, retry with a shorter response rather than silently overflowing."""
    for b in (budget, budget - 16, budget - 32):
        resp = truncate_story(doc, tok, b)
        if resp is None:
            return None
        if kind == "kw":
            # content words that are neither ubiquitous nor so rare that they're typos/odd names
            cands = sorted({w for w in re.findall(r"[a-z]{5,}", resp) if 30 <= df[w] <= 1500})
        elif kind == "name":
            cands = find_name(resp)
        else:
            cands = [""]
        if not cands:
            return None
        key = rng.choice(cands)
        ex = {"kind": kind, "key": key, "inst": rng.choice(TEMPLATES[kind]).format(k=key), "response": resp}
        if len(encode_example(ex, tok)[0]) <= max_len:
            return ex
    return None


def build_dataset(docs, n, tok, df, seed=0):
    rng, docs, out, i = random.Random(seed), list(docs), [], 0
    rng.shuffle(docs)
    for d in docs:
        ex = make_example(d, KINDS[len(out) % 3], tok, rng, df)
        if ex:
            out.append(ex)
        if len(out) == n:
            break
    return out


def encode_example(ex, tok):
    """-> (ids, labels). labels = -100 on the prompt, so the loss only teaches the response."""
    p = tok.encode(PROMPT.format(inst=ex["inst"]))
    r = tok.encode(ex["response"] + END)
    return p + r, [-100] * len(p) + r


def collate(encoded):
    """Right-pad to the longest sequence. Pads sit AFTER the real tokens, so causal attention never
    lets a real token see them: no attention mask needed. Pad targets are -100 (ignored)."""
    L = max(len(ids) for ids, _ in encoded) - 1
    x = torch.zeros(len(encoded), L, dtype=torch.long)
    y = torch.full((len(encoded), L), -100, dtype=torch.long)
    for i, (ids, lab) in enumerate(encoded):
        x[i, : len(ids) - 1] = torch.tensor(ids[:-1])
        y[i, : len(ids) - 1] = torch.tensor(lab[1:])        # next-token targets
    return x, y


# ----------------------------------------------------------------------------- generation + metrics
@torch.no_grad()
def respond(model, tok, inst, temperature=0.7, top_k=40):
    prompt = tok.encode(PROMPT.format(inst=inst))
    n_new = model.cfg.block_size - len(prompt) - 1
    stop = lambda ids: END_MARK in tok.decode(ids[len(prompt):])
    out = model.generate(torch.tensor([prompt]), n_new, temperature, top_k, stop=stop)[0].tolist()[len(prompt):]
    text = tok.decode(out)
    done = END_MARK in text
    return text.split(END_MARK)[0].strip(), done, len(out)


def has_word(text, w):
    return bool(re.search(rf"\b{re.escape(w)}\b", text, re.I))


def evaluate_following(model, tok, examples, seed=0, temperature=0.7):
    """Generate for each held-out instruction and verify with code. Also a *shuffled control*: score
    each response against the NEXT prompt's key, i.e. how often the word shows up by coincidence."""
    model.eval()
    rows = []
    for i, ex in enumerate(examples):
        torch.manual_seed(seed + i)
        text, done, n = respond(model, tok, ex["inst"], temperature)
        rows.append({**ex, "out": text, "done": done, "n_tok": n})
    res = {}
    for kind in KINDS:
        sub = [(i, r) for i, r in enumerate(rows) if r["kind"] == kind]
        if kind != "plain":
            ok = sum(has_word(r["out"], r["key"]) for _, r in sub)
            others = [r for r in rows if r["kind"] == kind]
            ctl = sum(has_word(others[(j + 1) % len(others)]["out"], r["key"]) for j, r in enumerate(others))
            res[kind] = {"n": len(sub), "ok": ok / len(sub), "ci": wilson(ok, len(sub)), "control": ctl / len(sub)}
    res["terminated"] = sum(r["done"] for r in rows) / len(rows)
    res["mean_tokens"] = float(np.mean([r["n_tok"] for r in rows]))
    return res, rows


@torch.no_grad()
def response_loss(model, tok, examples, batch=32):
    model.eval()
    tot, n = 0.0, 0
    for i in range(0, len(examples), batch):
        x, y = collate([encode_example(e, tok) for e in examples[i : i + batch]])
        logits = model(x)[0]
        l = torch.nn.functional.cross_entropy(logits.transpose(1, 2), y, ignore_index=-100, reduction="sum")
        tot += l.item(); n += (y != -100).sum().item()
    return tot / n


@torch.no_grad()
def key_nll(model, tok, examples):
    """Teacher-forced probe, smoother than a pass/fail rate: how many nats does the model pay to emit
    the required word at its place in the reference response, (a) with the real instruction and
    (b) with the neutral "Write a short story." instruction? gain = (b) - (a) = how much the model
    *uses* the instruction. Returns {kind: (nll_with_instruction, nll_neutral)} for kw and name."""
    model.eval()
    neutral = TEMPLATES["plain"][0]
    out = {}
    for kind in ("kw", "name"):
        a_tot = b_tot = 0.0; n = 0
        for ex in (e for e in examples if e["kind"] == kind):
            resp, key = ex["response"], ex["key"]
            m = re.search(rf"(?<= )\b{re.escape(key)}\b", resp)       # first occurrence preceded by a space
            if not m:
                continue
            n_before = len(tok.encode(resp[: m.start() - 1]))             # prefix ends at a chunk boundary
            span = len(tok.encode(resp[: m.end()])) - n_before            # tokens of " key"
            for inst, which in ((ex["inst"], 0), (neutral, 1)):
                ids = tok.encode(PROMPT.format(inst=inst)) + tok.encode(resp + END)
                p_len = len(tok.encode(PROMPT.format(inst=inst)))
                logp = torch.log_softmax(model(torch.tensor([ids[:-1]]))[0][0], -1)
                tgt = range(p_len + n_before - 1, p_len + n_before - 1 + span)  # positions predicting the span
                nll = -sum(logp[i, ids[i + 1]].item() for i in tgt)
                if which == 0:
                    a_tot += nll
                else:
                    b_tot += nll
            n += 1
        out[kind] = (a_tot / n, b_tot / n)
    return out


def load_sft_model(base_dir, ckpt_dir):
    """Load base, then apply whatever lives in ckpt_dir: adapter.pt (LoRA, merged) or model.pt (full)."""
    model, tok = load_model(base_dir)
    if os.path.exists(os.path.join(ckpt_dir, "adapter.pt")):
        ad = torch.load(os.path.join(ckpt_dir, "adapter.pt"))
        apply_lora(model, ad["rank"], ad["alpha"], targets=tuple(ad["targets"].split(",")))
        model.load_state_dict(ad["state"], strict=False)
        merge_lora(model)
    elif os.path.exists(os.path.join(ckpt_dir, "model.pt")):
        model, tok = load_model(ckpt_dir)
    return model.eval(), tok


# ----------------------------------------------------------------------------- training
def finetune(model, tok, train, val, args):
    enc = [encode_example(e, tok) for e in train]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
    steps = args.epochs * math.ceil(len(enc) / args.batch_size)
    rng, step, t0 = random.Random(args.seed), 0, time.perf_counter()
    model.train()
    for ep in range(args.epochs):
        order = list(range(len(enc))); rng.shuffle(order)
        for b in range(0, len(order), args.batch_size):
            lr = args.lr * min(1.0, (step + 1) / args.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps)))
            for g in opt.param_groups:
                g["lr"] = lr
            x, y = collate([enc[i] for i in order[b : b + args.batch_size]])
            loss = model(x, y)[1]
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); step += 1
            if step % 50 == 0:
                print(f"  step {step}/{steps} loss {loss.item():.3f} lr {lr:.1e}", flush=True)
        print(f"epoch {ep + 1}: held-out response loss {response_loss(model, tok, val):.3f} "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)
        model.train()


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--text", required=True); b.add_argument("--out", required=True)
    b.add_argument("--base", default="checkpoints/tinystories", help="for the tokenizer (token budgets)")
    b.add_argument("--n_train", type=int, default=6000); b.add_argument("--n_val", type=int, default=300)
    pr = sub.add_parser("probe", help="key-word teacher-forced probe on an existing checkpoint")
    pr.add_argument("--base", required=True); pr.add_argument("--ckpt", default=None); pr.add_argument("--data", required=True)
    r = sub.add_parser("run")
    r.add_argument("--base", required=True); r.add_argument("--data", required=True)
    r.add_argument("--data_dir", required=True, help="pretraining data (to measure forgetting)")
    r.add_argument("--mode", choices=["none", "full", "lora"], required=True)
    r.add_argument("--rank", type=int, default=8); r.add_argument("--alpha", type=float, default=0)
    r.add_argument("--targets", default="qkv,proj,fc,gate")
    r.add_argument("--lr", type=float, default=3e-4); r.add_argument("--epochs", type=int, default=3)
    r.add_argument("--batch_size", type=int, default=32); r.add_argument("--warmup", type=int, default=20)
    r.add_argument("--n_train", type=int, default=0, help="use only the first N training examples")
    r.add_argument("--seed", type=int, default=0); r.add_argument("--threads", type=int, default=0)
    r.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.cmd == "build":
        _, tok = load_model(a.base)
        train_docs, val_docs = split_docs(open(a.text, encoding="utf-8").read())
        df = doc_freq(train_docs)
        os.makedirs(a.out, exist_ok=True)
        for name, docs, n in (("train", train_docs, a.n_train), ("val", val_docs, a.n_val)):
            ex = build_dataset(docs, n, tok, df, seed=1 if name == "train" else 2)
            with open(os.path.join(a.out, f"{name}.jsonl"), "w") as f:
                f.writelines(json.dumps(e) + "\n" for e in ex)
            print(name, len(ex), Counter(e["kind"] for e in ex))
        print(PROMPT.format(inst=ex[0]["inst"]) + ex[0]["response"] + END)
        return

    if a.cmd == "probe":
        model, tok = load_sft_model(a.base, a.ckpt or a.base)
        val = [json.loads(l) for l in open(os.path.join(a.data, "val.jsonl"))]
        print({k: tuple(round(x, 3) for x in v) for k, v in key_nll(model, tok, val).items()})
        return
    if a.threads:
        torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    load = lambda n: [json.loads(l) for l in open(os.path.join(a.data, n))]
    train, val = load("train.jsonl"), load("val.jsonl")
    if a.n_train:
        train = train[: a.n_train]
    model, tok = load_model(a.base)
    base_params = count_params(model)[1]
    if a.mode == "lora":
        apply_lora(model, a.rank, a.alpha or 2 * a.rank, targets=tuple(a.targets.split(",")))
    trainable, total = count_params(model)
    if a.mode == "none":
        trainable = 0
    print(f"mode={a.mode} trainable={trainable:,} ({100 * trainable / base_params:.2f}% of base {base_params:,})")
    if a.mode != "none":
        finetune(model, tok, train, val, a)
    os.makedirs(a.out, exist_ok=True)
    if a.mode == "lora":
        torch.save({"rank": a.rank, "alpha": a.alpha or 2 * a.rank, "targets": a.targets, "state": lora_state_dict(model)},
                   os.path.join(a.out, "adapter.pt"))
        merge_lora(model)                                  # evaluate/serve the merged model
    elif a.mode == "full":
        torch.save({"config": model.cfg.__dict__, "model": model.state_dict()}, os.path.join(a.out, "model.pt"))
    tok.save(os.path.join(a.out, "tokenizer.json"))

    res, rows = evaluate_following(model, tok, val)
    heldout = response_loss(model, tok, val)
    val_tok = np.asarray(load_bin(os.path.join(a.data_dir, "val.bin")))[:60_000]
    bpb = ppl_report(model, tok, val_tok, model.cfg.block_size, 64)[0]["bpb"]
    kn = key_nll(model, tok, val)
    out = {"key_nll": kn, "mode": a.mode, "rank": a.rank, "trainable": trainable, "lr": a.lr, "epochs": a.epochs,
           "n_train": len(train), "seed": a.seed, "sft_val_loss": heldout, "pretrain_val_bpb": bpb, **res}
    json.dump(out, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    print(f"held-out response loss {heldout:.3f} | pretraining val bits/byte {bpb:.3f}")
    for k in ("kw", "name"):
        print(f"{k}: followed {res[k]['ok'] * 100:.1f}% (95% CI {res[k]['ci'][0] * 100:.0f}-{res[k]['ci'][1] * 100:.0f}), "
              f"coincidence control {res[k]['control'] * 100:.1f}%")
    print("teacher-forced nats to emit the required word (with instruction / neutral instruction): "
          + ", ".join(f"{k} {v[0]:.2f}/{v[1]:.2f}" for k, v in kn.items()))
    print(f"terminated (emitted End) {res['terminated'] * 100:.1f}%, mean new tokens {res['mean_tokens']:.0f}")
    for r in rows[:6:2]:
        print(f"--- [{r['kind']}] {r['inst']}\n{r['out'][:300]}")


if __name__ == "__main__":
    main()
