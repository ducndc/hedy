"""Evaluation toolkit. Run:  python -m hedy.eval --ckpt checkpoints/tinystories --data_dir data/tinystories \
                                 --text data/tinystories-valid.txt

Sections: (1) perplexity done properly, (2) n-gram baselines for context, (3) a multiple-choice
likelihood benchmark scored the way lm-evaluation-harness does it, (4) generation diagnostics.
"""
import argparse
import json
import math
import os
import random
import re

import numpy as np
import torch
import torch.nn.functional as F

from .data import load_bin
from .model import GPT, GPTConfig
from .prepare import split_docs
from .tokenizer import BPETokenizer, load_tokenizer

LN2 = math.log(2)


# ----------------------------------------------------------------------------- utilities
def load_model(ckpt_dir: str):
    tok = load_tokenizer(os.path.join(ckpt_dir, "tokenizer.json"))
    ck = torch.load(os.path.join(ckpt_dir, "model.pt"), map_location="cpu")
    model = GPT(GPTConfig(**ck["config"]))
    model.load_state_dict(ck["model"])
    return model.eval(), tok


def token_nbytes(tok) -> np.ndarray:
    """UTF-8 byte length of every token id; lets us convert token-level loss to bits per byte."""
    if isinstance(tok, BPETokenizer):
        return np.array([len(tok.vocab[i]) for i in range(tok.vocab_size)])
    return np.array([len(c.encode("utf-8")) for c in tok.chars])


def wilson(k: int, n: int, z: float = 1.96):
    """95% Wilson score interval for a binomial proportion (better than +-sqrt(p(1-p)/n) for small n)."""
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


# ----------------------------------------------------------------------------- 1. perplexity
@torch.no_grad()
def token_nlls(model, tokens: np.ndarray, window: int, stride: int, batch: int = 64):
    """Per-token negative log-likelihood (nats) with a sliding window.

    Window w covers tokens[s : s+window], s = 0, stride, 2*stride, ... In every window only the
    last `stride` targets are scored (the first window scores all of them), so every token except
    tokens[0] is scored exactly once and always sees (almost) a full window of left context.
    stride == window reproduces the naive non-overlapping "chunked" evaluation.
    Returns (nll[len-1], position_in_window[len-1]); entry i is the loss of predicting tokens[i+1].
    """
    t = torch.from_numpy(np.asarray(tokens, dtype=np.int64))
    n = len(t)
    nll = np.zeros(n - 1)
    pos = np.zeros(n - 1, dtype=np.int64)
    seen = np.zeros(n - 1, dtype=np.int64)
    starts = list(range(0, max(n - 1 - (window - stride), 1), stride))
    for b in range(0, len(starts), batch):
        chunk = starts[b : b + batch]
        # windows near the end can be shorter; evaluate those one by one
        full = [s for s in chunk if s + window + 1 <= n]
        for group, ss in ((True, full), (False, [s for s in chunk if s + window + 1 > n])):
            if not ss:
                continue
            lens = [min(window, n - 1 - s) for s in ss]
            if group:
                x = torch.stack([t[s : s + window] for s in ss]); y = torch.stack([t[s + 1 : s + window + 1] for s in ss])
                lp = F.cross_entropy(model(x)[0].transpose(1, 2), y, reduction="none").numpy()
                rows = [(s, lp[i], window) for i, s in enumerate(ss)]
            else:
                rows = []
                for s, L in zip(ss, lens):
                    x, y = t[s : s + L][None], t[s + 1 : s + L + 1][None]
                    rows.append((s, F.cross_entropy(model(x)[0].transpose(1, 2), y, reduction="none").numpy()[0], L))
            for s, l, L in rows:
                # Score only the "new" tail: the previous window already scored everything before
                # in-window index window-stride (also true for a short final window).
                first = 0 if s == 0 else window - stride
                for j in range(first, L):
                    nll[s + j] = l[j]; pos[s + j] = j
                    seen[s + j] += 1
    assert (seen == 1).all(), "every token must be scored exactly once"
    return nll, pos


def ppl_report(model, tok, tokens, window, stride):
    nll, pos = token_nlls(model, tokens, window, stride)
    nb = token_nbytes(tok)[np.asarray(tokens)[1:]]
    return {"loss": nll.mean(), "ppl": math.exp(nll.mean()), "bpb": nll.sum() / nb.sum() / LN2, "n": len(nll)}, nll, pos


# ----------------------------------------------------------------------------- 2. n-gram baselines
def ngram_baselines(train: np.ndarray, val: np.ndarray, V: int, nbytes: np.ndarray):
    """Unigram and bigram language models with a Dirichlet-style prior (the bigram backs off to the
    unigram). alpha is tuned on the last 200k *training* tokens, never on validation."""
    hold = train[-200_000:]; fit = train[:-200_000]

    def fit_counts(x):
        uni = np.bincount(x, minlength=V).astype(np.float64)
        bi = np.bincount(x[:-1].astype(np.int64) * V + x[1:], minlength=V * V).reshape(V, V).astype(np.float64)
        return uni, bi

    def nll_uni(uni, x):
        p = (uni + 1) / (uni.sum() + V)
        return -np.log(p[x[1:]])

    def nll_bi(uni, bi, x, alpha):
        p1 = (uni + 1) / (uni.sum() + V)
        a, b = x[:-1], x[1:]
        p = (bi[a, b] + alpha * p1[b]) / (bi[a].sum(1) + alpha)
        return -np.log(p)

    uni, bi = fit_counts(fit)
    best = min((nll_bi(uni, bi, hold, al).mean(), al) for al in (0.1, 0.3, 1, 3, 10, 30))[1]
    uni, bi = fit_counts(train)
    out = {}
    for name, nll in (("unigram", nll_uni(uni, val)), ("bigram", nll_bi(uni, bi, val, best))):
        out[name] = {"loss": nll.mean(), "bpb": nll.sum() / nbytes[val[1:]].sum() / LN2}
    out["bigram"]["alpha"] = best
    return out


# ----------------------------------------------------------------------------- 3. likelihood MCQ
_SENT_END = re.compile(r'(?:(?<=[.!?])|(?<=[.!?]["\']))\s+(?=[A-Z"\'0-9])')  # not before a lowercase dialogue tag


def sentences(story: str) -> list[str]:
    return [s for s in _SENT_END.split(story.strip().replace("\n", " ")) if s]


def build_cloze(val_docs, tok, n_items=500, n_ctx=3, n_choices=4, seed=0):
    """HellaSwag-style items from held-out stories: given the first `n_ctx` sentences, pick the true
    next sentence among distractors = the same-position sentence from *other* stories.
    Context ends at the sentence's final punctuation and each choice starts with a space, so
    encode(ctx) + encode(choice) == encode(ctx + choice) (no BPE merge across the boundary).
    Items where that doesn't hold are dropped."""
    rng = random.Random(seed)
    pool = [sentences(d) for d in val_docs]
    pool = [s for s in pool if len(s) > n_ctx]
    items, dropped = [], 0
    for _ in range(n_items * 3):
        if len(items) == n_items:
            break
        i = rng.randrange(len(pool))
        ctx = " ".join(pool[i][:n_ctx])
        truth = " " + pool[i][n_ctx]
        others = [" " + pool[j][n_ctx] for j in rng.sample(range(len(pool)), n_choices + 2) if j != i][: n_choices - 1]
        choices = others + [truth]
        if len(set(choices)) < n_choices:
            continue
        rng.shuffle(choices)
        if any(tok.encode(ctx) + tok.encode(c) != tok.encode(ctx + c) for c in choices):
            dropped += 1
            continue
        items.append({"ctx": ctx, "choices": choices, "label": choices.index(truth)})
    return items, dropped


@torch.no_grad()
def loglikelihood(model, tok, ctx: str, cont: str) -> float:
    """log P(cont | ctx): sum over continuation tokens of log p(token | everything before it)."""
    c_ids, k_ids = tok.encode(ctx), tok.encode(cont)
    ids = (c_ids + k_ids)[-(model.cfg.block_size + 1):]
    n_cont = min(len(k_ids), len(ids) - 1)
    x = torch.tensor([ids[:-1]])
    logp = F.log_softmax(model(x)[0][0], dim=-1)
    tgt = torch.tensor(ids[1:])
    return logp[torch.arange(len(tgt)), tgt][-n_cont:].sum().item()


def score_mcq(model, tok, items):
    """Returns accuracy variants. acc: argmax total log-likelihood (biased toward SHORT choices,
    every extra token costs probability). acc_norm: argmax of log-likelihood per byte of the choice
    (what lm-evaluation-harness reports as acc_norm). Also the trivial length-only baselines."""
    hit = {"acc": 0, "acc_norm": 0, "shortest": 0, "longest": 0}
    for it in items:
        ll = np.array([loglikelihood(model, tok, it["ctx"], c) for c in it["choices"]])
        nb = np.array([len(c.encode("utf-8")) for c in it["choices"]])
        hit["acc"] += int(ll.argmax() == it["label"])
        hit["acc_norm"] += int((ll / nb).argmax() == it["label"])
        hit["shortest"] += int(nb.argmin() == it["label"])
        hit["longest"] += int(nb.argmax() == it["label"])
    n = len(items)
    return {k: (v / n, *wilson(v, n)) for k, v in hit.items()}


# ----------------------------------------------------------------------------- 4. generation
def distinct_n(ids: list[int], n: int) -> float:
    grams = [tuple(ids[i : i + n]) for i in range(len(ids) - n + 1)]
    return len(set(grams)) / max(len(grams), 1)


@torch.no_grad()
def gen_diagnostics(model, tok, prompts, temps=(0.5, 0.8, 1.0, 1.3), n_tokens=120, seed=0):
    torch.manual_seed(seed)
    rows = {}
    for T in temps:
        d1 = d2 = d3 = 0.0
        for p in prompts:
            out = model.generate(torch.tensor([tok.encode(p)]), n_tokens, temperature=T)[0].tolist()[len(tok.encode(p)):]
            d1 += distinct_n(out, 1); d2 += distinct_n(out, 2); d3 += distinct_n(out, 3)
        rows[T] = (d1 / len(prompts), d2 / len(prompts), d3 / len(prompts))
    return rows


# ----------------------------------------------------------------------------- CLI
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data_dir", required=True, help="hedy.prepare output (for val.bin/train.bin)")
    p.add_argument("--text", required=True, help="raw text corpus, to rebuild the held-out documents")
    p.add_argument("--n_items", type=int, default=500)
    p.add_argument("--max_tokens", type=int, default=0, help="limit validation tokens (0 = all)")
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--strides", default="", help="comma list of sliding-window strides (default: window, window/2, window/8)")
    a = p.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    model, tok = load_model(a.ckpt)
    V, W = tok.vocab_size, model.cfg.block_size
    val = np.asarray(load_bin(os.path.join(a.data_dir, "val.bin")))
    train = np.asarray(load_bin(os.path.join(a.data_dir, "train.bin")))
    if a.max_tokens:
        val = val[: a.max_tokens]
    nb = token_nbytes(tok)
    print(f"model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, window {W}, vocab {V}; "
          f"val tokens {len(val):,}")

    print("\n== 1. perplexity (val) ==")
    print(f"{'method':<28} {'loss':>6} {'ppl':>7} {'bits/byte':>10} {'tokens scored':>14}")
    strides = [int(x) for x in a.strides.split(",")] if a.strides else [W, W // 2, W // 8]
    chunk_pos = None
    for stride in strides:
        name = "chunked (stride=window)" if stride == W else f"sliding stride={stride}"
        r, nll, pos = ppl_report(model, tok, val, W, stride)
        print(f"{name:<28} {r['loss']:>6.3f} {r['ppl']:>7.2f} {r['bpb']:>10.3f} {r['n']:>14,}")
        if stride == W:
            chunk_pos = np.array([nll[pos == j].mean() for j in range(W)])
    if chunk_pos is not None:
        print("loss by position inside a chunk (nats):  " + "  ".join(f"pos{j}:{chunk_pos[j]:.2f}" for j in (0, 1, 3, 7, 15, 31, 63, W - 1)))

    print("\n== 2. baselines (val bits/byte) ==")
    print(f"{'uniform over tokens':<28} {math.log(V) / (nb[val[1:]].mean() * LN2):>10.3f}")
    for k, v in ngram_baselines(train, val, V, nb).items():
        print(f"{k:<28} {v['bpb']:>10.3f}" + (f"   (alpha={v['alpha']})" if "alpha" in v else ""))

    print("\n== 3. next-sentence multiple choice (4 choices, chance = 25%) ==")
    docs = split_docs(open(a.text, encoding="utf-8").read())[1]
    items, dropped = build_cloze(docs, tok, a.n_items)
    print(f"{len(items)} items from {len(docs)} held-out stories ({dropped} dropped for BPE-boundary mismatch)")
    for k, (acc, lo, hi) in score_mcq(model, tok, items).items():
        print(f"{k:<10} {acc * 100:5.1f}%   95% CI [{lo * 100:.1f}, {hi * 100:.1f}]")

    print("\n== 4. generation diversity (distinct-1/2/3 of 120 sampled tokens, 30 prompts) ==")
    prompts = [d.split(".")[0] + "." for d in docs[:30]]
    ref = [tok.encode(" ".join(d.split()[:90]))[:120] for d in docs[:30]]
    print(f"{'real held-out text':<20} " + " ".join(f"{np.mean([distinct_n(r, n) for r in ref]):.3f}" for n in (1, 2, 3)))
    for T, (d1, d2, d3) in gen_diagnostics(model, tok, prompts).items():
        print(f"{'temperature ' + str(T):<20} {d1:.3f} {d2:.3f} {d3:.3f}")


if __name__ == "__main__":
    main()
