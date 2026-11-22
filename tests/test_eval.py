import math

import numpy as np
import torch
import torch.nn.functional as F

from hedy.eval import (build_cloze, distinct_n, loglikelihood, ngram_baselines, score_mcq, sentences,
                       token_nbytes, token_nlls, wilson)
from hedy.model import GPT, GPTConfig
from hedy.tokenizer import BPETokenizer

TEXT = ("Tim had a red ball. He liked to play with it. One day he lost it. Then he cried. "
        "Mia had a blue kite. She flew it high. The wind was strong. She laughed. ") * 30


def tiny(vocab=30, block=16):
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=vocab, block_size=block, n_layer=2, n_head=2, n_embd=32, dropout=0.0)).eval()


def test_sliding_window_scores_every_token_once_with_full_context():
    m = tiny()
    toks = np.random.RandomState(0).randint(0, 30, 103)
    for window, stride in [(16, 16), (16, 8), (16, 4), (16, 1)]:
        nll, pos = token_nlls(m, toks, window, stride)       # asserts "exactly once" internally
        assert len(nll) == len(toks) - 1
        assert (pos[window:] >= window - stride).all()       # later tokens always see >= window-stride+1 tokens
    # matches an independent per-token reference
    nll, pos = token_nlls(m, toks, 16, 4)
    t = torch.from_numpy(toks)
    for i in (0, 5, 20, 57, 100, 101):
        s, j = i - pos[i], pos[i]
        ref = F.cross_entropy(m(t[s : s + j + 1][None])[0][0, -1:], t[i + 1 : i + 2]).item()
        assert abs(ref - nll[i]) < 1e-5


def test_smaller_stride_gives_tokens_more_context_so_position_zero_is_rare():
    m = tiny()
    toks = np.random.RandomState(1).randint(0, 30, 200)
    _, pos_chunk = token_nlls(m, toks, 16, 16)
    _, pos_slide = token_nlls(m, toks, 16, 4)
    assert (pos_chunk == 0).mean() > 5 * (pos_slide == 0).mean()


def test_loglikelihood_matches_manual_sum():
    tok = BPETokenizer.train(TEXT, 300)
    m = GPT(GPTConfig(vocab_size=tok.vocab_size, block_size=64, n_layer=2, n_head=2, n_embd=32, dropout=0.0)).eval()
    ctx, cont = "Tim had a red ball.", " He liked to play."
    ids = tok.encode(ctx) + tok.encode(cont)
    lp = F.log_softmax(m(torch.tensor([ids[:-1]]))[0][0], -1)
    manual = sum(lp[i, ids[i + 1]].item() for i in range(len(tok.encode(ctx)) - 1, len(ids) - 1))
    assert abs(loglikelihood(m, tok, ctx, cont) - manual) < 1e-4


def test_cloze_builder_alignment_and_labels():
    tok = BPETokenizer.train(TEXT, 300)
    docs = [f"Tim had a ball number {i}. He liked it a lot. It was red. Then he ran fast number {i}. The end came soon." for i in range(40)]
    items, dropped = build_cloze(docs, tok, n_items=20, n_ctx=3, n_choices=4, seed=1)
    assert len(items) == 20
    for it in items:
        assert len(set(it["choices"])) == 4 and 0 <= it["label"] < 4
        for c in it["choices"]:                              # no BPE merge across the ctx/choice boundary
            assert tok.encode(it["ctx"]) + tok.encode(c) == tok.encode(it["ctx"] + c)


def test_mcq_scores_a_model_that_knows_the_answer():
    tok = BPETokenizer.train(TEXT, 300)
    m = GPT(GPTConfig(vocab_size=tok.vocab_size, block_size=64, n_layer=2, n_head=2, n_embd=64, dropout=0.0))
    ids = torch.tensor(tok.encode(TEXT[:400]))
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(150):                                     # memorize the passage
        i = torch.randint(0, len(ids) - 33, (8,))
        x = torch.stack([ids[j : j + 32] for j in i]); y = torch.stack([ids[j + 1 : j + 33] for j in i])
        loss = m(x, y)[1]; opt.zero_grad(); loss.backward(); opt.step()
    m.eval()
    item = {"ctx": "Tim had a red ball.", "choices": [" She flew it high.", " He liked to play with it.", " The wind was strong.", " Then he cried."], "label": 1}
    assert score_mcq(m, tok, [item])["acc"][0] == 1.0


def test_wilson_interval_and_distinct_n():
    lo, hi = wilson(50, 100)
    assert lo < 0.5 < hi and 0.40 < lo and hi < 0.60
    assert wilson(0, 10)[0] == 0.0 or wilson(0, 10)[0] < 1e-9
    assert distinct_n([1, 1, 1, 1], 2) == 1 / 3 and distinct_n([1, 2, 3, 4], 2) == 1.0


def test_sentences_and_ngram_baselines():
    assert sentences('Hi there. "Wow!" said Tim. Is it? Yes it is.') == ['Hi there.', '"Wow!" said Tim.', 'Is it?', 'Yes it is.']
    rng = np.random.RandomState(0)
    # a deterministic cycle: bigram should predict it almost perfectly, unigram cannot
    train = np.tile(np.arange(20), 20_000).astype(np.int64); val = np.tile(np.arange(20), 200).astype(np.int64)
    r = ngram_baselines(train, val, 20, np.ones(20, dtype=np.int64))
    assert r["bigram"]["loss"] < 0.1 < math.log(20) - 0.5 < r["unigram"]["loss"] + 0.01
