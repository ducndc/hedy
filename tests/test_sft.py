from collections import Counter

import torch

from hedy.model import GPT, GPTConfig
from hedy.sft import (END, END_MARK, PROMPT, build_dataset, collate, doc_freq, encode_example, find_name,
                      has_word, make_example, response_loss, truncate_story)
from hedy.tokenizer import BPETokenizer

DOCS = [f"Tim and Mia went to the {w}. Tim loved the {w} very much. Mia smiled at Tim. They played all day long. "
        f"Then they went home to sleep." for w in ["garden", "market", "forest", "castle", "village", "meadow"] * 8]
TOK = BPETokenizer.train(" ".join(DOCS), 320)


def test_labels_mask_the_prompt_only():
    ex = {"kind": "plain", "key": "", "inst": "Write a short story.", "response": "Tim ran. Mia hid."}
    ids, lab = encode_example(ex, TOK)
    p = len(TOK.encode(PROMPT.format(inst=ex["inst"])))
    assert len(ids) == len(lab) and lab[:p] == [-100] * p and lab[p:] == ids[p:]
    assert TOK.decode(ids[p:]) == ex["response"] + END                # response + End marker are trained on


def test_collate_shifts_pads_and_ignores_padding():
    a = ([5, 6, 7, 8], [-100, -100, 7, 8]); b = ([5, 6, 9], [-100, -100, 9])
    x, y = collate([a, b])
    assert x.shape == y.shape == (2, 3)
    assert y.tolist() == [[-100, 7, 8], [-100, 9, -100]]            # shifted by one; pad target ignored
    assert x.tolist() == [[5, 6, 7], [5, 6, 0]]


def test_padding_does_not_change_loss():
    """Right-padding is invisible to causal attention: loss of a sequence alone == inside a padded batch."""
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=TOK.vocab_size, block_size=128, n_layer=2, n_head=2, n_embd=32, dropout=0.0)).eval()
    exs = [{"kind": "plain", "key": "", "inst": "Write a short story.", "response": r}
           for r in ("Tim ran home.", "Tim and Mia went to the garden. They played all day long.")]
    alone = [response_loss(m, TOK, [e]) for e in exs]
    together = response_loss(m, TOK, exs)
    n = [sum(1 for t in encode_example(e, TOK)[1] if t != -100) for e in exs]
    assert abs(together - (alone[0] * n[0] + alone[1] * n[1]) / sum(n)) < 1e-4


def test_dataset_examples_are_verifiable_and_fit_context():
    df = Counter({w: 100 for w in ["garden", "market", "forest", "castle", "village", "meadow"]})
    ds = build_dataset(DOCS, 30, TOK, df, seed=0)
    assert {e["kind"] for e in ds} == {"kw", "name", "plain"}
    for e in ds:
        if e["kind"] in ("kw", "name"):
            assert has_word(e["response"], e["key"]) and e["key"] in e["inst"]
        assert len(encode_example(e, TOK)[0]) <= 128


def test_helpers():
    assert find_name("Tim ran home. Then Tim saw Mia. Mia waved at Tim.") == ["Mia", "Tim"]
    assert find_name("The cat sat. The dog ran.") == []
    assert has_word("A big Garden!", "garden") and not has_word("gardener", "garden")
    assert truncate_story(DOCS[0], TOK, 10_000) == DOCS[0]
    assert truncate_story(DOCS[0], TOK, 5) is None                       # first sentence alone doesn't fit twice
    assert len(TOK.encode(truncate_story(DOCS[0], TOK, 40))) <= 40


def test_generate_stop_rule():
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=30, block_size=32, n_layer=1, n_head=2, n_embd=16, dropout=0.0)).eval()
    out = m.generate(torch.zeros((1, 1), dtype=torch.long), 20, stop=lambda ids: len(ids) >= 6)
    assert out.shape == (1, 6)


def test_key_nll_runs_and_instruction_helps_a_model_that_learned_to_copy():
    from hedy.sft import TEMPLATES, key_nll
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=TOK.vocab_size, block_size=128, n_layer=2, n_head=2, n_embd=64, dropout=0.0))
    exs = []
    for w in ["garden", "market", "forest", "castle", "village", "meadow"]:
        resp = f"Tim and Mia went to the {w}. Tim loved the {w} very much."
        exs.append({"kind": "kw", "key": w, "inst": TEMPLATES["kw"][0].format(k=w), "response": resp})
        exs.append({"kind": "name", "key": "Mia", "inst": TEMPLATES["name"][0].format(k="Mia"), "response": resp})
    enc = [encode_example(e, TOK) for e in exs]
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    for _ in range(200):                                           # overfit: the instruction word predicts the story word
        x, y = collate(enc); loss = m(x, y)[1]; opt.zero_grad(); loss.backward(); opt.step()
    r = key_nll(m.eval(), TOK, exs)
    with_inst, neutral = r["kw"]
    assert with_inst < neutral - 0.5                               # knowing the instruction makes the word likelier
