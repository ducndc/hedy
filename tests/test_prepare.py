import json
import sys

import numpy as np

from hedy.prepare import encode_stream, main, split_docs
from hedy.tokenizer import BPETokenizer, load_tokenizer

STORIES = [f"Tim had a ball number {i}. He liked it a lot. Then he went home to sleep." for i in range(60)]
TEXT = "<|endoftext|>\n".join(STORIES)


def run_prepare(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["prepare", *args])
    main()


def test_prepare_split_then_train_only_with_shared_validation(tmp_path, monkeypatch):
    src = tmp_path / "small.txt"; src.write_text(TEXT)
    run_prepare(monkeypatch, "--input", str(src), "--out_dir", str(tmp_path / "a"), "--vocab_size", "280", "--val_frac", "0.1")
    meta_a = json.load(open(tmp_path / "a" / "meta.json"))
    assert meta_a["val_tokens"] > 0 and meta_a["train_tokens"] > meta_a["val_tokens"]
    # the big corpus reuses a's tokenizer and a's held-out set
    big = tmp_path / "big.txt"; big.write_text(TEXT + "<|endoftext|>cut mid-sto")
    run_prepare(monkeypatch, "--input", str(big), "--out_dir", str(tmp_path / "b"), "--tokenizer", str(tmp_path / "a" / "tokenizer.json"),
                "--train_only", "--drop_last", "--val_from", str(tmp_path / "a"))
    meta_b = json.load(open(tmp_path / "b" / "meta.json"))
    assert meta_b["val_tokens"] == meta_a["val_tokens"] and meta_b["val_bytes"] == meta_a["val_bytes"]
    assert (tmp_path / "b" / "val.bin").read_bytes() == (tmp_path / "a" / "val.bin").read_bytes()
    ids = np.fromfile(tmp_path / "b" / "train.bin", dtype=np.uint16)
    assert len(ids) == meta_b["train_tokens"] and ids.max() < meta_b["vocab_size"]
    tok = load_tokenizer(str(tmp_path / "b" / "tokenizer.json"))
    assert "cut mid-sto" not in tok.decode(ids.tolist())                  # the truncated final story was dropped


def test_encode_stream_matches_in_memory_encoding(tmp_path):
    tok = BPETokenizer.train(" ".join(STORIES), 300)
    n, nbytes = encode_stream(tok, STORIES, str(tmp_path / "t.bin"), chunk_docs=7)       # chunked, as for big corpora
    got = np.fromfile(tmp_path / "t.bin", dtype=np.uint16).tolist()
    want = [t for i in range(0, len(STORIES), 7) for t in tok.encode("\n\n".join(STORIES[i : i + 7]) + "\n\n")]
    assert got == want and n == len(want)
    assert tok.decode(got).startswith(STORIES[0])


def test_split_docs_holds_out_the_tail():
    train, val = split_docs(TEXT, 0.1)
    assert len(val) == 6 and train + val == [s.strip() for s in STORIES]
