"""Tokenize a text corpus once into flat binary files, so training streams tokens from disk.

    python -m hedy.prepare --input data/tinystories-valid.txt --out_dir data/tinystories --vocab_size 1024

Writes: tokenizer.json, train.bin, val.bin (uint16 token ids), meta.json.
The split is by *document* (the tail of the file becomes validation), never inside a story.
"""
import argparse
import json
import os

import numpy as np

from .tokenizer import BPETokenizer

EOT = "<|endoftext|>"


def split_docs(text: str, val_frac: float = 0.05):
    """(train_docs, val_docs). The tail of the file is validation. Shared with hedy.eval so the
    benchmark is built from exactly the documents the model never trained on."""
    docs = [d.strip() for d in text.split(EOT) if d.strip()]
    n_val = max(1, int(len(docs) * val_frac))
    return docs[:-n_val], docs[-n_val:]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--vocab_size", type=int, default=1024)
    p.add_argument("--tok_chars", type=int, default=3_000_000, help="chars used to learn BPE merges")
    p.add_argument("--val_frac", type=float, default=0.05)
    a = p.parse_args()
    assert a.vocab_size <= 65536, "uint16 storage"

    text = open(a.input, encoding="utf-8").read()
    # Our BPE has no special tokens yet (see docs/02), so a document boundary becomes a blank line.
    train_docs, val_docs = split_docs(text, a.val_frac)
    docs = train_docs + val_docs
    print(f"{len(docs)} documents -> {len(train_docs)} train / {len(val_docs)} val")

    sample = "\n\n".join(train_docs)[: a.tok_chars]
    print(f"training BPE (vocab {a.vocab_size}) on {len(sample):,} chars ...")
    tok = BPETokenizer.train(sample, a.vocab_size)

    os.makedirs(a.out_dir, exist_ok=True)
    tok.save(os.path.join(a.out_dir, "tokenizer.json"))
    meta = {"vocab_size": tok.vocab_size}
    for name, d in (("train", train_docs), ("val", val_docs)):
        joined = "\n\n".join(d) + "\n\n"
        ids = np.array(tok.encode(joined), dtype=np.uint16)
        ids.tofile(os.path.join(a.out_dir, f"{name}.bin"))
        meta[f"{name}_tokens"] = int(len(ids))
        meta[f"{name}_bytes"] = len(joined.encode("utf-8"))
        print(f"{name}: {len(ids):,} tokens, {meta[f'{name}_bytes'] / len(ids):.2f} bytes/token")
    json.dump(meta, open(os.path.join(a.out_dir, "meta.json"), "w"))


if __name__ == "__main__":
    main()
