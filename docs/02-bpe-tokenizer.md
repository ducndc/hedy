# 02 — Byte-level BPE tokenizer

## Goal
Replace the character tokenizer with **byte-pair encoding (BPE)** written from scratch, and measure what it changes. Code: `hedy/tokenizer.py` (`BPETokenizer`), tests in `tests/test_tokenizer.py`.

## 1. Theory

### Why not characters or words?
| Unit | Problem |
|---|---|
| Characters/bytes | Sequences are long (attention cost is O(T²)), model must learn spelling, each position carries little information |
| Words | Huge vocab, rare words are unseen (`<unk>`), no sharing between `run`/`running`/`runs` |
| **Subwords (BPE)** | Frequent strings become single tokens, rare strings fall back to smaller pieces. Fixed vocab, no unknown inputs |

### The BPE algorithm
Training (learns an ordered list of merges):
1. Start with every text as a sequence of **bytes** (vocab = 256 byte values).
2. Count all adjacent pairs. Pick the most frequent pair, e.g. `(t, h)`.
3. Create a new token `th` (id 256), replace all occurrences of the pair with it.
4. Repeat until the vocab reaches the target size. Each iteration adds one merge: `vocab_size = 256 + n_merges`.

Encoding new text: split into bytes, then apply the learned merges **in the order they were learned** (earliest first). Decoding: concatenate each token's bytes, decode as UTF-8.

### Why *byte*-level
Starting from bytes (not Unicode characters) means **every possible string is representable** — emoji, Chinese, garbage — with no `<unk>`. The cost: non-English text costs more tokens (a Chinese character is 3 bytes and rarely gets merged in an English-trained vocab).

### Pre-tokenization
Before BPE, text is split by a regex into chunks (`" the"`, `"ing"`, `" "`, `"."`, `"123"` …) and merges happen **only inside a chunk**. Without this, BPE would learn junk tokens like `"e t"` spanning word boundaries or `"dog."` vs `"dog!"` as unrelated tokens. Our pattern is GPT-2's simplified to ASCII classes (so no third-party `regex` dependency). Side effect: a leading space attaches to the word (`" the"` ≠ `"the"`), which is why tokenizers are sensitive to whitespace.

### Trade-offs of vocabulary size
Bigger vocab → shorter sequences (cheaper attention, more text per context window) but a larger embedding/output matrix and rarer tokens that are seen few times and so learned poorly. Real models use 32k–200k; we use 512 because our corpus is only ~1 MB.

### Comparing models with different tokenizers
Loss is *per token*. A BPE token covers ~2 bytes, a char token ~1, so the raw losses are **not comparable** (BPE's loss per token will be higher simply because each token carries more information). The fair metric is **bits per byte**:

    bpb = loss_nats_per_token × tokens / bytes / ln 2

`train.py` now logs `val_bpb`. (Lower is better; a uniform guess over 256 bytes would be 8.)

## 2. Code walkthrough
| Piece | Notes |
|---|---|
| `SPLIT_PATTERN` | regex pre-tokenizer |
| `BPETokenizer.train` | collapses text into `{chunk_bytes: frequency}` first, so each iteration counts pairs over ~25k distinct chunks rather than 1M characters (≈16 s for 256 merges). Ties in pair count are broken by the pair itself so training is deterministic |
| `_merge` | replaces non-overlapping pair occurrences left→right |
| `encode` / `_encode_chunk` | repeatedly merge the pair with the **lowest rank** (earliest learned) until none apply; per-chunk cache gives a big speedup since words repeat |
| `decode` | join byte strings, `errors="replace"` (a model can emit a token sequence that's invalid UTF-8 mid-character) |
| `load_tokenizer` | saved JSON has a `type` field so `sample.py` works with either tokenizer |

Complexity note: this is the simple O(merges × unique-chunks) trainer — fine for MBs of text. Production trainers (HF `tokenizers`, `tiktoken`, SentencePiece) update pair counts incrementally and are written in Rust/C++ to handle 100s of GB.

**Not implemented (on purpose, for later):** special tokens (`<|endoftext|>`), which are needed once we train on many documents or do chat/SFT.

## 3. Experiment
`python -m hedy.train --tokenizer {char,bpe} --steps 1000` (same model: 4 layers, 128 dim, block 128, CPU):

| tokenizer | vocab | params | bytes/token | step 0 val_bpb | step 500 val_bpb | step 1000 val_bpb | step 1000 val loss (per token) |
|---|---|---|---|---|---|---|---|
| char | 65 | 0.81M | 1.00 | 6.03 | 2.96 | 2.66 | 1.844 |
| BPE | 512 | 0.87M | 1.94 | 4.75 | 2.59 | **2.39** | 3.145 |

Notice the per-token val loss is *higher* for BPE (3.145 vs 1.844) even though it is the better model by bits-per-byte (2.39 vs 2.66). This is exactly why raw losses can't be compared across tokenizers.

Things to notice:
- BPE compresses the corpus (`bytes/token` ≈ 1.94 vs 1.0), so with the same `block_size=128` the model sees ~2× more text per window.
- Compare **val_bpb**, not val loss.
- Caveats: one seed, 1000 steps, and BPE sees ~2× more text per step (same batch/block size in tokens), so part of its edge is just more data per step. A rigorous comparison would match bytes seen and average over seeds (exercise for you).
- Also note the BPE train loss (2.83) vs val (3.15) gap: with only ~575k tokens and a 512-way softmax, the model starts to overfit sooner than the char model.

## 4. Exercises
1. Run `BPETokenizer.train(text, 300, verbose=True)` and read the first 30 merges. What does BPE learn first? (spaces + common letters, then `" the"`, `"ing"` …)
2. Plot `bytes/token` vs `vocab_size` (256, 400, 512, 1000, 2000). Where do the returns diminish?
3. Tokenize `"hello"`, `" hello"`, `"Hello"`, `"hello!"`. See how whitespace and case change tokens.
4. Tokenize some Chinese/Vietnamese text. How many tokens per character? Why?
5. Remove the pre-tokenizer (`SPLIT_PATTERN` → whole text as one chunk) and inspect the vocab: find tokens that span words.
6. Implement incremental pair counting to make `train` faster; verify it gives identical merges (the determinism test helps).
7. Compare with `tiktoken`'s `gpt2` encoding on the same text.

## 5. Further reading
- Sennrich et al., *Neural Machine Translation of Rare Words with Subword Units* (2016) — BPE for NLP
- GPT-2 paper, section 2.2 (byte-level BPE)
- Karpathy, *Let's build the GPT Tokenizer* (video) and `minbpe`
- Kudo & Richardson, *SentencePiece* (2018); Kudo, *Unigram LM* (2018) — the main alternative to BPE
- Petrov et al., *Language Model Tokenizers Introduce Unfairness Between Languages* (2023)

**Next:** 03 — architecture upgrades (RoPE, RMSNorm, SwiGLU), one ablation at a time.
