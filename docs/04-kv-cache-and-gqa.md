# 04 — KV-cache and Grouped-Query Attention

## Goal
Make generation fast (KV-cache) and make the cache small (GQA). These two ideas are most of what "LLM inference engineering" starts from. Code: `hedy/model.py` (`KVCache`, `CausalSelfAttention`, `GPT.generate`), `hedy/bench.py`, tests in `tests/test_kvcache.py`.

## 1. The problem: naive generation recomputes everything
Generating token `t+1` needs the model's output at position `t`. Our step-01 `generate` re-ran the **whole** window every step. But because attention is causal, the hidden states of positions `< t` **do not change** when we append a new token. Recomputing them is pure waste: generating N tokens costs ~N²/2 token-forwards instead of N.

## 2. KV-cache

### What to cache
Position `t`'s output depends on its own query `q_t` and the **keys and values of all positions ≤ t**. Past queries are never needed again (each was used once, for its own output). Past keys/values are needed at every future step. So we cache K and V per layer:

    step 0 (prefill): run the whole prompt, store K,V of every layer for every prompt token
    step t (decode):  run ONE token; compute its q,k,v; append k,v to the cache; attend q over the full cached K,V

### Two phases (a key mental model)
| | Prefill | Decode |
|---|---|---|
| input | whole prompt (T tokens) | 1 token |
| bound by | compute (big matmuls) | **memory bandwidth** (each step reads all weights + the whole cache to do very little math) |
| metric | time-to-first-token | tokens/second |

Decode being bandwidth-bound is why shrinking the cache (GQA, quantization, paging) directly raises throughput.

### What changed in the code
- `KVCache` stores `k[layer]`, `v[layer]` of shape `(B, n_kv_head, T_past, head_dim)`; `update` concatenates the new k/v.
- **Positions must continue where the cache ended.** `offset = cache.seq_len`. Learned positions index `offset…offset+T`; **RoPE rotates the new q/k by their absolute positions** (`rope_cache(..., offset=offset)`). Note the cache holds keys *already rotated*, which is correct because RoPE's rotation depends only on the key's own position; this is a nice property of RoPE.
- **Masking** has three cases: empty cache → ordinary causal mask; one new token → no mask needed (it may see everything before it); a multi-token chunk after a non-empty cache → explicit mask `tril(diagonal=offset)` (needed e.g. for chunked prefill or speculative decoding).
- **Window limit.** When the sequence reaches `block_size`, `generate` falls back to the old sliding-window recompute. Reason: cached K/V were computed with a left context that the sliding window would drop, and learned positions can't exceed `block_size`. Real systems handle this with sliding-window attention or longer trained contexts.

### Correctness is testable, and should be tested
A cache is an *optimization*; it must not change results. `tests/test_kvcache.py` checks (for learned/RoPE × MHA/GQA/MQA):
1. logits from prefill + token-by-token decode `==` one full forward pass (atol 1e-4);
2. chunked decode with a non-empty cache (the explicit-mask branch) also matches;
3. greedy generation with and without cache produces **identical tokens**.

Bugs here are silent (text still looks plausible but is subtly wrong), so equivalence tests are the only real safety net. Off-by-one position errors are the classic one.

### Measured speed-up (`python -m hedy.bench`)
Model: 6 layers, 256 dim, 8 heads, RoPE+RMSNorm+SwiGLU, random weights, CPU 4 threads, greedy, 16-token prompt.

| new tokens | no cache | with cache | speed-up | tok/s (cached) |
|---|---|---|---|---|
| 64 | 0.38 s | 0.20 s | 1.9× | 315 |
| 128 | 1.02 s | 0.42 s | 2.4× | 305 |
| 256 | 3.64 s | 0.83 s | 4.4× | 307 |
| 480 | 12.30 s | 1.83 s | **6.7×** | 262 |

Read it as: uncached cost grows ~quadratically with length, cached grows ~linearly, so **the speed-up grows with sequence length** (it would be far larger at 4k+ tokens and on GPU). Cached tok/s still drops slightly at the end: attention over the cache is O(T) per token, and our `torch.cat` copies the whole cache every step (a preallocated buffer would avoid this).

## 3. The cost of the cache: memory
Per sequence:

    KV bytes = 2 (K and V) × n_layer × n_kv_head × head_dim × bytes_per_value × seq_len

It grows linearly with context and batch size, and at serving scale it — not the weights — is often what limits how many users fit on a GPU. Real example, LLaMA-2-70B (80 layers, 64 query heads, head_dim 128, bf16), 4096 tokens: with full multi-head attention the cache would be `2·80·64·128·2·4096 ≈ 10.7 GB` **per sequence**.

## 4. GQA — Grouped-Query Attention (`--n_kv_head`)

### Idea
Multi-head attention (MHA) gives every query head its own K and V head. Observation: query heads need to be diverse, but K/V heads can be **shared**.
- **MHA**: `n_kv_head = n_head`
- **MQA** (multi-query, Shazeer 2019): `n_kv_head = 1` — all query heads share one K/V
- **GQA** (Ainslie et al. 2023): `1 < n_kv_head < n_head` — query heads split into groups; each group shares a K/V head

LLaMA-2-70B uses GQA with 8 KV heads for 64 query heads: cache shrinks **8×** (10.7 GB → 1.3 GB at 4096 tokens), and decode gets faster because there is 8× less to read.

### Implementation
- One fused projection now outputs `[ q: n_head·d | k: n_kv_head·d | v: n_kv_head·d ]` (so the parameter savings are real too: fewer K/V weights).
- The **cache stores only `n_kv_head` heads**; just before the attention call, `repeat_interleave(n_head // n_kv_head)` broadcasts each K/V head to its group of query heads. (Newer PyTorch has `enable_gqa` in SDPA that avoids materializing the copy; a fused kernel never does.)
- `n_head` must be divisible by `n_kv_head`.
- With `n_kv_head == n_head` the model is bit-identical to before: step-0 loss matches the previous run exactly (4.231).

### Memory & parameters (`hedy.bench`, 6 layers, 8 heads, head_dim 32, fp32)
| n_kv_head | type | cache @512 tok | cache @4096 tok | params |
|---|---|---|---|---|
| 8 | MHA | 6.3 MB | 50.3 MB | 4.88M |
| 4 | GQA | 3.1 MB | 25.2 MB | 4.48M |
| 2 | GQA | 1.6 MB | 12.6 MB | 4.29M |
| 1 | MQA | 0.8 MB | 6.3 MB | 4.19M |

## 5. Does sharing K/V hurt quality? (experiment)
Setup as in step 03 "all three": char tokenizer, 4 layers, 128 dim, **4 query heads**, RoPE+RMSNorm+SwiGLU, 1000 steps, seed 1337. Metric: val bits-per-byte.

| n_kv_head | type | params | KV-cache size | val_bpb @500 | val_bpb @1000 | Δ vs MHA | val loss @1000 |
|---|---|---|---|---|---|---|---|
| 4 | MHA | 0.80M | 1× | 2.487 | **2.313** | — | 1.604 |
| 2 | GQA | 0.73M | 0.5× | 2.513 | 2.341 | +0.028 | 1.622 |
| 1 | MQA | 0.70M | 0.25× | 2.517 | 2.359 | +0.046 | 1.635 |

**How to read this**
- The ordering is monotone (more sharing → slightly worse), which matches the literature: GQA is a small quality cost for a large memory saving, MQA costs a bit more than GQA.
- But the gaps (+0.028, +0.046) are the same size as the ~0.03 seed/thread noise we measured in step 03, so with one seed we can only say "the cost is small", not rank GQA vs MHA precisely. Exercise 7 settles it.
- This is not a free lunch to compare at equal parameters: sharing K/V also removes parameters (0.80M → 0.70M, −12%). Part of any loss gap is just that. Papers compensate by widening the MLP or by "uptraining"; we did not.
- Tiny model with only 4 query heads, so there's little redundancy to share. The saving matters at scale (many heads, long context, many concurrent users), which a 1000-step char model can't show. Treat this as a sanity check that GQA trains fine, not as evidence about magnitudes.

## 6. Exercises
1. Print `cache.nbytes()` after generating 200 tokens for `n_kv_head` 4/2/1 and confirm the formula in §3.
2. Replace `torch.cat` with a preallocated `(B, Hk, max_len, d)` buffer and a write index. Re-run `hedy.bench`; how much does cached tok/s improve at 480 tokens?
3. Use `F.scaled_dot_product_attention(..., enable_gqa=True)` instead of `repeat_interleave` (torch ≥ 2.5). Same outputs? Faster?
4. Batched generation with different prompt lengths needs left-padding plus an attention mask. Why does the cache make this harder?
5. Implement **sliding-window attention** (each token sees only the last W tokens) with a rolling cache so generation can go past `block_size`. What happens to quality beyond the trained length?
6. Implement **speculative decoding**: a small model drafts k tokens, the big model verifies all k in one forward using the chunked-with-cache branch (`T>1`, non-empty cache). Prove the output distribution is unchanged.
7. Sweep seeds for the GQA experiment (3 seeds each) and decide whether the gap is real.
8. Compute the cache size for your favorite open model from its config (`num_hidden_layers`, `num_key_value_heads`, `head_dim`).

## 7. Further reading
- Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need* (2019) — MQA
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models* (2023)
- Pope et al., *Efficiently Scaling Transformer Inference* (2022) — prefill/decode, memory-bandwidth analysis
- Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention* (vLLM, 2023)
- Leviathan et al., *Fast Inference from Transformers via Speculative Decoding* (2023)
- DeepSeek-V2 (2024) — MLA, compressing the cache even further than GQA

**Next:** 05 — training at scale (mixed precision, gradient accumulation, `torch.compile`, a bigger real corpus).
