# 03 — Modern architecture: RoPE, RMSNorm, SwiGLU

## Goal
Upgrade the GPT-2-style block to the recipe used by LLaMA/Mistral/Qwen-class models, **one change at a time**, so we can measure each. All three are config switches in `GPTConfig` (`pos`, `norm`, `mlp`); defaults reproduce step 01/02 exactly. Code: `hedy/model.py`, tests: `tests/test_arch.py`.

Ablation discipline: same seed, data, steps and hyper-parameters; change one flag; compare `val_bpb`.

## 1. RoPE — Rotary Position Embedding (`--pos rope`)

### Problem with learned absolute positions
GPT-2 adds a learned vector per position index to the token embedding. It (a) has a hard maximum length (`block_size`), (b) must learn from scratch that position 7 and 8 are "close", and (c) encodes *absolute* position although language mostly cares about *relative* offsets ("the word 3 tokens back").

### Idea
Don't add position to the embedding. Instead **rotate the query and key vectors** by an angle proportional to their position. Split the head dimension into pairs `(x_{2i}, x_{2i+1})`; treat each pair as a 2-D point and rotate it by `θ = pos · ω_i`, with frequencies `ω_i = 10000^(-2i/d)` (fast rotation for early pairs, slow for later ones — like digits of different significance).

Key property: a rotation by angle `a` followed by the transpose of a rotation by `b` is a rotation by `a-b`. So

    ⟨R(m)q, R(n)k⟩ = ⟨q, R(n-m)k⟩   → the attention score depends only on the offset n-m.

`tests/test_arch.py::test_rope_scores_depend_only_on_relative_position` verifies this numerically, and `test_rope_preserves_norm` checks that rotation doesn't change vector length.

### Details
- Applied to **q and k only**, never to v (values carry content, not position).
- No new parameters; no `pos_emb` table (`GPT.pos_emb` is `None`).
- Applied *inside* every attention layer, so position information is re-injected at each depth rather than only at the input.
- Better length extrapolation than learned positions (still limited; real systems extend context with tricks like position interpolation / NTK / YaRN, all of which modify these frequencies). We keep `block_size` as a cap for now.
- Code: `rope_cache` builds `cos/sin` tables `(T, d/2)`; `apply_rope` computes `(x1·cos − x2·sin, x1·sin + x2·cos)` on even/odd channel pairs. (LLaMA's reference code pairs channel `i` with `i+d/2` instead; it's an equivalent re-ordering of channels.)

## 2. RMSNorm (`--norm rmsnorm`)

LayerNorm: `(x − mean) / std · g + b`. RMSNorm: `x / rms(x) · g`, with `rms = √(mean(x²) + ε)`.

Claim (Zhang & Sennrich 2019): the re-centering (mean subtraction) contributes little; the **re-scaling** is what stabilizes training. Dropping it removes a reduction and the bias parameter: slightly cheaper, same quality, and now standard in LLaMA-family models. Our implementation is 3 lines; `test_rmsnorm_unit_rms` checks output has RMS ≈ 1 regardless of input scale/shift.

Expect a **tiny** effect on loss at our scale — the benefit is mostly speed at large scale. Honest ablations often show "no difference"; that is a valid result.

## 3. SwiGLU (`--mlp swiglu`)

Standard MLP: `proj(GELU(fc(x)))`, hidden width 4C.
Gated MLP (Shazeer 2020): `proj( SiLU(gate(x)) ⊙ fc(x) )`. One linear branch is passed through a nonlinearity and **multiplies** the other elementwise, so the network can learn per-feature, input-dependent gating of information. `SiLU(x) = x·sigmoid(x)` (aka Swish); the "Swi" in SwiGLU.

Cost: three weight matrices instead of two. To compare fairly we shrink the hidden width to `⌈(8/3)·C⌉` (rounded to a multiple of 8) so total parameters stay within ~1% of the GELU MLP (`test_swiglu_param_count_close_to_gelu`). Parameter-matched comparisons matter: otherwise a "win" may just be "more parameters".

Shazeer's paper famously offers no theory: *"We offer no explanation as to why these architectures seem to work; we attribute their success, as all else, to divine benevolence."* It is an empirical finding that gated units consistently beat plain ones.

## 4. Other pieces you will see in LLaMA-style models (not yet implemented)
- **Bias-free linears** — we already do this.
- **GQA / MQA** (grouped/multi-query attention): share k/v heads across several q heads to shrink the KV cache. Planned with KV-cache work.
- **Untied embeddings** at large scale (we tie).

## 5. Experiment
Setup: char tokenizer, 4 layers, 128 dim, 4 heads, block 128, batch 32, 1000 steps, seed 1337, CPU. Metric: val bits-per-byte (lower is better).

| variant | params | val_bpb @500 | val_bpb @1000 | Δ vs baseline | val loss @1000 |
|---|---|---|---|---|---|
| baseline (learned pos, LayerNorm, GELU) | 0.81M | 2.932 | 2.630 | — | 1.823 |
| + RoPE | 0.80M | 2.630 | **2.395** | −0.235 | 1.660 |
| + RMSNorm | 0.81M | 2.953 | 2.655 | +0.025 | 1.841 |
| + SwiGLU | 0.82M | 2.901 | 2.581 | −0.049 | 1.789 |
| all three | 0.80M | 2.487 | **2.313** | −0.317 | 1.604 |

**How to read this honestly**
- **Noise level.** The same baseline config gave val_bpb 2.660 in step 01 (default threads) and 2.630 here (`--threads 6`): same seed, different floating-point summation order, ~0.03 apart. So treat differences below ~0.05 as unresolved with a single seed.
- **RoPE: clear win** (−0.235, 8× the noise). It is also visibly ahead at step 500 (2.63 vs 2.93), i.e. it learns faster, not just ends lower. Plausible reason: the model no longer has to learn a position table from 1M characters of data; relative offsets come for free.
- **RMSNorm: no measurable difference** (+0.025, inside the noise). That's the expected outcome; its advantage is compute savings at scale, not loss.
- **SwiGLU: small gain** (−0.049), at the edge of the noise. Direction matches the literature, but we can't claim it from one seed. Exercise 1 settles it.
- **All three together** is best (2.313), roughly the RoPE gain plus a bit more. The effects look roughly additive here, but we did not run pairwise combinations.
- Caveats: tiny model, 1000 steps (far from converged), one seed, one dataset. Do not extrapolate magnitudes to large models; the *ranking* (RoPE ≫ SwiGLU > RMSNorm≈LayerNorm) is what generally holds in practice.

## 6. Exercises
1. Re-run each ablation with seeds 1, 2, 3. What is the seed-to-seed standard deviation? Is any difference in the table larger than noise? (This is the single most important habit in empirical ML.)
2. Train with RoPE at `block_size 128`, then evaluate/sample with a context of 256. Compare against learned positions (which can't even run). How fast does quality degrade?
3. Replace the RoPE base `10000` by `500` and `1e6`. What changes?
4. Plot RoPE rotation angle vs position for a few channel pairs (`rope_cache`) — see the multi-scale "clock hands".
5. Implement LayerNorm-without-bias and RMSNorm-with-mean-subtraction to isolate which part of LayerNorm matters.
6. Replace SiLU with GELU or ReLU in the gate (GEGLU / ReGLU). Differences?
7. Count FLOPs per token for GELU MLP vs SwiGLU at equal parameters.

## 7. Further reading
- Su et al., *RoFormer: Enhanced Transformer with Rotary Position Embedding* (2021)
- Zhang & Sennrich, *Root Mean Square Layer Normalization* (2019)
- Shazeer, *GLU Variants Improve Transformer* (2020)
- Touvron et al., *LLaMA* (2023) — the combination of all three
- Press et al., *ALiBi* (2021) — another relative-position scheme; Peng et al., *YaRN* (2023) — RoPE context extension

**Next:** 04 — KV-cache for fast generation (and GQA).
