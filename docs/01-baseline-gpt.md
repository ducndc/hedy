# 01 — Baseline GPT (character-level)

## Goal
Build the smallest complete LLM pipeline: **text → tokens → transformer → next-token loss → training → sampling**. Everything later (BPE, RoPE, scaling) is a modification of this loop.

## 1. Theory

### Language modeling
An LLM models `P(x_t | x_1 … x_{t-1})`: the probability of the next token given all previous ones. The probability of a whole text is the product of these. Training = maximize the likelihood of real text, i.e. minimize **cross-entropy**:

    loss = -(1/T) Σ_t log P(x_t | x_<t)

Sanity check: a random model over a vocab of V=65 characters gives `ln 65 ≈ 4.17`. Our first logged loss was 4.18. If your initial loss is far from `ln(V)`, something is wrong.

### Tokenization
Models consume integers, not text. Here each character is a token (vocab ≈ 65). Simple, but sequences are long and the model must learn spelling. Step 02 replaces this with BPE.

### The transformer (decoder-only)
Pipeline for input ids of shape `(B, T)`:

1. **Embeddings**: `tok_emb(ids) + pos_emb(positions)` → `(B, T, C)`. Attention itself is order-blind, so position info is added explicitly.
2. **N × Block**, each with two residual sub-layers:
   - `x = x + Attention(LayerNorm(x))` — tokens exchange information.
   - `x = x + MLP(LayerNorm(x))` — each token is transformed independently (4× expansion, GELU).
3. **Final LayerNorm → linear head** → logits `(B, T, V)`.

**Self-attention** (the core idea): each token makes a query `q`, key `k`, value `v` (`qkv` linear layer). Token i looks at token j with weight `softmax_j(q_i·k_j / √d)`, and its output is the weighted sum of values. **Causal mask**: j ≤ i only, so the model can't peek at the future — this is what lets one forward pass train on all T positions at once. **Multi-head**: C channels are split into `n_head` groups that attend independently, then are concatenated and mixed by `proj`.

**Pre-norm + residuals**: normalizing *before* each sub-layer keeps the residual stream clean, making deep stacks train stably.

## 2. Code walkthrough

| Concept | Where | Notes |
|---|---|---|
| Config | `GPTConfig` in `model.py` | `n_embd` must divide by `n_head`; head dim = `n_embd / n_head` |
| Attention | `CausalSelfAttention` | one fused `qkv` matmul, reshape to `(B, heads, T, d)`, `F.scaled_dot_product_attention(is_causal=True)` does softmax(QKᵀ/√d)V with the mask (and uses flash-attention kernels on GPU) |
| MLP | `MLP` | `C → 4C → C` |
| Weight tying | `self.head.weight = self.tok_emb.weight` | input and output embeddings share parameters: fewer params, usually better for small models |
| Init | `_init`, scaled `proj` | N(0, 0.02); residual output projections are shrunk by `1/√(2·n_layer)` so the residual stream's variance doesn't grow with depth (GPT-2 trick) |
| Loss | end of `GPT.forward` | `cross_entropy` over flattened `(B·T, V)` |
| Batching | `get_batch` | random windows; `y` is `x` shifted by one token — that shift *is* the training signal |
| Optimizer | `train.py` | AdamW, β2=0.95, weight decay 0.1, grad-clip 1.0 |
| LR schedule | `lr_at` | linear warmup (100 steps) then cosine decay to 10% — warmup avoids early instability, decay helps final loss |
| Sampling | `GPT.generate` | loop: forward → last position logits → ÷temperature → top-k filter → softmax → `multinomial` → append |

**Sampling knobs**: temperature <1 sharpens (safer, repetitive), >1 flattens (creative, noisy). Top-k cuts the unlikely tail.

## 3. Experiment (what we ran)
`python -m hedy.train --steps 300` (0.81M params, CPU):

    step 0:   train 4.184  val 4.179   # ≈ ln(65): random
    step 150: train 2.443  val 2.460
    step 300: train 2.261  val 2.282

Output already has the *format* of a play (speaker names, colons, line breaks) but nonsense words. Typical longer runs (2000+ steps) reach val ≈ 1.5–1.7 and produce plausible words.

Always compare **train vs val**: val ≫ train means overfitting (Tiny Shakespeare is only ~1MB, so larger models overfit fast).

## 4. Exercises
1. Run 2000 steps; record val loss. Then change one thing at a time: `--n_layer 2`, `--n_embd 256`, `--dropout 0`, `--lr 1e-3`. Which matters most?
2. Remove the causal mask (`is_causal=False`). Training loss will drop absurdly low — why? What happens when you sample?
3. Delete `pos_emb`. What does the model lose? (Hint: shuffle the context.)
4. Print the attention weights of one head (compute `softmax(QKᵀ/√d)` manually) and look at what tokens attend to.
5. Verify the parameter count by hand: embeddings `V·C + block_size·C`, per block `12·C²` (+ norms).

## 5. Further reading
- Vaswani et al., *Attention Is All You Need* (2017)
- Radford et al., *Language Models are Unsupervised Multitask Learners* (GPT-2, 2019)
- Karpathy, *Let's build GPT from scratch* (video) and `nanoGPT`
- Xiong et al., *On Layer Normalization in the Transformer Architecture* (pre-norm)

**Next:** 02 — byte-level BPE tokenizer.
