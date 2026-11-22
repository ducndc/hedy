# 05 — Training at scale: accumulation, mixed precision, compile, real data, resume

## Goal
Turn the toy training loop into one that has the *mechanisms* of a real pretraining run: big effective batches on small memory, low-precision math, a compiled model, a disk-backed token stream, and checkpoints you can resume from bit-exactly. Code: `hedy/train.py` (rewritten), `hedy/prepare.py`, `hedy/data.py`; tests: `tests/test_train.py`.

> **Honest scope note.** This machine has no GPU. Everything here runs and is tested on CPU, but some paths only matter (or only run) on CUDA: the fp16 `GradScaler`, fused AdamW, and multi-GPU. They are written but **not exercised** here; the note marks them. And the CPU has no bf16 hardware, so the speed numbers below show what *not* to expect on a GPU.

## 1. What "scale" changes
Three quantities govern a run:
- **Tokens per optimizer step** = `micro_batch × grad_accum × block_size` (batch size is measured in *tokens*, not sequences). Real runs use 0.5M–4M tokens/step; ours is 16k.
- **Throughput** = tokens/second. Training time = total tokens ÷ throughput.
- **Total compute** ≈ `6 · N · D` FLOPs for `N` parameters and `D` tokens (2 for the forward matmuls, 4 for backward). Chinchilla (2022) found compute-optimal training uses about **20 tokens per parameter**. Our TinyStories run: ~0.9M params × ~6M-token corpus ≈ 7 tokens/param per epoch.

Everything below is either *fitting a bigger batch in memory*, *doing the same math faster*, or *not losing work when the job dies*.

## 2. Gradient accumulation
**Problem:** the batch you want doesn't fit in memory (activation memory scales with batch × sequence).
**Trick:** run `k` micro-batches, let `.backward()` **add** gradients into `.grad`, step the optimizer once.

Why it is exact: the loss over a big batch is the mean over examples, so its gradient is the mean of micro-batch gradients. Dividing each micro-loss by `k` before `backward()` makes the *sum* equal that mean (`accumulate` in `train.py`). `tests/test_train.py::test_grad_accumulation_equals_big_batch` checks that 1 batch of 8 and 4 micro-batches of 2 give the same loss and gradients to 1e-6.

Caveats worth knowing:
- Exact **only if** nothing couples examples within a batch. Transformers with LayerNorm/RMSNorm are fine (BatchNorm would not be). Dropout draws different masks per micro-batch (harmless; it's random anyway).
- With padded variable-length batches you must weight by *token count*, not by micro-batch; we have fixed-length windows so equal weights are correct.
- It costs nothing in math but trades memory for **time**: k sequential passes instead of one big parallel one.
- **How big should the batch be?** Too small: noisy gradients, poor hardware use. Beyond the "critical batch size" extra batch gives little gain per token. When you scale the batch, you usually raise the LR (roughly ∝ √ or linearly) and lengthen warmup. We keep `lr` explicit so you can experiment (exercise 2).

## 3. Mixed precision
### Number formats
| format | bits (sign/exp/mantissa) | range | precision | notes |
|---|---|---|---|---|
| fp32 | 1/8/23 | ±3e38 | ~7 digits | master copy |
| fp16 | 1/5/10 | ±65504 | ~3 digits | small range → gradients underflow → needs loss scaling |
| **bf16** | 1/8/7 | ±3e38 | ~2 digits | **same range as fp32**, coarser; no scaling needed → the modern default |

### What autocast does
`torch.autocast` runs **matmuls/linear layers in low precision** (where >90% of FLOPs are, and where tensor cores give 2–16× speed) while keeping **numerically sensitive ops in fp32** (softmax, normalization, loss reduction, reductions generally). Parameters and the AdamW state stay **fp32 master weights**: a bf16 weight update of `lr·grad ≈ 1e-5` on a weight of `0.02` would be rounded away (bf16 has ~3 significant digits), so training would stall.

### Loss scaling (fp16 only)
fp16 gradients below ~6e-8 flush to zero. `GradScaler` multiplies the loss by a large factor (dynamically adjusted; backs off on inf/NaN), then divides gradients back (`unscale_`) *before* clipping so the clip threshold is in true units. In `run`: `scaler.scale(loss).backward()` → `scaler.unscale_(opt)` → clip → `scaler.step(opt)`. **Written but not exercised here (CUDA-only).**

### Memory accounting (why big runs need sharding)
Per parameter with AdamW + fp32 master weights: weights 4 B + grads 4 B + two Adam moments 8 B = **16 B/param**, *before* activations. A 7B model is ~112 GB of state; this is what ZeRO/FSDP shard across GPUs, and why people use 8-bit optimizers and bf16 params with fp32 accumulation tricks.

## 4. `torch.compile`
Eager PyTorch launches one kernel per op and writes every intermediate to memory. Many transformer ops (norms, residual adds, SiLU×mul, RoPE) are **memory-bound**: they spend time moving data, not computing. `torch.compile` traces the model, **fuses** such ops into single kernels (so intermediates stay in registers/cache), and generates code (needs a C++ compiler on CPU, Triton on GPU). Costs: seconds-to-minutes of compile time on the first steps (our logs show it), and recompilation if shapes change. We compile `model` for the forward/backward but keep `raw` for state-dict/clipping: a compiled module's keys are prefixed `_orig_mod.`, which would make checkpoints incompatible with the uncompiled model.

## 5. Optimizer & schedule details
- **Weight decay only on matrices** (`ndim ≥ 2`): decaying norm gains or biases isn't regularization. (`make_optimizer`, tested.) Note the tied embedding is a matrix and is decayed, as in GPT-2/nanoGPT.
- **AdamW β=(0.9, 0.95)**: β2=0.95 (not 0.999) reacts faster to loss-scale changes; standard for LLM pretraining.
- **Gradient clipping at 1.0** guards against loss spikes; we log the pre-clip `gnorm`, a very useful health signal (sudden jumps precede divergence).
- **Warmup + cosine to 10%**: now `--warmup` is a flag.
- `fused=True` AdamW (one kernel for all params) is enabled on CUDA only.

## 6. Data pipeline for a real corpus
`hedy/prepare.py` tokenizes **once** and writes flat `uint16` arrays (`train.bin`, `val.bin`). Training memory-maps them (`np.memmap`): the OS pages data in on demand, so a 100 GB corpus costs no RAM, and `get_batch` just slices random windows. Details:
- `uint16` holds ids < 65,536: fine for vocabs up to 64k; larger needs `uint32`.
- **Split by document**, not by position inside a document, and validation is held out *before* tokenizer training, so the tokenizer never sees it (no leakage).
- Our BPE has no special tokens yet, so `<|endoftext|>` becomes a blank line. Adding a real `<|endoftext|>` token is a natural follow-up (exercise 5).
- Random windows sample *with replacement*; "epochs" lose meaning at LLM scale. Report **tokens seen**.
- Corpus quality dominates at scale: deduplication and filtering (FineWeb-Edu, DCLM) move results more than most architecture tweaks. TinyStories is synthetic, simple, and deliberately small: great for studying, not representative.

## 7. Checkpointing and exact resume
A real run will be interrupted. `ckpt.pt` stores **model + optimizer state + step** (Adam moments matter: resuming without them silently changes training), written atomically (`.tmp` then `os.replace`) so a crash mid-write can't corrupt the last good checkpoint. Exactness trick: every step starts with `torch.manual_seed(seed·1000003 + step)`, so the data and dropout at step *t* are a function of *t*, not of how many steps ran in this process. `test_resume_reproduces_uninterrupted_run` trains 6 steps straight vs 3 + resume + 3 and asserts identical weights. (This test caught a real bug while I wrote it: I had dropped the initial seed, so two runs started from different weights.)

## 8. Experiments

### 8.1 Throughput on this CPU
Model: 4 layers, 128 dim, RoPE+RMSNorm+SwiGLU, vocab 1024, micro-batch 32 × 128 tokens, 12 CPU threads, steady-state window (steps 41–60).

| config | tok/s (steady) | vs fp32 | notes |
|---|---|---|---|
| fp32 (eager) | ~9,600 | 1.00× | baseline |
| bf16 autocast | ~1,280 | **0.13×** | 7.5× *slower* |
| fp32 + `torch.compile` | ~11,200–12,300 | ~1.2–1.3× | first window 2,700 tok/s includes compile time |
| bf16 + compile | ~1,230 | 0.13× | compile can't fix emulated bf16 |

Reading it:
- **bf16 is much slower here, and that is the correct, expected result for this hardware.** The Ryzen 5 5625U has no bf16 instructions (no AVX512-BF16/AMX), so autocast converts to bf16 and back around every matmul and emulates the arithmetic. On an A100/H100 (or a CPU with AMX) the same flag gives a large *speed-up* and ~2× less activation memory. Never assume a feature is a win; measure on the target hardware.
- **Numerics were fine.** The loss at steps 20/40/60 is 6.305/5.646/4.900 (fp32) vs 6.306/5.639/4.899 (bf16): indistinguishable over this window. That's what bf16's fp32-sized exponent buys.
- **`torch.compile` gives a modest ~20–30% speed-up** on CPU by fusing the memory-bound ops, after a one-time compile cost visible in the first 20 steps.
- Single run per config, 60 steps. I re-ran fp32 eager and fp32+compile later on the idle machine and got 10.2k and 12.9k tok/s: close to the table (the load average on this box is ~5, so absolute numbers drift ~10%; ratios held).
- **Dropout is a hidden speed tax.** The runs above use the default `--dropout 0.1`. Re-running fp32 eager with `--dropout 0.0` gave **15.4k tok/s (+50%)**. Dropout generates a random mask for every activation it touches and forces attention onto a slower kernel path. Modern pretraining runs use dropout 0 (with a single pass over huge data there is little to regularize), which is why the TinyStories run below is faster than this table.

### 8.2 A real (small) corpus: TinyStories
Setup: BPE vocab 1024 trained on 3M chars of the train split; 5.91M training tokens (3.08 bytes/token), 312k validation tokens (held-out stories). Model: 4 layers, 128 dim, 4 query / 2 KV heads (GQA), RoPE+RMSNorm+SwiGLU, **0.86M parameters**, dropout 0, fp32 + `torch.compile`. Batch 32 × 4 accumulation × 128 tokens = **16,384 tokens/step**, 1000 steps = **16.4M tokens** (≈2.8 passes over the data, ≈19 tokens/param: right at the Chinchilla ratio). AdamW lr 3e-3, warmup 100, cosine to 10%. Throughput ~20,600 tok/s (≈14 min of training).

| step | train loss | val loss | val bits/byte |
|---|---|---|---|
| 0 | 6.948 | 6.946 | 3.265 |
| 200 | 2.724 | 2.834 | 1.332 |
| 400 | 2.387 | 2.502 | 1.176 |
| 600 | 2.209 | 2.339 | 1.099 |
| 800 | 2.113 | 2.241 | 1.053 |
| 1000 (final) | – | 2.214 | **1.041** |

Step-0 sanity check: `ln(1024) = 6.93`, observed 6.95.

Sample (`python -m hedy.sample --ckpt <out> --prompt "Once upon a time"`, temperature 0.8, top-k 40):

> Once upon a time, there was a little fish named Alice. Alice loved to explore the world and look for new things. One day, she was running up the pit. She walked up and found a diamond. It was a basket of herbs! She picked it up and said, "Let's go!"
> When she opened the pencil, she heard a big, wide tree shake. She ran closer, but the tree was not as lily. She was so sad. [...]

Notice what 0.86M parameters *can* and *can't* do: grammar, dialogue quotes, sentence rhythm, character names and a "story shape" are all there (compare to the nonsense words of step 01), but **meaning is not**: a fish "running up the pit", opening "the pencil". Local syntax is cheap to learn; coherence over many sentences needs far more capacity. This is the TinyStories paper's central observation, reproduced at a scale that trains on a laptop CPU in 14 minutes.

Note the comparison with step 01: val_bpb 1.04 on TinyStories is **not comparable** to Shakespeare's 2.3–2.6, because the *text* is different (simple, repetitive synthetic children's stories are far more predictable). bits-per-byte is comparable across tokenizers, not across datasets.

## 9. What I could not test here (needs GPU / multiple devices)
- fp16 + `GradScaler`, fused AdamW, tensor-core speed-ups, Flash-attention kernels
- **DDP** (replicate model, shard batch, all-reduce gradients) and **FSDP/ZeRO** (shard params/grads/optimizer state). With accumulation under DDP you skip gradient all-reduce on non-final micro-batches (`no_sync`).
- Measuring **MFU** (model FLOPs utilization = achieved FLOPs ÷ hardware peak; 40–60% is excellent on GPUs). The formula is easy; the number is meaningless on this CPU.

## 10. Exercises
1. Run the same config with `--batch_size 32 --grad_accum 1` and `--batch_size 8 --grad_accum 4`. Do the loss curves agree (they should, up to dropout/sampling)? Compare tok/s and peak memory (`/usr/bin/time -v`).
2. Sweep `--lr` ∈ {1e-3, 3e-3, 1e-2, 3e-2} at fixed batch; then double the batch (via accumulation) and re-sweep. Does the best LR move?
3. Turn off the weight-decay split (decay everything) and compare val_bpb over 3 seeds.
4. Kill a run with Ctrl-C mid-way, `--resume`, and diff its log against an uninterrupted run.
5. Add a real `<|endoftext|>` special token to the BPE (id = vocab_size), insert it between documents in `prepare.py`, and make `generate` stop on it.
6. Switch `uint16` memmap reads from per-sample Python slicing to a vectorized gather; measure data-loading share of step time with a profiler.
7. Compute FLOPs/token as `6N + 12·L·C·T` (attention term) and compare with measured tok/s to get a CPU "utilization" number; compare with your CPU's peak (cores × clock × FLOPs/cycle).
8. Estimate memory for AdamW training of a 1B-parameter model (state + activations at batch 1M tokens), and what ZeRO-3 across 8 GPUs would leave per GPU.

## 11. Further reading
- Micikevicius et al., *Mixed Precision Training* (2017); Kalamkar et al., *A Study of BFLOAT16 for Deep Learning Training* (2019)
- Hoffmann et al., *Training Compute-Optimal LLMs* (Chinchilla, 2022); Kaplan et al., *Scaling Laws for Neural Language Models* (2020)
- McCandlish et al., *An Empirical Model of Large-Batch Training* (critical batch size, 2018)
- Rajbhandari et al., *ZeRO* (2019); PyTorch FSDP docs
- Penedo et al., *The FineWeb Datasets* (2024) — data quality at scale
- Eldan & Li, *TinyStories* (2023)
- Karpathy `nanoGPT` / `llm.c` — compact, readable references for everything in this step

**Next:** 06 — evaluation (perplexity properly, then downstream benchmarks with the lm-evaluation-harness).
