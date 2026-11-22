# Roadmap

1. **Baseline** (done, see [01](01-baseline-gpt.md)): char-level GPT, train on Tiny Shakespeare, sample.
2. **Tokenizer** (done, see [02](02-bpe-tokenizer.md)): byte-level BPE from scratch; compare loss/bits-per-byte vs char-level.
3. **Architecture upgrades** (done for RoPE/RMSNorm/SwiGLU, see [03](03-modern-architecture.md)); KV-cache + GQA done in [04](04-kv-cache-and-gqa.md).
4. **Training at scale** (done except DDP/GPU-only paths, see [05](05-training-at-scale.md)): bf16/AMP, torch.compile, gradient accumulation, resumable checkpoints, a real corpus (TinyStories).
5. **Evaluation** (done, see [06](06-evaluation.md)): sliding-window perplexity, n-gram baselines, likelihood MCQ with confidence intervals, generation diagnostics. (Real HellaSwag/ARC need a bigger model; adapter left as an exercise.)
6. **Post-training**: SFT + LoRA done (see [07](07-post-training-sft-lora.md)); DPO with a verifier done in [08](08-preference-optimization-dpo.md); online verifier-reward RL (GRPO/RLOO) next.
7. **Research**: scaling-law sweeps, optimizer comparisons (AdamW vs Muon), data-quality experiments.

Rule: change one thing at a time, log config + seed + val loss for every run.
