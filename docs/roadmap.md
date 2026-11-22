# Roadmap

1. **Baseline** (done, see [01](01-baseline-gpt.md)): char-level GPT, train on Tiny Shakespeare, sample.
2. **Tokenizer** (done, see [02](02-bpe-tokenizer.md)): byte-level BPE from scratch; compare loss/bits-per-byte vs char-level.
3. **Architecture upgrades** (done for RoPE/RMSNorm/SwiGLU, see [03](03-modern-architecture.md)); KV-cache + GQA done in [04](04-kv-cache-and-gqa.md).
4. **Training at scale**: bf16/AMP, torch.compile, gradient accumulation, DDP, a real corpus (FineWeb-Edu sample).
5. **Evaluation**: perplexity, HellaSwag/ARC via lm-evaluation-harness.
6. **Post-training**: SFT, then DPO / LoRA on an open base model.
7. **Research**: scaling-law sweeps, optimizer comparisons (AdamW vs Muon), data-quality experiments.

Rule: change one thing at a time, log config + seed + val loss for every run.
