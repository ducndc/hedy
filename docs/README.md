# Study notes

One note per step, written to be read in order. Each note has: **goal → theory → code walkthrough → experiment → exercises → further reading**.

| # | Note | Code |
|---|------|------|
| 01 | [Baseline GPT](01-baseline-gpt.md) | `hedy/model.py`, `tokenizer.py`, `data.py`, `train.py`, `sample.py` |
| 02 | [Byte-level BPE tokenizer](02-bpe-tokenizer.md) | `hedy/tokenizer.py`, `tests/test_tokenizer.py` |
| 03 | [Modern architecture: RoPE, RMSNorm, SwiGLU](03-modern-architecture.md) | `hedy/model.py`, `tests/test_arch.py` |
| 04 | [KV-cache and Grouped-Query Attention](04-kv-cache-and-gqa.md) | `hedy/model.py`, `hedy/bench.py`, `tests/test_kvcache.py` |
| 05 | [Training at scale](05-training-at-scale.md) | `hedy/train.py`, `hedy/prepare.py`, `hedy/data.py`, `tests/test_train.py` |
| 06 | [Evaluation](06-evaluation.md) | `hedy/eval.py`, `tests/test_eval.py` |
| 07 | [Post-training: SFT and LoRA](07-post-training-sft-lora.md) | `hedy/sft.py`, `hedy/lora.py`, `tests/test_sft.py`, `tests/test_lora.py` |
| 08 | [Preference optimization: DPO with a verifier](08-preference-optimization-dpo.md) | `hedy/dpo.py`, `tests/test_dpo.py` |
| 09 | [Online RL with a verifiable reward (GRPO/RLOO)](09-online-rl-grpo.md) | `hedy/rl.py`, `tests/test_rl.py` |
| 10 | [Scaling laws](10-scaling-laws.md) | `hedy/scaling.py`, `hedy/scalefit.py`, `hedy/scaleplots.py`, `tests/test_scaling.py`, `docs/data/`, `docs/figures/` |

Plan: [roadmap.md](roadmap.md)
