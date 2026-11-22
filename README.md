# hedy

A small GPT built from scratch in PyTorch, for **studying and researching LLMs**, with a written study note for every step. It runs on a CPU. Everything from the tokenizer to online RL is implemented in this repository (PyTorch only; no training frameworks), tested (96 tests), and measured with explicit baselines, confidence intervals and stated limitations.

## The eleven steps
Read the notes in order: [docs/README.md](docs/README.md) (theory, code walkthrough, experiment, exercises, reading).

| # | topic | code | headline result |
|---|---|---|---|
| 01 | [Baseline GPT](docs/01-baseline-gpt.md) | `model.py`, `train.py` | a char-level transformer, loss starts at ln V as it should |
| 02 | [BPE tokenizer](docs/02-bpe-tokenizer.md) | `tokenizer.py` | 2.39 vs 2.66 bits/byte against characters (one seed) |
| 03 | [RoPE, RMSNorm, SwiGLU](docs/03-modern-architecture.md) | `model.py` | RoPE is the clear win; RMSNorm is neutral |
| 04 | [KV-cache and GQA](docs/04-kv-cache-and-gqa.md) | `model.py`, `bench.py` | 6.7× faster decoding at 480 tokens, bit-identical output |
| 05 | [Training at scale](docs/05-training-at-scale.md) | `train.py`, `prepare.py` | accumulation, mixed precision, compile, exact resume; bf16 was 7× *slower* on this CPU |
| 06 | [Evaluation](docs/06-evaluation.md) | `eval.py` | sliding-window perplexity, baselines, likelihood multiple choice with intervals |
| 07 | [SFT and LoRA](docs/07-post-training-sft-lora.md) | `sft.py`, `lora.py` | format is learned with 1% of the parameters; a skill the base lacks is not |
| 08 | [DPO with a verifier](docs/08-preference-optimization-dpo.md) | `dpo.py` | sharpens pass@1, not pass@4; collapse is visible in chosen log-prob |
| 09 | [Online RL (GRPO/RLOO)](docs/09-online-rl-grpo.md) | `rl.py` | gentlest route to DPO-level accuracy; a hard task gets gamed |
| 10 | [Scaling laws](docs/10-scaling-laws.md) | `scaling.py`, `scalefit.py` | IsoFLOP + parametric fit; predicts held-out runs to ~1% |
| 11 | [Capstone](docs/11-capstone-final-model.md) | all of it | a scaling-law-chosen 2M model, pre-registered prediction, full scorecard |

**The capstone in one paragraph.** The step-10 law chose a 1.97M-parameter model for a 2·10¹⁴-FLOP budget and predicted its validation loss **before training** (1.975); the trained model scored **1.992** (+0.9%), 10× beyond the largest budget the law was fitted on. Pretrain → SFT → online RL then took name-following from 60% to 76% to **85%** (the 0.86M pipeline: 38% → 63% → 77%). The "include this keyword" task stayed at ~2% for both sizes: not a capacity fix, as I had guessed in step 07.

## Setup
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch   # or the default index for a GPU build
pip install -e ".[dev,plots]"
pytest                                                               # ~2-3 minutes on CPU
```

## Quick start
```bash
python -m hedy.train --steps 2000                 # Tiny Shakespeare, character level, saves checkpoints/
python -m hedy.sample --prompt "ROMEO:"
```
To reproduce the final model end to end (about an hour of CPU after the data is prepared) use the commands in [docs/11-capstone-final-model.md](docs/11-capstone-final-model.md#6-reproduce-the-final-model). Each module is a command-line tool; `python -m hedy.<module> --help` lists its options:

| module | what it does |
|---|---|
| `hedy.prepare` | tokenize a corpus once into memory-mapped files (also streams big corpora) |
| `hedy.train` | pretrain: accumulation, mixed precision, `--compile`, resumable checkpoints |
| `hedy.eval` | sliding-window perplexity, n-gram baselines, multiple-choice scoring, diversity |
| `hedy.sft` | build verifiable instruction data; full or LoRA fine-tuning; teacher-forced probes |
| `hedy.dpo` | sample, build verifier-labelled preference pairs, train DPO/RFT, evaluate with pass@k |
| `hedy.rl` | online RL with a verifiable reward (GRPO / RLOO) |
| `hedy.scaling` | run, plan and fit scaling experiments; check a prediction (`loss`) |
| `hedy.bench` | KV-cache speed-up and cache memory |

Plan and what remains: [docs/roadmap.md](docs/roadmap.md).
