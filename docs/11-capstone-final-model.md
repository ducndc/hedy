# 11 — Capstone: one model, the whole recipe, an honest scorecard

## Goal
Steps 01–10 each isolated one idea. This step combines them into **one final model** and asks three questions:

1. **Does the recipe work end to end?** Pretrain → SFT → online RL, each stage measured with the same tools.
2. **Did the scaling law earn its keep?** In step 10 I fitted `L(N, D)` on budgets up to 2e13 FLOPs and could not test it far outside that range. Here the law *chooses the model size* for a 10× larger budget (2e14 FLOPs) and **I wrote down its predicted loss before training** (`docs/data/final_prediction.json`).
3. **Does a better base model change what post-training can do?** Steps 07 and 09 found that the 0.86M-parameter model could not learn the "include this keyword" task whatever the method, and I attributed it to capacity without testing. A 2.3× larger, better-trained base is the test.

> Scope: still toy scale (a ~2M-parameter model on a 17M-token budget, trained on a 6-core CPU). The point is the *method and the honesty of the accounting*, not the model.

## 1. The recipe, and the evidence for each choice

| choice | what the final model uses | evidence |
|---|---|---|
| Tokenizer | byte-level BPE, vocab 1,024 (same as steps 05–10) | step 02: better bits/byte than characters, no `<unk>` |
| Position | **RoPE** | step 03: the one clear architectural win (−0.235 bits/byte vs learned) |
| MLP / norm | **SwiGLU**, **RMSNorm** | step 03: small gain / no loss difference, RMSNorm is cheaper |
| Attention | plain multi-head (no GQA) | step 04: GQA saves KV-cache memory at a small quality cost; at context 128 the cache is irrelevant, so I keep the quality |
| Optimizer | AdamW (β=0.9/0.95), weight decay 0.1 on matrices only, clip 1.0, 100-step warmup + cosine to 10% | step 05 |
| Regularization | **dropout 0** | step 05: dropout cost ~50% throughput and a single pass over fresh data needs no regularizer |
| Precision / speed | fp32, `torch.compile` | step 05: bf16 was 7× *slower* on this CPU; compile was ~10–25% faster (measured again here: 6.9k vs 6.2k tok/s) |
| Data | 67.4M-token TinyStories slice, no repeated epochs | step 10 |
| Size and tokens | **chosen by the scaling law** (below) | step 10 |
| Learning rate | 1.2e-3 | step 10: best LR falls with width (≈1.4–1.5e-3 at d=192/256 in short runs); I used a slightly lower value because this run is ~10× longer than the sweeps. **Not tuned at this length** (a caveat for the prediction test). |
| Post-training | full-parameter SFT with response-only loss (lr 3e-4, 3 epochs, 6,000 examples) → online RL (GRPO, verifier reward, KL 0.02) | steps 07, 09 |
| Evaluation | sliding-window bits/byte, n-gram baselines, likelihood multiple choice with CIs, verifier-based instruction following with pass@k | steps 06, 07, 08 |

**Deliberately left out**, and why: *GQA* (no benefit at this context length), *LoRA* (its advantage is operational and negligible at 2M parameters; step 07 found no quality edge), *DPO* (step 08: no better than online RL here and needs a tuning/monitoring burden; I compare against it only by reference).

## 2. How the model size was chosen, and the pre-registered prediction

The step-10 parametric fit is `L(N, D) = 1.363 + 80.5 / N^0.371 + 8772 / D^0.631`. For a compute budget `C = 6·N·D = 2·10^14` FLOPs (10× the largest budget in the fit, a roughly 40-minute run on this machine), it says:

| candidate (d, layers) | N | tokens D | predicted loss |
|---|---|---|---|
| 160, 4 | 1.40M | 23.7M | 1.978 |
| **192, 4** | **1.97M** | **16.9M** | **1.975** |
| 224, 4 | 2.65M | 12.6M | 1.986 |
| 192, 5 | 2.41M | 13.8M | 1.981 |
| 256, 5 | 4.22M | 7.9M | 2.032 |

(The continuous optimum is N* ≈ 1.8M, D* ≈ 19M, about **10 tokens per parameter** at this budget; the fit's exponent above ½ means the ratio keeps falling as compute grows.) The loss surface is flat near the optimum: every candidate between 1.4M and 2.65M parameters is predicted within 0.011 nats, which is *below* the seed noise of 0.02–0.04. So d=192, 4 layers is not a sharp choice, only a good one. I picked it for its position at the fitted optimum.

**Pre-registered prediction** (written to `docs/data/final_prediction.json` before the model was trained): fixed-window validation loss **1.975** nats/token. The 95% interval from a 300-resample bootstrap of the fit alone is **1.895–2.038**; adding the measured seed noise (σ ≈ 0.03) widens it to **1.883–2.067**. The check uses `python -m hedy.scaling loss`, the same 2,400 fixed validation windows as every step-10 run.

What would count as success or failure: inside the 1.88–2.07 interval means "consistent with the law at this noise level"; a miss by more than ~0.1 nats would mean the law (or my untuned learning rate) does not extrapolate 10× in compute.

## 3. Results

### 3.1 The prediction test: did the law extrapolate 10×?

The final model (1,967,808 parameters, 4,136 steps × 4,096 tokens = 16.94M tokens, about 43 minutes at ~6.7k tokens/s with `torch.compile`) finished with:

| quantity | value |
|---|---|
| **pre-registered prediction** | **1.975** nats/token (fit-only 95% interval 1.895–2.038; with seed noise 1.883–2.067) |
| **measured** (fixed windows, `hedy.scaling loss`) | **1.992** nats/token = 0.936 bits/byte |
| error | **+0.017 nats (+0.86%)** |
| inside the fit-only interval? | yes |
| inside the interval including seed noise? | yes |

(The training log's own random-window estimate at the end was 1.996; train loss at step 4000 was 1.912, a 0.08 train–validation gap.)

**Verdict: consistent with the law, at the noise level.** A fit on 22 runs up to 2e13 FLOPs predicted the loss of a run with 10× more compute to within 0.9%, and the error is about the size of the seed-to-seed noise (0.02–0.04). Caveats that matter:
- It is **one run**, so the error has an unknown sign-and-size distribution; +0.017 is a single draw.
- The error is positive (the model is slightly *worse* than predicted). That is the direction I would expect from a learning rate I did not tune at this length (1.2e-3, chosen by interpolation from short sweeps), but I did not test it, so I can't separate "mild mistuning" from "noise" from "the law bends a little".
- The law was fitted on the same data, tokenizer, architecture family and optimizer recipe; it tells you nothing about changing those.
- The prediction was *also* made for a size at the flat bottom of the loss surface (every candidate from 1.4M to 2.65M parameters predicted within 0.011 nats), so it does not show that the law picks the best size, only that it predicts the loss of a good one.

### 3.2 The pretrained model, scored with the step-06 toolkit

| metric | step-05 model (0.86M, ~16M tokens) | **final model (1.97M, 17M tokens)** |
|---|---|---|
| validation loss, chunked | 2.200 (1.034 bits/byte) | **1.992 (0.937)** |
| validation loss, sliding window (stride 64) | 2.128 (**1.000** bits/byte) | **1.897 (0.892)** |
| unigram / bigram baseline (bits/byte) | 2.725 / 1.633 | same baselines; model is 45% below the bigram |
| next-sentence multiple choice, `acc` | 42.6% [38.3, 47.0] | **49.6%** [45.2, 54.0] |
| next-sentence multiple choice, `acc_norm` | 51.6% [47.2, 56.0] | **61.2%** [56.9, 65.4] |
| loss at position 0 / 7 / 15 in the window (nats) | 3.83 / 2.38 / 2.16 | 3.76 / 2.27 / 1.99 |
| distinct-2 at T=0.8 (real text: 0.883) | 0.930 | **0.903** |

- The bits/byte gain is **0.108 (−10.8%)**; the chunked-versus-sliding gap is now about 5% of the loss (it was 3.4% for the 0.86M model), so honest evaluation matters a little more as the model gets better at using context.
- Both models are evaluated on the *same* 500 multiple-choice items, so the 9–10 point gains are paired (I did not compute a paired test; the unpaired intervals are barely disjoint).
- **Diversity at the same temperature is now closer to real text**: the temperature at which distinct-2 matches real stories moved from ≈0.65–0.7 (0.86M) to ≈0.75 (final), interpolating the table values, meaning the larger model can be sampled hotter before it starts producing words real stories wouldn't.
- Sample (T=0.8, top-k 40, prompt "Once upon a time"): *"Once upon a time there was a little girl named Lily. She loved to play with her toys and draw pictures. One day, Lily's mom took her to a big book with her crayons and a picture of a crayon…"* Fluent and story-shaped; the semantics still wobble ("took her to a big book").

### 3.3 The whole pipeline, both model sizes

Each stage is evaluated with the step-08/09 protocol (150 held-out name prompts, 150 held-out keyword prompts, K=4 samples, T=0.7/top-k 40, bootstrap intervals over prompts). *mentions/pass* = how often the name appears in responses that contain it (real stories: 2.5); *bpb(60k)* = bits/byte on plain stories (first 60k validation tokens), so its growth is forgetting.

| model and stage | name pass@1 [95% CI] | name pass@4 | keyword pass@1 [95% CI] | keyword pass@4 | finished | mentions / pass | bpb (60k) | SFT val loss |
|---|---|---|---|---|---|---|---|---|
| 0.86M pretrained (step 05) | 37.8% [32, 43] | 69.3% | 1.2% [0.3, 2.2] | 4.0% | 0% | 2.28 | 1.058 | 2.439 |
| 0.86M + SFT (step 07) | 63.0% [57, 69] | 82.7% | 0.7% [0.2, 1.3] | 2.7% | 100% | 2.71 | 1.190 | 1.616 |
| 0.86M + SFT + RL, names only, 50 steps (step 09) | 77.0% [71, 83] | 86.7% | 0.3% [0.0, 0.8] | 1.3% | 100% | 3.63 | 1.195 | 1.634 |
| **1.97M pretrained** | 60.3% [55, 66] | 87.3% | 1.5% [0.7, 2.5] | 6.0% | 0% | 2.27 | 0.946 | 2.341 |
| **1.97M + SFT** | 75.5% [70, 81] | 89.3% | 1.8% [0.8, 3.0] | 7.3% | 99% | 2.61 | 1.081 | 1.465 |
| **1.97M + SFT + RL** (names + keywords, 40 steps) | **85.0%** [80, 90] | **91.3%** | 2.0% [1.0, 3.2] | 8.0% | 100% | 3.39 | 1.090 | 1.476 |

(SFT = full fine-tuning, lr 3e-4, 3 epochs, 6,000 examples, **identical settings to step 07**. RL = GRPO, lr 2e-5, KL 0.02, 32 prompts × 8 samples per step, **mixed name+keyword prompts**.)

### 3.4 What the scorecard says

**1. The recipe composes.** Every stage improves the model it is applied to, in both sizes, and in the same pattern: pretraining → SFT adds instruction-following and the stop marker (0% → 99–100% finished); RL adds a further nine-to-fourteen points of name-following with almost no collateral damage (forgetting +0.005 to +0.009 bits/byte; SFT validation loss unchanged).

**2. A better base lifts every stage, and the gap narrows with post-training.** The 1.97M model leads at every stage: +22.5 points of name pass@1 pretrained, +12.5 after SFT, +8.0 after RL (85.0% vs 77.0%, intervals 80–90 vs 71–83, a gap that is suggestive and not conclusive from one run each). It also copies the name from the prompt 60% of the time *with no SFT at all*, versus 38% for the small model. So post-training gains are larger when the base is weaker (SFT: +25 points for 0.86M, +15 for 1.97M): post-training elicits what the base can already do, and a stronger base has less left to elicit.

**3. It still sharpens rather than expands.** For the final model, name pass@4 only moves 87.3% → 89.3% → 91.3% across the stages while pass@1 climbs 60.3% → 75.5% → 85.0%: the model becomes reliable at what it could already sometimes do (pass@1 after RL, 85%, is close to the SFT model's pass@4, 89%). Same story as steps 08 and 09, now at two sizes.

**4. Keywords: the capacity hypothesis from steps 07 and 09 is not confirmed.** In step 07 I attributed the zero keyword-following to the model being too small. Here the base is 2.3× larger and has 2.4× the pretraining compute. Keyword pass@1 is **1.8% [0.8, 3.0]** after SFT, versus 0.7% [0.2, 1.3] before: overlapping intervals, a movement of at most a point or two. pass@4 rose from 2.7% to 7.3% (4 prompts → 11 of 150 had any success), and the teacher-forced gain from 0.15 to 0.23 nats; both are in the direction of "more capacity helps", but both are tiny and within what a different seed could produce. **After RL the picture gets worse, not better**: all 8 keyword successes (out of 600 samples) are for the words *explore, years, three, forest*, the same template-word coincidences as in step 09, whereas the SFT model's 9 successes were spread over eight different keys (explore, years, daddy, garden, world, yellow, popular, dolls). So the (few) real keyword successes the SFT model had did not get amplified by RL; the coincidental ones did. Two clear design lessons: (a) half of my RL prompts were keyword prompts, which have almost no signal, so ~50% of the sampling budget was wasted (the fraction of groups with any signal fell from 41% to 26% over 40 steps); (b) a binary reward on a skill the model can't yet do gives RL nothing real to amplify. What would test the capacity hypothesis: a base at least 10× larger, or a curriculum/partial-credit reward (exercises).

**5. The reward is gamed a little, as before.** Names appear 2.27 → 2.61 → **3.39** times per passing response (real stories: 2.5), the same mild inflation as the small model (2.28 → 2.71 → 3.63), still driven by the verifier accepting repetition.

**6. Forgetting is the same price at both sizes.** SFT costs +0.135 bits/byte on plain stories for the 1.97M model and +0.132 for the 0.86M model, and RL costs <0.01 more. The loss on plain text is the price of learning a format; the bigger model doesn't pay less.

### 3.5 Samples from the final model (SFT + RL; seed 5; anecdotes, not evidence)
| instruction | response |
|---|---|
| `Write a short story about a character named Lily.` | "Once upon a time, there was a little girl named Lily. Lily loved to play with her toys and eat yummy food. One day, Lily's mommy said, "Lily, we have to clean this room." Lily was happy and said, "I think much more about you."" |
| `Tell a story about Sam.` | "Once upon a time, there was a little boy named Sam. Sam had a toy car that he loved very much. One day, Sam's toy car went by. He brought it to Sam's toy car and said, "I have a new toy." Sam and Sam were playing with the car." |
| `Write a short story that includes the word "bridge".` | "Once upon a time, there was a little boy named Timmy. Timmy loved to play with his toy cars and trucks. One day, Timmy's toy car got very dirty and didn't like them…" (ignores the keyword) |
| `Write a short story.` | "Once upon a time, there was a little girl named Lily. Lily loved to play outside with her toys. One day, she saw a pretty flower on the ground…" |

Names are followed; keywords are not; grammar and story shape are good; coherence over several sentences ("Sam and Sam were playing", "Lily's mommy said … I think much more about you") is where it fails.

### 3.6 Cost of the whole pipeline on this machine
Pretraining 43 min (2e14 nominal FLOPs); SFT 5 min of training (303 s); RL 7 min (415 s); evaluations a few minutes each. About **one hour of CPU** from raw text to an instruction-following, RL-tuned model, plus the 4+ hours of experiments in steps 01–10 that informed each choice.

## 4. Model card (what this model is, and isn't)
- **Model**: decoder-only transformer, 1,967,808 parameters (4 layers, width 192, 6 heads, context 128, vocabulary 1,024), RoPE + RMSNorm + SwiGLU, tied embeddings, trained from scratch with the code in this repository.
- **Data**: 67.4M tokens of TinyStories (synthetic children's stories generated by large language models), tokenized with a 1,024-token byte-level BPE; SFT on 6,000 synthetic instruction/response pairs derived from the same stories; RL against a programmatic verifier. Check the dataset's license before redistributing any weights built from it.
- **Intended use**: studying language-model training. **Not** for any deployment.
- **What it does**: writes short, fluent, simple English stories; follows "include this character name"; stops at an end marker.
- **What it can't do**: follow any other instruction, know facts, reason, or stay coherent beyond a few sentences; it **does not follow the "include this word" instruction** (≈2%); its vocabulary and world are those of 2- to 5-year-old story-books.
- **Evaluation caveats**: single seed per stage; evaluation prompts are generated from the same templates as the training instructions (no held-out phrasing); the verifier only checks that a word appears, and the model partly games it (repeating the name).
- **Weights** are not committed to the repository (they are 8 MB each and retraining takes under an hour); the checkpoints live in `checkpoints/final-{pretrain,sft,rl}/`.

## 5. What I'd trust, what I wouldn't

**Trust** (replicated, or large relative to the measured noise):
- The pipeline works end to end and each stage helps (stage gaps of 9–25 points against intervals of ±5–6; the same pattern at two model sizes).
- A fit on small runs predicted a 10× larger run to within 1%.
- Sliding-window evaluation, baselines and a sanity check against an untrained model (step 06) were necessary: without them these numbers would not mean much.
- RL's side effects are small and its gains are real but bounded by pass@k.

**Don't over-read** (single seed, small effects, or untested):
- The size comparisons between the two models (one run each; I did not run a paired test).
- Scaling exponents (they depend on bookkeeping; step 10), the optimal size at the flat bottom of the curve, and the learning rate (not tuned at this length).
- The keyword conclusions: all keyword numbers are within a couple of points of zero.
- Anything about *quality* of text beyond the proxy metrics: nothing here measures it directly.

## 6. Reproduce the final model
```bash
# environment
python3 -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch && pip install -e ".[dev,plots]"

# data (the TinyStories text files are downloaded from Hugging Face; see steps 05 and 10)
python -m hedy.prepare --input data/tinystories-valid.txt --out_dir data/tinystories --vocab_size 1024
python -m hedy.prepare --input data/raw/tinystories-train-200mb.txt --out_dir data/tinystories-big \
    --tokenizer data/tinystories/tokenizer.json --train_only --drop_last --val_from data/tinystories
python -m hedy.sft build --text data/tinystories-valid.txt --out data/sft
python -m hedy.dpo prompts --text data/tinystories-valid.txt --out data/dpo      # held-out instruction prompts

# pretrain the scaling-law-chosen model (~45 min on 6 CPU cores), then check the pre-registered prediction
python -m hedy.train --data_dir data/tinystories-big --out checkpoints/final-pretrain \
    --n_layer 4 --n_head 6 --n_embd 192 --pos rope --norm rmsnorm --mlp swiglu --dropout 0 \
    --lr 1.2e-3 --steps 4136 --batch_size 32 --block_size 128 --warmup 100 --threads 4 --compile
python -m hedy.scaling loss --ckpt checkpoints/final-pretrain --prediction docs/data/final_prediction.json

# evaluate, post-train, evaluate
python -m hedy.eval --ckpt checkpoints/final-pretrain --data_dir data/tinystories --text data/tinystories-valid.txt --strides 128,64
python -m hedy.sft run --base checkpoints/final-pretrain --data data/sft --data_dir data/tinystories --mode full --lr 3e-4 --out checkpoints/final-sft
python -m hedy.dpo eval --policy checkpoints/final-sft
python -m hedy.rl --policy checkpoints/final-sft --kinds name,kw --steps 40 --lr 2e-5 --kl_coef 0.02 --out checkpoints/final-rl
```
The numbers behind this note are in `docs/data/final_scorecard.json`, the pre-registered prediction in `docs/data/final_prediction.json`, and the pretraining log in `docs/data/final_pretrain.log`.

## 7. Exercises
1. **Replicate with seeds.** Retrain the pretrained model with two more seeds and re-run every stage. How much of the 8-point final-versus-small gap survives, and what is the seed-to-seed spread of the prediction error (+0.017 here)?
2. **Tune the learning rate at the real length** (e.g. 1.2e-3 vs 8e-4 vs 1.8e-3 at 1/4 of the tokens) and see whether the +0.017 error moves.
3. **A real extrapolation.** Use the law to pick a model for 1e15 FLOPs (≈5M parameters, 34M tokens, a ~3.5-hour run here) and compare with its prediction (≈1.78 nats), a 50× jump beyond the fit.
4. **Fix the RL budget**: run name-only RL, then a keyword curriculum (easy keywords first, then harder), and compare with the mixed run's wasted signal.
5. **Test the capacity hypothesis properly**: train a 10× larger base (≈20M parameters; ~10–15 h here) or give partial credit for keyword proximity, and see whether keyword-following leaves the floor.
6. **Held-out instruction phrasings.** Add paraphrases at evaluation that were not in training. How much of the name-following is the template and how much the skill?
7. **Real downstream benchmarks.** A 2M model scores near chance on HellaSwag/ARC; find the model size at which the `lm-evaluation-harness` numbers leave chance, and compare with the scaling law's loss at that size.
8. **Publish it properly**: write a complete model card, attach the weights to a GitHub release, and add a one-command evaluation script that anyone can run.

## 8. Where to go next
You now have the vocabulary and a small working lab for every stage of building a language model. The natural next learning steps, none of which fit on this CPU: **multi-GPU training** (DDP, FSDP/ZeRO, tensor/pipeline parallelism), **data engineering** (deduplication, filtering, mixture design, tokenizer scaling), **longer context** (RoPE scaling, sliding-window and linear attention, KV-cache paging), **mixture-of-experts**, **reasoning-oriented RL with verifiable rewards at scale**, **real benchmark suites and contamination checks**, **safety and alignment evaluation**, and **interpretability** (what the induction heads and copy circuits behind "include this name" actually look like, which is what this toy model is small enough to open up).

Further reading for the whole arc: Karpathy's *nanoGPT*/*llm.c*/"Neural Networks: Zero to Hero"; the Hugging Face *Smol Training Playbook* and `nanotron`; Stanford CS336 (Language Modeling from Scratch); the Chinchilla, LLaMA, DeepSeek-R1 and Tülu 3 papers (cited in steps 10, 04, 09, 09); Mitchell et al., *Model Cards for Model Reporting* (2019).
