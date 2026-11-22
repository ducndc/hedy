# 07 — Post-training: supervised fine-tuning (SFT) and LoRA

## Goal
A pretrained model only *continues text*. Post-training turns it into something that **follows instructions**. This step implements the first stage of that pipeline, SFT, in two ways (update all weights vs. LoRA), and measures what each buys and costs. Code: `hedy/sft.py`, `hedy/lora.py`; tests: `tests/test_sft.py`, `tests/test_lora.py`.

Base model: the 0.86M-parameter TinyStories model from step 05 (`checkpoints/tinystories`, local/gitignored).

## 1. Theory

### The post-training pipeline (what we do and don't do)
1. **Pretraining**: next-token prediction on raw text → knowledge and fluency (steps 01–05).
2. **SFT** (this step): continue training on (instruction → good response) pairs → follows the *format* and the *task*.
3. **Preference tuning** (RLHF / DPO / RLVR; not here): push the model toward responses people (or a verifier) prefer rather than just imitating examples.

SFT is *the same loss as pretraining*, cross-entropy on next tokens. The only differences are the **data** (instruction/response pairs) and the **mask**.

### Loss masking
A training example is `prompt + response`. We compute the loss **only on the response** (and the end marker), setting prompt targets to `-100`, which `cross_entropy` ignores. Why: the prompt is given at inference time; spending capacity (and gradient) on predicting user text teaches nothing useful and, with short responses, can dominate the loss. `test_labels_mask_the_prompt_only` pins this down. Some pipelines also train on the prompt; masking is the common default.

### Teaching the model to stop
At inference you need to know where the response ends. Production models have a special `<|end|>`/EOS token. Our BPE has none, so we end each response with the literal text `\n### End\n`, train on those tokens (so the model learns to emit them), and stop generation when that string appears (`GPT.generate(stop=...)`). A model that never learned the marker never stops: the base model emits it **0%** of the time.

### Padding without attention masks
Batches need equal lengths. We pad on the **right**, and because attention is causal, a real token never attends to later positions, so pads cannot influence it. Pad *targets* are `-100`. `test_padding_does_not_change_loss` verifies that a sequence's loss alone equals its contribution inside a padded batch.

### Full fine-tuning vs LoRA
**Full fine-tuning** updates every weight: maximum flexibility, but you must store gradients + Adam state for all of them (16 B/param, step 05), and each task needs a whole new copy of the model.

**LoRA** (Hu et al. 2021) rests on the observation that the *change* needed to adapt a pretrained model has low intrinsic rank. Freeze `W` and learn

    y = W x + (α/r) · B (A x),     A ∈ R^{r×in}, B ∈ R^{out×r}

- `B` is initialized to **0**, so the adapted model *starts exactly equal to the base* (`test_zero_init_is_identity`).
- Parameters per wrapped layer: `r·(in+out)` instead of `in·out`; with r=8 on a 128→256 layer that's 3,072 vs 32,768.
- A subtle fact (tested): on the very first step `dL/dA = 0`, because `A` only matters through `B=0`; only `B` learns first, then `A` joins in.
- `α/r` is a fixed scale; we use `α = 2r`. Changing `r` at fixed `α/r` keeps the update's size comparable.
- **Merging**: after training, fold the update into the weights, `W' = W + (α/r)·BA`, and the model is architecturally the same as the base, with **no inference overhead** (`merge_lora`; `test_merge_equals_adapted_and_removes_overhead`).
- **Adapters are tiny and swappable**: ship a few KB per task and keep one base model in memory (serving many fine-tunes). `lora_state_dict` extracts just those tensors.
- Where to put it: we wrap all linear layers (`qkv, proj, fc, gate`). The original paper used attention only; later work found adapting the MLPs matters too (compare in §3).
- **QLoRA** (not here) additionally stores the frozen base in 4-bit, making 65B-parameter fine-tuning fit on one GPU.

### Catastrophic forgetting
Fine-tuning on a narrow task can erase general ability. We measure it directly: **bits/byte on the original pretraining validation set** before vs. after. LoRA constrains the update to a low-rank subspace, so it is often (not always) said to "learn less and forget less" (Biderman et al. 2024); we test that claim in miniature.

## 2. The task: synthetic *verifiable* instructions
Real SFT data is human- or model-written. We need something a 0.86M model can learn and a **program can grade**, so we derive instructions from TinyStories itself (`python -m hedy.sft build`):

| type | instruction | check |
|---|---|---|
| `kw` | `Write a short story that includes the word "money".` | the word appears (whole word, case-insensitive) |
| `name` | `Write a short story about a character named Sam.` | the name appears |
| `plain` | `Write a short story.` | (format only) |

Each type has 2 paraphrased templates. The response is the story's first sentences, cut to fit the 128-token context (prompt + response + End ≤ 128; the builder retries with a shorter response if the prompt tokenizes long; found by a failing test). Keywords are content words with **document frequency between 30 and 1500** (out of ~21k stories) so they are neither trivially common nor typos; names are capitalized words that appear mid-sentence at least twice. Train: 6,000 examples built from training stories. Eval: 300 prompts (100/type) from the **held-out** stories.

Example training pair:

    ### Instruction:
    Write a short story that includes the word "money".
    ### Response:
    Once upon a time there were two brothers, Jack and John. Every day they bickered and fought with each other. One day, they decided to spend their money on a rare license. They hoped it would make them feel special.
    ### End

### How instruction-following is scored
Generate once per prompt (temperature 0.7, top-k 40, fixed seeds) and check with code. Two safeguards:
- **Coincidence control**: score each response against the *next prompt's* keyword. That's how often the required word shows up by luck. A model that "follows" at the control rate hasn't learned anything.
- **Confidence intervals** (Wilson), since n = 100 per type.

Honest limitations: both instructions are *copy-a-word-from-the-prompt* tasks (induction/copying, not reasoning); the templates at eval are the same two paraphrases per type as in training (no held-out phrasing); a program checking "word appears" doesn't check the story is *good* or that the word is used sensibly.

## 3. Results

Setup: base = step-05 model (0.857M params, window 128). 6,000 training examples, 3 epochs (561 steps), batch 32, AdamW (no weight decay), 20-step warmup + cosine to 10%, fp32 CPU. LoRA: α=2r, on all linears unless noted. Metrics on 300 held-out prompts (100/type), temperature 0.7, top-k 40. Columns:
- **SFT loss**: held-out response-token loss (nats; the base starts at 2.44).
- **pretrain bpb**: bits/byte on *plain* pretraining validation text (60k tokens, sliding window): lower is better; the base model's value, **1.058**, is the "no forgetting" reference.
- **kw % / name %**: generated response contains the required word (n=100 each; 95% CI roughly ±10 points). Controls (word-appears-by-coincidence) were 0–6% everywhere.
- **kw gain / name gain** (teacher-forced, in nats): how much cheaper the required word becomes when the real instruction replaces the neutral "Write a short story." Higher = the model uses the instruction. Averaged over 100 prompts, far less noisy than the pass rate.
- **end %**: response emitted the `### End` marker.

| run | trainable params | SFT loss | pretrain bpb | kw % | kw gain | name % | name gain | end % |
|---|---|---|---|---|---|---|---|---|
| base (no SFT) | 0 | 2.439 | **1.058** | 1 | 0.12 | 37 | 1.34 | 0 |
| **full** lr 1e-4 | 857,216 (100%) | 1.627 | 1.137 | 1 | 0.13 | 44 | 2.03 | 99 |
| **full** lr 3e-4 (seed 0 / seed 1) | 857,216 | 1.616 / 1.614 | 1.190 / 1.192 | 1 / 2 | 0.15 / 0.15 | 64 / 60 | 2.44 / 2.37 | 98 / 99 |
| **full** lr 1e-3 | 857,216 | 1.633 | 1.290 | 2 | 0.19 | 66 | 2.70 | 99 |
| LoRA r=8 lr 1e-3 | 65,792 (7.7%) | 1.650 | 1.157 | 1 | 0.12 | 35 | 1.76 | 98 |
| LoRA r=8 lr 3e-3 (seed 0 / seed 1) | 65,792 | 1.645 / 1.645 | 1.183 / 1.183 | 2 / 1 | 0.13 / 0.13 | 42 / 51 | 2.16 / 2.16 | 99 / 100 |
| LoRA r=8 lr 1e-2 | 65,792 | 1.648 | 1.188 | 1 | 0.13 | 50 | 2.19 | 99 |
| LoRA r=1 | 8,224 (1.0%) | 1.670 | 1.151 | 2 | 0.13 | 16 | 1.19 | 99 |
| LoRA r=4 | 32,896 (3.8%) | 1.655 | 1.161 | 1 | 0.11 | 40 | 1.94 | 98 |
| LoRA r=16 | 131,584 (15.4%) | 1.638 | 1.199 | 2 | 0.14 | 57 | 2.32 | 100 |
| LoRA r=8, attention only | 35,584 (4.2%) | 1.655 | 1.138 | 1 | 0.13 | 46 | 2.00 | 99 |
| full, 1,000 examples × 6 ep | 857,216 | 1.711 | 1.204 | 1 | 0.16 | 40 | 1.94 | 99 |
| LoRA r=8, 1,000 examples × 6 ep | 65,792 | 1.722 | 1.206 | 2 | 0.15 | 18 | 1.36 | 98 |

(All runs: `python -m hedy.sft run …`; logs/results are in `checkpoints/sft-*/results.json`, local/gitignored.)

### What the numbers say

**1. Format is learned by everything, even 1% of the parameters.** Every SFT variant moves held-out response loss from 2.44 to 1.61–1.72 and starts emitting `### End` 98–100% of the time (base: 0%). Rank-1 LoRA (8,224 parameters, a 44 KB adapter file) gets to 1.670 against full fine-tuning's 1.616: ~93% of the loss improvement from 1% of the weights. Format and style are a *low-dimensional* change: this is the empirical heart of LoRA, and of the "superficial alignment" idea that SFT mostly surfaces behaviour the base model already has (Zhou et al., LIMA, 2023: a hypothesis, not a law).

**2. Names: the instruction is used, but only partly, and capacity matters.** The base model already copies a name from the prompt 37% of the time (in-context copying, which pretraining gave it). SFT raises the teacher-forced gain from 1.34 to 2.0–2.7 nats. Among LoRA runs the gain is **monotone in rank** (r=1: 1.19, r=4: 1.94, r=8: 2.16, r=16: 2.32) and full fine-tuning is highest (2.37–2.70). At r=8, LoRA delivers ~75% of full fine-tuning's gain over the base ((2.16−1.34)/(2.44−1.34)); r=16 delivers ~90%. Note r=1 is *worse than the untrained base* on this measure (1.19 < 1.34): too little capacity to add the skill, yet enough to shift the format, which slightly disturbs the copying the base already had.

**3. Keywords: nothing learned, and it is not a learning-rate problem.** Across all 14 fine-tuned runs, learning rates spanning 100× (full 1e-4…1e-3, LoRA 1e-3…1e-2), ranks 1–16 and full fine-tuning, keyword inclusion stays 1–2% (control 0–1%) and the teacher-forced gain stays 0.11–0.19 nats, i.e. the model pays ~5.7 nats (p ≈ 0.3%) for the word whether or not the instruction asked for it. (A unit test confirms the probe works: a model overfit to the pattern shows a large gain.) Why the two tasks differ: a name appears early with an obvious cue ("a girl named …"), so copying suffices. A keyword has to be worked into a story at a place the model *chooses*: that needs planning ("I must use *bridge* somewhere, and here it fits"), which a 0.86M-parameter model apparently cannot do. Even an ideal model could not predict the exact position, so its gain per slot is bounded, but ~0.1 nats is far below that bound. **SFT cannot conjure a capability the base model lacks.** What SFT does teach is format, style and the use of skills the model already has. Try: a bigger base model, or an easier variant ("begin the story with the word …") (exercise 3).

**4. Forgetting: how far you move matters, and LoRA is not automatically safer.** The "no forgetting" reference is 1.058 bits/byte.
- Full fine-tuning rises with learning rate: 1.137 (1e-4) → 1.190 (3e-4) → **1.290** (1e-3, +0.23, severe).
- LoRA r=8 sits at 1.157–1.188 and rank adds a little (r=1 1.151 → r=16 1.199).
- Plot forgetting against task gain (name gain / pretrain bpb): full 1e-4 (2.03 / 1.137), **LoRA attention-only (2.00 / 1.138)**, LoRA r=4 (1.94 / 1.161), LoRA r=8 (2.16 / 1.183), LoRA r=16 (2.32 / 1.199), full 3e-4 (2.44 / 1.190), full 1e-3 (2.70 / 1.290). Attention-only LoRA and 1e-4 full fine-tuning land on the same point; full 3e-4 gets *more* gain than LoRA r=16 with *less* forgetting. **At matched task performance there is no forgetting advantage for LoRA in this experiment** (the slogan "LoRA learns less and forgets less" is only the trade-off curve: both move along it). Caveats: coarse LR grid; one narrow forgetting probe; on a model this tiny full fine-tuning's memory cost is irrelevant, which is the one thing LoRA is really for.

**5. Less data/steps hurts LoRA more.** With 1,000 examples × 6 epochs (190 steps), full fine-tuning still reaches name gain 1.94 (40%) but LoRA r=8 stays at 1.36 (18%), the base level. This is confounded (fewer optimizer steps, and `B=0` makes LoRA start slowly), so read it as "LoRA at this LR needs more steps", not as a statement about data volume.

**6. Noise.** Same config, different seed: full 64% vs 60% name pass rate; LoRA r=8 42% vs **51%**, with CIs of ±10 points, so **pass-rate differences under ~10 points between runs are not meaningful**. The teacher-forced gain is much steadier (LoRA r=8: 2.16 / 2.16, full: 2.44 / 2.37), which is exactly why that probe exists: *a smooth, teacher-forced metric on the critical tokens separates runs where a thresholded generation metric cannot.* The learning-rate effect on LoRA (1e-3 → 3e-3 → 1e-2: gain 1.76 → 2.16 → 2.19) is real; LoRA wants roughly 10× the full fine-tuning rate and plateaus.

### Samples (same seed; sampling is noisy, so anecdotes, not evidence)
Prompt: `Write a short story about a character named Lily.`

| model | response |
|---|---|
| base | "Once upon a time, there was a party. The path was very pleased … Lily's mommy put the package on the table …" (no End; runs on) |
| full | "Once upon a time, there was a boy named **Timmy**. Timmy loved to play outside … " (ends cleanly; **ignores the name**) |
| LoRA r=8 | "Once upon a time, there was a little party. … One day, **Lily** wanted to go to the park. She asked her mom …" (ends cleanly; uses the name) |

The format is learned (story opener, stops on its own); the content constraint is followed only sometimes, matching 64% / 42% above.

## 4. Practical lessons
- **Mask the prompt, teach the stop token, test both.** Silent bugs here produce models that look fine on loss.
- **Tune the learning rate per method** (LoRA ≈ 10× full). A single shared LR would have made one method look bad.
- **Measure capability with a verifier and a control**, and prefer a smooth teacher-forced probe to a pass/fail rate when comparing close runs.
- **Always measure forgetting** against the original ability, and compare at matched task performance, not matched hyper-parameters.
- **LoRA's real win is operational** (tiny per-task files, one shared frozen base, far less optimizer memory), and the saving grows with model size: adapter cost is `2r·d` per layer vs `d²`, i.e. `2r/d`: 12.5% at our d=128, r=8, but 0.4% at d=4096. At 0.86M parameters the saving is almost irrelevant, so judge LoRA's *quality* here, not its cost.

## 5. What this step doesn't do
- **Preference optimization** (RLHF/DPO) and RL on verifiable rewards: natural next step (step 08?). Our verifier is exactly what an RLVR reward would be.
- Real chat templates, multi-turn data, special tokens, and system prompts. (Exercise 5.)
- QLoRA/quantization, LoRA variants (DoRA, rsLoRA with α/√r), merging multiple adapters.

## 6. Exercises
1. Add weight decay and LoRA dropout (`apply_lora(..., dropout=0.1)`) and see whether name gain/forgetting move.
2. Add a held-out **template** at eval (e.g. "Please write a story about a girl called {k}."). Does instruction-following survive a paraphrase it wasn't trained on?
3. Make the keyword task learnable: require the word in the **first sentence**, or ask for the word to *start* the story. Does kw gain jump? Then scale the base model (e.g. 4× params) on the original task and see whether planning emerges.
4. Mix 10% *pretraining* text into the SFT batches (rehearsal/replay) and re-measure forgetting vs. name gain. Does it bend the trade-off curve?
5. Add a real `<|end|>` special token (vocab + 1; resize the embedding and tied head) in place of the `### End` text, and stop generation on its id.
6. Train on the prompt too (no masking). What happens to held-out response loss and to the first tokens of generations?
7. Implement rsLoRA (`scale = α/√r`) and test whether it removes the rank-dependence of the best LR.
8. Merge two adapters trained on different tasks by adding their `BA` updates. When does it work, when does it interfere?
9. Plot the full trade-off curve (name gain vs pretrain bpb) with more LR points for each method; where does each curve end?

## 7. Further reading
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models* (2021)
- Dettmers et al., *QLoRA* (2023)
- Biderman et al., *LoRA Learns Less and Forgets Less* (2024): the claim we tested in miniature
- Zhou et al., *LIMA: Less Is More for Alignment* (2023): superficial alignment hypothesis
- Ouyang et al., *Training language models to follow instructions with human feedback* (InstructGPT, 2022): the SFT → reward model → RL pipeline
- Wei et al., *Finetuned Language Models are Zero-Shot Learners* (FLAN, 2021); Taori et al., *Alpaca* (2023), the instruction format we borrowed
- Kalajdzievski, *A Rank Stabilization Scaling Factor for LoRA* (2023)

**Next:** 08 — preference optimization: DPO (and a verifier-reward RL loop), reusing this step's verifiable instructions.
