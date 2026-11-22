# 06 — Evaluation: what do the numbers actually mean?

## Goal
Learn to measure a language model *honestly*. A loss of 2.2 or an accuracy of 51.6% means nothing alone; it needs a **definition, a baseline, a noise estimate, and a leak check**. This step builds a small evaluation toolkit (`hedy/eval.py`, tests in `tests/test_eval.py`) and applies it to the 0.86M-parameter TinyStories model from step 05.

Reproduce (the checkpoint is local/gitignored; train it with the step-05 command, copy `model.pt` + `tokenizer.json` to `checkpoints/tinystories/`):

    python -m hedy.eval --ckpt checkpoints/tinystories --data_dir data/tinystories --text data/tinystories-valid.txt

## 1. Perplexity, done properly

### Definitions
- **Loss** = mean negative log-likelihood per token (nats). **Perplexity** = `exp(loss)`: the model is as uncertain as a uniform choice among `ppl` tokens at each step.
- **Bits per byte** = `total_NLL / (total_bytes · ln 2)`. Perplexity depends on the tokenizer (a BPE token covers ~3 bytes, a char token 1), so only bits/byte compares models across tokenizers (step 02). It still doesn't compare across *datasets* (step 05).

### The context-length trap
A model scores token *t* using the tokens before it. If you cut the validation set into non-overlapping windows of `block_size` ("chunked", what the training loop's eval does), the first tokens of every window are predicted with almost **no context**, which is unfairly hard and inflates loss. **Sliding window**: advance by `stride < window` and score only the *newest* `stride` tokens of each window, so every token (except the very first) is scored **exactly once** with at least `window − stride` tokens of context. `stride == window` reduces to chunked.

Implementation notes (`token_nlls`): windows are batched; an internal assertion checks every token is scored exactly once. I wrote the offset of the short final window wrongly at first (it re-scored tokens that already had better context); I caught it by re-reading, and `test_sliding_window_scores_every_token_once_with_full_context` now checks coverage, minimum context, and agreement with an independent per-token reference.

### Results (val = 312k tokens of held-out stories, window 128)
| method | loss | ppl | bits/byte | cost |
|---|---|---|---|---|
| chunked (stride 128) | 2.200 | 9.02 | 1.034 | 1× |
| sliding, stride 64 | 2.128 | 8.40 | **1.000** | 2× |
| sliding, stride 16 | 2.126 | 8.39 | **1.000** | 8× |

- Chunked **overstates loss by 3.4%** (1.034 vs 1.000). The training logs of step 05 (val_bpb 1.041) are the pessimistic, chunked-style number.
- Stride 64 already captures essentially all of the gain; stride 16 buys nothing more. Pick the largest stride where the number stops moving.

Loss by position inside a chunk (nats, ~2,400 samples per position):

| position | 0 | 1 | 3 | 7 | 15 | 31 | 63 | 127 |
|---|---|---|---|---|---|---|---|---|
| loss | 3.83 | 3.20 | 2.72 | 2.38 | 2.16 | 2.25 | 2.12 | 2.15 |

The model needs a handful of tokens of context to "know" it's in a story; after ~16 tokens the curve is flat (the wiggle after that is sampling noise). This is a nice diagnostic for **context use**: for long-context models, plot it out to the whole window and check whether loss still falls.

## 2. Baselines: what is 1.000 bits/byte *good* for?
| model | val bits/byte |
|---|---|
| uniform over 1024 tokens | 3.258 |
| unigram (token frequencies) | 2.725 |
| bigram (previous token only, backs off to unigram) | 1.633 |
| **our 0.86M transformer (sliding)** | **1.000** |

The bigram's smoothing strength α=30 was tuned on the last 200k *training* tokens, never on validation (tuning on the test set is the most common way to fool yourself). Reading: the transformer removes ~39% of the bigram's remaining uncertainty by using longer context. Without the baselines "1.0" has no meaning; with them, it says "much better than local statistics, but how much of the true entropy is left we can't know". Cheap strong baselines (n-grams, a smaller model, the previous checkpoint) belong in every report.

## 3. Multiple-choice likelihood scoring (how benchmarks work)

### The method
Almost every "base model" benchmark (HellaSwag, ARC, PIQA, MMLU in its likelihood form, …) is scored **without generating anything**: for each choice compute `log P(choice | question)` (sum of the choice tokens' log-probs given the context and the preceding choice tokens), then pick the argmax. `lm-evaluation-harness` reports two accuracies:
- **acc**: argmax of total log-likelihood. Biased toward *short* choices (each extra token costs probability).
- **acc_norm**: argmax of log-likelihood **per byte (or char)** of the choice, removing the length bias.

Implementation: `loglikelihood(model, tok, ctx, cont)` and `score_mcq`. The harness's three primitives are `loglikelihood`, `loglikelihood_rolling` (whole-text perplexity, our §1) and `generate_until`; we implement the first two.

### Tokenization boundary gotcha
Score the continuation's tokens given the context's tokens, and tokenize them **separately** only if that equals tokenizing the joined string. Our pre-tokenizer attaches a leading space to the next word, so a context ending in `"... ball."` and a continuation starting with `" He liked ..."` tokenizes identically alone or joined. The benchmark builder checks `encode(ctx)+encode(choice) == encode(ctx+choice)` for every item (0 of 500 dropped here). Getting this wrong silently changes scores.

### Our benchmark: next-sentence selection (HellaSwag-style)
With a 0.86M model, real benchmarks (below) score at chance, so there is nothing to learn from them. I built one the model can actually be measured on: from **held-out** stories (same document split as `hedy.prepare`, via `split_docs`), give the first 3 sentences and ask for the 4th, among **4 choices** = the true sentence + 3 sentences at the same position from *other* stories. Example:

> Callie was so excited for her trip. She had never been on a kayak before. It was a beautiful day, with the sun shining and the gentle breeze blowing.
> - **✔ Callie's parents helped her climb into the kayak and showed her what to do.**
> - Tom practiced walking and running.
> - The mother told her, "I will support you and make the case easy for you."
> - The little girl didn't like the idea at first, but then she remembered that the police are there to support her.

### Results (500 items; chance 25%)
| scorer | accuracy | 95% CI (Wilson) |
|---|---|---|
| acc (total log-lik) | 42.6% | 38.3–47.0 |
| **acc_norm (per byte)** | **51.6%** | 47.2–56.0 |
| always pick shortest choice | 24.8% | 21.2–28.8 |
| always pick longest choice | 22.8% | 19.3–26.7 |
| *untrained model, acc* | 23.0% | 19.5–26.9 |
| *untrained model, acc_norm* | 24.6% | 21.0–28.6 |

How to read it, point by point:
1. **Validate the benchmark itself.** An untrained model must score at chance (it does: 23.0/24.6%) and trivial shortcuts like "shortest/longest" must too (they do: no length leak). If either had been high, the benchmark, not the model, would be what we were measuring.
2. **acc < acc_norm** is the length bias in action, not a shortcut in the data (length-only baselines are at chance).
3. **Uncertainty.** With n=500, the 95% CI is about ±4.4 points. Re-sampling the items with two other seeds gave acc 44.0 / 40.8% and acc_norm 50.0 / 47.4%, i.e. ±2–3 points of *item-sampling* noise on top. So two models closer than roughly 5–8 points on this benchmark are **not distinguishable** without more items or a paired test (McNemar / paired bootstrap on the same items, which is much tighter).
4. **This benchmark is easy by design.** Distractors come from other stories, so the model can win by topical/name consistency alone ("Callie", "kayak"). HellaSwag instead picks distractors *adversarially* (ones a language model finds plausible), which is why it stays hard. As our models improve, this benchmark will saturate; harder distractors (same story, shuffled positions; or mined by a baseline model) would be the next version.

## 4. Real benchmarks and `lm-evaluation-harness`
- HellaSwag (commonsense continuation), ARC (grade-school science), PIQA, WinoGrande, MMLU (knowledge, 4-choice) all use the likelihood scoring above. A *random* model scores 25% on the 4-choice ones.
- **Small models sit near chance for a long time.** From memory (approximate): the 124M-parameter GPT-2 gets only about 29–31% on HellaSwag. Signal appears as parameters × tokens grow; MMLU stays at chance until models are far larger. So running these on our 0.86M model would produce chance-level numbers: honest, but uninformative. That is why §3 uses a task-matched benchmark.
- **I did not run `lm-evaluation-harness` here.** It needs a model wrapper that implements the three primitives above (our `loglikelihood`/rolling functions are the core of one), plus a large dependency stack (`transformers`, `datasets`) and downloads. Writing and running that adapter is exercise 6; do it once you have a model large enough to score above chance.
- **Contamination:** benchmark text appearing in the training corpus inflates scores. Standard defenses: n-gram overlap/deduplication between training data and test sets, and held-out private sets. We split by *document* and never tune on validation; the same discipline at small scale.

## 5. Generation diagnostics
Likelihood doesn't tell you whether *sampled* text is any good. Cheap, automatic probes: **distinct-n** (fraction of unique n-grams in a sample; low = repetitive loops). 30 prompts × 120 sampled tokens (the real-text row uses 120-token excerpts of held-out stories; distinct-n is length-dependent, so compare only against this):

| source | distinct-1 | distinct-2 | distinct-3 |
|---|---|---|---|
| real held-out text | 0.634 | 0.883 | 0.947 |
| temperature 0.5 | 0.557 | 0.824 | 0.912 |
| temperature 0.8 | 0.674 | 0.930 | 0.981 |
| temperature 1.0 | 0.721 | 0.956 | 0.991 |
| temperature 1.3 | 0.775 | 0.981 | 0.998 |

At T=0.5 the model is **more repetitive than real text**; from T≈0.8 up it is **more varied than real text**: not because it's more creative, but because it's emitting words real stories wouldn't (step 05's sample). The "right" temperature sits where diversity matches real text, near 0.6–0.7 by this probe. **Diversity is not quality**: a random-token generator maximizes distinct-n. Quality needs human or LLM-judge ratings; MAUVE compares the distribution of generated vs real text.

## 6. Evaluation hygiene checklist
- Held-out data split **by document**, built before tokenizer/model/hyper-parameter choices; tune on a *training* holdout, not the test set. ✔ (n-gram α)
- Use a tokenizer-independent metric when tokenizers differ (bits/byte). ✔
- Always report baselines and a sanity check with a model whose answer you know (untrained/shuffled). ✔
- Report uncertainty (CIs, multiple item/seed samples), and don't over-read gaps smaller than it. ✔
- Evaluate with full context (sliding window), and say how. ✔
- **Test the evaluation code** like any other code: eval bugs don't crash, they just produce plausible wrong numbers.

## 7. Exercises
1. Evaluate with `--max_tokens 20000` and compare to the full set. How noisy is a 20k-token validation estimate? Bootstrap-resample documents to get a CI for bits/byte.
2. Make a **trigram** baseline (interpolated with bigram/unigram). How close does it get to the transformer?
3. Evaluate the step-01..04 style models (retrain small ones) with `hedy.eval`'s MCQ and see whether acc_norm tracks bits/byte ordering. Use a paired bootstrap on the shared items.
4. Make the benchmark harder: distractors = other sentences from the **same story** (positions shuffled), or mined as the highest-likelihood sentences from a bigram model. Where does the model land?
5. Compute `acc` normalized by *token count* instead of bytes ("acc_token_norm"). When do the three disagree?
6. Write the `lm-evaluation-harness` adapter (`loglikelihood`, `loglikelihood_rolling`, `generate_until`) and run it on a ≥100M-parameter public model to see real HellaSwag/ARC numbers.
7. Plot loss-by-position for a RoPE model evaluated beyond its trained window (needs removing the `block_size` assert and a longer RoPE table). Where does it break?
8. Calibration: bucket predictions by max-prob and compare confidence vs accuracy (reliability diagram); compute expected calibration error.

## 8. Further reading
- Gao et al., *A framework for few-shot language model evaluation* (`lm-evaluation-harness`)
- Zellers et al., *HellaSwag* (2019): adversarial filtering of distractors
- Hugging Face docs, *Perplexity of fixed-length models* (the sliding-window argument)
- Jacovi et al., *Stop Uploading Test Data in Plain Text* and Magar & Schwartz, *Data Contamination: From Memorization to Exploitation* (2022)
- Pillutla et al., *MAUVE* (2021); Li et al., *A Diversity-Promoting Objective Function for Neural Conversation Models* (distinct-n, 2016)
- Wilson, *Probable inference, the law of succession, and statistical inference* (1927), and Miller, *Adding Error Bars to Evals* (2024)

**Next:** 07 — post-training: supervised fine-tuning (SFT) and parameter-efficient LoRA.
