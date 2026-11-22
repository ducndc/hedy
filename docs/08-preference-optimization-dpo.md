# 08 — Preference optimization: DPO with a program as the judge

## Goal
SFT (step 07) teaches a model to *imitate* examples. Preference optimization teaches it to *prefer* better outputs over worse ones, which can fix failures that imitation doesn't. We implement **DPO** from scratch, build the preference data automatically with a **verifier** (code that checks a response), and compare it against a simpler baseline to see what DPO adds. Code: `hedy/dpo.py`; tests: `tests/test_dpo.py`.

Starting point: the step-07 full-fine-tuned model (`checkpoints/sft-full`, name-following ≈ 60%, keyword-following ≈ 1%).

## 1. Theory

### From RLHF to DPO
Classic RLHF: (1) collect human comparisons, (2) fit a **reward model** `r(x, y)` with the Bradley–Terry model `P(y_c ≻ y_r) = σ(r(x,y_c) − r(x,y_r))`, (3) run RL (PPO) to maximize reward **minus a KL penalty** that keeps the policy near the SFT/reference model:

    max_π  E[ r(x,y) ] − β · KL( π(·|x) ‖ π_ref(·|x) )

DPO (Rafailov et al. 2023) observes this KL-regularized problem has a closed-form optimum, `π*(y|x) ∝ π_ref(y|x) · exp(r(x,y)/β)`, which can be inverted: the reward is `r(x,y) = β · log(π(y|x)/π_ref(y|x)) + const`. Substituting into Bradley–Terry, the constant cancels between chosen and rejected, leaving a **supervised loss on preference pairs, with no reward model and no sampling loop**:

    L = − log σ( β · [ (log π(y_c|x) − log π_ref(y_c|x)) − (log π(y_r|x) − log π_ref(y_r|x)) ] )

Read it as: raise the chosen response's probability *relative to the reference* and lower the rejected one's, and stop caring once the margin is large (the sigmoid saturates).

### The pieces
- **Log-probability of a sequence** = sum of token log-probs of the *response* given the prompt (`seq_logps`; prompt tokens are masked exactly like SFT's labels). It is a *sum*, not a mean, so DPO implicitly weights longer responses more and can drift toward length changes (we measure length).
- **Reference model** `π_ref`: frozen copy of the starting policy. We don't keep two models: reference log-probs of every pair are computed **once, before training** (`ref_logps`) and cached.
- **β** controls how far the policy may move: small β → large moves allowed (`exp(r/β)` is sharp), large β → stay close to the reference. At initialization `π = π_ref`, margin 0, loss `= ln 2` (tested).
- **Implicit reward accuracy**: fraction of pairs where the implicit reward of chosen > rejected; the standard training diagnostic, along with the *margin*.
- **Known pathology**: DPO often lowers the log-probability of **both** responses (it only constrains the *difference*), shifting probability mass to outputs seen in neither. We log `Δlogp(chosen)` and `Δlogp(rejected)` to watch for it. A common remedy is adding an NLL term on the chosen response (`--sft_coef`, as in RPO).
- **On- vs off-policy data**: the pairs here are sampled from the model itself (on-policy), which tends to work better than comparing responses from other models, whose text the policy would assign negligible probability.

### Preference data from a verifier
Instead of human labels, step 07's checker is the judge:
1. For each training instruction, sample **K = 8** responses from the SFT model at temperature 1.0 (`sample_k`: one batched `generate` call per prompt).
2. A response **passes** if it finished (emitted `### End`) *and* contains the required word.
3. Pair a passing response (chosen) with a **finished** failing one (rejected). Requiring the rejected one to finish keeps the preference about *content*, not about whether it stopped.
4. Prompts where all 8 samples pass or all fail give no pair: **no contrast, no learning signal**, which is why task difficulty matters (below).

We keep the sampled **token ids** instead of re-encoding the decoded text: the model may sample a non-canonical tokenization that re-encoding would change, and DPO's math is about the policy's *own* sequence probabilities.

### Controls worth having
- **Rejection-sampling fine-tuning (RFT)** (`--objective rft`): plain NLL on the *chosen* responses only, same data. If DPO doesn't beat this, you didn't need the contrast.
- **pass@k vs pass@1** at evaluation: pass@4 = "at least one of 4 samples passes", pass@1 = average. A large gap means the model *can* produce correct answers but not reliably; preference/RL training mostly **sharpens** (moves pass@1 toward pass@k), it rarely creates behaviour the model never samples. Recent RL-with-verifiers work debates exactly this (Yue et al., 2025).
- **Reward-hacking probes**: the verifier only checks "the word appears", so the cheap way to please it is to *mention* the word, repeat it, or stuff it in. We track mean key count, the fraction of responses containing it ≥3×, distinct-2 (repetitiveness), and length. (The first printed training pair already shows an imperfect reward: "…a little boy named Judy. June was very sleepy…" passes the name-check but isn't a story about June.)
- **Forgetting/drift**: pretraining bits/byte and SFT validation loss relative to the reference.

## 2. Data we built
From 2,000 training instructions × 8 samples (temperature 1.0; about 4 minutes of wall time with 3 processes of 4 threads each):

| type | prompts | sample pass-rate | prompts with ≥1 pass | pairs | length chosen vs rejected |
|---|---|---|---|---|---|
| name | 1,000 | 53.7% | 89% | 1,592 | 70.5 vs 69.7 tokens |
| keyword | 1,000 | 0.8% | 6% | 116 | 68.1 vs 67.4 tokens |

Two things to notice. (1) **Contrast is the fuel.** Names pass about half the time, so most prompts yield a (pass, fail) pair; keywords almost never pass, so 94% of keyword prompts yield *nothing*: the same wall SFT hit in step 07. Preference learning cannot teach what the model never samples. (2) Chosen and rejected lengths are matched, so DPO's length bias can't be what we measure.

Evaluation: 150 held-out name prompts and 150 held-out keyword prompts, K = 4 samples each at temperature 0.7/top-k 40 (the same setting as step 07). Intervals are **bootstrap over prompts** (samples for one prompt are correlated, so resampling samples would give falsely tight CIs). All runs use the same prompts and seeds, which makes cross-run comparisons somewhat tighter than the unpaired intervals suggest, but I did not compute paired tests.

## 3. Results (name-following; start = the SFT model, pass@1 63.0%)

Everything trains from `sft-full` on the same 1,708 pairs, 3 epochs, batch 16 (321 steps), full fine-tuning unless noted.

| run | name pass@1 (95% CI) | pass@4 | finished | mentions / passing response | Δ pretrain bpb | SFT val loss | Δlogp chosen (train) |
|---|---|---|---|---|---|---|---|
| SFT reference (start) | 63.0% (57–69) | 82.7% | 100% | 2.71 | +0.000 | 1.616 | – |
| RFT lr 2e-5 | 73.5% (68–79) | 86.7% | 100% | 2.71 | +0.004 | 1.623 | – |
| RFT lr 1e-4 | 73.8% (68–80) | 84.7% | 100% | 2.75 | +0.025 | 1.654 | – |
| DPO β=0.1 lr 2e-5 | 76.0% (70–82) | 85.3% | 98% | 4.20 | +0.026 | 1.684 | −1.6 |
| DPO β=0.1 lr 2e-5, seed 1 | 77.2% (71–83) | 86.0% | 98% | 4.14 | +0.026 | 1.682 | −1.5 |
| DPO β=0.1 lr 5e-5 | 79.0% (73–85) | 84.7% | 93% | 5.19 | +0.079 | 1.812 | −7.0 |
| DPO β=0.1 lr 1e-4 | 78.5% (72–85) | 83.3% | **10%** | 6.08 | +0.147 | 2.084 | **−20.5** |
| DPO β=0.03 lr 2e-5 | 69.8% (64–76) | 82.0% | 98% | 4.51 | +0.044 | 1.751 | −5.8 |
| DPO β=0.3 lr 2e-5 | 78.2% (72–84) | 86.0% | 99% | 3.70 | +0.014 | 1.647 | +0.7 |
| DPO β=0.3 lr 1e-4 | 81.2% (75–87) | 86.0% | 92% | 4.66 | +0.081 | 1.802 | −3.6 |
| DPO β=0.1 lr 2e-5 + NLL anchor | 77.3% (71–83) | 86.0% | 99% | 3.38 | +0.012 | 1.645 | +1.3 |
| DPO β=0.1 lr 1e-4 + NLL anchor | 79.7% (74–85) | 87.3% | 99% | 2.92 | +0.042 | 1.700 | +6.4 |
| DPO LoRA r=8 lr 1e-3 | 78.2% (72–84) | 84.0% | **2%** | 6.44 | +0.158 | 2.211 | **−29.4** |

Column guide: *finished* = fraction of responses that emitted `### End` (a basic health check); *mentions / passing response* = mean number of times the name appears in responses that contain it (real stories: **2.5**; chosen responses in the pairs: **2.0**); *Δ pretrain bpb* = forgetting versus the SFT model's 1.190 on plain stories; *Δlogp chosen* = how the chosen responses' log-probability moved from its reference value during the last training epoch (nats, summed over the response).

### What the numbers say

**1. Every method helps, and DPO's edge over the simple baseline is modest.** Name pass@1 rises from 63% to 73–81%. Plain RFT (just SFT on the passing samples) already gets **73.5–73.8%** with essentially no side effects: still 100% finished, the same 2.7 mentions per passing response as the starting model, and almost no drift (+0.004 bpb at the gentle setting). The better-behaved DPO runs reach **76–80%**: about 2.5–6 points above RFT, with confidence intervals (±6) that overlap, and the same direction across five configurations. Suggestive, not conclusive. If you only look at pass@1 you'd say "DPO wins"; if you also look at the other columns, the real picture is a trade.

**2. It sharpens; it doesn't expand.** pass@4 of the starting model is 82.7%. Training moves pass@4 only to 82–87% (intervals ±5), but pass@1 climbs toward it: the best pass@1 anywhere (81.2%) is about the starting model's pass@4. The model learned to produce, *reliably*, something it could already produce *sometimes*. The keyword task confirms it from the other side: only 116 keyword pairs existed, and keyword pass@1 stays 0.5–2.0% (pass@4 2–6%, i.e. 3–9 prompts out of 150) in every run: noise. No contrast, no learning.

**3. Plain DPO over-optimizes, and the warning signs are visible before the metric breaks.**
- *The proxy plateaus while everything else degrades.* Raising lr at β=0.1: pass@1 76.0 → 79.0 → 78.5 (flat), but finished 98% → 93% → **10%**, SFT val loss 1.68 → 1.81 → 2.08, forgetting +0.026 → +0.079 → +0.147. At lr 1e-4 the model has largely forgotten *to stop*. Its 78.5% pass@1 is real by the verifier's definition and meaningless as a story generator. **The verifier never asks for termination, so it could not notice.** This is Goodhart's law in one table.
- *Training accuracy is a poor guide.* Implicit-reward accuracy hits 0.90–1.00 on the training pairs in every run; it only says the pairs are separated, not that the model is better.
- *The leading indicator is the chosen response's log-probability.* Collapse coincides with Δlogp(chosen) of −20 to −29 nats (the LoRA run, −29.4, collapsed the same way: not a LoRA problem, a step-size problem). Runs that keep it near or above 0 (β=0.3: +0.7; anchored runs: +1.3, +6.4) stay healthy. DPO only constrains the *difference* between chosen and rejected; pushing the rejected one far down is easiest, and the chosen one is dragged down with it.

**4. DPO extrapolates and the reward is hackable.** Among responses that pass, the name appears 2.7 times for the SFT model and for RFT, and **4.1–6.4 times** for DPO. The real stories average 2.5 and the *chosen* responses we trained on average 2.0. So DPO is not imitating the data: it found that repeating the name is a safer way to satisfy "the name appears" and pushed beyond anything in its training set. (I did not run an ablation that proves this causal story; it is the explanation consistent with the evidence.) A better reward (penalize repetition, require the name to be the protagonist) would be needed to close that gap, and it is the reason real reward models and verifiers get attacked.

**5. The knobs.**
- **β**: larger β = tighter leash. At the same lr 2e-5, β = 0.03 / 0.1 / 0.3 gives 69.8% / 76.0% / **78.2%** pass@1 with forgetting +0.044 / +0.026 / **+0.014** and mentions 4.51 / 4.20 / 3.70. Counter-intuitively, the largest β is *both* the most effective and the safest here, because small β lets the log-probabilities drift far for the same pair accuracy (the chosen log-prob falls 5.8 nats at β = 0.03 and *rises* 0.7 at β = 0.3).
- **NLL anchor** (`--sft_coef 1`, adding an NLL term on the chosen response): the cheapest fix for collapse. At lr 1e-4 plain DPO finishes 10% of responses; with the anchor, **99%**, with pass@1 79.7%, the highest pass@4 (87.3%), and the lowest hacking (2.92 mentions, closest to real stories).
- **Seeds**: the same config with seeds 0/1 gives 76.0/77.2%, 4.20/4.14 mentions, identical drift: training noise is small next to the evaluation intervals.

**6. Cost.** RFT needs no reference model and no rejected responses: it trained in about 2 minutes vs. 3–4 for DPO on this machine (timings are rough, with 3 runs sharing the CPU) and had no failure mode we could find. DPO's extra contrast buys a few points at the price of tuning (β, lr, anchor) and monitoring.

### Practical lessons
- **Build the evaluation before the training.** The "finished" and "mentions per passing response" columns exist because I distrusted the headline metric; they are what exposed the collapse.
- **Compare against RFT.** If the simple baseline gets most of the gain, you have learned what you are paying DPO for.
- **Monitor chosen log-probability, not just reward accuracy.** A falling chosen log-prob is the early warning.
- **Keep β moderate and add an NLL anchor** unless you've shown you don't need them.
- **Measure what the verifier can't see** (termination, repetition, length, drift): reward models have blind spots, and optimizers find them.
- **Task difficulty sets the ceiling**: preference/RL training needs successes to learn from. The fix for "0.8% pass rate" is a curriculum, a better SFT model, or a larger model, not more DPO.

## 4. What this step doesn't do
- **Online RL** (PPO / GRPO / RLOO with the verifier as the reward) samples fresh responses *during* training, so the data stays on-policy and the model can discover new successes; our pairs are fixed after one sampling pass. This is the natural next step (and the real way "RL with verifiable rewards" is done).
- Learned reward models, human preference data, and judge models.
- Safety-style preference tuning, and length-controlled variants (SimPO, length-normalized DPO), IPO/KTO/ORPO.

## 5. Exercises
1. Normalize the log-probabilities by length (mean per token, as in SimPO/length-normalized DPO). Does it change drift, finishing, or pass@1?
2. **Close the loophole**: make the verifier reject responses with the name ≥ 4 times (or reward `distinct-2`), rebuild the pairs, retrain. Does the repetition disappear, and what does it cost?
3. Add a "finished" term to the verifier (done); now *remove* it from the rejected-side rule so that unfinished failures can be rejected too. Does DPO learn to stop faster?
4. Compute a **paired** bootstrap (same prompts) of DPO − RFT. Is the 2.5–6 point gap significant? (Store per-prompt pass rates in `results.json` to do it.)
5. Iterate: sample from the DPO model, build new pairs, train again ("iterative / online DPO"). Where does it saturate?
6. Increase pair count by sampling 32 samples per prompt, or use only the *hardest* prompts (pass rate 10–50%). Which data helps more, volume or informativeness?
7. Implement **RLOO/GRPO-style** online RL: sample K, reward = verifier, advantage = reward − mean of the group, policy-gradient loss + KL penalty to the reference. Compare with DPO at equal sampling compute.
8. Train DPO on a *mixture* of name and keyword pairs with the keyword pairs upsampled 10×. Does keyword pass@4 move beyond noise with 116 pairs repeated, or does it only memorize?
9. Plot the trade-off curve (pass@1 vs forgetting bpb) for all runs, and find the Pareto frontier.

## 6. Further reading
- Rafailov et al., *Direct Preference Optimization* (2023)
- Ouyang et al., *Training language models to follow instructions with human feedback* (InstructGPT, 2022); Schulman et al., *PPO* (2017)
- Dong et al., *RAFT: Reward rAnked FineTuning* (2023) and Yuan et al., *Scaling relationship on learning mathematical reasoning* (rejection-sampling fine-tuning)
- Pal et al., *Smaug: Fixing Failure Modes of Preference Optimisation with DPO-Positive* (2024) and Pang et al., *Iterative Reasoning Preference Optimization* (2024): the falling chosen-likelihood problem and the NLL-anchor fix
- Meng et al., *SimPO* (2024); Azar et al., *IPO* (2023); Ethayarajh et al., *KTO* (2024)
- Xu et al., *Is DPO Superior to PPO for LLM Alignment?* (2024)
- Gao et al., *Scaling Laws for Reward Model Overoptimization* (2022); Skalse et al., *Defining and Characterizing Reward Hacking* (2022)
- Shao et al., *DeepSeekMath / GRPO* (2024); Yue et al., *Does RL Really Incentivize Reasoning Capacity Beyond the Base Model?* (2025): sharpening vs. expansion

**Next:** 09, online RL with a verifiable reward (GRPO/RLOO-style), the on-policy alternative to the fixed pairs used here.
