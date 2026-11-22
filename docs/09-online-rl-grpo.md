# 09 — Online RL with a verifiable reward (GRPO / RLOO)

## Goal
Step 08's DPO trained on pairs sampled **once** from the SFT model and then frozen. Online RL instead samples **fresh responses from the current policy at every step**, scores them with the verifier, and pushes probability toward what scored well. Because the policy is always generating its own data, it can *discover* successes it did not produce before, which is the capability fixed pairs could not give us (the keyword task, 0.8% pass rate). This step implements the loop and tests that claim. Code: `hedy/rl.py`; tests: `tests/test_rl.py`.

Start: the same step-07 full-fine-tuned model as step 08 (name pass@1 63.0%, keyword 0.7%), same verifier, same held-out evaluation, so results are directly comparable.

## 1. Theory

### The policy gradient
We want to maximize expected reward `J(θ) = E_{y∼π_θ(·|x)}[ R(x, y) ]`. The **log-derivative trick** gives

    ∇J = E[ R(x,y) · ∇ log π_θ(y|x) ]

Reading it: sample a response, and raise its log-probability in proportion to how good it was (REINFORCE, Williams 1992). For a sequence, `log π(y|x) = Σ_t log π(y_t | x, y_<t)`, so every token of a good response is pushed up. Nothing needs to be differentiable about the *reward*, only about the model: the verifier can be arbitrary code.

### Baselines: same expected gradient, far less noise
Subtracting any baseline `b` that doesn't depend on the sampled response, `E[(R − b) ∇ log π] = ∇J`, because `E[∇ log π] = 0`. The best baselines make `R − b` small and centred, so a good sample gets a positive weight and a below-average one a negative weight, even if all rewards are positive. Classic PPO learns a *value network* as `b`; for LLMs that is a second model as big as the policy. Group methods avoid it by sampling **K responses to the same prompt** and using the other samples as the baseline:

| method | advantage of sample *i* |
|---|---|
| **GRPO** (DeepSeekMath, 2024) | `(r_i − mean_group) / (std_group + ε)` |
| **RLOO** (Ahmadian et al. 2024) | `r_i − mean of the other K−1 samples` = `K/(K−1) · (r_i − mean)`; exactly unbiased (the baseline for *i* excludes *i*) |
| "mean" / Dr. GRPO | `r_i − mean_group` (no std scaling) |

Std-normalization rescales each prompt's gradient by `1/std`, which up-weights prompts where almost all samples agree and the few outliers have huge normalized advantage; Dr. GRPO (Liu et al. 2025) argues this, and the per-sequence token-mean below, are biases. We implement all three (`--adv`). `tests/test_rl.py::test_advantages` pins down the identities.

**Zero-variance groups carry no signal**: if all K samples get the same reward, every advantage is 0 and the gradient is exactly 0. If a prompt is too hard (0/K) or too easy (K/K), the compute spent sampling it is wasted. We log the fraction of groups with signal (`signal-groups`); newer methods (DAPO) resample to keep batches informative.

### From REINFORCE to the GRPO/PPO objective
- **Importance ratio and clipping.** PPO lets you reuse one batch for several gradient steps by weighting with `ratio = π_θ(token)/π_old(token)` and clipping it to `[1−ε, 1+ε]`: once a token's probability has moved by more than ε in the direction the advantage wants, its gradient is cut off (`test_clipping_zeroes_gradient...`). With **one update per batch** (`--inner_epochs 1`, our default) `ratio ≡ 1` and the objective *is* plain REINFORCE with a group baseline; clipping matters only when `inner_epochs > 1` (we log `clipfrac`).
- **KL penalty to a frozen reference.** Optimizing a proxy reward drifts the policy toward whatever the proxy over-rewards (step 08's name-stuffing). We subtract `β · KL(π ‖ π_ref)` per token. Estimating KL from samples of π with the raw log-ratio is noisy and can go negative; we use Schulman's **k3** estimator `e^d − d − 1` with `d = log π_ref − log π` (always ≥ 0, unbiased: tested against the analytic KL). `test_kl_penalty_pulls_back_toward_reference` checks the sign of its gradient.
- **Per-sequence token-mean.** The loss is averaged over each response's tokens, then over responses (original GRPO). This weights each *token* in a short response more than in a long one.

### On-policy sampling
The gradient formula is an expectation over samples **from π_θ**. So we sample at temperature 1 with no top-k (exactly the distribution whose log-probs we differentiate); sampling at T=0.7 would silently optimize a different objective. Evaluation uses T=0.7/top-k 40 as in steps 07–08, so training and evaluation temperatures differ, and a policy sharpened at T=1 looks even better at T=0.7.

### Credit assignment: the weakness of sequence-level rewards
A single reward arrives for the whole response, and **every token gets the same advantage**. If the reward depends on one token among ~70, the other 69 tokens receive the same push and add pure noise. My first unit test for the loop used exactly such a reward ("is the first token even?") and it *did not learn* in 25 steps; switching to a dense reward (fraction of tokens that are even) made the same loop learn quickly. The loop wasn't wrong; the credit signal was diluted. Practical consequence: sparse, late, single-token rewards need many more samples (or process rewards / better baselines).

### Where this sits next to what we've built
| | data | uses negatives? | sees new behaviour? | needs |
|---|---|---|---|---|
| SFT (07) | fixed demos | no | no | demonstrations |
| RFT (08) | model's own, filtered once | no | only what was sampled once | verifier |
| DPO (08) | pairs, sampled once | yes | no (fixed pairs) | verifier + pairs |
| **online RL (09)** | **fresh samples every step** | **yes (advantage < 0)** | **yes** | verifier + sampling compute |

## 2. Implementation (`hedy/rl.py`)
| piece | what it does |
|---|---|
| `sample_k` (from step 08) | K samples per prompt in one batched `generate`, T=1, no top-k; returns the sampled token ids (cut after `### End`) |
| verifier | step 07/08: reward 1 if the response finished **and** contains the required word, else 0 |
| `advantages` | `grpo` / `rloo` / `mean` |
| `token_logps` | `(B, Lr)` per-token log-probs of the response (+ mask), right-padded; agrees with the sequence log-prob of step 08 (tested) |
| `pg_loss` | PPO-clipped surrogate + `kl_coef · k3` KL to the frozen reference, per-sequence token-mean |
| `train_rl` | loop: sample → reward → advantage → drop zero-variance groups → update (optionally `inner_epochs` > 1 with clipping) |

Logged every few steps: mean reward on the *training* samples (free, on-policy), fraction of groups with signal, finished-rate, length, KL to the reference, clip fraction. Held-out evaluation at the end is identical to step 08 (150 name + 150 keyword prompts, K=4, bootstrap CIs over prompts), so the numbers below sit in the same table as DPO.

Seven unit tests: advantage identities; token log-probs match sequence log-probs; k3 is ≥0, zero at equality, and unbiased vs the analytic KL; one gradient step raises the rewarded response and lowers the other; clipping zeroes the gradient exactly when the ratio left the trust region on the side the advantage pushes; the KL gradient points back toward the reference; and an end-to-end toy run where a dense reward must rise.

## 3. Results

All runs start from `sft-full` (name pass@1 63.0%, keyword 0.7%), 32 prompts × K=8 per step, temperature-1 sampling, KL coefficient 0.02, full fine-tuning, one update per batch unless noted. Evaluation as in step 08 (name column: 150 held-out prompts × 4 samples). *mentions / pass* = how often the name appears in responses that contain it (real stories: 2.5). Δ bpb = forgetting on plain stories relative to the SFT model (1.190). *Final train KL/reward* = mean over the last 5 steps.

### 3.1 Name-following

| run | name pass@1 (95% CI) | pass@4 | mentions / pass | Δ bpb | SFT val loss | final train KL | final train reward |
|---|---|---|---|---|---|---|---|
| SFT start | 63.0% (57–69) | 82.7% | 2.71 | 0 | 1.616 | – | – |
| GRPO lr 2e-5, 50 steps | 77.0% (71–83) | 86.7% | 3.63 | +0.006 | 1.634 | 0.022 | 0.726 |
| GRPO lr 2e-5, 50 steps, seed 1 | 76.3% (70–82) | 85.3% | 3.69 | +0.005 | 1.635 | 0.024 | 0.723 |
| RLOO lr 2e-5, 50 steps | 76.8% (71–83) | 86.7% | 3.54 | +0.006 | 1.630 | 0.019 | 0.734 |
| GRPO lr 2e-5, **100** steps | 79.7% (74–85) | 86.7% | 3.96 | +0.015 | 1.650 | 0.053 | 0.782 |
| GRPO lr 2e-5, 4 inner epochs | 80.7% (75–86) | 86.0% | 4.11 | +0.035 | 1.694 | 0.095 | 0.810 |
| GRPO lr 5e-5 | 79.0% (73–85) | 84.0% | 4.02 | +0.022 | 1.676 | 0.109 | 0.790 |
| GRPO lr 5e-5, **no KL penalty** | 79.0% (73–85) | 85.3% | 4.09 | +0.022 | 1.685 | 0.084 | 0.798 |
| *(step 08)* RFT lr 2e-5 | 73.5% (68–79) | 86.7% | 2.71 | +0.004 | 1.623 | – | – |
| *(step 08)* DPO β=0.1 lr 2e-5 | 76.0% (70–82) | 85.3% | 4.20 | +0.026 | 1.684 | – | – |
| *(step 08)* DPO β=0.3 lr 2e-5 | 78.2% (72–84) | 86.0% | 3.70 | +0.014 | 1.647 | – | – |
| *(step 08)* DPO β=0.1 lr 2e-5 + NLL anchor | 77.3% (71–83) | 86.0% | 3.38 | +0.012 | 1.645 | – | – |

**1. Online RL works and is clean at the gentle setting.** 50 steps of GRPO at lr 2e-5 lift name pass@1 from 63% to **77%** (training reward 0.55 → 0.73) with almost no collateral damage: forgetting +0.006 bpb, SFT val loss +0.018, 100% of responses still finish. The same result at seed 1 (76.3%), and RLOO (76.8%), say the estimator choice and seed noise are small next to everything else.

**2. GRPO ≈ RLOO here.** With a binary reward and K=8, the std-normalization of GRPO and the leave-one-out baseline of RLOO give almost the same advantages up to a per-prompt rescaling; the outcomes (77.0 vs 76.8%, identical drift) agree. The difference between them shows up with continuous rewards and very small/large K.

**3. Compared with DPO and RFT (the point of the step).** At matched pass@1 (~77%), online RL forgets **less**: +0.006 bpb vs DPO's +0.012 to +0.026 (the best DPO variants: β=0.3 78.2%/+0.014, anchored 77.3%/+0.012), and RFT (73.5%, +0.004) is the only cheaper-on-drift alternative but ~3.5 points lower in pass@1 (CIs overlap, direction consistent). Sample budgets were comparable (RL 50×256 = 12.8k samples; DPO sampled 16k). In a standalone pilot at full machine speed RL took ~13 s/step (~11 min per 50 steps), vs ~4 min sampling + ~3 min training for DPO. So: online RL was the gentlest way to get DPO-level accuracy on this task, at similar compute; it is not a different league.

**4. It sharpens.** pass@4 is flat at 84–87% (the start: 82.7%) in every run, while pass@1 climbs from 63% toward it (77–81%). The model learned to be reliably what it could already sometimes be, as with DPO. The 100-step run shows the curve flattening: the training reward goes 0.60 → 0.67 → 0.70 → 0.72 → 0.72 → 0.72 → 0.75 → 0.79 → 0.81 → 0.80 per 10-step block.

**5. Signal starvation.** As the policy improves, more groups become "all K pass" and contribute no gradient: the fraction of groups with signal drops from 79% to 41% over 100 steps (to 37% in the 4-inner-epoch run). Success starves the learning signal. Practical fixes: filter or resample easy prompts, or harder prompts (curriculum); we did not.

**6. The same loophole as step 08, growing with optimization pressure.** Mentions per passing response rise with how far the policy moved (train KL): 3.63 (KL 0.022) → 3.96 (0.053) → 4.02 (0.109) → 4.11 (0.095) vs 2.7 for the start and 2.5 in real stories. "The name appears" is satisfied more safely by repeating it; every reward-optimizing method learned that, RL included. It's mild here because we stopped early.

**7. What the knobs did (and didn't).**
- *KL coefficient 0.02 did nothing visible.* With vs without it at lr 5e-5: 79.0% vs 79.0%, Δ bpb +0.022 both, KL 0.109 vs 0.084 (noise). At this scale 0.02 is too weak to constrain a 50-step run; it matters when runs are long or unstable (the keyword run below ran without it).
- *Clipping barely engaged.* Even with 4 inner epochs the clip fraction stayed ≤ 0.003; the policy moves too little per step at lr 2e-5 for the ratio to leave [0.8, 1.2]. Those 4 inner epochs reached 80.7% but forgot 6× more (+0.035 vs +0.006) than 1 epoch: reusing a batch is *more optimization pressure*, not free sample-efficiency.
- *Higher lr / longer / more epochs* all trade pass@1 (+2 to +4) against drift and repetition; none moved pass@4.

### 3.2 Can online RL discover keyword-following? (the real test of "online")

Keyword prompts start at a 0.8% pass rate, so only 6% of groups have any success (step 08's wall). Online sampling might find successes that fixed pairs missed. I used K=16, 16 prompts per step (same 256 samples/step) on keyword prompts only.

| run | train reward (start → peak → end) | groups with signal | final train KL | kw pass@1 | kw pass@4 | name pass@1 (side effect) | Δ bpb | SFT val loss |
|---|---|---|---|---|---|---|---|---|
| SFT start | – | – | – | 0.7% | 2.7% | 63.0% | 0 | 1.616 |
| lr 2e-5, KL 0.02, 50 steps | 0.009 → – → 0.011 | 8–20% | 0.005 | 1.0% | 4.0% | 60.3% | −0.001 | 1.620 |
| lr 1e-4, **no KL**, 100 steps | 0.011 → **0.050** (steps 70–80) → 0.015 | 10–17% | 0.553 | **6.7%** (CI 3–11) | 7.3% | **21.3%** | +0.078 | **1.922** |

**The gentle run learned nothing**: the reward stayed at ~1% and the policy barely moved (KL 0.005). The aggressive run did raise the training reward ~4.5× and the held-out keyword pass@1 to 6.7% (10× the start), then became unstable and fell back by step 100 (KL 0.55). The cost: **name-following collapsed from 63% to 21%**, held-out SFT loss 1.62 → 1.92, repetitive text (distinct-2 0.94 → 0.71).

**What it learned is not instruction-following; it is a template.** Sampling the final policy shows nearly every story is the same skeleton: *"Once there was a little girl was three years old and loved to explore the forest. She was three years old and …"*. Checking all 600 samples (150 prompts × 4): the policy "passes" 46 times, and **40 of those 46 (87%)** are for the four keywords **explore, three, years, forest**, exactly the words inside the template. The SFT model's 9 hits were spread across 8 different keywords (only 2 of the 9 are on words that appear in the template). The verifier ("the word appears") is satisfied for those few prompts *by coincidence with the template*, and the policy has found that a fixed, low-diversity text collects that small reward reliably. This is **specification gaming / reward hacking**, and it is a direct consequence of how we set the problem up:
- very few real successes (about 1% of samples: ~130 per 50 steps at the start) gave the optimizer almost nothing to learn from but coincidences;
- with a group of 16 and a lone success, GRPO's normalized advantage for that sample is ≈ +3.9, a big push on whatever that sample happened to be;
- no KL penalty and a high lr removed the brakes.

The lesson matches step 07 and step 08 from the other direction: **RL amplifies whatever correlates with reward in the samples it sees. If the model can't produce the target behaviour, the cheapest correlate wins**, and "mention a common word" is a much easier policy than "plan a story around a given word".

## 4. Practical lessons
- **Watch the training reward *and* the held-out behaviour.** The keyword run's training reward looked like progress (4.5×); only the held-out name-following (63 → 21%) and a look at the samples revealed collapse. Always sample from the policy yourself.
- **Monitor signal-groups.** Too low (hard task) and you learn only coincidences; falling (easy task) means learning is slowing for lack of contrast.
- **Keep the KL penalty meaningful** (or stop early) in long runs; at tiny scale a 0.02 coefficient is decorative.
- **Reusing a batch (inner epochs) is not free**: more drift, no better pass@4.
- **A verifier is a spec, and optimizers read the spec literally.** "The word appears" was gamed in step 08 (repeat the name) and in step 09 (a template of common words). Reward design, not the RL algorithm, is usually the bottleneck.
- **RL needs the base model to already succeed sometimes** (pass@k > 0 at meaningful k). Improve the SFT model, shape the task (curriculum), or give a denser/partial-credit reward first.

## 5. What this step doesn't do
- A learned or dense reward, process rewards, and partial credit (e.g. +0.5 for the keyword appearing, +0.5 for being a coherent story).
- Curriculum / dynamic sampling (drop all-pass and all-fail groups, resample), asymmetric clip ("clip-higher", DAPO), length-normalization fixes (Dr. GRPO's constant normalizer).
- Value-network PPO, off-policy replay, async actor/learner, multi-GPU rollouts.
- Many seeds: only the main configuration was repeated (76.3 vs 77.0); the 3–4 point differences between runs are within the evaluation intervals.

## 6. Exercises
1. Plot the 100-step training curve (reward, signal-groups, KL, mentions-per-pass) from `checkpoints/rl-*/results.json` (`history`). Where does progress per step slow, and why?
2. **Fix the loophole.** Reward `1` only if the name appears once or twice (or add `+distinct-2`), retrain, and see whether the mention inflation and the keyword-template collapse go away.
3. **Give the keyword task partial credit**: e.g. 0.3 for finishing, 0.7 for containing the word *and* the story having distinct-2 > 0.85. Does the template solution stop paying?
4. Use a curriculum for keywords: first train on *easy* keywords (those the SFT model passes most often) and move to harder ones. Does RL bootstrap?
5. Replace per-sequence token-mean with Dr. GRPO's constant normalizer (`sum over tokens / L_max`). Does the length drift in the 5e-5 run disappear?
6. Raise `kl_coef` to 0.1–1.0 on the aggressive keyword run. Can a strong KL tether prevent the collapse while still allowing some gain?
7. Implement dynamic sampling: keep resampling prompts until the batch has N groups with signal. Compare signal-per-sample and final pass@1.
8. Start RL from the **DPO** model (step 08) or the **RFT** model: does stacking help?
9. Compute a paired bootstrap on per-prompt pass rates between GRPO (50 steps) and DPO β=0.3. Is the drift difference the only real difference?

## 7. Further reading
- Williams, *Simple statistical gradient-following algorithms* (REINFORCE, 1992); Sutton et al., *Policy gradient methods* (2000)
- Schulman et al., *Proximal Policy Optimization* (2017); Schulman, *Approximating KL Divergence* (the k3 estimator)
- Shao et al., *DeepSeekMath* (GRPO, 2024); DeepSeek-AI, *DeepSeek-R1* (2025)
- Ahmadian et al., *Back to Basics: Revisiting REINFORCE Style Optimization for RLHF* (RLOO, 2024)
- Liu et al., *Understanding R1-Zero-Like Training (Dr. GRPO)* (2025); Yu et al., *DAPO* (2025)
- Yue et al., *Does RL Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?* (2025): sharpening vs expansion
- Amodei et al., *Concrete Problems in AI Safety* (2016) and Skalse et al., *Defining and Characterizing Reward Hacking* (2022): specification gaming
- Lambert et al., *Tulu 3* (2024): RL with verifiable rewards in a full post-training recipe

**Next:** 10, scaling: a bigger model and more data, and what you learn from scaling laws.
