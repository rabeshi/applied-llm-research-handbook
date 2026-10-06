# 5. Fine-tuning, post-training and reinforcement learning

## 5.1 Pre-training

A base autoregressive LM learns next-token prediction over very large
corpora.

Simplified:
`internet/books/code/etc. → token sequences → predict next token → update weights`

Pre-training builds broad representations and capabilities.

## 5.2 Fine-tuning

Fine-tuning continues optimization from pretrained weights on a smaller
targeted dataset.

Use it when prompting alone is insufficient and you have suitable
data/resources.

## 5.3 Supervised Fine-Tuning (SFT)

SFT trains on desired input-output demonstrations.

Example: Input: `Summarize: <clinical note>` Target:
`<high-quality summary>`

The model is optimized to assign higher probability to target responses.

### Experiment

-   base/instruction model;
-   500 curated examples;
-   held-out 200 examples;
-   compare before/after on task score and unrelated capability checks.

Always evaluate regressions, not only target-task improvement.

## 5.4 LoRA / parameter-efficient fine-tuning

Instead of updating every weight, LoRA learns low-rank adapter updates
in selected layers. It can substantially reduce trainable parameters and
memory requirements.

Questions to test: - adapter rank; - target modules; - data size; -
learning rate; - domain shift.

QLoRA combines quantized base weights with trainable adapters, making
adaptation of larger models more memory-efficient.

## 5.5 Post-training

Post-training is the broader stage after pre-training used to shape
behavior.

It may include: - SFT; - preference optimization; - reward modeling; -
reinforcement learning; - safety tuning; - distillation.

Fine-tuning is therefore a method; post-training is a broader
phase/category.

## 5.6 Preference data

A preference example often contains: - prompt; - chosen/preferred
response; - rejected/non-preferred response.

Example: Prompt: "Explain hypertension to a patient." Chosen: accurate,
clear, appropriately cautious. Rejected: jargon-heavy and makes
unsupported treatment claims.

## 5.7 Reward model

A reward model learns a scalar preference signal, conceptually:

`r(prompt, response) → score`

It can be trained from pairwise human preferences. The challenge: the
learned reward is only a proxy for what people actually want.

## 5.8 Reinforcement learning fundamentals

RL has: - **agent**: learner/decision maker; - **environment**: world it
interacts with; - **state/observation**: information available; -
**action**: choice; - **reward**: feedback; - **policy π(a\|s)**: action
distribution; - **trajectory**: sequence of interactions; - **return**:
accumulated future reward.

Goal: learn a policy maximizing expected return.

### Tiny bandit

Three buttons have unknown reward probabilities.

At each round: 1. choose button; 2. observe reward 0/1; 3. update
estimates; 4. balance exploration vs exploitation.

This teaches reward optimization without LLM complexity.

## 5.9 Q-learning intuition

For discrete settings, Q(s,a) estimates expected return from taking
action `a` in state `s` and behaving well afterward.

A classic update:

`Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]`

-   α: learning rate;
-   γ: discount factor.

Do this in GridWorld before studying RLHF.

## 5.10 Policy gradients

Instead of estimating action values only, directly optimize a
parameterized policy.

REINFORCE intuition: `∇J(θ) ≈ Σ_t ∇ log π_θ(a_t|s_t) * return`

Actions producing higher-than-expected reward should become more likely.

Modern LLM RL uses more sophisticated objectives and variance/control
techniques, but this is the conceptual bridge.

## 5.11 RLHF

A common conceptual RLHF pipeline:

`pretrained model → SFT → human preference data → reward/preference signal → RL optimization → evaluation`

Historically PPO has been prominent, but "RLHF" is broader than one
optimizer.

Key risks: - reward hacking; - overoptimization; - preference annotator
bias; - capability regressions; - verbosity/style being rewarded instead
of correctness.

## 5.12 DPO

Direct Preference Optimization uses preference pairs to optimize a
policy relative to a reference without separately running the classic
reward-model-plus-RL loop.

Conceptually:
`(prompt, chosen, rejected) → increase relative preference for chosen`

DPO is post-training/preference optimization; it is often discussed
alongside RLHF, but it is not the same training procedure as online RL
with an environment.

## 5.13 RL with verifiable rewards / RLVR

For tasks with mechanically checkable outcomes---e.g., math answers or
code tests---reward can come from a verifier.

Example: - model generates solution; - checker verifies final numeric
answer; - correct = reward 1, incorrect = 0.

This avoids some ambiguity of subjective human preference, but can still
reward shortcuts or exploit weaknesses in the verifier.

## 5.14 GRPO

Group Relative Policy Optimization is an online policy-optimization
approach used in modern reasoning-model post-training. A useful
intuition is that multiple candidate completions are sampled for a
prompt and their rewards are compared within the group to form a
relative learning signal.

Do not learn it only as an acronym. Implement a toy group-relative
update or run a small TRL example after understanding policy gradients
and reward design.

## 5.15 Recommended learning sequence

`bandit → GridWorld/Q-learning → policy gradients → preference data → reward models → SFT → DPO → online LLM RL/RLVR`

This order prevents RLHF terminology from becoming memorized vocabulary
without intuition.
