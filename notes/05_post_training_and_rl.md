# 5. Fine-tuning, post-training and reinforcement learning

The worked examples use toy numbers and fictional scenarios unless stated otherwise. They illustrate the method; they are not measured results.

## 5.1 Pre-training

A base autoregressive LM learns next-token prediction over very large
corpora.

Simplified:
`internet/books/code/etc. → token sequences → predict next token → update weights`

Pre-training builds broad representations and capabilities.

### Worked example

For the token sequence "the cat sleeps," next-token training uses contexts such as "the" to predict "cat," then "the cat" to predict "sleeps." Training can compute losses for multiple positions in parallel with causal masking; generation normally appends tokens sequentially. Predicting text well does not automatically teach compliance with user instructions.

## 5.2 Fine-tuning

Fine-tuning continues optimization from pretrained weights on a smaller
targeted dataset.

Use it when prompting alone is insufficient and you have suitable
data/resources.

### Worked example

Suppose a support assistant must produce a consistent category schema. First establish a prompting baseline. If it still fails, fine-tune on curated input/label pairs, using separate validation and test conversations. Fine-tuning can improve the target behavior while hurting other tasks, so reserve regression evaluations.

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

### Worked example

A training pair might be input "I cannot sign in" and target `{"category": "access"}`. Response-token cross-entropy increases the likelihood of the desired output. In instruction SFT, implementations often mask prompt tokens from the loss; verify that preprocessing and chat formatting match your intended objective.

## 5.4 LoRA / parameter-efficient fine-tuning

Instead of updating every weight, LoRA learns low-rank adapter updates
in selected layers. It can substantially reduce trainable parameters and
memory requirements.

Questions to test:

- adapter rank;
- target modules;
- data size;
- learning rate;
- domain shift.

QLoRA combines quantized base weights with trainable adapters, making
adaptation of larger models more memory-efficient.

### Worked example

For a toy `100 × 100` weight matrix, full tuning changes 10,000 parameters. A rank-4 update `ΔW = B A`, with `B` shaped `100 × 4` and `A` shaped `4 × 100`, has 800 trainable parameters. This is 8% for this matrix, not an estimate for a whole model.

The base matrix is commonly frozen. Quantizing it reduces storage, but training still needs memory for activations and adapter optimization. See the original [LoRA paper](https://arxiv.org/abs/2106.09685) and [QLoRA paper](https://arxiv.org/abs/2305.14314).

## 5.5 Post-training

Post-training is the broader stage after pre-training used to shape
behavior.

It may include:

- SFT;
- preference optimization;
- reward modeling;
- reinforcement learning;
- safety tuning;
- distillation.

Fine-tuning is therefore a method; post-training is a broader
phase/category.

### Worked example

A model can undergo SFT for task formatting, then preference optimization for response quality, then evaluation for regressions. These are stages in a post-training workflow; every model need not use all of them. Distillation trains a student from teacher outputs or distributions and may occur during post-training.

## 5.6 Preference data

A preference example often contains:

- prompt;
- chosen/preferred response;
- rejected/non-preferred response.

Example: Prompt: "Explain hypertension to a patient." Chosen: accurate,
clear, appropriately cautious. Rejected: jargon-heavy and makes
unsupported treatment claims.

### Worked example

For "Explain a password reset," response A clearly describes the approved reset flow; response B invents a shortcut. Label A chosen and B rejected under a correctness-and-policy rubric. Record ties, uncertainty, and annotator disagreement instead of forcing arbitrary preferences. Preference labels express the rubric and annotator population.

## 5.7 Reward model

A reward model learns a scalar preference signal, conceptually:

`r(prompt, response) → score`

It can be trained from pairwise human preferences. The challenge: the
learned reward is only a proxy for what people actually want.

### Worked example

If a reward model scores A at 2 and B at 0, a common pairwise model gives `P(A preferred) = sigmoid(2 - 0) ≈ 0.881`. The scores are not percentages, and their absolute values need not have a universal meaning. Check whether verbose but wrong responses receive inflated reward.

## 5.8 Reinforcement learning fundamentals

RL has:

- **agent**: learner/decision maker;
- **environment**: world it interacts with;
- **state/observation**: information available;
- **action**: choice;
- **reward**: feedback;
- **policy π(a\|s)**: action distribution;
- **trajectory**: sequence of interactions;
- **return**: accumulated future reward.

Goal: learn a policy maximizing expected return.

### Tiny bandit

Three buttons have unknown reward probabilities.

At each round:

1. choose button;
2. observe reward 0/1;
3. update estimates;
4. balance exploration vs exploitation.

This teaches reward optimization without LLM complexity.

### Worked example

Consider a two-step episode with rewards 1 and 2 and discount `γ = 0.9`. The initial return is `1 + 0.9 × 2 = 2.8`. A policy chooses actions given observations; a trajectory records the resulting sequence. Observations may reveal only part of the environment's true state.

In a bandit, action A has estimated reward 0.6 and B has 0.5. Exploitation chooses A; exploration sometimes tries B to improve estimates. A short-term exploratory loss can provide useful information.

## 5.9 Q-learning intuition

For discrete settings, Q(s,a) estimates expected return from taking
action `a` in state `s` and following the specified policy afterward. The optimal value
function Q* instead assumes optimal future behavior; Q-learning targets it.

A classic update:

`Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]`

-   α: learning rate;
-   γ: discount factor.

Do this in GridWorld before studying RLHF.

### Worked example

Let old `Q(s,a) = 2`, reward `r = 1`, best next-state Q-value `= 3`, learning rate `α = 0.5`, and discount `γ = 0.9`. The target is `1 + 0.9 × 3 = 3.7`; the update gives `2 + 0.5 × (3.7 - 2) = 2.85`. For a terminal next state, the bootstrap term is zero. Q-learning is an off-policy value-learning method; it is not the standard direct training recipe for an LLM.

## 5.10 Policy gradients

Instead of estimating action values only, directly optimize a
parameterized policy.

REINFORCE intuition: `∇J(θ) ≈ Σ_t ∇ log π_θ(a_t|s_t) * return`

Actions producing higher-than-expected reward should become more likely.

Modern LLM RL uses more sophisticated objectives and variance/control
techniques, but this is the conceptual bridge.

### Worked example

If a sampled action gets return 3 and a baseline predicts 2, its advantage is `+1`; gradient ascent encourages it. Return 1 against the same baseline yields advantage `-1`, discouraging it. A baseline reduces variance. This describes a learning signal, not a guarantee that every sampled action becomes better after one optimizer step.

## 5.11 RLHF

A common conceptual RLHF pipeline:

`pretrained model → SFT → human preference data → reward/preference signal → RL optimization → evaluation`

Historically PPO has been prominent, but "RLHF" is broader than one
optimizer.

Key risks:

- reward hacking;
- overoptimization;
- preference annotator bias;
- capability regressions;
- verbosity/style being rewarded instead of correctness.

### Worked example

Imagine a preference-trained reward model favors accurate concise answers. RL samples outputs, scores them, and updates the policy; a reference-policy penalty can limit drift. If reward improves while held-out correctness falls, the policy may be exploiting the proxy. Evaluate independent task and safety metrics. See [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) for one concrete pipeline.

## 5.12 DPO

Direct Preference Optimization uses preference pairs to optimize a
policy relative to a reference without separately running the classic
reward-model-plus-RL loop.

Conceptually:
`(prompt, chosen, rejected) → increase relative preference for chosen`

DPO is post-training/preference optimization; it is often discussed
alongside RLHF, but it is not the same training procedure as online RL
with an environment.

### Worked example

For one preference pair, suppose the reference assigns chosen/rejected probabilities `0.2/0.1` and the policy assigns `0.4/0.1`. The log-ratio margin is `ln(0.4/0.2) - ln(0.1/0.1) = ln(2)`. DPO's preference loss is `-ln(sigmoid(β × margin))`; `β` controls the scaling. Real response probabilities are products of conditional token probabilities, usually computed as sums of log probabilities.

The objective compares policy changes relative to the reference, rather than merely copying chosen responses. See the original [DPO paper](https://arxiv.org/abs/2305.18290).

## 5.13 RL with verifiable rewards / RLVR

For tasks with mechanically checkable outcomes---e.g., math answers or
code tests---reward can come from a verifier.

Example:

- model generates solution;
- checker verifies final numeric answer;
- correct = reward 1, incorrect = 0.

This avoids some ambiguity of subjective human preference, but can still
reward shortcuts or exploit weaknesses in the verifier.

### Worked example

For a toy sorting task, award reward 1 only when outputs are sorted and preserve the input elements. A verifier checking just "is sorted" would incorrectly reward an empty list. Use held-out and adversarial tests to audit the checker; passing a finite test suite does not prove arbitrary-program correctness.

## 5.14 GRPO

Group Relative Policy Optimization is an online policy-optimization
approach used in modern reasoning-model post-training. A useful
intuition is that multiple candidate completions are sampled for a
prompt and their rewards are compared within the group to form a
relative learning signal.

Do not learn it only as an acronym. Implement a toy group-relative
update or run a small TRL example after understanding policy gradients
and reward design.

### Worked example

For four completions with rewards `[1, 0, 1, 0]`, the group mean is `0.5` and population standard deviation is `0.5`. Standardized relative advantages are approximately `[1, -1, 1, -1]` when using `(reward - mean)/(std + ε)`. If rewards are identical, this centered reward signal is zero.

This illustrates advantage construction only. GRPO also includes a policy objective with probability ratios and clipping; reference-policy regularization and normalization details depend on the formulation. See [DeepSeekMath](https://arxiv.org/abs/2402.03300), which introduced GRPO.

## 5.15 Recommended learning sequence

`bandit → GridWorld/Q-learning → policy gradients → preference data → reward models → SFT → DPO → online LLM RL/RLVR`

This order prevents RLHF terminology from becoming memorized vocabulary
without intuition.

### Worked example

Use concrete milestones: estimate arm rewards in a bandit; update a GridWorld Q-table; compute one policy-gradient advantage; annotate ten preference pairs; calculate an SFT token loss; and calculate a DPO pair margin. Move to online LLM RL only when you can explain what the reward measures and how the policy update uses it.
