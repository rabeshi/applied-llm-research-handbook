# 2. Prompting and inference

## 2.1 Zero-shot

**Zero-shot** means asking the model to perform a task without giving
labeled demonstrations in the prompt.

Example:

`Classify the sentiment as positive, negative, or neutral: "The interface is attractive but painfully slow."`

No worked classification examples were supplied.

## 2.2 One-shot and few-shot

**One-shot:** one demonstration.

**Few-shot:** a small number of demonstrations in context.

Example:

`Text: "Amazing battery." Label: positive`
`Text: "It crashes constantly." Label: negative`
`Text: "It arrived Tuesday." Label: neutral`
`Text: "Beautiful screen, terrible keyboard." Label: ?`

This is **in-context learning**: the model's weights are not updated.

### Critical distinction

Few-shot prompting ≠ fine-tuning.

-   Few-shot: examples live in the prompt; weights unchanged.
-   Fine-tuning: optimization updates model parameters (or adapter
    parameters).

## 2.3 Controlled experiment: zero vs few-shot

Dataset: 200 held-out sentiment examples.

Conditions: A. zero-shot instruction; B. 1-shot; C. 3-shot; D. 5-shot.

Keep model, decoding and test examples fixed.

Measure: - accuracy / macro-F1; - tokens per request; - latency/cost; -
per-class errors.

Repeat with several randomly selected demonstration sets. Otherwise you
may accidentally conclude "few-shot works" when one unusually good set
of examples caused the gain.

## 2.4 Prompt components

A useful decomposition: - role/system instruction; - task; -
definitions/decision rules; - demonstrations; - input; - output
schema; - constraints.

For evaluation, version prompts exactly as you would version code.

## 2.5 Structured outputs

Instead of free text, request a schema such as:

`{"label": "positive", "confidence": 0.82}`

Then validate the structure programmatically. Structured output makes
scoring and downstream use easier, but a self-reported confidence is not
automatically calibrated probability.

## 2.6 Temperature

Temperature rescales logits before softmax:

`P(i) ∝ exp(z_i / T)`

-   lower T: sharper distribution, usually more deterministic;
-   higher T: flatter distribution, usually more diverse.

Toy logits `[2,1,0]`: - low T exaggerates differences; - high T reduces
differences.

For factual benchmark evaluation, deterministic or low-variance settings
are often useful. For creativity/diversity research, sampling is itself
an experimental variable.

## 2.7 Greedy, top-k and top-p

**Greedy:** choose highest-probability token each step.

**Top-k:** restrict sampling to k highest-probability candidates.

**Top-p / nucleus:** take the smallest high-probability set whose
cumulative probability reaches approximately p, then sample from it.

These affect generation behavior; do not compare models with materially
different decoding settings and attribute all differences to the models.

## 2.8 Repeated sampling

One output can hide stochastic instability.

For each prompt, sample 10--30 responses and measure: - answer
agreement; - accuracy distribution; - refusal frequency; - format
compliance; - semantic diversity.

A model that scores 90% once but varies wildly across runs may be less
reliable than a stable 88% model.

## 2.9 Prompt sensitivity

Paraphrase the same question five ways. Ideally, semantically equivalent
prompts should not cause large task-irrelevant performance swings.

This becomes a robustness benchmark:
`robustness gap = best paraphrase score - worst paraphrase score`

## 2.10 Reasoning prompts

Terms such as "chain-of-thought" are often used for prompts that elicit
intermediate reasoning. For research, distinguish: - answer accuracy; -
generated rationale quality; - faithfulness of rationale to actual model
computation.

A plausible explanation is not proof that the model used that reasoning
internally.

A safe evaluation pattern is to score the final answer and, when needed,
request concise supporting evidence or verifiable intermediate outputs
rather than assuming free-form reasoning is faithful.
