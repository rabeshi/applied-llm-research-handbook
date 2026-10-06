# 2. Prompting and inference

The worked examples use toy numbers and fictional scenarios unless stated otherwise. They illustrate the method; they are not measured results.

## 2.1 Zero-shot

**Zero-shot** means asking the model to perform a task without giving
labeled demonstrations in the prompt.

Example:

`Classify the sentiment as positive, negative, or neutral: "The interface is attractive but painfully slow."`

No worked classification examples were supplied.

### Worked example

Make the task unambiguous before measuring it:

```text
Classify the overall sentiment as positive, negative, neutral, or mixed.
Use mixed when the text contains both clear praise and criticism.
Return only the label.
Text: The interface is attractive but painfully slow.
```

Under this rule, the intended label is `mixed`. If your dataset has only three classes, define how mixed sentiment should be labeled rather than treating disagreement as an obvious model error.

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

### Worked example

Use the same instruction and target input across conditions. Add one labeled demonstration for one-shot, or three for three-shot. Place examples inside clear boundaries, and never draw them from the test set. Demonstration order and class balance can change results even when the example count is unchanged.

## 2.3 Controlled experiment: zero vs few-shot

Dataset: 200 held-out sentiment examples.

Conditions: A. zero-shot instruction; B. 1-shot; C. 3-shot; D. 5-shot.

Keep model, decoding and test examples fixed.

Measure:

- accuracy / macro-F1;
- tokens per request;
- latency/cost;
- per-class errors.

Repeat with several randomly selected demonstration sets. Otherwise you
may accidentally conclude "few-shot works" when one unusually good set
of examples caused the gain.

### Worked example

Illustrative results: zero-shot gets 160/200 correct (80%); three-shot gets 168/200 (84%). The gain is **4 percentage points**, not 4% relative improvement; relative improvement is `4/80 = 5%`. Repeat demonstration selection and compute a paired uncertainty interval before claiming a dependable gain. Track the added tokens per request alongside the score.

## 2.4 Prompt components

A useful decomposition:

- role/system instruction;
- task;
- definitions/decision rules;
- demonstrations;
- input;
- output schema;
- constraints.

For evaluation, version prompts exactly as you would version code.

### Worked example

For invoice extraction, specify the task (extract total), definition (amount payable after tax), input boundary (invoice text), and schema (`{"total": number, "currency": string}`). Record the complete prompt under a version such as `invoice_v1`. If you change the decision rule, give the prompt a new version so results remain attributable.

## 2.5 Structured outputs

Instead of free text, request a schema such as:

`{"label": "positive", "confidence": 0.82}`

Then validate the structure programmatically. Structured output makes
scoring and downstream use easier, but a self-reported confidence is not
automatically calibrated probability.

### Worked example

An output can be syntactically valid JSON and still contain an invalid label:

```python
import json

def parse_label(raw):
    obj = json.loads(raw)
    allowed = {"positive", "negative", "neutral", "mixed"}
    if not isinstance(obj, dict) or obj.get("label") not in allowed:
        raise ValueError("Invalid sentiment label")
    return obj["label"]

print(parse_label('{"label": "mixed"}'))
```

Define how parse failures and retries affect scoring before running the evaluation. Report first-attempt format compliance separately from correctness.

## 2.6 Temperature

Temperature rescales logits before softmax:

`P(i) ∝ exp(z_i / T)`

-   lower T: sharper distribution, usually more deterministic;
-   higher T: flatter distribution, usually more diverse.

Toy logits `[2,1,0]`:

- low T exaggerates differences;
- high T reduces differences.

For factual benchmark evaluation, deterministic or low-variance settings
are often useful. For creativity/diversity research, sampling is itself
an experimental variable.

### Worked example

For logits `[2, 1, 0]`, temperatures `0.5`, `1`, and `2` give approximately `[0.867, 0.117, 0.016]`, `[0.665, 0.245, 0.090]`, and `[0.506, 0.307, 0.186]`. Temperature changes relative probabilities before sampling. `T = 0` cannot be inserted into the division formula; systems commonly handle it as a greedy setting. Greedy decoding still does not guarantee identical outputs across every serving environment.

## 2.7 Greedy, top-k and top-p

**Greedy:** choose highest-probability token each step.

**Top-k:** restrict sampling to k highest-probability candidates.

**Top-p / nucleus:** take the smallest high-probability set whose
cumulative probability reaches approximately p, then sample from it.

These affect generation behavior; do not compare models with materially
different decoding settings and attribute all differences to the models.

### Worked example

Start with probabilities `[0.50, 0.30, 0.15, 0.05]`. Greedy selects the first token. Top-k with `k = 2` keeps the first two and renormalizes them to `[0.625, 0.375]`. Top-p with `p = 0.7` also keeps the first two because their cumulative mass reaches `0.8`; it does not keep exactly 70% of tokens. Keep filtering order fixed when combining temperature and truncation.

## 2.8 Repeated sampling

One output can hide stochastic instability.

For each prompt, sample 10--30 responses and measure:

- answer agreement;
- accuracy distribution;
- refusal frequency;
- format compliance;
- semantic diversity.

A model that scores 90% once but varies wildly across runs may be less
reliable than a stable 88% model.

### Worked example

If a prompt produces labels A, A, B, A, B across five runs, modal agreement is `3/5 = 60%`. Agreement does not establish correctness: a model can repeat the same wrong answer. Repeated completions from one item also do not replace evaluating a representative set of distinct items.

## 2.9 Prompt sensitivity

Paraphrase the same question five ways. Ideally, semantically equivalent
prompts should not cause large task-irrelevant performance swings.

This becomes a robustness benchmark:
`robustness gap = best paraphrase score - worst paraphrase score`

### Worked example

Suppose equivalent prompts score 82%, 81%, 79%, 80%, and 74% on the same examples. The robustness gap is 8 percentage points. Inspect whether the weakest paraphrase accidentally changes the task or output format before attributing the gap to model sensitivity. Choose prompt variants on development data, not by searching the final test for the best wording.

## 2.10 Reasoning prompts

Terms such as "chain-of-thought" are often used for prompts that elicit intermediate reasoning. For research, distinguish:

- answer accuracy;
- generated rationale quality;
- faithfulness of rationale to actual model computation.

A plausible explanation is not proof that the model used that reasoning
internally.

A safe evaluation pattern is to score the final answer and, when needed,
request concise supporting evidence or verifiable intermediate outputs
rather than assuming free-form reasoning is faithful.

### Worked example

For `3 items × $12 each`, request the quantity, unit price, and final total as checkable outputs. A verifier can check `3 × 12 = 36`. Compare final-answer accuracy with and without these intermediate outputs while measuring extra tokens. Correct arithmetic in an explanation supports that calculation; it does not reveal the model's internal computation.
