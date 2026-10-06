# 3. Evaluation and benchmarking

The worked examples use toy numbers and fictional scenarios unless stated otherwise. They illustrate the method; they are not measured results.

## 3.1 Evaluation

Evaluation asks: **How well does the system satisfy a defined
objective?**

A complete evaluation specifies:

- task;
- population/domain;
- dataset;
- model version;
- prompt;
- decoding;
- tools/retrieval;
- metric;
- scorer;
- uncertainty;
- failure analysis.

### Worked example

A concrete objective is "route English support tickets to billing, access, or technical support." Specify the customer population, labeling rules, held-out conversations, frozen prompt, and macro-F1 scorer. Add a constraint such as a maximum invalid-output rate so a score cannot conceal unusable responses.

## 3.2 Common metrics

### Classification

Accuracy: `correct / total`

Precision: `TP / (TP + FP)`

Recall: `TP / (TP + FN)`

F1: `2PR / (P + R)`

Use macro-F1 when class imbalance makes equal treatment of classes
important.

### QA

-   exact match;
-   token/string F1;
-   multiple-choice accuracy;
-   evidence correctness;
-   human/LLM rubric scoring for open-ended answers.

### Generation

Automatic overlap metrics (BLEU/ROUGE) can be useful in constrained
settings but may correlate poorly with factuality or usefulness for
open-ended generation. Consider task-specific criteria, semantic metrics
and human evaluation.

### Worked example

For a positive-class detector with `TP = 8`, `FP = 2`, `FN = 4`, and `TN = 86`:

| Metric | Calculation | Result |
| --- | --- | --- |
| Accuracy | `(8 + 86) / 100` | 94% |
| Precision | `8 / (8 + 2)` | 80% |
| Recall | `8 / (8 + 4)` | 66.7% |
| F1 | `2 × 8 / (2 × 8 + 2 + 4)` | 72.7% |

An always-negative baseline reaches 88% accuracy while finding no positives. Macro-F1 averages class-specific F1 values equally; it is not the F1 computed from aggregate precision and recall. Specify zero-denominator handling.

For QA, "Paris" and "Paris, France" can disagree under strict exact match yet answer the same question. Freeze normalization rules. For summaries, shared wording can yield high overlap scores while a changed date makes the answer wrong.

## 3.3 Benchmark

A benchmark is a standardized task/dataset/evaluation protocol enabling
systematic comparison.

A good benchmark needs more than questions:

- construct definition;
- inclusion/exclusion criteria;
- categories;
- difficulty;
- data provenance;
- answer key/rubric;
- evaluation script;
- baselines;
- documentation;
- leakage/contamination discussion.

### Worked example

To benchmark invoice extraction, include typed and scanned invoices, several currencies, and ambiguous totals. Define what "total" means, publish a scorer, and include an OCR-plus-rules baseline. A dataset of only clean templates measures a narrower construct than general invoice understanding.

## 3.4 Capability vs safety

Capability: **Can the system do X?**

Safety: **Does the system do X without unacceptable harmful/unreliable
behavior?**

Example medical QA:

- capability: diagnosis-question accuracy;
- safety: unsupported treatment recommendations, failure to express uncertainty, contradiction of supplied allergies.

### Worked example

For a fictional document assistant, score whether it answers a lookup question correctly and whether it invents an authorization absent from the source. A correct lookup does not excuse an unsupported action. Treat capability and constraint adherence as separate outcomes; the examples here are evaluation exercises, not domain advice.

## 3.5 Benchmark design worked example

Question: "Do few-shot demonstrations improve small-LM clinical
abbreviation expansion?"

1.  Define target population: de-identified clinical-style sentences.
2.  Define task: expand one marked abbreviation.
3.  Build 500 examples.
4.  Stratify by common/rare and ambiguous/unambiguous.
5.  Hold out test set.
6.  Compare zero-shot vs 3-shot.
7.  Use exact match + clinically acceptable synonym rubric.
8.  Bootstrap confidence intervals.
9.  Analyze ambiguity-related failures.
10. Repeat across 3 models.
11. Report prompt/token cost.

Now the work answers a research question rather than merely producing a
leaderboard.

### Worked example

A non-clinical version asks whether demonstrations improve business abbreviation expansion. Mark `PO` in "The supplier received the PO" and use "purchase order" as the target. Include ambiguous contexts such as "send it to the PO box." Split by source document and define acceptable expansions before comparing zero-shot with three-shot.

## 3.6 Baselines

Possible baselines:

- majority/random;
- simple lexical/statistical method;
- classical ML;
- smaller model;
- previous benchmark result;
- zero-shot baseline;
- no-RAG baseline;
- no-fine-tuning baseline.

A sophisticated system without a sensible baseline tells you little
about what produced the gain.

### Worked example

If 70% of tickets are billing issues, always predicting billing yields 70% accuracy. A proposed model scoring 72% has only a 2-point improvement over that baseline. Compare per-class recall and macro-F1 to determine whether the model helps minority categories.

## 3.7 Ablations

An ablation removes or changes one component.

RAG example:

- full system;
- no reranker;
- smaller chunk size;
- no metadata;
- no retrieval.

If performance falls only when retrieval is removed, that supports a
claim about retrieval's contribution.

### Worked example

Illustrative QA accuracy: full RAG 84%, no reranker 83%, no retrieval 71%. Retrieval appears to contribute more than reranking under these conditions. Hold the other components fixed and quantify paired differences. Component interactions mean a single ablation does not establish a universal causal effect.

## 3.8 Data leakage and contamination

Leakage occurs when information unavailable at genuine prediction time
improperly reaches training/development.

Benchmark contamination is especially important when evaluation items or
near-duplicates may have appeared in pretraining or post-training.

Mitigations:

- new/private test sets;
- time-based splits;
- deduplication;
- canary strings where appropriate;
- held-out generated variants;
- explicit limitations.

### Worked example

Two paraphrases of the same support conversation appear in train and test. The model may recognize the conversation rather than generalize to a new one. Deduplicate and group by conversation before splitting. New synthetic items can still imitate public benchmark patterns; "newly generated" does not prove absence of contamination.

## 3.9 LLM-as-a-judge

An LLM can score outputs using a rubric, but the judge is itself a model
and can be biased.

Validate judge scores against human annotations. Test:

- position/order effects;
- verbosity preference;
- self-preference;
- prompt sensitivity;
- inter-rater agreement.

### Worked example

Show the judge answer A followed by B, then reverse the order. If its preferred answer changes frequently, order bias may distort rankings. Use a concrete rubric such as correctness, source support, and completeness; compare judge decisions with blinded human labels on a representative subset.

## 3.10 Confidence intervals and significance

Do not report only `Model A = 81%, Model B = 80%`.

Ask whether the difference is stable.

Bootstrap:

1. sample benchmark items with replacement;
2. recompute score difference;
3. repeat many times;
4. inspect confidence interval.

For paired model outputs on the same classification items, paired tests
are usually preferable to pretending samples are independent.

### Worked example

Evaluate A and B on the same items. On each bootstrap repetition, resample item indices once and apply them to both systems:

```python
import random

a = [1, 1, 0, 1, 0, 1, 1, 0]
b = [1, 0, 0, 1, 0, 1, 0, 0]
rng = random.Random(7)
differences = []
for _ in range(2000):
    indices = [rng.randrange(len(a)) for _ in a]
    differences.append(sum(a[i] - b[i] for i in indices) / len(a))
differences.sort()
print("Observed difference:", sum(x - y for x, y in zip(a, b)) / len(a))
print("Approximate 95% percentile interval:", differences[50], differences[1949])
```

Eight items demonstrate mechanics, not adequate evidence for a general claim. If items share a document or user, resample independent groups rather than treating every item as independent. An interval crossing zero does not prove equivalence.

## 3.11 Error taxonomy

Create categories before or during blinded error review:

- knowledge error;
- reasoning error;
- instruction failure;
- extraction error;
- hallucination;
- refusal;
- formatting;
- retrieval failure;
- tool failure.

Quantifying *why* systems fail is often more useful than one aggregate
score.

### Worked example

A response can contain both a wrong retrieved document and a fabricated citation. Decide whether categories are mutually exclusive or multi-label. If 12 of 30 failed answers have retrieval errors, report both `12/30` among failures and `12/N` among all evaluated answers; the denominators answer different questions.
