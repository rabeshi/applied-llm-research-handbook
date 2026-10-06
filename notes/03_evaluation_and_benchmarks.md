# 3. Evaluation and benchmarking

## 3.1 Evaluation

Evaluation asks: **How well does the system satisfy a defined
objective?**

A complete evaluation specifies: - task; - population/domain; -
dataset; - model version; - prompt; - decoding; - tools/retrieval; -
metric; - scorer; - uncertainty; - failure analysis.

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

## 3.3 Benchmark

A benchmark is a standardized task/dataset/evaluation protocol enabling
systematic comparison.

A good benchmark needs more than questions: - construct definition; -
inclusion/exclusion criteria; - categories; - difficulty; - data
provenance; - answer key/rubric; - evaluation script; - baselines; -
documentation; - leakage/contamination discussion.

## 3.4 Capability vs safety

Capability: **Can the system do X?**

Safety: **Does the system do X without unacceptable harmful/unreliable
behavior?**

Example medical QA: - capability: diagnosis-question accuracy; - safety:
unsupported treatment recommendations, failure to express uncertainty,
contradiction of supplied allergies.

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

## 3.6 Baselines

Possible baselines: - majority/random; - simple lexical/statistical
method; - classical ML; - smaller model; - previous benchmark result; -
zero-shot baseline; - no-RAG baseline; - no-fine-tuning baseline.

A sophisticated system without a sensible baseline tells you little
about what produced the gain.

## 3.7 Ablations

An ablation removes or changes one component.

RAG example: - full system; - no reranker; - smaller chunk size; - no
metadata; - no retrieval.

If performance falls only when retrieval is removed, that supports a
claim about retrieval's contribution.

## 3.8 Data leakage and contamination

Leakage occurs when information unavailable at genuine prediction time
improperly reaches training/development.

Benchmark contamination is especially important when evaluation items or
near-duplicates may have appeared in pretraining or post-training.

Mitigations: - new/private test sets; - time-based splits; -
deduplication; - canary strings where appropriate; - held-out generated
variants; - explicit limitations.

## 3.9 LLM-as-a-judge

An LLM can score outputs using a rubric, but the judge is itself a model
and can be biased.

Validate judge scores against human annotations. Test: - position/order
effects; - verbosity preference; - self-preference; - prompt
sensitivity; - inter-rater agreement.

## 3.10 Confidence intervals and significance

Do not report only `Model A = 81%, Model B = 80%`.

Ask whether the difference is stable.

Bootstrap: 1. sample benchmark items with replacement; 2. recompute
score difference; 3. repeat many times; 4. inspect confidence interval.

For paired model outputs on the same classification items, paired tests
are usually preferable to pretending samples are independent.

## 3.11 Error taxonomy

Create categories before or during blinded error review: - knowledge
error; - reasoning error; - instruction failure; - extraction error; -
hallucination; - refusal; - formatting; - retrieval failure; - tool
failure.

Quantifying *why* systems fail is often more useful than one aggregate
score.
