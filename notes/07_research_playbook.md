# 7. Research playbook: from notebook to paper

## 7.1 Start with a falsifiable question

Weak: "Compare LLMs on healthcare."

Better: "Does retrieval reduce unsupported medication claims under
longitudinal record updates, and does the effect persist as context
length increases?"

## 7.2 Write hypotheses before running

H1: RAG lowers stale-information errors. H2: Benefit grows when relevant
updates are outside the model's short local context. H3: Retrieval
errors become the dominant failure mode at high distractor volume.

Pre-specifying hypotheses reduces retrospective storytelling.

## 7.3 Define independent/dependent/control variables

Example: - independent: retrieval on/off; context length; model; -
dependent: current-state accuracy, stale-error rate; - controls:
prompts, decoding, test items, tool configuration.

## 7.4 Build data carefully

Document: - source/provenance; - licenses/permissions; - inclusion
criteria; - annotation instructions; - annotator expertise; -
adjudication; - class distribution; - splits; -
privacy/de-identification where relevant.

## 7.5 Pilot before scaling

Run 20--50 examples first.

Check: - task ambiguity; - parser failures; - metric bugs; -
ceiling/floor effects; - cost; - whether expected answers are actually
defensible.

## 7.6 Baselines and ablations

Minimum useful comparison often includes: - simple baseline; - current
standard approach; - proposed method; - ablations of key components.

## 7.7 Reproducibility

Record: - model identifier/version; - date; - package versions; -
seeds; - prompts; - decoding parameters; - hardware; - dataset
commit/hash; - evaluation code commit.

For hosted models, behavior can change; save outputs and metadata when
permitted.

## 7.8 Error analysis

Sample errors blindly if possible. Create categories, annotate them,
measure category prevalence, and include representative examples.

A useful result can be: "Overall accuracy changed little, but
unsupported high-severity recommendations fell substantially."

## 7.9 Statistical reporting

Report: - n; - point estimate; - uncertainty/confidence interval; -
paired comparison where appropriate; - multiple-run variability for
stochastic procedures; - effect size, not only p-values.

## 7.10 Publication-quality benchmark checklist

Before claiming a benchmark contribution: - Is the construct clearly
defined? - Does the dataset actually instantiate it? - Are examples
difficult for the intended reason? - Is scoring valid? - Are human
baselines available where useful? - Are strong model baselines
included? - Is contamination discussed? - Are categories
balanced/representative? - Are uncertainty and error analyses
reported? - Can others reproduce evaluation? - Are ethical risks and
intended use documented?
