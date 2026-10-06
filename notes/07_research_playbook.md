# 7. Research playbook: from notebook to paper

The worked examples use toy numbers and fictional scenarios unless stated otherwise. They illustrate the method; they are not measured results.

## 7.1 Start with a falsifiable question

Weak: "Compare LLMs on healthcare."

Better: "Does retrieval reduce unsupported medication claims under
longitudinal record updates, and does the effect persist as context
length increases?"

### Worked example

A testable question is "Does retrieval reduce stale-stock answers on a held-out set of timestamped inventory records?" Specify the population and comparison. "Retrieval makes agents smarter" leaves the behavior, baseline, and measurement undefined.

## 7.2 Write hypotheses before running

H1: RAG lowers stale-information errors. H2: Benefit grows when relevant
updates are outside the model's short local context. H3: Retrieval
errors become the dominant failure mode at high distractor volume.

Pre-specifying hypotheses reduces retrospective storytelling.

### Worked example

Before collecting final results, write H1: retrieval decreases stale-answer rate; primary metric: stale answers divided by all test queries; comparison: identical model with retrieval disabled. State whether the analysis is directional, the stopping rule, and how secondary outcomes are handled. Label new ideas discovered afterward as exploratory.

## 7.3 Define independent/dependent/control variables

Example:

- independent: retrieval on/off; context length; model;
- dependent: current-state accuracy, stale-error rate;
- controls: prompts, decoding, test items, tool configuration.

### Worked example

A `2 × 3` design combines retrieval on/off with three distractor counts, giving six conditions. Run all six on the same test items with the same model and prompt template. Compare retrieval's benefit at each distractor level; this tests an interaction that one overall average could hide.

## 7.4 Build data carefully

Document:

- source/provenance;
- licenses/permissions;
- inclusion criteria;
- annotation instructions;
- annotator expertise;
- adjudication;
- class distribution;
- splits;
- privacy/de-identification where relevant.

### Worked example

If several questions come from one inventory history, split by history before selecting demonstrations. Write an annotation rule: use the latest applicable update from the authoritative source. Have two annotators independently label a subset, inspect disagreement, and revise ambiguous rules before freezing the test labels.

## 7.5 Pilot before scaling

Run 20--50 examples first.

Check:

- task ambiguity;
- parser failures;
- metric bugs;
- ceiling/floor effects;
- cost;
- whether expected answers are actually defensible.

### Worked example

In a 30-item development pilot, suppose five inputs lack enough evidence and three outputs fail parsing. Repair the data specification and parser before scaling. Keep pilot items out of the untouched final test if they influenced design decisions. Estimate total cost using actual pilot token counts and call counts.

## 7.6 Baselines and ablations

Minimum useful comparison often includes:

- simple baseline;
- current standard approach;
- proposed method;
- ablations of key components.

### Worked example

Compare a latest-authoritative-record rule, a model without retrieval, a retrieval model, and an oracle-evidence model. Disable reranking as an ablation. The simple rule may be strong for structured records; the oracle condition diagnoses whether retrieval or generation limits performance.

## 7.7 Reproducibility

Record:

- model identifier/version;
- date;
- package versions;
- seeds;
- prompts;
- decoding parameters;
- hardware;
- dataset commit/hash;
- evaluation code commit.

For hosted models, behavior can change; save outputs and metadata when
permitted.

### Worked example

Store one record per item and run, for example:

```json
{"item_id":"stock_017","run_id":"run_03","model":"record-exact-version","prompt_version":"v2","seed":7,"retrieval_ids":["record_41"],"raw_output":"unavailable","score":1}
```

Also retain dataset and code hashes, decoding settings, dates, token counts, and tool traces. A seed is useful metadata but does not guarantee reproducibility across hardware or hosted implementations.

## 7.8 Error analysis

Sample errors blindly if possible. Create categories, annotate them,
measure category prevalence, and include representative examples.

A useful result can be: "Overall accuracy changed little, but
unsupported high-severity recommendations fell substantially."

### Worked example

Blind reviewers to the system name when classifying failures. Suppose retrieval reduces invented facts but introduces stale-source errors. Report both categories and their denominators. Include representative cases selected by a declared rule rather than only the most persuasive anecdotes.

## 7.9 Statistical reporting

Report:

- n;
- point estimate;
- uncertainty/confidence interval;
- paired comparison where appropriate;
- multiple-run variability for stochastic procedures;
- effect size, not only p-values.

### Worked example

If A gets 160/200 and B gets 170/200, the observed improvement is 5 percentage points. Compute a paired interval using the item-level outcomes; totals alone cannot reveal which items changed. Report run-to-run variation separately from uncertainty due to the sample of test items. Treat extensive comparisons as exploratory or account for multiplicity.

## 7.10 Publication-quality benchmark checklist

Before claiming a benchmark contribution:

- Is the construct clearly defined?
- Does the dataset actually instantiate it?
- Are examples difficult for the intended reason?
- Is scoring valid?
- Are human baselines available where useful?
- Are strong model baselines included?
- Is contamination discussed?
- Are categories balanced/representative?
- Are uncertainty and error analyses reported?
- Can others reproduce evaluation?
- Are ethical risks and intended use documented?

### Worked example

A benchmark release should include a dataset description, annotation rubric, split identifiers, evaluation script, baseline configuration, uncertainty analysis, and intended-use limitations. Ask another person to reproduce one baseline from the documentation. If they must guess label normalization or prompt formatting, the protocol is not yet complete.
