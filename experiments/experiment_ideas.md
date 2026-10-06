# Experiment ladder

## Beginner

### A. Tokenization audit

Compare token efficiency across prose, code, URLs, names and multiple
languages.

### B. Temperature stability

Generate 20 answers per prompt at several temperatures. Measure
exact-answer agreement and semantic diversity.

### C. Zero-shot vs few-shot

Run the same labeled dataset with 0, 1, 3 and 5 demonstrations. Repeat
with multiple demonstration selections.

### D. Prompt robustness

Create five semantically equivalent prompt templates and quantify score
spread.

## Intermediate

### E. Embedding retrieval

Create 100 documents and 50 queries with known relevant passages.
Compare Recall@1/3/5.

### F. RAG ablation

No RAG vs RAG; then vary chunk size, k and reranking. Separate retrieval
and answer metrics.

### G. Hallucination benchmark

Give models source passages and questions. Annotate answer claims as
supported, contradicted or unsupported.

### H. LLM judge validation

Have humans and two judge models score 200 responses. Measure agreement
and investigate order/verbosity bias.

### I. Small-model SFT

Fine-tune a small open model on a narrow task. Compare base vs SFT on
held-out task performance and general capability checks.

## Advanced

### J. Preference optimization

Create chosen/rejected pairs; run a small DPO experiment; compare to
SFT-only.

### K. Verifiable-reward reasoning

Use math/programming tasks with automatic checkers. Compare pass@k and
reward vs true correctness.

### L. Agent tool robustness

Sandbox an agent with tools. Inject tool errors, stale results and
untrusted instructions. Score trajectories.

### M. Longitudinal safety benchmark

Construct evolving records where newer facts supersede older facts.
Measure stale-state errors as history length and distractors increase.

## For every experiment, save

`config.json` `prompts/` `raw_outputs.jsonl` `scores.csv`
`analysis.ipynb` `README.md`

Never keep only aggregate scores. Raw outputs enable later error
analysis.
