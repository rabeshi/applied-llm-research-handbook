# 4. RAG and agents

## 4.1 Retrieval-Augmented Generation (RAG)

RAG supplies retrieved external information to a generator at inference
time.

Pipeline:

`documents → chunks → embeddings/index → query → retrieval → optional reranking → context → LLM → answer`

RAG changes context, not necessarily model weights.

## 4.2 Chunking

Documents are split into retrievable pieces. Variables: - chunk
length; - overlap; - semantic vs fixed-length boundaries; - metadata.

Experiment: compare 128-, 256-, 512- and 1024-token chunks on the same
QA set.

Measure: - Recall@k of gold evidence; - answer accuracy; - context
tokens; - latency.

## 4.3 Retrieval metrics

If relevant evidence is known: - Recall@k: was relevant evidence in top
k? - Precision@k: how much of top k was relevant? - MRR: how high did
the first relevant result rank? - nDCG: ranking quality with graded
relevance.

Separate retrieval quality from generation quality.

## 4.4 Groundedness

An answer can be factually correct but unsupported by retrieved
evidence, or well-supported by evidence that itself is wrong.

Evaluate separately: - answer correctness; - citation/evidence
correctness; - completeness; - groundedness/faithfulness.

## 4.5 Agents

An agent uses a model within an iterative decision loop, often with
tools.

`observation → choose action → tool/environment → new observation → ... → final result`

Examples: - search agent; - coding agent; - calendar assistant; -
database analyst.

## 4.6 Agent evaluation

Outcome-only scoring is insufficient.

Measure: - task success; - number of steps; - tool-call correctness; -
invalid calls; - recovery after tool error; - cost/latency; -
policy/safety violations; - unnecessary actions.

## 4.7 Worked agent test

Task: "Find the cheapest valid flight under these constraints" in a
sandbox.

Perturbations: - tool returns timeout; - one price changes; - a
malicious page contains prompt injection; - user constraint conflicts
with a tool suggestion.

Measure whether the agent: 1. preserves user constraints; 2. ignores
untrusted instructions; 3. retries safely; 4. avoids irreversible action
without authorization; 5. reports uncertainty.

This turns agentic safety into measurable behavior.
