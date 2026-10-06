# 4. RAG and agents

The worked examples use toy numbers and fictional scenarios unless stated otherwise. They illustrate the method; they are not measured results.

## 4.1 Retrieval-Augmented Generation (RAG)

RAG supplies retrieved external information to a generator at inference
time.

Pipeline:

`documents → chunks → embeddings/index → query → retrieval → optional reranking → context → LLM → answer`

RAG changes context, not necessarily model weights.

### Worked example

A refund policy states "returns accepted within 30 days." Retrieval selects this passage for "Can I return an item after 20 days?" and the generator answers using it. Compare with no retrieval and an oracle condition supplied the correct passage. Oracle context helps separate retrieval failures from answer-generation failures.

## 4.2 Chunking

Documents are split into retrievable pieces. Variables:

- chunk length;
- overlap;
- semantic vs fixed-length boundaries;
- metadata.

Experiment: compare 128-, 256-, 512- and 1024-token chunks on the same
QA set.

Measure:

- Recall@k of gold evidence;
- answer accuracy;
- context tokens;
- latency.

### Worked example

A chunk boundary separates "returns accepted" from "within 30 days," leaving each fragment incomplete. Sentence-aware boundaries or overlap may preserve the rule. Overlap also adds near-duplicate candidates and consumes context. Vary chunking while keeping the corpus, queries, retriever, and evidence labels fixed.

## 4.3 Retrieval metrics

If relevant evidence is known:

- Recall@k: fraction of relevant items retrieved in the top k.
- Precision@k: fraction of the top k results that are relevant.
- MRR: mean reciprocal rank of the first relevant result across queries.
- nDCG: ranking quality with graded relevance.

Separate retrieval quality from generation quality.

### Worked example

Suppose two chunks are relevant, at ranks 2 and 5. At `k = 3`, Recall@3 is `1/2 = 0.5`, Precision@3 is `1/3`, and reciprocal rank is `1/2`. MRR averages reciprocal rank across queries, assigning zero when no relevant result is retrieved. The indicator "at least one relevant chunk retrieved" is **Hit@k**, not general Recall@k; they coincide when exactly one relevant item exists.

nDCG discounts lower-ranked results and normalizes against an ideal ordering, allowing graded relevance. Specify the relevance scale, gain formula, cutoff, and treatment of queries with no relevant items.

## 4.4 Groundedness

An answer can be factually correct but unsupported by retrieved
evidence, or well-supported by evidence that itself is wrong.

Evaluate separately:

- answer correctness;
- citation/evidence correctness;
- completeness;
- groundedness/faithfulness.

### Worked example

Retrieved evidence says "standard delivery takes 3–5 business days." The answer "delivery takes 3–5 days and is free" adds an unsupported price claim. Score each claim: the timing statement is supported if its business-day qualifier is preserved; the free-delivery claim is not. A citation attached to the sentence does not automatically support every claim.

## 4.5 Agents

An agent uses a model within an iterative decision loop, often with
tools.

`observation → choose action → tool/environment → new observation → ... → final result`

Examples:

- search agent;
- coding agent;
- calendar assistant;
- database analyst.

### Worked example

A stock-checking agent observes a product ID, calls `get_stock(product_id)`, receives `{"available": false}`, and reports unavailability. The tool response changes its next decision. Give the loop a step budget and a termination condition so repeated failed actions cannot run indefinitely.

## 4.6 Agent evaluation

Outcome-only scoring is insufficient.

Measure:

- task success;
- number of steps;
- tool-call correctness;
- invalid calls;
- recovery after tool error;
- cost/latency;
- policy/safety violations;
- unnecessary actions.

### Worked example

Agent A completes 90/100 tasks with 4 calls per task; B completes 92/100 with 12 calls. Compare task success, latency, and total cost under the same budget. Also inspect invalid actions: a final answer that looks correct can conceal an unauthorized intermediate operation.

## 4.7 Worked agent test

Task: "Find the cheapest valid flight under these constraints" in a
sandbox.

Perturbations:

- tool returns timeout;
- one price changes;
- a malicious page contains prompt injection;
- user constraint conflicts with a tool suggestion.

Measure whether the agent:

1. preserves user constraints;
2. ignores untrusted instructions;
3. retries safely;
4. avoids irreversible action without authorization;
5. reports uncertainty.

This turns agentic safety into measurable behavior.

### Worked example

Use a mock flight tool with candidate A costing 200 units with one stop and candidate B costing 240 with no stops. For a nonstop constraint, B is the valid answer. Inject a timeout on the first call, then make B unavailable. The agent should re-query or report that no valid option remains rather than silently substituting A or booking anything.
