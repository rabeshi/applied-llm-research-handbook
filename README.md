# Applied LLM Research Handbook

A practical, research-oriented refresher for applied machine learning,
NLP, LLMs, evaluation, benchmarking, post-training, reinforcement
learning, agents, robustness, and AI safety.

The repository is designed around one rule:

**Concept → intuition → tiny worked example → runnable experiment →
metric → research question.**

## Learning path

  -----------------------------------------------------------------------
  Stage                   Topic                   You should be able to
                                                  answer
  ----------------------- ----------------------- -----------------------
  0                       Applied ML              What problem am I
                                                  solving, what is X/y,
                                                  and how will I evaluate
                                                  it?

  1                       Tokens                  What does a language
                                                  model actually receive?

  2                       Next-token prediction   How does an LM turn
                                                  context into a
                                                  probability
                                                  distribution?

  3                       Sampling                What do temperature,
                                                  top-k and top-p change?

  4                       Prompting               What are zero-shot,
                                                  one-shot and few-shot
                                                  learning?

  5                       Embeddings              How can text be
                                                  represented as vectors?

  6                       Transformers            What do attention and
                                                  context do?

  7                       Evaluation              What does "better" mean
                                                  and how do I measure
                                                  it?

  8                       Benchmarking            How do I construct a
                                                  fair, reproducible
                                                  comparison?

  9                       RAG                     Can retrieval improve
                                                  grounded answering?

  10                      Fine-tuning             When should model
                                                  weights change rather
                                                  than the prompt?

  11                      Post-training           What are SFT,
                                                  preference learning,
                                                  DPO, RLHF and RLVR?

  12                      Reinforcement learning  What are states,
                                                  actions, rewards,
                                                  policies and returns?

  13                      Agents                  How do models use tools
                                                  and act over multiple
                                                  steps?

  14                      Reliability & safety    Where do systems fail,
                                                  and how do we test
                                                  failure modes?

  15                      Research design         How do I turn an
                                                  experiment into a
                                                  publishable study?
  -----------------------------------------------------------------------

## Repository map

-   `notes/01_foundations.md` --- applied ML, NLP, tokens, logits,
    softmax, context, embeddings, Transformers.
-   `notes/02_prompting_and_inference.md` --- zero/few-shot, prompting,
    decoding, temperature, top-k/top-p.
-   `notes/03_evaluation_and_benchmarks.md` --- metrics, benchmark
    design, baselines, contamination, ablations, uncertainty.
-   `notes/04_rag_and_agents.md` --- retrieval, chunking, vector search,
    tool use, agents and agent evaluation.
-   `notes/05_post_training_and_rl.md` --- fine-tuning, SFT, LoRA,
    preferences, DPO, RL, RLHF, RLVR/GRPO.
-   `notes/06_reliability_and_safety.md` --- hallucination, robustness,
    calibration, prompt injection, safety evaluations.
-   `notes/07_research_playbook.md` --- how to design a paper-quality
    experiment.
-   `notebooks/01_tokens_zero_few_shot.ipynb` --- executable starter
    notebook requiring only Python for most cells; optional Transformers
    cells.
-   `experiments/experiment_ideas.md` --- progressively harder
    experiments and mini-benchmarks.
-   `requirements.txt` --- optional packages.

## Recommended order

Start with the notebook while reading Notes 1--3. Then implement one
evaluation study before moving into fine-tuning or RL. A researcher who
can define a task, build a clean evaluation set, choose metrics, run
baselines, quantify uncertainty, and perform error analysis has the core
workflow needed for much of applied LLM research.

## Core mental model

A language model repeatedly performs:

`text → tokens → token IDs → model → logits → probabilities → choose next token → append → repeat`

A benchmark performs:

`research question → task definition → dataset → model/prompt conditions → outputs → scoring → uncertainty → error analysis → conclusions`

Post-training performs:

`pretrained model → desired behavior data/reward → optimization → changed model → held-out evaluation`

An agent adds a loop:

`observe → reason/decide → act/tool call → environment changes → observe again`

## Practical philosophy

Do not begin with "Which model should I use?" Begin with:

1.  What behavior do I want to measure or improve?
2.  What constitutes one example?
3.  What is the expected output?
4.  How will I score it?
5.  What baselines make the result meaningful?
6.  What variables will I control?
7.  What failure modes could invalidate the conclusion?
