# Applied LLM Research Handbook

A practical guide to understanding language models and designing sound applied LLM experiments. Topics include foundations, prompting, evaluation, retrieval, agents, post-training, reinforcement learning, reliability, and safety.

Each topic connects a concept to intuition, a worked example, an experiment, a metric, and a research question.

## Start here

1. Read [Foundations](notes/01_foundations.md) and [Prompting and inference](notes/02_prompting_and_inference.md).
2. Work through the [starter notebook](notebooks/01_tokens_zero_few_shot.ipynb) to explore probabilities, temperature, sampling, prompting, and classification metrics.
3. Read [Evaluation and benchmarks](notes/03_evaluation_and_benchmarks.md), then choose a study from the [experiment ideas](experiments/experiment_ideas.md).
4. Use the [research playbook](notes/07_research_playbook.md) to define baselines, control variables, analyze errors, and report results.

The notes need no installation. The notebook uses small numerical examples and requires no API key or GPU. Its optional tokenizer section downloads a Hugging Face tokenizer when enabled.

## Run the notebook

```sh
git clone https://github.com/rabeshi/applied-llm-research-handbook.git
cd applied-llm-research-handbook
python -m venv .venv
```

Activate your environment:

| Shell | Command |
| --- | --- |
| Windows PowerShell | `.\.venv\Scripts\Activate.ps1` |
| macOS / Linux | `source .venv/bin/activate` |

Install the packages needed by the active notebook cells and launch Jupyter:

```sh
python -m pip install numpy pandas matplotlib scikit-learn jupyter
jupyter notebook notebooks/01_tokens_zero_few_shot.ipynb
```

For the optional tokenizer section, also install `transformers`. To install the broader set of tools for further experiments, use `python -m pip install -r requirements.txt`.

## Learning path

| Stage | Topic | Guiding question |
| --- | --- | --- |
| 0 | Applied ML | What problem am I solving, and how will I evaluate it? |
| 1 | Tokens | What does a language model receive as input? |
| 2 | Next-token prediction | How does context become a probability distribution? |
| 3 | Sampling | What do temperature, top-k, and top-p change? |
| 4 | Prompting | How do zero-shot, one-shot, and few-shot prompts differ? |
| 5 | Embeddings | How can text be represented as vectors? |
| 6 | Transformers | How do attention and context affect predictions? |
| 7 | Evaluation | What does better mean, and how do I measure it? |
| 8 | Benchmarking | How do I make comparisons fair and reproducible? |
| 9 | Retrieval-augmented generation (RAG) | Can retrieval improve grounded answering? |
| 10 | Fine-tuning | When should I change model weights? |
| 11 | Post-training | How do supervised and preference-based methods shape behavior? |
| 12 | Reinforcement learning | How do policies, rewards, and returns relate? |
| 13 | Agents | How do models use tools across multiple steps? |
| 14 | Reliability and safety | Where do systems fail, and how can I test those failures? |
| 15 | Research design | How do I turn an experiment into a defensible study? |

## Repository guide

| Resource | Contents |
| --- | --- |
| [Foundations](notes/01_foundations.md) | Applied ML, NLP, tokens, logits, softmax, embeddings, and Transformers |
| [Prompting and inference](notes/02_prompting_and_inference.md) | Prompt conditions, decoding, temperature, top-k, and top-p |
| [Evaluation and benchmarks](notes/03_evaluation_and_benchmarks.md) | Metrics, baselines, contamination, ablations, and uncertainty |
| [RAG and agents](notes/04_rag_and_agents.md) | Retrieval, chunking, vector search, tool use, and agent evaluation |
| [Post-training and RL](notes/05_post_training_and_rl.md) | Fine-tuning, SFT, LoRA, preferences, DPO, RLHF, and RLVR/GRPO |
| [Reliability and safety](notes/06_reliability_and_safety.md) | Hallucination, robustness, calibration, prompt injection, and safety evaluation |
| [Research playbook](notes/07_research_playbook.md) | Study design and reporting |
| [Starter notebook](notebooks/01_tokens_zero_few_shot.ipynb) | Numerical examples, prompt comparisons, metrics, and optional tokenization |
| [Experiment ideas](experiments/experiment_ideas.md) | Progressively harder studies and mini-benchmarks |
| [Requirements](requirements.txt) | Python packages for the notebook and further experiments |

## Design your first study

Before choosing a model, define:

1. The behavior you want to measure or improve.
2. What counts as one example and its expected output.
3. The scoring method and meaningful baselines.
4. The variables you will control and the conditions you will compare.
5. The failure modes that could invalidate your conclusions.

Begin with one small evaluation study before moving to fine-tuning or reinforcement learning. Preserve raw outputs, report variation across runs, and use error analysis to explain what aggregate scores miss.
