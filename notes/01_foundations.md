# 1. Foundations: from applied ML to tokens

The worked examples use toy numbers and fictional scenarios unless stated otherwise. They illustrate the method; they are not measured results.

## 1.1 Applied machine learning

Applied ML uses statistical/ML methods to solve or investigate a
concrete problem.

A supervised dataset is often written as pairs `(x_i, y_i)`.

Example: spam detection.

-   `x`: email text.
-   `y`: spam / not spam.
-   model: logistic regression, Transformer classifier, or LLM.
-   metric: precision, recall, F1, AUROC, etc.

The research question is not "Can I run an LLM?" A stronger question is:
**Does an LLM improve recall on difficult phishing emails without
increasing false positives compared with a conventional classifier?**

### Train / validation / test

-   **Train:** fit parameters.
-   **Validation:** choose hyperparameters/model/prompt.
-   **Test:** final unbiased estimate.
-   Never repeatedly inspect test results and tune against them; the
    test set then becomes part of development.

### Worked example

Suppose you have 1,000 labeled support tickets. Use 600 for training, 200 for validation, and 200 for the final test. Choose the prompt or classifier on validation data; evaluate the chosen system once on the test set. If multiple tickets belong to the same conversation, split by conversation so near-duplicate messages cannot cross splits.

**Try it:** compare a majority-class baseline with a text classifier. Report macro-F1 as well as accuracy, and inspect which categories are missed.

## 1.2 NLP, language models, LLMs and generative AI

**NLP** is the broad field of computational methods for language.

**Language modeling** estimates probabilities over token sequences.
Autoregressive LMs factorize:

`P(x_1,...,x_T) = Π_t P(x_t | x_<t)`

**LLM** usually means a large pretrained language model, commonly
Transformer-based.

**Generative AI** is broader: systems generating text, code, images,
audio, video, etc.

### Worked example

For a toy two-token sequence, suppose `P("red") = 0.2` and `P("ball" | "red") = 0.5`. Its joint probability is `0.2 × 0.5 = 0.1`. A sentiment classifier produces a label; an autoregressive language model predicts successive tokens. You can use the latter for classification by prompting it to generate a label.

## 1.3 Tokens

Models do not directly read words. A **tokenizer** maps text into
discrete units and IDs.

Example conceptually:

`"Machine learning is useful."` →
`["Machine", " learning", " is", " useful", "."]` →
`[43121, 6975, 374, 4465, 13]`

Exact tokenization depends on the tokenizer. A token can be a whole
word, part of a word, punctuation, whitespace-associated unit, byte
sequence, etc.

### Why tokens matter

They determine:

- input/output length;
- context-window consumption;
- inference/training cost;
- how unusual names, code, numbers and languages are represented;
- the units over which next-token probabilities are predicted.

### Experiment

Tokenize:

- common English;
- a rare surname;
- a URL;
- Python;
- an African language;
- a long number.

Measure token count / character count. Ask whether some data types are
represented less efficiently.

### Worked example

Imagine tokenizer A represents a name with two tokens and tokenizer B uses six. Repeating that name 100 times contributes 200 versus 600 tokens, before counting surrounding text. These counts are illustrative: measure them with the actual tokenizer rather than estimating tokens from words.

**Try it:** tokenize the same multilingual sentences with two tokenizers. Compare token counts, then discuss why tokenizer efficiency alone does not establish language understanding.

## 1.4 Vocabulary and token IDs

A tokenizer has a finite vocabulary. Each token maps to an integer. IDs
themselves have no semantic ordering: ID 9000 is not "more meaningful"
than ID 10.

### Worked example

A toy vocabulary might map `"cat" → 0`, `"dog" → 1`, and `"bird" → 2`. The ID selects a row in an embedding table; arithmetic on IDs has no linguistic meaning. Changing token IDs without changing the matching model's embedding table changes what the model receives. Always pair a model with its compatible tokenizer.

## 1.5 Embeddings

An embedding maps a discrete item to a dense vector.

Token ID: `42`

Embedding: `[0.12, -0.41, 0.03, ...]`

During model processing, contextual representations change depending on
surrounding tokens.

Separately, **sentence/document embeddings** represent larger pieces of
text and are often used for semantic search, clustering and RAG.

Cosine similarity:

`cos(a,b) = (a·b) / (||a|| ||b||)`

Values closer to 1 indicate vectors pointing in similar directions,
though interpretation depends on the embedding model.

### Worked example

Let `a = [1, 0]`, `b = [2, 0]`, and `c = [0, 1]`. Then `cos(a, b) = 1` and `cos(a, c) = 0`: cosine compares direction rather than magnitude. In semantic search, a query such as "reset my password" may match "recover account access" despite limited word overlap. Similarity is a ranking signal, not proof that two statements are factually equivalent.

## 1.6 Context window

The context window is the amount of token context a model can process in
one inference/training setup. More context does not guarantee perfect
use of all information.

Experiment: place a key fact at the beginning, middle and end of
progressively longer documents. Ask the same question and measure
retrieval accuracy by position.

### Worked example

For a hypothetical 8,192-token total budget, reserving 1,024 output tokens leaves at most 7,168 for input, including instructions, history, and retrieved text. Actual systems can impose separate input/output limits. A document fitting the budget can still be poorly used.

**Try it:** move the same lookup fact between beginning, middle, and end positions while keeping document length and question fixed.

## 1.7 Logits, softmax and probabilities

For each next-token position, the model outputs **logits**: unnormalized
scores over vocabulary items.

Toy logits:

- `cat = 2.0`
- `dog = 1.0`
- `car = 0.0`

Softmax converts logits to probabilities:

`P(i) = exp(z_i) / Σ_j exp(z_j)`

The highest-logit token gets the highest probability.

### Worked example

For logits `[2, 1, 0]`, softmax is approximately `[0.665, 0.245, 0.090]`. Adding the same constant to every logit does not change the probabilities.

```python
import math

logits = [2.0, 1.0, 0.0]
weights = [math.exp(z - max(logits)) for z in logits]
probs = [w / sum(weights) for w in weights]
print([round(p, 3) for p in probs])  # [0.665, 0.245, 0.09]
```

Subtracting the maximum improves numerical stability. These are probabilities over next tokens, not probabilities that an entire answer is true.

## 1.8 Cross-entropy and perplexity

Training next-token LMs commonly minimizes negative
log-likelihood/cross-entropy.

For the correct next token `y`:

`loss = -log P(y)`

If the model assigns high probability to the correct token, loss is low.

Perplexity is often:

`PPL = exp(mean cross-entropy)`

Lower perplexity means the model assigns higher probability to observed
sequences, but it is not a universal measure of downstream usefulness.

### Worked example

Using natural logarithms, assigning the observed token probability `0.8` gives loss `-ln(0.8) ≈ 0.223`; assigning `0.2` gives `1.609`. If two observed tokens each receive probability `0.5`, their mean loss is `0.693` and perplexity is `2`.

Compare perplexity under compatible tokenization and evaluation protocols. A different tokenizer changes the prediction units, so raw per-token perplexities need not be comparable.

## 1.9 Transformer intuition

A Transformer repeatedly transforms token representations.

Important components:

- embeddings;
- positional information;
- self-attention;
- feed-forward/MLP blocks;
- residual connections;
- normalization;
- output projection to vocabulary logits.

### Self-attention intuition

For each token, the model creates query, key and value representations.
A simplified attention operation is:

`Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V`

Intuition: each position computes how much it should use information
from other allowed positions.

Do not interpret individual attention weights automatically as
explanations of model reasoning.

### Worked example

Suppose one attention head assigns weights `[0.75, 0.25]` to value vectors `[2, 0]` and `[0, 4]`. Its weighted output is `[1.5, 1.0]`. The model learns the projections producing these weights and values. Multiple heads can combine different relationships.

In a causal language model, a position cannot attend to future tokens. Positional information distinguishes "dog bites person" from "person bites dog"; MLP blocks transform features, residual connections preserve a path for information and gradients, and normalization controls activation scale. The final projection turns hidden representations into vocabulary logits.

See [Attention Is All You Need](https://arxiv.org/abs/1706.03762) for the original Transformer architecture; modern implementations vary.
