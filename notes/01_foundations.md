# 1. Foundations: from applied ML to tokens

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

## 1.2 NLP, language models, LLMs and generative AI

**NLP** is the broad field of computational methods for language.

**Language modeling** estimates probabilities over token sequences.
Autoregressive LMs factorize:

`P(x_1,...,x_T) = Π_t P(x_t | x_<t)`

**LLM** usually means a large pretrained language model, commonly
Transformer-based.

**Generative AI** is broader: systems generating text, code, images,
audio, video, etc.

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

They determine: - input/output length; - context-window consumption; -
inference/training cost; - how unusual names, code, numbers and
languages are represented; - the units over which next-token
probabilities are predicted.

### Experiment

Tokenize: - common English; - a rare surname; - a URL; - Python; - an
African language; - a long number.

Measure token count / character count. Ask whether some data types are
represented less efficiently.

## 1.4 Vocabulary and token IDs

A tokenizer has a finite vocabulary. Each token maps to an integer. IDs
themselves have no semantic ordering: ID 9000 is not "more meaningful"
than ID 10.

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

## 1.6 Context window

The context window is the amount of token context a model can process in
one inference/training setup. More context does not guarantee perfect
use of all information.

Experiment: place a key fact at the beginning, middle and end of
progressively longer documents. Ask the same question and measure
retrieval accuracy by position.

## 1.7 Logits, softmax and probabilities

For each next-token position, the model outputs **logits**: unnormalized
scores over vocabulary items.

Toy logits: - `cat = 2.0` - `dog = 1.0` - `car = 0.0`

Softmax converts logits to probabilities:

`P(i) = exp(z_i) / Σ_j exp(z_j)`

The highest-logit token gets the highest probability.

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

## 1.9 Transformer intuition

A Transformer repeatedly transforms token representations.

Important components: - embeddings; - positional information; -
self-attention; - feed-forward/MLP blocks; - residual connections; -
normalization; - output projection to vocabulary logits.

### Self-attention intuition

For each token, the model creates query, key and value representations.
A simplified attention operation is:

`Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) V`

Intuition: each position computes how much it should use information
from other allowed positions.

Do not interpret individual attention weights automatically as
explanations of model reasoning.
