# Word2Vec (Skip-Gram with Negative Sampling) — NumPy Implementation

This project implements **Word2Vec Skip-Gram with Negative Sampling (SGNS)** from scratch using **pure NumPy**, without machine learning frameworks such as PyTorch or TensorFlow.

The goal is to learn **dense vector representations (word embeddings)** such that words appearing in similar contexts have similar vector representations.

---

# Model Overview

Word2Vec trains embeddings by predicting **context words from a center word**.

Example sentence:

the quick brown fox jumps

If the center word is:

brown

and the window size is 1, the context words are:

quick, fox

Training pairs:

brown → quick  
brown → fox

The model learns embeddings so that **center and context words have high similarity**.

---

# Architecture

Two embedding matrices are learned:

- \( W_{in} \in \mathbb{R}^{V \times D} \)
- \( W_{out} \in \mathbb{R}^{V \times D} \)

Where:
- **V** = vocabulary size
- **D** = embedding dimension

The score for a word pair is:

\[
\text{score} = v_c^\top u_o
\]

Where:
- \( v_c = W_{in}[\text{center}] \)
- \( u_o = W_{out}[\text{context}] \)

---

# Training Objective

For each pair:
- `(center, positive_context)`

and \(K\) negative samples:
- `negative_1, ..., negative_K`

the SGNS loss is:

\[
L = -\log \sigma(v_c^\top u_{pos}) - \sum_{k=1}^{K} \log \sigma(-v_c^\top u_{neg_k})
\]

The implementation uses a **numerically stable log-sigmoid computation** based on `logaddexp`.

---

# Key Components

### Tokenization
Text is lowercased and tokenized using:

`[a-z0-9']+`

### Skip-Gram Sampling
Training pairs are generated using a sliding window around each word.

### Dynamic Window
Instead of using a fixed context size, the actual window radius is randomly sampled for each center word:

`window ∈ [1, max_window]`

### Negative Sampling
Negative words are sampled using the distribution:

\[
P(w) \propto \text{frequency}(w)^{0.75}
\]

Additionally, negative samples are filtered so that:

`negative ≠ positive_context`

to avoid conflicting training signals.

### Subsampling Frequent Words
Very frequent words such as:

`the, of, and, to`

are randomly removed during training to reduce noise.

Retention probability:

\[
P_{\text{keep}}(w) = \min\left(1, \sqrt{\frac{t}{f(w)}} + \frac{t}{f(w)}\right)
\]

where \(f(w)\) is the relative frequency of the word in the corpus.

---

# Optimization

Training uses **mini-batch stochastic gradient descent**.

Updates are applied using:

`np.add.at`

to correctly accumulate gradients when words repeat within a batch.

A simple linear learning rate decay is used:

`lr_t = lr * max(0.1, 1 - step / steps)`

---

# Hyperparameters

Example configuration:

```python
embedding_dim = 50
window_size = 5
batch_size = 256
negative_K = 5
learning_rate = 0.05
steps = 2000
```

# Running the Project

1. Place your training dataset in the folder `data` and name the file `text.txt`.

2. Open the training notebook.

3. Run all cells in the notebook to start the training process.

During training, you should see output similar to:

```
step=0   loss=4.16
step=100 loss=3.92
step=200 loss=3.75
```

A decreasing loss indicates that the model is successfully learning meaningful word embeddings.
