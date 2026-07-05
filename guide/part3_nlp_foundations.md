# Part 3: NLP Foundations

---

## Chapter 6: Text as Numbers - Tokenization and Embeddings

### 6.1 Why Text Needs Processing

Neural networks operate on numbers, not text. Before we can train a model to understand language, we need to convert text into numerical representations. This process involves three key steps:

1. **Tokenization**: Breaking text into pieces (tokens)
2. **Numericalization**: Mapping tokens to integer IDs
3. **Embedding**: Converting integer IDs to dense vector representations

As a web developer, you can think of this as similar to how a browser processes HTML: raw text is parsed into a DOM tree (tokenization), elements get assigned identifiers (numericalization), and then the rendering engine uses those to compute layouts and paint pixels (embedding/forward pass).

### 6.2 Tokenization Strategies

#### Character-Level Tokenization

The simplest approach: each character is a token.

```python
text = "Hello, world!"
tokens = list(text)
print(tokens)  # ['H', 'e', 'l', 'l', 'o', ',', ' ', 'w', 'o', 'r', 'l', 'd', '!']

# Build vocabulary
vocab = sorted(set(text))
char_to_id = {ch: i for i, ch in enumerate(vocab)}
id_to_char = {i: ch for ch, i in char_to_id.items()}
vocab_size = len(vocab)

print(f"Vocabulary: {vocab}")
print(f"Vocabulary size: {vocab_size}")
print(f"'H' -> {char_to_id['H']}")

# Encode text
encoded = [char_to_id[ch] for ch in text]
print(f"Encoded: {encoded}")

# Decode back
decoded = ''.join(id_to_char[i] for i in encoded)
print(f"Decoded: {decoded}")
```

**Pros**: Tiny vocabulary, handles any text
**Cons**: No semantic meaning, long sequences, hard to learn patterns

#### Word-Level Tokenization

Split on whitespace and punctuation:

```python
import re

text = "The cat sat on the mat."

# Simple split
words = text.lower().split()
print(words)  # ['the', 'cat', 'sat', 'on', 'the', 'mat.']

# Better: regex split keeping punctuation
words = re.findall(r'\w+|\S', text.lower())
print(words)  # ['the', 'cat', 'sat', 'on', 'the', 'mat', '.']

# Build vocabulary
vocab = sorted(set(words))
word_to_id = {w: i for i, w in enumerate(vocab)}
vocab_size = len(vocab)
```

**Pros**: More semantic meaning per token, shorter sequences
**Cons**: Huge vocabulary, can't handle unknown words, different forms ("run"/"running") are separate tokens

#### Subword Tokenization (Modern Standard)

Modern NLP uses subword tokenization, a balance between character and word level. Common algorithms include BPE (Byte Pair Encoding), WordPiece, and SentencePiece.

Let's implement a simple BPE tokenizer:

```python
"""Simple BPE (Byte Pair Encoding) Tokenizer implementation."""

class SimpleBPETokenizer:
    """A minimal BPE tokenizer for educational purposes."""

    def __init__(self):
        self.merges = {}       # pair -> merged token
        self.vocab = {}        # token -> id
        self.id_to_token = {}  # id -> token

    def train(self, text, vocab_size=256, num_merges=100):
        """Train BPE on text to learn merge rules."""
        # Start with character-level tokens
        tokens = list(text)
        vocab = sorted(set(tokens))
        self.vocab = {t: i for i, t in enumerate(vocab)}

        # Count pairs
        for i in range(num_merges):
            pairs = {}
            for j in range(len(tokens) - 1):
                pair = (tokens[j], tokens[j + 1])
                pairs[pair] = pairs.get(pair, 0) + 1

            if not pairs:
                break

            # Find most common pair
            best_pair = max(pairs, key=pairs.get)
            new_token = best_pair[0] + best_pair[1]

            # Record merge
            self.merges[best_pair] = new_token
            if new_token not in self.vocab:
                self.vocab[new_token] = len(self.vocab)

            # Apply merge to token list
            new_tokens = []
            j = 0
            while j < len(tokens):
                if (j < len(tokens) - 1 and
                    tokens[j] == best_pair[0] and
                    tokens[j + 1] == best_pair[1]):
                    new_tokens.append(new_token)
                    j += 2
                else:
                    new_tokens.append(tokens[j])
                    j += 1
            tokens = new_tokens

            if len(self.vocab) >= vocab_size:
                break

        self.id_to_token = {i: t for t, i in self.vocab.items()}

    def encode(self, text):
        """Encode text to token IDs."""
        tokens = list(text)
        # Apply merges in order
        for pair, merged in self.merges.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (i < len(tokens) - 1 and
                    tokens[i] == pair[0] and
                    tokens[i + 1] == pair[1]):
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return [self.vocab.get(t, 0) for t in tokens]

    def decode(self, ids):
        """Decode token IDs back to text."""
        return ''.join(self.id_to_token.get(i, '?') for i in ids)

    @property
    def vocab_size(self):
        return len(self.vocab)


# Usage
tokenizer = SimpleBPETokenizer()
training_text = "the cat sat on the mat. the dog ran on the mat. the cat and the dog."
tokenizer.train(training_text, vocab_size=64, num_merges=30)

print(f"Vocabulary size: {tokenizer.vocab_size}")
print(f"Merges learned: {len(tokenizer.merges)}")

test = "the cat"
encoded = tokenizer.encode(test)
print(f"Encoded '{test}': {encoded}")
print(f"Decoded: {tokenizer.decode(encoded)}")
```

### 6.3 Using HuggingFace Tokenizers in Practice

For real projects, use production-ready tokenizers from HuggingFace:

```python
from transformers import AutoTokenizer

# Load a pre-trained tokenizer (e.g., GPT-2 style)
tokenizer = AutoTokenizer.from_pretrained("gpt2")

text = "Hello, world! Machine learning is amazing."

# Tokenize
tokens = tokenizer.encode(text)
print(f"Token IDs: {tokens}")
print(f"Number of tokens: {len(tokens)}")

# See the actual tokens
token_strings = tokenizer.convert_ids_to_tokens(tokens)
print(f"Tokens: {token_strings}")

# Decode back
decoded = tokenizer.decode(tokens)
print(f"Decoded: {decoded}")

# Vocabulary size
print(f"Vocabulary size: {tokenizer.vocab_size}")
```

### 6.4 From Tokens to Arrays in MLX

Once we have token IDs, we convert them to MLX arrays:

```python
import mlx.core as mx
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

def tokenize(text, max_length=128):
    """Convert text to MLX array of token IDs."""
    ids = tokenizer.encode(
        text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )
    return mx.array(ids)

# Single example
text = "The quick brown fox jumps over the lazy dog."
input_ids = tokenize(text)
print(f"Shape: {input_ids.shape}")  # (128,)
print(f"Dtype: {input_ids.dtype}")  # int32

# Batch of examples
texts = [
    "The quick brown fox.",
    "Machine learning with MLX.",
    "Apple silicon is fast.",
]
batch_ids = mx.stack([tokenize(t) for t in texts])
print(f"Batch shape: {batch_ids.shape}")  # (3, 128)
```

### 6.5 Embeddings Giving Tokens Meaning

An embedding converts sparse token IDs into dense vectors that capture semantic relationships. Similar words get similar vectors.

```python
import mlx.core as mx
import mlx.nn as nn

# An embedding layer maps token IDs to vectors
vocab_size = 10000
embed_dim = 256

embedding = nn.Embedding(vocab_size, embed_dim)

# Token IDs (batch of 4 sequences, each 32 tokens long)
token_ids = mx.random.randint(0, vocab_size, (4, 32))

# Get embeddings
embedded = embedding(token_ids)
print(f"Token IDs shape: {token_ids.shape}")     # (4, 32)
print(f"Embedded shape: {embedded.shape}")        # (4, 32, 256)

# Each token ID is now a 256-dimensional vector
# Similar tokens will have similar vectors after training
```

**Web Developer Analogy**: An embedding is like a CSS color code. Just as `#FF0000` represents "red" and `#FF0100` is a very similar shade, embedding vectors for "king" and "queen" would be numerically close.

### 6.6 Understanding Embedding Geometry

The magic of embeddings is that they capture relationships in their vector geometry:

```python
import mlx.core as mx
import mlx.nn as mx_nn

# After training, embeddings capture relationships like:
# vec("king") - vec("man") + vec("woman") ≈ vec("queen")
# vec("paris") - vec("france") + vec("italy") ≈ vec("rome")

# Let's simulate this concept:
vocab_size = 10000
embed_dim = 128
embedding = mx_nn.Embedding(vocab_size, embed_dim)

# Compute cosine similarity between two embeddings
def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    a_norm = a / mx.sqrt(mx.sum(a ** 2))
    b_norm = b / mx.sqrt(mx.sum(b ** 2))
    return mx.sum(a_norm * b_norm)

# Before training, embeddings are random
king_emb = embedding(mx.array(100))   # Token 100 ("king")
queen_emb = embedding(mx.array(101))  # Token 101 ("queen")
apple_emb = embedding(mx.array(200))  # Token 200 ("apple")

# Similarity (before training random)
sim_king_queen = cosine_similarity(king_emb, queen_emb)
sim_king_apple = cosine_similarity(king_emb, apple_emb)
mx.eval(sim_king_queen, sim_king_apple)
print(f"Similarity(king, queen): {sim_king_queen.item():.4f}")
print(f"Similarity(king, apple): {sim_king_apple.item():.4f}")
# Before training, these are random after training, king/queen should be higher
```

### 6.7 Positional Encoding

Transformers don't inherently understand token order. We need to add position information:

```python
import mlx.core as mx
import math

def sinusoidal_position_encoding(max_length, embed_dim):
    """
    Create sinusoidal positional encodings as described in
    'Attention Is All You Need' (Vaswani et al., 2017).

    Args:
        max_length: Maximum sequence length
        embed_dim: Embedding dimension

    Returns:
        Array of shape (max_length, embed_dim)
    """
    position = mx.arange(max_length).reshape(max_length, 1)
    div_term = mx.exp(
        mx.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
    )

    pe = mx.zeros((max_length, embed_dim))
    pe[:, 0::2] = mx.sin(position * div_term)  # Even dimensions
    pe[:, 1::2] = mx.cos(position * div_term)  # Odd dimensions

    return pe

# Create positional encodings
max_len = 512
d_model = 256
pos_enc = sinusoidal_position_encoding(max_len, d_model)
print(f"Position encoding shape: {pos_enc.shape}")  # (512, 256)

# Add to embeddings
seq_len = 32
batch_size = 4

token_embeddings = mx.random.normal((batch_size, seq_len, d_model))
position_embeddings = pos_enc[:seq_len]  # (32, 256)

# Broadcasting: position_embeddings is added to each sequence in the batch
combined = token_embeddings + position_embeddings
print(f"Combined shape: {combined.shape}")  # (4, 32, 256)
```

MLX also provides built-in positional encoding layers:

```python
import mlx.nn as nn

# Sinusoidal positional encoding (built-in)
sin_pe = nn.SinusoidalPositionalEncoding(d_model)

# Rotary Position Embedding (RoPE) used in modern models like LLaMA
rope = nn.RoPE(dims=d_model)

# ALiBi (Attention with Linear Biases) another positional encoding approach
alibi = nn.ALiBi()
```

### 6.8 Building a Complete Text Preprocessing Pipeline

Let's put it all together into a reusable preprocessing pipeline:

```python
"""Text preprocessing pipeline for NLP tasks in MLX."""

import mlx.core as mx
import mlx.nn as nn
from transformers import AutoTokenizer
import math


class TextPreprocessor:
    """Complete text preprocessing for transformer models."""

    def __init__(self, model_name="gpt2", max_length=128, embed_dim=256):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.embed_dim = embed_dim
        self.vocab_size = self.tokenizer.vocab_size

        # Padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Embedding layer
        self.token_embedding = nn.Embedding(self.vocab_size, embed_dim)

        # Positional encoding
        self.pos_encoding = self._create_pos_encoding(max_length, embed_dim)

    def _create_pos_encoding(self, max_length, embed_dim):
        """Sinusoidal positional encoding."""
        position = mx.arange(max_length).reshape(max_length, 1)
        div_term = mx.exp(
            mx.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )
        pe = mx.zeros((max_length, embed_dim))
        pe[:, 0::2] = mx.sin(position * div_term)
        pe[:, 1::2] = mx.cos(position * div_term)
        return pe

    def process(self, texts):
        """
        Process a list of texts into embedded tensors.

        Args:
            texts: List of strings

        Returns:
            tuple of (input_ids, embeddings, attention_mask)
        """
        # Tokenize
        encoded = self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors=None,
        )

        # Convert to MLX arrays
        input_ids = mx.array(encoded["input_ids"])
        attention_mask = mx.array(encoded["attention_mask"])

        # Embed tokens
        token_emb = self.token_embedding(input_ids)

        # Add positional encoding
        positions = self.pos_encoding[:self.max_length]
        embeddings = token_emb + positions

        return input_ids, embeddings, attention_mask

    def decode(self, ids):
        """Decode token IDs back to text."""
        if isinstance(ids, mx.array):
            ids = ids.tolist()
        return self.tokenizer.decode(ids, skip_special_tokens=True)
```

---

## Chapter 7: NLP Tasks and Data Preparation

### 7.1 Common NLP Tasks

As a machine learning engineer, here are some NLP tasks and patterns you'll encounter. This section includes task-specific python classes and not self-contained executable scripts. They are considered foundational in machine learning and therefore adapted here to MLX.  

| Task | Description | Example |
|------|-------------|---------|
| **Text Classification** | Assign a category to text | Spam detection, sentiment analysis |
| **Token Classification** | Assign a label to each token | Named Entity Recognition (NER) |
| **Text Generation** | Generate new text | Chatbots, code completion |
| **Sequence-to-Sequence** | Transform input to output | Translation, summarization |
| **Masked Language Modeling** | Predict masked tokens | BERT pre-training |
| **Causal Language Modeling** | Predict next token | GPT pre-training |

### 7.2 The Dataset Object

MLX doesn't have a built-in dataset class, but we can use the HuggingFace datasets library and wrap it for MLX. The data examples and scripts are for demonstrative purposes. For practice, feel free to use your own data. Free datasets from Kaggle and HuggingFace are good options as well.

```python
"""Dataset utilities for MLX-based NLP training."""

import mlx.core as mx
import numpy as np
from torch.utils.data import Dataset as TorchDataset


class TextDataset:
    """A simple text dataset for language modeling."""

    def __init__(self, texts, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        for text in texts:
            encoding = tokenizer(
                text,
                max_length=max_length,
                truncation=True,
                padding="max_length",
            )
            self.examples.append({
                "input_ids": encoding["input_ids"],
                "attention_mask": encoding["attention_mask"],
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        return {
            "input_ids": mx.array(example["input_ids"]),
            "attention_mask": mx.array(example["attention_mask"]),
        }

    def get_batch(self, indices):
        """Get a batch of examples."""
        batch = [self[i] for i in indices]
        return {
            "input_ids": mx.stack([b["input_ids"] for b in batch]),
            "attention_mask": mx.stack([b["attention_mask"] for b in batch]),
        }


class CausalLMDataset(TextDataset):
    """Dataset for causal (autoregressive) language modeling.

    For causal LM, the labels are the input_ids shifted by one position.
    The model learns to predict the next token.
    """

    def __getitem__(self, idx):
        example = self.examples[idx]
        input_ids = mx.array(example["input_ids"])

        # For causal LM: input is [t1, t2, t3, ...], target is [t2, t3, t4, ...]
        return {
            "input_ids": input_ids[:-1],      # All tokens except last
            "targets": input_ids[1:],          # All tokens except first
            "attention_mask": mx.array(example["attention_mask"][:-1]),
        }
```

### 7.3 Loading Real Datasets  

This example loads IMDB's movie review data from Hugging Face https://huggingface.co/datasets/stanfordnlp/imdb  

```python
from datasets import load_dataset
from transformers import AutoTokenizer

# Load a dataset (e.g., IMDB for sentiment analysis)
dataset = load_dataset("imdb")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Explore the data
print(f"Train size: {len(dataset['train'])}")
print(f"Test size: {len(dataset['test'])}")
print(f"Example: {dataset['train'][0]['text'][:200]}...")
print(f"Label: {dataset['train'][0]['label']}")  # 0 = negative, 1 = positive

# Tokenize the dataset
def tokenize_fn(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=256,
        padding="max_length",
    )

tokenized = dataset.map(tokenize_fn, batched=True)
```

### 7.4 Batching for Training

Efficient batching is crucial for training speed:

```python
import mlx.core as mx
import numpy as np

class DataLoader:
    """Simple DataLoader for MLX."""

    def __init__(self, dataset, batch_size=32, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(dataset))

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

        for start in range(0, len(self.indices), self.batch_size):
            batch_indices = self.indices[start:start + self.batch_size]
            yield self.dataset.get_batch(batch_indices)

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
```

### 7.5 The Training Loop Pattern

Here's the standard training loop you'll use throughout this guide. Think of it as the "request-response cycle" of ML training:

```python
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from tqdm import tqdm

def train(model, dataset, num_epochs=10, batch_size=32, lr=1e-3):
    """Standard training loop for an MLX model."""

    # Define loss function
    def loss_fn(model, batch):
        input_ids = batch["input_ids"]
        targets = batch["targets"]
        logits = model(input_ids)
        # Cross-entropy loss
        return nn.losses.cross_entropy(logits, targets, reduction="mean")

    # Create gradient function
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Set up optimizer
    optimizer = optim.AdamW(learning_rate=lr)

    # Set model to training mode
    model.train()

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            # Forward pass + gradient computation
            loss, grads = loss_and_grad(model, batch)

            # Update parameters
            optimizer.update(model, grads)

            # Evaluate (actually run the computation)
            mx.eval(model.parameters(), optimizer.state, loss)

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    return model
```

### 7.6 Evaluating Models  

Evaluations are crucial in determining model quality. Data is batched processed measuring the total loss value then determining average loss and perplexity.

```python
def evaluate(model, dataset, batch_size=32):
    """Evaluate model on a dataset."""
    model.eval()  # Set to evaluation mode (disables dropout, etc.)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        input_ids = batch["input_ids"]
        targets = batch["targets"]
        logits = model(input_ids)
        loss = nn.losses.cross_entropy(logits, targets, reduction="mean")
        mx.eval(loss)
        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    perplexity = float(mx.exp(mx.array(avg_loss)).item())
    return avg_loss, perplexity
```

### 7.7 Compute and Measure Perplexity of Language Models

Perplexity is the standard metric for language models. It measures how "surprised" the model is by the test data:

- Lower perplexity = better model (less surprised)
- Perplexity of 1 = perfect predictions
- Perplexity of vocab_size = random guessing

```python
import mlx.core as mx

def compute_perplexity(model, dataset, batch_size=32):
    """Compute perplexity of a language model."""
    avg_loss, perplexity = evaluate(model, dataset, batch_size)
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.2f}")
    return perplexity

# Interpretation guide:
# Perplexity > 100: Model is basically random for this data
# Perplexity ~ 30-50: Decent small model
# Perplexity ~ 10-20: Good model
# Perplexity < 10: Strong model
# Perplexity ~ 1: Overfitting (memorized training data)
```

### 7.8 Data Augmentation for NLP  

Including randomness such as swapping words and inserting stop words are augmentation techniques used in NLP.  

```python
import random
import re

class TextAugmenter:
    """Simple text augmentation techniques for NLP."""

    @staticmethod
    def random_delete(text, p=0.1):
        """Randomly delete words with probability p."""
        words = text.split()
        return ' '.join(w for w in words if random.random() > p)

    @staticmethod
    def random_swap(text, n=1):
        """Randomly swap two words n times."""
        words = text.split()
        for _ in range(n):
            if len(words) >= 2:
                i, j = random.sample(range(len(words)), 2)
                words[i], words[j] = words[j], words[i]
        return ' '.join(words)

    @staticmethod
    def random_stopword_insert(text, stopwords=None):
        """Insert a random stopword at a random position."""
        if stopwords is None:
            stopwords = ["the", "a", "an", "is", "was", "are", "were",
                         "in", "on", "at", "to", "for", "of", "with"]
        words = text.split()
        if words:
            pos = random.randint(0, len(words))
            words.insert(pos, random.choice(stopwords))
        return ' '.join(words)
```

### 7.9 Handling Large Datasets with Streaming

For datasets too large to fit in memory, the preferred processing method is through streaming.  

```python
from datasets import load_dataset

# Stream a dataset (doesn't download all at once)
dataset = load_dataset("wikitext", "wikitext-103-raw-v1", streaming=True)

# Process in chunks
def stream_batches(dataset, tokenizer, batch_size=32, max_length=128):
    """Yield batches from a streaming dataset."""
    buffer = []
    for example in dataset:
        text = example["text"]
        if not text.strip():
            continue

        tokens = tokenizer.encode(
            text,
            max_length=max_length,
            truncation=True,
        )

        if len(tokens) > 1:
            buffer.append({
                "input_ids": tokens[:-1],
                "targets": tokens[1:],
            })

        if len(buffer) >= batch_size:
            batch = buffer[:batch_size]
            buffer = buffer[batch_size:]

            yield {
                "input_ids": mx.stack([mx.array(b["input_ids"]) for b in batch]),
                "targets": mx.stack([mx.array(b["targets"]) for b in batch]),
            }
```

### 7.10 Train/Validation/Test Splits  

Split ratios depend on the total size of the dataset. For small datasets, under 10k rows use a 60/20/20 split. is typical to ensure stable estimates. For medium datasets 10k–1M rows, 80/10/10 are standard defaults. For large datasets >1M rows, ratios like 98/1/1 are used because even 1% provides statistically significant validation and test sets. 

```python
from datasets import load_dataset

# Load with splits
dataset = load_dataset("imdb")
train_data = dataset["train"]
test_data = dataset["test"]

# Create validation split from training data
split = train_data.train_test_split(test_size=0.1, seed=42)
train_data = split["train"]
val_data = split["test"]

print(f"Train: {len(train_data)} examples")
print(f"Validation: {len(val_data)} examples")
print(f"Test: {len(test_data)} examples")
```

---

**Next**:  
[In Part 4](https://github.com/rutkat/MLX_training/blob/main/guide/part4_transformer_architecture.md), we'll dive deep into the Transformer architecture.
