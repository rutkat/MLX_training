# Part 5: Training Language Models

---

## Chapter 10: Training a Character-Level Language Model

### 10.1 Project Overview

We'll build a character-level language model from scratch. This small model will teach you the complete training pipeline that scales to much larger models.

**Goal**: Train a model to generate text character by character, learning from a training corpus.

### 10.2 The Complete Training Script

This section's code lives in `projects/text_generator/train.py`. Here we'll walk through each component:

```python
"""
Character-level language model training with MLX.

This script trains a small transformer to generate text one character
at a time, learning patterns from a training corpus.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import math
from dataclasses import dataclass


@dataclass
class Config:
    """Model configuration."""
    vocab_size: int = 65        # Unique characters in dataset
    d_model: int = 128          # Embedding dimension
    num_heads: int = 4          # Attention heads
    num_layers: int = 4         # Transformer layers
    d_ff: int = 512             # Feed-forward dimension (4x d_model)
    max_seq_len: int = 256      # Maximum sequence length
    dropout: float = 0.1        # Dropout rate
    learning_rate: float = 3e-4 # Learning rate
    batch_size: int = 64        # Batch size
    num_epochs: int = 20        # Number of epochs
    eval_interval: int = 100    # Evaluate every N steps
```

### 10.3 The Model Architecture

```python
class CharTransformer(nn.Module):
    """A small character-level transformer language model."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)

        # Transformer layers
        self.layers = [
            nn.TransformerEncoderLayer(
                dims=config.d_model,
                num_heads=config.num_heads,
                mlp_dims=config.d_ff,
                dropout=config.dropout,
            )
            for _ in range(config.num_layers)
        ]

        self.norm = nn.LayerNorm(config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.vocab_size)
        self.dropout = nn.Dropout(config.dropout)

    def __call__(self, x):
        B, T = x.shape

        # Token + position embeddings
        tok_emb = self.token_embedding(x)
        pos_emb = self.pos_embedding(mx.arange(T))
        x = self.dropout(tok_emb + pos_emb)

        # Causal mask
        mask = mx.triu(mx.full((T, T), -1e9), k=1)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)
        logits = self.output_proj(x)
        return logits
```

### 10.4 Data Preparation

```python
class CharDataset:
    """Character-level dataset for language modeling."""

    def __init__(self, text, seq_len=256):
        self.text = text
        self.seq_len = seq_len

        # Build vocabulary
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        self.char_to_id = {ch: i for i, ch in enumerate(self.chars)}
        self.id_to_char = {i: ch for ch, i in self.char_to_id.items()}

        # Encode entire text
        self.encoded = np.array([self.char_to_id[ch] for ch in text], dtype=np.int32)

    def __len__(self):
        return max(0, (len(self.encoded) - self.seq_len) // self.seq_len)

    def get_batch(self, batch_size):
        """Get a random batch of (input, target) pairs."""
        # Random starting positions
        indices = np.random.randint(
            0, len(self.encoded) - self.seq_len - 1, size=batch_size
        )

        x = mx.array(np.stack([
            self.encoded[i:i + self.seq_len] for i in indices
        ]))
        y = mx.array(np.stack([
            self.encoded[i + 1:i + self.seq_len + 1] for i in indices
        ]))

        return x, y

    def encode(self, text):
        """Encode text to token IDs."""
        return [self.char_to_id[ch] for ch in text]

    def decode(self, ids):
        """Decode token IDs to text."""
        return ''.join(self.id_to_char[i] for i in ids)
```

### 10.5 The Training Loop

```python
def train_char_model():
    """Train a character-level language model."""

    # Load data (Shakespeare is a common small dataset for this)
    # You can replace this with any text file
    text = open("shakespeare.txt", "r").read()
    print(f"Dataset size: {len(text)} characters")

    # Create dataset
    config = Config()
    dataset = CharDataset(text, seq_len=config.max_seq_len)
    config.vocab_size = dataset.vocab_size

    print(f"Vocabulary: {dataset.vocab_size} characters")
    print(f"Vocabulary: {''.join(dataset.chars)}")

    # Create model
    model = CharTransformer(config)

    # Count parameters
    from mlx.utils import tree_flatten
    total_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"Total parameters: {total_params:,}")

    # Loss function
    def loss_fn(model, x, y):
        logits = model(x)
        return nn.losses.cross_entropy(
            logits.reshape(-1, config.vocab_size),
            y.reshape(-1),
            reduction="mean"
        )

    # Gradient function
    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Optimizer with cosine decay
    lr_schedule = optim.cosine_decay(config.learning_rate, 1000, 1e-5)
    optimizer = optim.AdamW(learning_rate=lr_schedule)

    # Training loop
    model.train()
    steps_per_epoch = len(dataset) // config.batch_size

    for epoch in range(config.num_epochs):
        total_loss = 0.0
        num_steps = 0

        for step in range(steps_per_epoch):
            x, y = dataset.get_batch(config.batch_size)

            # Forward + backward
            loss, grads = loss_and_grad(model, x, y)

            # Update
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)

            total_loss += loss.item()
            num_steps += 1

            if step % config.eval_interval == 0:
                avg_loss = total_loss / max(num_steps, 1)
                print(f"Epoch {epoch+1}, Step {step}: loss = {avg_loss:.4f}")

        avg_loss = total_loss / num_steps
        perplexity = math.exp(avg_loss)
        print(f"Epoch {epoch+1} complete: loss = {avg_loss:.4f}, "
              f"perplexity = {perplexity:.2f}")

        # Generate sample
        sample = generate(model, dataset, "The ", max_tokens=200)
        print(f"Sample: {sample}\n")

    return model, dataset


def generate(model, dataset, prompt, max_tokens=200, temperature=0.8):
    """Generate text from the model."""
    model.eval()

    tokens = dataset.encode(prompt)
    input_ids = mx.array([tokens])

    for _ in range(max_tokens):
        # Get logits for the last position
        logits = model(input_ids)
        next_logits = logits[:, -1, :] / temperature

        # Sample
        probs = mx.softmax(next_logits, axis=-1)
        next_token = mx.random.categorical(mx.log(probs + 1e-10)[:, None, :])[:, 0]

        input_ids = mx.concatenate([input_ids, next_token[:, None]], axis=1)
        mx.eval(input_ids)

    return dataset.decode(input_ids[0].tolist())
```

### 10.6 Understanding the Training Dynamics

When training a character-level model, you should observe:

**Early training** (loss ~4.0, perplexity ~55):
```
The the the the the the the the
```
The model learns common characters first.

**Mid training** (loss ~2.0, perplexity ~7):
```
The caled the shall the pake the sone
```
The model learns word-like patterns.

**Late training** (loss ~1.5, perplexity ~4.5):
```
The king shall hear the common voice
And speak the truth of noble deed
```
The model captures sentence structure and style.

---

## Chapter 11: Training a BERT-Style Classifier

### 11.1 Transfer Learning for Classification

Instead of training from scratch, we can fine-tune a pre-trained model for specific tasks. This is much more efficient and effective.

### 11.2 Building a Sentiment Classifier

The complete code for this project is in `projects/sentiment_classifier/`. Here's the walkthrough:

```python
"""
Sentiment classification using a Transformer encoder trained with MLX.

This trains a BERT-style model on the IMDB dataset for binary
sentiment classification (positive/negative).
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import numpy as np


# Configuration
VOCAB_SIZE = 30522      # GPT-2 tokenizer vocab size
D_MODEL = 256
NUM_HEADS = 8
NUM_LAYERS = 4
D_FF = 1024
MAX_SEQ_LEN = 256
DROPOUT = 0.1
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3
```

### 11.3 The Classification Model

```python
class SentimentClassifier(nn.Module):
    """Transformer-based text classifier."""

    def __init__(self, vocab_size, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, D_MODEL)
        self.pos_embedding = nn.Embedding(MAX_SEQ_LEN, D_MODEL)
        self.embed_dropout = nn.Dropout(DROPOUT)

        self.layers = [
            nn.TransformerEncoderLayer(
                dims=D_MODEL,
                num_heads=NUM_HEADS,
                mlp_dims=D_FF,
                dropout=DROPOUT,
            )
            for _ in range(NUM_LAYERS)
        ]

        self.norm = nn.LayerNorm(D_MODEL)
        self.classifier = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(D_MODEL, num_classes),
        )

    def __call__(self, input_ids, mask=None):
        B, T = input_ids.shape

        # Embeddings
        x = self.embedding(input_ids) + self.pos_embedding(mx.arange(T))
        x = self.embed_dropout(x)

        # Encode
        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)

        # Pool: mean of non-padding tokens
        if mask is not None:
            mask_expanded = mask[:, :, None].astype(mx.float32)
            x = mx.sum(x * mask_expanded, axis=1) / mx.sum(mask_expanded, axis=1)
        else:
            x = mx.mean(x, axis=1)

        # Classify
        return self.classifier(x)
```

### 11.4 Data Preparation

```python
def prepare_data():
    """Load and prepare the IMDB dataset."""
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("imdb")

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LEN,
        )

    tokenized = dataset.map(tokenize, batched=True)

    train_data = []
    for example in tokenized["train"]:
        train_data.append({
            "input_ids": np.array(example["input_ids"], dtype=np.int32),
            "attention_mask": np.array(example["attention_mask"], dtype=np.int32),
            "label": np.array(example["label"], dtype=np.int32),
        })

    test_data = []
    for example in tokenized["test"]:
        test_data.append({
            "input_ids": np.array(example["input_ids"], dtype=np.int32),
            "attention_mask": np.array(example["attention_mask"], dtype=np.int32),
            "label": np.array(example["label"], dtype=np.int32),
        })

    return train_data, test_data, tokenizer
```

### 11.5 Training with Accuracy Tracking

```python
def train_classifier():
    """Train the sentiment classifier."""
    train_data, test_data, tokenizer = prepare_data()

    model = SentimentClassifier(vocab_size=tokenizer.vocab_size, num_classes=2)

    def loss_fn(model, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["label"]

        logits = model(input_ids, mask=None)
        return nn.losses.cross_entropy(logits, labels, reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    optimizer = optim.AdamW(learning_rate=LEARNING_RATE, weight_decay=0.01)

    model.train()

    for epoch in range(NUM_EPOCHS):
        np.random.shuffle(train_data)
        total_loss = 0.0
        correct = 0
        total = 0

        for i in tqdm(range(0, len(train_data), BATCH_SIZE),
                      desc=f"Epoch {epoch+1}"):
            batch_data = train_data[i:i + BATCH_SIZE]

            batch = {
                "input_ids": mx.stack([mx.array(d["input_ids"]) for d in batch_data]),
                "attention_mask": mx.stack([mx.array(d["attention_mask"]) for d in batch_data]),
                "label": mx.array([d["label"] for d in batch_data]),
            }

            loss, grads = loss_and_grad(model, batch)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)

            total_loss += loss.item()

            # Compute accuracy
            logits = model(batch["input_ids"])
            predictions = mx.argmax(logits, axis=-1)
            mx.eval(predictions)
            correct += mx.sum(predictions == batch["label"]).item()
            total += len(batch_data)

        avg_loss = total_loss / (len(train_data) / BATCH_SIZE)
        accuracy = correct / total
        print(f"Epoch {epoch+1}: loss = {avg_loss:.4f}, "
              f"accuracy = {accuracy:.4f}")

    return model
```

### 11.6 Understanding Loss Functions for Classification

```python
import mlx.core as mx
import mlx.nn as nn

# Cross-entropy loss: the standard for classification
# Measures how far the predicted probability distribution is from the true one

# For a single example:
logits = mx.array([[2.0, -1.0]])    # Model output (2 classes)
target = mx.array([0])               # True class is 0

loss = nn.losses.cross_entropy(logits, target)
mx.eval(loss)
print(f"Cross-entropy loss: {loss.item():.4f}")

# The loss is lower when the model assigns higher probability to the correct class
good_logits = mx.array([[10.0, -10.0]])  # Very confident correct prediction
bad_logits = mx.array([[-10.0, 10.0]])   # Very confident wrong prediction
print(f"Good prediction loss: {nn.losses.cross_entropy(good_logits, target).item():.4f}")
print(f"Bad prediction loss: {nn.losses.cross_entropy(bad_logits, target).item():.4f}")
```

---

## Chapter 12: Fine-Tuning Large Language Models with LoRA

### 12.1 Why Fine-Tuning?

Training a large language model from scratch requires enormous compute and data. Fine-tuning adapts a pre-trained model to a specific task using much less data and compute. It's like how a web developer who knows JavaScript can quickly learn TypeScript the foundation is already there.

### 12.2 LoRA Low-Rank Adaptation

LoRA is a parameter-efficient fine-tuning technique. Instead of updating all model weights, LoRA adds small "adapter" matrices that are trained while the original weights stay frozen:

```
Original: y = W @ x           (W is large, e.g., 4096 x 4096)
LoRA:     y = W @ x + B @ A @ x  (A is 4096 x r, B is r x 4096, r << 4096)
```

Where `r` (rank) is typically 8-64. This reduces trainable parameters by 90%+ while maintaining quality.

```python
import mlx.core as mx
import mlx.nn as nn


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation."""

    def __init__(self, base_linear: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base_linear = base_linear
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        # Original dimensions
        in_features = base_linear.weight.shape[1]
        out_features = base_linear.weight.shape[0]

        # LoRA matrices
        # A: (in_features, rank) initialized with Kaiming
        # B: (rank, out_features) initialized with zeros
        self.lora_A = mx.random.normal(shape=(in_features, rank)) * 0.01
        self.lora_B = mx.zeros((rank, out_features))

    def __call__(self, x):
        # Original forward pass (frozen weights)
        base_output = self.base_linear(x)

        # LoRA adaptation
        lora_output = (x @ self.lora_A @ self.lora_B) * self.scale

        return base_output + lora_output
```

### 12.3 Applying LoRA to a Model

```python
def apply_lora(model, rank=8, alpha=16.0, target_modules=None):
    """
    Apply LoRA to specific layers of a model.

    Args:
        model: The nn.Module to modify
        rank: LoRA rank
        alpha: LoRA alpha (scaling factor)
        target_modules: List of module name patterns to apply LoRA to
    """
    if target_modules is None:
        # Default: apply to attention projection layers
        target_modules = ["w_q", "w_k", "w_v", "w_o"]

    def _apply_to_module(name, module):
        for attr_name in dir(module):
            child = getattr(module, attr_name)
            if isinstance(child, nn.Linear):
                # Check if this module name matches any target
                should_apply = any(t in attr_name for t in target_modules)
                if should_apply:
                    lora_layer = LoRALinear(child, rank=rank, alpha=alpha)
                    setattr(module, attr_name, lora_layer)

    # Apply LoRA and freeze base weights
    model.apply_to_modules(lambda n, m: _apply_to_module(n, m))

    # Freeze all parameters except LoRA
    model.freeze()
    # Unfreeze LoRA parameters
    def _unfreeze_lora(module):
        if isinstance(module, LoRALinear):
            module.lora_A.requires_grad = True
            module.lora_B.requires_grad = True
    model.apply_to_modules(lambda n, m: _unfreeze_lora(m))

    return model
```

### 12.4 Using MLX-LM for Fine-Tuning

The `mlx-lm` package provides a ready-to-use fine-tuning pipeline:

```bash
# Fine-tune with LoRA using mlx-lm
# Prepare your data in JSONL format:
# {"text": "prompt and response text"}
# {"text": "another example"}

# Run fine-tuning
mlx_lm.lora \
    --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
    --data ./my_data \
    --batch-size 4 \
    --lora-layers 16 \
    --iters 1000 \
    --learning-rate 1e-5
```

### 12.5 Fine-Tuning Data Format

For supervised fine-tuning (SFT), organize your data as JSONL:

```jsonl
{"messages": [{"role": "system", "content": "You are a helpful coding assistant."}, {"role": "user", "content": "Write a Python function to reverse a string."}, {"role": "assistant", "content": "def reverse_string(s):\n    return s[::-1]"}]}
{"messages": [{"role": "system", "content": "You are a helpful coding assistant."}, {"role": "user", "content": "Explain REST APIs."}, {"role": "assistant", "content": "REST (Representational State Transfer) is an architectural style for web services..."}]}
```

### 12.6 A Complete LoRA Fine-Tuning Script

The full project code is in `projects/lora_finetune/`. Here's the core training loop:

```python
"""
LoRA fine-tuning script for MLX.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import numpy as np
from tqdm import tqdm


# Configuration
MODEL_NAME = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
LORA_RANK = 8
LORA_ALPHA = 16.0
LEARNING_RATE = 1e-5
BATCH_SIZE = 4
NUM_ITERS = 1000
SEQ_LEN = 256


def load_data(path):
    """Load JSONL training data."""
    data = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                example = json.loads(line)
                text = example.get("text", "")
                if text:
                    data.append(text)
    return data


def create_batches(data, tokenizer, batch_size, seq_len):
    """Tokenize and batch data."""
    all_tokens = []
    for text in data:
        tokens = tokenizer.encode(text)[:seq_len]
        if len(tokens) > 1:
            all_tokens.append(tokens)

    # Pad to uniform length
    max_len = min(max(len(t) for t in all_tokens), seq_len)
    padded = []
    for tokens in all_tokens:
        if len(tokens) < max_len:
            tokens = tokens + [0] * (max_len - len(tokens))
        padded.append(tokens[:max_len])

    # Create batches
    batches = []
    for i in range(0, len(padded), batch_size):
        batch = padded[i:i + batch_size]
        if len(batch) == batch_size:
            input_ids = mx.array(batch)[:, :-1]
            targets = mx.array(batch)[:, 1:]
            batches.append((input_ids, targets))

    return batches


def train_lora():
    """Fine-tune a model with LoRA."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load data
    train_data = load_data("train.jsonl")
    val_data = load_data("val.jsonl")

    train_batches = create_batches(train_data, tokenizer, BATCH_SIZE, SEQ_LEN)
    val_batches = create_batches(val_data, tokenizer, BATCH_SIZE, SEQ_LEN)

    print(f"Training batches: {len(train_batches)}")
    print(f"Validation batches: {len(val_batches)}")

    # Note: In practice, use mlx-lm's built-in LoRA support
    # This shows the conceptual training loop
    print("Use mlx_lm.lora for production fine-tuning")
    print(f"  mlx_lm.lora --model {MODEL_NAME} --data ./data --iters {NUM_ITERS}")
```

### 12.7 Evaluating Fine-Tuned Models

```python
def evaluate_generation(model, tokenizer, prompts, max_tokens=100):
    """Evaluate a fine-tuned model on generation quality."""
    model.eval()

    for prompt in prompts:
        # Tokenize
        input_ids = mx.array([tokenizer.encode(prompt)])

        # Generate
        output_ids = model.generate(input_ids, max_new_tokens=max_tokens)

        # Decode
        text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
        print(f"Prompt: {prompt}")
        print(f"Output: {text}")
        print("-" * 80)
```

### 12.8 Quantization for Efficient Inference

After fine-tuning, you can quantize the model for even more efficient inference:

```python
import mlx.core as mx
import mlx.nn as nn

# Quantize a model to 4-bit (reduces memory by ~4x)
def quantize_model(model, bits=4, group_size=64):
    """Quantize model weights for efficient inference."""
    nn.quantize(model, bits=bits, group_size=group_size)
    return model

# Or use mlx-lm from the command line:
# mlx_lm.convert --model ./my_finetuned_model -q --quantize --bits 4
```

### 12.9 Saving and Loading Fine-Tuned Models

```python
import mlx.core as mx

# Save LoRA weights (only the adapter weights, not the full model)
def save_lora_weights(model, path):
    """Save only the LoRA adapter weights."""
    lora_params = {}
    for name, param in model.parameters().items():
        if "lora" in name:
            lora_params[name] = param
    mx.save_safetensors(path, lora_params)

# Load weights
def load_weights(model, path):
    """Load weights from a safetensors file."""
    weights = mx.load(path)
    model.load_weights(list(weights.items()))
    return model
```

### 12.10 Training Tips and Best Practices

**Learning Rate Scheduling:**

```python
import mlx.optimizers as optim

# Warmup + cosine decay (standard for fine-tuning)
warmup_steps = 100
total_steps = 1000

# Linear warmup
warmup_schedule = optim.linear_schedule(0, LEARNING_RATE, warmup_steps)

# Cosine decay after warmup
decay_schedule = optim.cosine_decay(LEARNING_RATE, total_steps - warmup_steps, 1e-6)

# Combine
lr_schedule = optim.join_schedules([warmup_schedule, decay_schedule], [warmup_steps])
optimizer = optim.AdamW(learning_rate=lr_schedule, weight_decay=0.01)
```

**Gradient Clipping** (prevent exploding gradients):

```python
# Clip gradient norm to prevent instability
grads = optim.clip_grad_norm(grads, max_norm=1.0)
```

**Mixed Precision Training**:

```python
# Use float16 for forward pass, float32 for optimizer
model.set_dtype(mx.float16)
# MLX handles the mixed precision automatically in most cases
```

**Checkpointing** (save training progress):

```python
def save_checkpoint(model, optimizer, step, path):
    """Save training checkpoint."""
    checkpoint = {
        "step": step,
        "model_state": model.parameters(),
        "optimizer_state": optimizer.state,
    }
    mx.savez(path, checkpoint)

def load_checkpoint(model, optimizer, path):
    """Load training checkpoint."""
    checkpoint = mx.load(path)
    model.update(checkpoint["model_state"])
    optimizer.state = checkpoint["optimizer_state"]
    return checkpoint["step"]
```

---

**Next**:  
[In Part 6](https://github.com/rutkat/MLX_training/blob/main/guide/part6_advanced_topics.md) advanced topics including model optimization, distributed training, and deploying MLX models.
