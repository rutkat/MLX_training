"""
Character-Level Language Model -- Full Training Script
======================================================

Trains a small transformer model to generate text character by character.
Uses the Shakespeare dataset as a default training corpus.

Usage:
    python projects/text_generator/train.py [--data PATH] [--epochs N]

Download Shakespeare:
    curl -O https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import numpy as np
import math
import argparse
import os
import time
from dataclasses import dataclass


@dataclass
class Config:
    """Model and training configuration."""
    # Model
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    max_seq_len: int = 256

    # Training
    batch_size: int = 64
    learning_rate: float = 3e-4
    num_epochs: int = 20
    warmup_steps: int = 100
    weight_decay: float = 0.01

    # Generation
    temperature: float = 0.8
    top_k: int = 40
    gen_length: int = 500


class CharDataset:
    """Character-level dataset for language modeling."""

    def __init__(self, text, seq_len=256):
        self.text = text
        self.seq_len = seq_len

        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        self.char_to_id = {ch: i for i, ch in enumerate(self.chars)}
        self.id_to_char = {i: ch for ch, i in self.char_to_id.items()}
        self.encoded = np.array([self.char_to_id[ch] for ch in text], dtype=np.int32)

    def __len__(self):
        return max(0, (len(self.encoded) - self.seq_len) // self.seq_len)

    def get_batch(self, batch_size):
        indices = np.random.randint(0, len(self.encoded) - self.seq_len - 1, size=batch_size)
        x = mx.array(np.stack([self.encoded[i:i + self.seq_len] for i in indices]))
        y = mx.array(np.stack([self.encoded[i+1:i + self.seq_len+1] for i in indices]))
        return x, y

    def encode(self, text):
        return [self.char_to_id[ch] for ch in text]

    def decode(self, ids):
        if isinstance(ids, mx.array):
            ids = ids.tolist()
        return ''.join(self.id_to_char.get(i, '?') for i in ids)


class CharTransformer(nn.Module):
    """Small character-level transformer language model."""

    def __init__(self, config: Config, vocab_size: int):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(vocab_size, config.d_model)
        self.pos_embedding = nn.Embedding(config.max_seq_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        self.layers = [
            self._make_block(config)
            for _ in range(config.num_layers)
        ]

        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, vocab_size)

    def _make_block(self, config):
        return nn.TransformerEncoderLayer(
            dims=config.d_model,
            num_heads=config.num_heads,
            mlp_dims=config.d_ff,
            dropout=config.dropout,
        )

    def __call__(self, x):
        B, T = x.shape

        tok_emb = self.token_embedding(x)
        pos_emb = self.pos_embedding(mx.arange(T))
        x = self.drop(tok_emb + pos_emb)

        mask = mx.triu(mx.full((T, T), -1e9), k=1)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)
        return self.head(x)


@mx.compile
def compiled_forward(model, x):
    """Compiled forward pass for faster generation."""
    return model(x)


def generate(model, dataset, prompt, max_tokens=500, temperature=0.8, top_k=40):
    """Generate text from the model."""
    model.eval()

    tokens = dataset.encode(prompt)
    input_ids = mx.array([tokens])

    for _ in range(max_tokens):
        context = input_ids[:, -dataset.seq_len:]
        logits = model(context)
        next_logits = logits[:, -1, :] / temperature

        if top_k > 0:
            top_vals, top_idx = mx.topk(next_logits, top_k)
            probs = mx.softmax(top_vals, axis=-1)
            sample_idx = mx.random.categorical(mx.log(probs + 1e-10)[:, None, :])
            next_token = mx.take_along_axis(top_idx, sample_idx[:, None], axis=1)[:, 0]
        else:
            probs = mx.softmax(next_logits, axis=-1)
            next_token = mx.random.categorical(mx.log(probs + 1e-10)[:, None, :])[:, 0]

        input_ids = mx.concatenate([input_ids, next_token[:, None]], axis=1)
        mx.eval(input_ids)

    return dataset.decode(input_ids[0].tolist())


def train(config, data_path):
    """Train the character-level language model."""
    # Load data
    if data_path and os.path.exists(data_path):
        with open(data_path, "r") as f:
            text = f.read()
    else:
        text = """
        First Citizen: Before we proceed any further, hear me speak.
        All: Speak, speak.
        First Citizen: You are all resolved rather to die than to famish?
        All: Resolved. resolved.
        First Citizen: First, you know Caius Marcius is chief enemy to the people.
        All: We know't, we know't.
        """ * 50  # Repeat to have enough data

    print(f"Dataset: {len(text):,} characters")

    dataset = CharDataset(text, seq_len=config.max_seq_len)
    print(f"Vocabulary size: {dataset.vocab_size}")
    print(f"Characters: {''.join(dataset.chars)}")

    # Create model
    model = CharTransformer(config, vocab_size=dataset.vocab_size)
    total_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"Model parameters: {total_params:,}")

    # Loss function
    def loss_fn(model, x, y):
        logits = model(x)
        return nn.losses.cross_entropy(
            logits.reshape(-1, dataset.vocab_size),
            y.reshape(-1),
            reduction="mean",
        )

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Optimizer
    lr_schedule = optim.cosine_decay(config.learning_rate, 1000, 1e-5)
    optimizer = optim.AdamW(learning_rate=lr_schedule, weight_decay=config.weight_decay)

    # Train
    model.train()
    steps_per_epoch = len(dataset) // config.batch_size

    print(f"\nTraining for {config.num_epochs} epochs ({steps_per_epoch} steps/epoch)")
    print("-" * 60)

    for epoch in range(config.num_epochs):
        epoch_loss = 0.0
        epoch_start = time.time()

        for step in range(steps_per_epoch):
            x, y = dataset.get_batch(config.batch_size)
            loss, grads = loss_and_grad(model, x, y)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)

            epoch_loss += loss.item()

        avg_loss = epoch_loss / steps_per_epoch
        perplexity = math.exp(min(avg_loss, 20))  # Cap to avoid overflow
        elapsed = time.time() - epoch_start

        print(f"Epoch {epoch+1:3d}/{config.num_epochs}: "
              f"loss={avg_loss:.4f}, "
              f"ppl={perplexity:.2f}, "
              f"time={elapsed:.1f}s")

        # Generate sample every 5 epochs
        if (epoch + 1) % 5 == 0:
            sample = generate(model, dataset, "The ", max_tokens=200,
                            temperature=config.temperature, top_k=config.top_k)
            print(f"\nSample: {sample[:200]}...\n")

    # Final generation
    print("\n" + "=" * 60)
    print("FINAL GENERATION")
    print("=" * 60)
    sample = generate(model, dataset, "The ", max_tokens=config.gen_length,
                      temperature=config.temperature, top_k=config.top_k)
    print(sample)

    return model, dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a character-level LM")
    parser.add_argument("--data", type=str, default=None, help="Path to text file")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--d-model", type=int, default=128, help="Model dimension")
    args = parser.parse_args()

    config = Config(num_epochs=args.epochs, d_model=args.d_model)
    train(config, args.data)
