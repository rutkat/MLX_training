"""
Sentiment Classifier -- Full Training Script
=============================================

Trains a Transformer encoder for binary sentiment classification
on the IMDB movie review dataset.

Usage:
    python projects/sentiment_classifier/train.py

Requires: pip install datasets transformers
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
import time
import os


# --- Configuration ---
VOCAB_SIZE = 50257      # GPT-2 tokenizer
D_MODEL = 256
NUM_HEADS = 8
NUM_LAYERS = 4
D_FF = 1024
MAX_SEQ_LEN = 256
DROPOUT = 0.1
BATCH_SIZE = 32
LEARNING_RATE = 5e-5
NUM_EPOCHS = 3
NUM_CLASSES = 2


# --- Model ---
class SentimentTransformer(nn.Module):
    """Transformer encoder for text classification."""

    def __init__(self, vocab_size, num_classes=2):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, D_MODEL)
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

    def __call__(self, input_ids):
        B, T = input_ids.shape

        x = self.token_embedding(input_ids) + self.pos_embedding(mx.arange(T))
        x = self.embed_dropout(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        # Mean pooling over sequence
        x = mx.mean(x, axis=1)

        return self.classifier(x)


# --- Data ---
def load_imdb_data(tokenizer, max_samples=None):
    """Load and tokenize IMDB dataset."""
    from datasets import load_dataset

    dataset = load_dataset("imdb")

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LEN,
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

    train_data = []
    for ex in tqdm(tokenized["train"], desc="Processing train"):
        if max_samples and len(train_data) >= max_samples:
            break
        train_data.append({
            "input_ids": np.array(ex["input_ids"], dtype=np.int32),
            "label": np.array(ex["label"], dtype=np.int32),
        })

    test_data = []
    for ex in tqdm(tokenized["test"], desc="Processing test"):
        if max_samples and len(test_data) >= max_samples // 5:
            break
        test_data.append({
            "input_ids": np.array(ex["input_ids"], dtype=np.int32),
            "label": np.array(ex["label"], dtype=np.int32),
        })

    return train_data, test_data


def get_batch(data, batch_size, indices):
    """Get a batch from data by indices."""
    batch = [data[i] for i in indices]
    return {
        "input_ids": mx.stack([mx.array(d["input_ids"]) for d in batch]),
        "label": mx.array([d["label"] for d in batch]),
    }


# --- Training ---
def train():
    """Train the sentiment classifier."""
    print("=" * 60)
    print("Sentiment Classifier Training")
    print("=" * 60)

    # Load tokenizer and data
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nLoading IMDB dataset...")
    train_data, test_data = load_imdb_data(tokenizer)
    print(f"Train samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")

    # Create model
    model = SentimentTransformer(vocab_size=VOCAB_SIZE, num_classes=NUM_CLASSES)
    total_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    print(f"\nModel parameters: {total_params:,}")

    # Loss
    def loss_fn(model, batch):
        logits = model(batch["input_ids"])
        return nn.losses.cross_entropy(logits, batch["label"], reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    optimizer = optim.AdamW(learning_rate=LEARNING_RATE, weight_decay=0.01)

    # Training loop
    model.train()
    steps_per_epoch = len(train_data) // BATCH_SIZE

    for epoch in range(NUM_EPOCHS):
        indices = np.random.permutation(len(train_data))
        total_loss = 0.0
        correct = 0
        total = 0
        start = time.time()

        pbar = tqdm(range(0, len(train_data), BATCH_SIZE),
                    desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

        for i in pbar:
            batch_idx = indices[i:i + BATCH_SIZE]
            if len(batch_idx) < 2:
                continue

            batch = get_batch(train_data, BATCH_SIZE, batch_idx)

            loss, grads = loss_and_grad(model, batch)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)

            total_loss += loss.item()

            # Accuracy
            logits = model(batch["input_ids"])
            preds = mx.argmax(logits, axis=-1)
            mx.eval(preds)
            correct += mx.sum(preds == batch["label"]).item()
            total += len(batch_idx)

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")

        avg_loss = total_loss / (len(train_data) / BATCH_SIZE)
        accuracy = correct / total
        elapsed = time.time() - start

        print(f"Epoch {epoch+1}: loss={avg_loss:.4f}, accuracy={accuracy:.4f}, "
              f"time={elapsed:.1f}s")

        # Evaluate on test
        model.eval()
        test_correct = 0
        test_total = 0

        for i in range(0, len(test_data), BATCH_SIZE):
            batch = get_batch(test_data, BATCH_SIZE,
                            list(range(i, min(i + BATCH_SIZE, len(test_data)))))
            logits = model(batch["input_ids"])
            preds = mx.argmax(logits, axis=-1)
            mx.eval(preds)
            test_correct += mx.sum(preds == batch["label"]).item()
            test_total += len(batch["label"])

        test_acc = test_correct / test_total
        print(f"  Test accuracy: {test_acc:.4f}")
        model.train()

    # Print some predictions
    print("\n--- Sample Predictions ---")
    model.eval()
    for i in range(5):
        sample = test_data[i]
        input_ids = mx.array([sample["input_ids"]])
        logits = model(input_ids)
        pred = mx.argmax(logits, axis=-1)
        mx.eval(pred)

        text = tokenizer.decode(sample["input_ids"][:100])
        label = "Positive" if sample["label"] == 1 else "Negative"
        pred_label = "Positive" if pred.item() == 1 else "Negative"

        print(f"\nText: {text}...")
        print(f"True: {label}, Predicted: {pred_label}")

    return model


if __name__ == "__main__":
    train()
