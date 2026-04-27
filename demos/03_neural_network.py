"""
Demo 03: Building and Training a Neural Network
Run: python demos/03_neural_network.py

This demo shows how to build a simple neural network using mlx.nn,
train it on a classification task, and evaluate it.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import numpy as np


def count_params(model):
    """Count all parameters in a model using tree_flatten."""
    return sum(v.size for _, v in tree_flatten(model.parameters()))


def main():
    print("=" * 60)
    print("Neural Network Training Demo")
    print("=" * 60)

    # --- Generate synthetic data ---
    print("\n--- Generating Synthetic Data ---")
    np.random.seed(42)
    mx.random.seed(42)

    # Two moons dataset (binary classification)
    n_samples = 500
    X = np.random.randn(n_samples, 2).astype(np.float32)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 < 1.5).astype(np.int32)

    # Convert to MLX arrays
    X_train = mx.array(X[:400])
    y_train = mx.array(y[:400])
    X_test = mx.array(X[400:])
    y_test = mx.array(y[400:])

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Classes: {len(set(y))}")

    # --- Define the model ---
    print("\n--- Building the Model ---")

    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(2, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 2),
            )

        def __call__(self, x):
            return self.layers(x)

    model = Classifier()
    total_params = count_params(model)
    print(f"Model created with {total_params} parameters")

    # --- Loss function ---
    def loss_fn(model, X, y):
        logits = model(X)
        return nn.losses.cross_entropy(logits, y, reduction="mean")

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # --- Optimizer ---
    lr_schedule = optim.cosine_decay(1e-2, 500, 1e-5)
    optimizer = optim.AdamW(learning_rate=lr_schedule)

    # --- Training loop ---
    print("\n--- Training ---")
    model.train()
    batch_size = 64

    for epoch in range(20):
        # Shuffle
        perm = np.random.permutation(400)
        total_loss = 0.0
        correct = 0

        for i in range(0, 400, batch_size):
            idx = perm[i:i + batch_size]
            batch_X = mx.array(X[idx])
            batch_y = mx.array(y[idx])

            loss, grads = loss_and_grad(model, batch_X, batch_y)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)

            total_loss += loss.item() * len(idx)

            # Accuracy
            preds = mx.argmax(model(batch_X), axis=-1)
            mx.eval(preds)
            correct += mx.sum(preds == batch_y).item()

        avg_loss = total_loss / 400
        accuracy = correct / 400

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d}: loss = {avg_loss:.4f}, "
                  f"accuracy = {accuracy:.4f}")

    # --- Evaluate ---
    print("\n--- Evaluation ---")
    model.eval()
    logits = model(X_test)
    predictions = mx.argmax(logits, axis=-1)
    mx.eval(predictions)
    test_accuracy = mx.mean(predictions == y_test).item()
    print(f"Test accuracy: {test_accuracy:.4f}")

    # --- Using different nn layers ---
    print("\n--- Available MLX NN Layers ---")
    print("Linear layers: nn.Linear, nn.QuantizedLinear")
    print("Convolutions: nn.Conv1d, nn.Conv2d, nn.Conv3d")
    print("Normalizations: nn.LayerNorm, nn.RMSNorm, nn.GroupNorm, nn.BatchNorm")
    print("Activations: nn.ReLU, nn.GELU, nn.SiLU, nn.Sigmoid, nn.Tanh")
    print("Attention: nn.MultiHeadAttention")
    print("Embeddings: nn.Embedding")
    print("Sequential: nn.Sequential")
    print("Dropout: nn.Dropout")
    print("RNN: nn.RNN, nn.GRU, nn.LSTM")
    print("Positional: nn.RoPE, nn.SinusoidalPositionalEncoding, nn.ALiBi")

    print("\n" + "=" * 60)
    print("Demo complete!")


if __name__ == "__main__":
    main()
