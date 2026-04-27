"""
Demo 06: LoRA Fine-Tuning
Run: python demos/06_lora_finetuning.py

This demo shows how to implement LoRA (Low-Rank Adaptation) from scratch
in MLX for parameter-efficient fine-tuning.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import numpy as np
import math


# --- LoRA Layer ---
class LoRALinear(nn.Module):
    """Linear layer with LoRA (Low-Rank Adaptation)."""

    def __init__(self, in_features, out_features, rank=8, alpha=16.0, bias=True):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank

        # Base weight (frozen)
        self.base_linear = nn.Linear(in_features, out_features, bias=bias)

        # LoRA matrices
        # A: (in_features, rank) - initialized with small random values
        # B: (rank, out_features) - initialized with zeros
        self.lora_A = mx.random.normal(shape=(in_features, rank)) * 0.01
        self.lora_B = mx.zeros((rank, out_features))

    def __call__(self, x):
        # Base output (frozen weights)
        base_out = self.base_linear(x)
        # LoRA output (trainable)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scale
        return base_out + lora_out


def main():
    print("=" * 60)
    print("LoRA Fine-Tuning Demo")
    print("=" * 60)

    # --- Compare parameter counts ---
    print("\n--- Parameter Comparison ---")

    # Standard model
    class StandardModel(nn.Module):
        def __init__(self, d_model=256, d_ff=1024):
            super().__init__()
            self.linear1 = nn.Linear(d_model, d_ff)
            self.linear2 = nn.Linear(d_ff, d_model)
            self.linear3 = nn.Linear(d_model, d_model)

        def __call__(self, x):
            x = nn.relu(self.linear1(x))
            x = self.linear2(x)
            return self.linear3(x)

    # LoRA model
    class LoRAModel(nn.Module):
        def __init__(self, d_model=256, d_ff=1024, rank=8):
            super().__init__()
            self.linear1 = LoRALinear(d_model, d_ff, rank=rank)
            self.linear2 = LoRALinear(d_ff, d_model, rank=rank)
            self.linear3 = LoRALinear(d_model, d_model, rank=rank)

        def __call__(self, x):
            x = nn.relu(self.linear1(x))
            x = self.linear2(x)
            return self.linear3(x)

    standard = StandardModel()
    lora_model = LoRAModel()

    # Count parameters
    def count_params(model):
        return sum(v.size for _, v in tree_flatten(model.parameters()))

    def count_trainable(model):
        return sum(v.size for _, v in tree_flatten(model.trainable_parameters()))

    std_params = count_params(standard)
    lora_params = count_params(lora_model)

    print(f"Standard model parameters: {std_params:,}")
    print(f"LoRA model total parameters: {lora_params:,}")

    # Freeze base weights, count only LoRA params
    # In practice, lora_A and lora_B are the only trainable params
    lora_trainable = (
        lora_model.linear1.lora_A.size + lora_model.linear1.lora_B.size +
        lora_model.linear2.lora_A.size + lora_model.linear2.lora_B.size +
        lora_model.linear3.lora_A.size + lora_model.linear3.lora_B.size
    )
    print(f"LoRA trainable parameters: {lora_trainable:,}")
    print(f"Reduction: {100 * (1 - lora_trainable / std_params):.1f}% fewer trainable params")

    # --- Training demonstration ---
    print("\n--- Training Demonstration ---")

    d_model = 64
    d_ff = 256
    rank = 4
    batch_size = 32
    num_samples = 500

    # Generate synthetic task: learn a non-linear function
    np.random.seed(42)
    mx.random.seed(42)

    X_np = np.random.randn(num_samples, d_model).astype(np.float32)
    W_np = np.random.randn(1, 1).astype(np.float32) * 0.5  # Single shared feature
    y_np = (np.tanh(X_np[:, :1] @ W_np) + 0.5).astype(np.float32)

    X = mx.array(X_np)
    y = mx.array(y_np)

    # Create LoRA model
    class SimpleLoRANet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                LoRALinear(d_model, d_ff, rank=rank),
                nn.ReLU(),
                LoRALinear(d_ff, d_model, rank=rank),
            )
            self.head = nn.Linear(d_model, 1)

        def __call__(self, x):
            return self.head(self.net(x))

    model = SimpleLoRANet()

    # Freeze base weights
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            module.base_linear.freeze()

    model.head.unfreeze()  # Keep head trainable

    trainable = count_trainable(model)
    total = count_params(model)
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # Training setup
    def loss_fn(model, X, y):
        pred = model(X)
        return mx.mean((pred - y) ** 2)

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    optimizer = optim.Adam(learning_rate=1e-3)

    # Train
    model.train()
    for epoch in range(30):
        indices = np.random.permutation(num_samples)
        total_loss = 0.0
        n_batches = 0

        for i in range(0, num_samples, batch_size):
            batch_idx = indices[i:i + batch_size]
            batch_X = mx.array(X_np[batch_idx])
            batch_y = mx.array(y_np[batch_idx])

            loss, grads = loss_and_grad(model, batch_X, batch_y)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)

            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / n_batches
            print(f"Epoch {epoch+1:3d}: loss = {avg_loss:.4f}")

    print("\n--- Rank Comparison ---")
    print("LoRA rank affects the capacity of the adaptation:")
    for r in [1, 2, 4, 8, 16, 32]:
        params = r * (d_model + d_ff + d_model)  # A and B matrices
        pct = 100 * params / total
        print(f"  rank={r:2d}: {params:6,} trainable params ({pct:.1f}% of total)")

    print("\n--- Tips for LoRA Fine-Tuning ---")
    print("1. Use rank 8-16 for most tasks")
    print("2. Apply LoRA to attention projection layers (Q, K, V, O)")
    print("3. Higher alpha = stronger adaptation")
    print("4. Use a low learning rate (1e-5 to 5e-4)")
    print("5. For production, use mlx-lm's built-in LoRA support:")
    print("   mlx_lm.lora --model <model> --data <data_dir>")

    print("\n" + "=" * 60)
    print("Demo complete!")


if __name__ == "__main__":
    main()
