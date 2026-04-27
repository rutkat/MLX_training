"""
Demo 04: Transformer Block from Scratch
Run: python demos/04_transformer_block.py

This demo builds a complete Transformer decoder block step by step,
showing each component and how they connect.
"""

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
import math


def count_params(model):
    """Count all parameters in a model using tree_flatten."""
    return sum(v.size for _, v in tree_flatten(model.parameters()))


def main():
    print("=" * 60)
    print("Transformer Block Demo")
    print("=" * 60)

    # --- Step 1: Scaled Dot-Product Attention ---
    print("\n--- Step 1: Scaled Dot-Product Attention ---")

    def scaled_dot_product_attention(Q, K, V, mask=None):
        d_k = Q.shape[-1]
        scores = mx.matmul(Q, mx.transpose(K)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores + mask
        weights = mx.softmax(scores, axis=-1)
        return mx.matmul(weights, V), weights

    seq_len = 4
    d_model = 8

    Q = mx.random.normal((seq_len, d_model))
    K = mx.random.normal((seq_len, d_model))
    V = mx.random.normal((seq_len, d_model))

    output, weights = scaled_dot_product_attention(Q, K, V)
    mx.eval(output, weights)
    print(f"Q shape: {Q.shape}")
    print(f"Attention output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")
    print(f"Weights sum (should be 1.0): {mx.sum(weights[0]).item():.4f}")

    # --- Step 2: Causal Mask ---
    print("\n--- Step 2: Causal Mask (for autoregressive models) ---")

    causal_mask = mx.triu(mx.full((seq_len, seq_len), -1e9), k=1)
    mx.eval(causal_mask)
    print("Causal mask (0=can attend, -inf=blocked):")
    print(causal_mask)

    output_causal, weights_causal = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
    mx.eval(output_causal, weights_causal)
    print(f"Causal attention weights (should be lower-triangular):")
    print(mx.round(weights_causal * 100) / 100)

    # --- Step 3: Multi-Head Attention ---
    print("\n--- Step 3: Multi-Head Attention (using MLX built-in) ---")

    num_heads = 4
    d_k = d_model // num_heads
    mha = nn.MultiHeadAttention(dims=d_model, num_heads=num_heads)

    x = mx.random.normal((2, seq_len, d_model))  # batch=2
    output = mha(x, x, x)  # Self-attention: queries, keys, values are all x
    mx.eval(output)
    print(f"Input shape: {x.shape}")
    print(f"Multi-head attention output shape: {output.shape}")

    # --- Step 4: Feed-Forward Network ---
    print("\n--- Step 4: Feed-Forward Network ---")

    d_ff = d_model * 4  # 4x expansion
    ffn = nn.Sequential(
        nn.Linear(d_model, d_ff),
        nn.GELU(),
        nn.Linear(d_ff, d_model),
    )

    ffn_output = ffn(x)
    mx.eval(ffn_output)
    print(f"FFN input shape: {x.shape}")
    print(f"FFN output shape: {ffn_output.shape}")

    # --- Step 5: Full Transformer Block ---
    print("\n--- Step 5: Full Transformer Decoder Block ---")

    class TransformerBlock(nn.Module):
        def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
            super().__init__()
            self.num_heads = num_heads
            self.d_k = d_model // num_heads

            # Attention
            self.w_q = nn.Linear(d_model, d_model)
            self.w_k = nn.Linear(d_model, d_model)
            self.w_v = nn.Linear(d_model, d_model)
            self.w_o = nn.Linear(d_model, d_model)

            # FFN
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model),
            )

            # Norms
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

            self.dropout = nn.Dropout(dropout)

        def __call__(self, x, mask=None):
            B, T, D = x.shape

            # Pre-norm attention
            residual = x
            x = self.norm1(x)

            Q = self.w_q(x).reshape(B, T, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
            K = self.w_k(x).reshape(B, T, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
            V = self.w_v(x).reshape(B, T, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

            scores = mx.matmul(Q, K.transpose(0, 1, 3, 2)) / math.sqrt(self.d_k)
            if mask is not None:
                scores = scores + mask
            attn = mx.softmax(scores, axis=-1)
            context = mx.matmul(attn, V)
            context = context.transpose(0, 2, 1, 3).reshape(B, T, D)

            x = self.w_o(context)
            x = self.dropout(x)
            x = residual + x

            # Pre-norm FFN
            residual = x
            x = self.norm2(x)
            x = self.ffn(x)
            x = self.dropout(x)
            x = residual + x

            return x

    # Test the block
    block = TransformerBlock(d_model=64, num_heads=4, d_ff=256)
    x = mx.random.normal((2, 8, 64))  # batch=2, seq=8, d=64
    mask = mx.triu(mx.full((8, 8), -1e9), k=1)

    output = block(x, mask=mask)
    mx.eval(output)
    print(f"Block input shape: {x.shape}")
    print(f"Block output shape: {output.shape}")

    # Count parameters
    params = count_params(block)
    print(f"Block parameters: {params:,}")

    # --- Step 6: Full GPT-style Model ---
    print("\n--- Step 6: Full GPT-style Language Model ---")

    class MiniGPT(nn.Module):
        def __init__(self, vocab_size=100, d_model=64, num_heads=4,
                     num_layers=2, d_ff=256, max_seq_len=32):
            super().__init__()
            self.max_seq_len = max_seq_len
            self.token_embed = nn.Embedding(vocab_size, d_model)
            self.pos_embed = nn.Embedding(max_seq_len, d_model)
            self.blocks = [
                TransformerBlock(d_model, num_heads, d_ff)
                for _ in range(num_layers)
            ]
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size)

        def __call__(self, input_ids):
            B, T = input_ids.shape
            x = self.token_embed(input_ids) + self.pos_embed(mx.arange(T))
            mask = mx.triu(mx.full((T, T), -1e9), k=1)
            for block in self.blocks:
                x = block(x, mask=mask)
            x = self.norm(x)
            return self.head(x)

    model = MiniGPT(vocab_size=100, d_model=64, num_heads=4, num_layers=2)
    input_ids = mx.random.randint(0, 100, (2, 16))
    logits = model(input_ids)
    mx.eval(logits)
    total_params = count_params(model)
    print(f"Model input: {input_ids.shape}")
    print(f"Model output (logits): {logits.shape}")
    print(f"Total parameters: {total_params:,}")

    print("\n" + "=" * 60)
    print("Demo complete!")


if __name__ == "__main__":
    main()
