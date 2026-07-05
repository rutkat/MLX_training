# Part 4: Transformer Architecture Deep Dive

---

## Chapter 8: "Attention Is All You Need" Understanding the Transformer

### 8.1 The Transformer Revolution

Before Transformers, NLP relied on Recurrent Neural Networks (RNNs) and LSTMs, which processed text one token at a time like reading a book word by word. The Transformer, introduced in the 2017 paper "Attention Is All You Need" by Vaswani et al., changed everything by processing all tokens in parallel and using "attention" to model relationships between any two positions in a sequence.

For web developers: if RNNs are like synchronous sequential code, Transformers are like event-driven asynchronous processing everything happens at once, with a coordination mechanism (attention) to manage dependencies.

### 8.2 Self-Attention The Core Mechanism

Self-attention allows each position in a sequence to "attend to" (focus on) all other positions. The mechanism uses three learned projections:

- Query (Q): "What am I looking for?"
- Key (K): "What do I contain?"
- Value (V): "What information do I provide?"

The attention score between two positions is the dot product of their Query and Key vectors, scaled and normalized. Let's implement it step by step:

```python
import mlx.core as mx
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute scaled dot-product attention.

    Args:
        Q: Queries, shape (batch, seq_len, d_k)
        K: Keys, shape (batch, seq_len, d_k)
        V: Values, shape (batch, seq_len, d_v)
        mask: Optional mask, shape (batch, seq_len, seq_len)

    Returns:
        Output: shape (batch, seq_len, d_v)
    """
    d_k = Q.shape[-1]

    # Compute attention scores: Q @ K^T
    # Shape: (batch, seq_len, d_k) @ (batch, d_k, seq_len) -> (batch, seq_len, seq_len)
    scores = mx.matmul(Q, mx.transpose(K, (0, 2, 1))) / math.sqrt(d_k)

    # Apply mask (for causal/decoder attention)
    if mask is not None:
        scores = scores + (mask * -1e9)

    # Softmax over the key dimension
    attention_weights = mx.softmax(scores, axis=-1)

    # Weighted sum of values
    # Shape: (batch, seq_len, seq_len) @ (batch, seq_len, d_v) -> (batch, seq_len, d_v)
    output = mx.matmul(attention_weights, V)

    return output, attention_weights


# Example: Attention in action
batch_size = 2
seq_len = 4
d_model = 8

Q = mx.random.normal((batch_size, seq_len, d_model))
K = mx.random.normal((batch_size, seq_len, d_model))
V = mx.random.normal((batch_size, seq_len, d_model))

output, weights = scaled_dot_product_attention(Q, K, V)
mx.eval(output, weights)
print(f"Output shape: {output.shape}")      # (2, 4, 8)
print(f"Weights shape: {weights.shape}")    # (2, 4, 4)
print(f"Weights sum per position: {mx.sum(weights[0, 0])}")  # Should be ~1.0 (softmax)
```

### 8.3 Understanding Attention Weights

The attention weights form a matrix where entry (i, j) represents how much position i attends to position j:

```python
# Visualize attention pattern (text-based)
def print_attention_pattern(weights, tokens):
    """Print attention weights as a readable pattern."""
    seq_len = len(tokens)
    print("      " + "  ".join(f"{t:>5}" for t in tokens))
    for i, token in enumerate(tokens):
        row = weights[0, i]  # First batch item
        mx.eval(row)
        scores = [f"{row[j].item():.2f}" for j in range(seq_len)]
        print(f"{token:>5}: " + "  ".join(f"{s:>5}" for s in scores))

tokens = ["The", "cat", "sat", "down"]
# After a trained model, you'd see patterns like:
# "sat" attending strongly to "cat" (subject-verb relationship)
```

### 8.4 Multi-Head Attention

Instead of one attention computation, Transformers run multiple "heads" in parallel, each learning different attention patterns:

```python
import mlx.core as mx
import mlx.nn as nn
import math


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""

    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for Q, K, V, and output
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def __call__(self, x, mask=None):
        """
        Forward pass.

        Args:
            x: Input tensor, shape (batch, seq_len, d_model)
            mask: Optional attention mask

        Returns:
            Output tensor, same shape as input
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        Q = self.W_q(x)  # (batch, seq_len, d_model)
        K = self.W_k(x)
        V = self.W_v(x)

        # Reshape for multi-head: (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
        Q = Q.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(0, 2, 1, 3)

        # Scaled dot-product attention
        d_k = self.d_k
        scores = mx.matmul(Q, K.transpose(0, 1, 3, 2)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores + (mask * -1e9)

        attention_weights = mx.softmax(scores, axis=-1)
        context = mx.matmul(attention_weights, V)

        # Reshape back: (batch, num_heads, seq_len, d_k) -> (batch, seq_len, d_model)
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)

        # Output projection
        output = self.W_o(context)
        return output


# Example usage
d_model = 256
num_heads = 8
mha = MultiHeadAttention(d_model, num_heads)

x = mx.random.normal((4, 32, d_model))  # batch=4, seq_len=32, d_model=256
output = mha(x)
mx.eval(output)
print(f"Input shape: {x.shape}")
print(f"Output shape: {output.shape}")  # Same as input: (4, 32, 256)
```

### 8.5 Using MLX's Built-in MultiHeadAttention

MLX provides a built-in `MultiHeadAttention` layer that you can use directly:

```python
import mlx.core as mx
import mlx.nn as nn

# Built-in multi-head attention
mha = nn.MultiHeadAttention(dims=256, num_heads=8)

x = mx.random.normal((4, 32, 256))
output = mha(x, x, x)  # Self-attention: queries, keys, values all from x
mx.eval(output)
print(f"Output shape: {output.shape}")  # (4, 32, 256)
```

MLX also provides a fast fused implementation via `mx.fast.scaled_dot_product_attention`:

```python
import mlx.core as mx

# Fast SDPA (fused kernel, more efficient)
# This is used internally by nn.MultiHeadAttention
Q = mx.random.normal((4, 8, 32, 32))  # (batch, heads, seq, d_k)
K = mx.random.normal((4, 8, 32, 32))
V = mx.random.normal((4, 8, 32, 32))

output = mx.fast.scaled_dot_product_attention(Q, K, V, scale=1.0 / 32.0)
mx.eval(output)
```

### 8.6 Causal Masking For Autoregressive Models

For language generation (GPT-style), we need to prevent positions from attending to future positions:

```python
import mlx.core as mx

def create_causal_mask(seq_len):
    """
    Create a causal (lower-triangular) mask.

    Position i can only attend to positions 0..i (not i+1..seq_len).
    This is essential for autoregressive language models.
    """
    # Upper triangular matrix of ones (above diagonal)
    mask = mx.triu(mx.ones((seq_len, seq_len)), k=1)
    return mask

# Example: 4x4 causal mask
mask = create_causal_mask(4)
mx.eval(mask)
print("Causal mask:")
print(mask)
# [[0, 1, 1, 1],
#  [0, 0, 1, 1],
#  [0, 0, 0, 1],
#  [0, 0, 0, 0]]
# 1 means "blocked" position can't see future tokens

# Use in attention:
# scores = scores + (mask * -1e9)  # Adds -infinity to masked positions
```

### 8.7 The Feed-Forward Network

Each Transformer layer has a position-wise feed-forward network (FFN):

```python
import mlx.nn as nn

class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)    # Expand
        self.w2 = nn.Linear(d_ff, d_model)    # Compress
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()            # GELU activation (standard in modern models)

    def __call__(self, x):
        return self.dropout(self.w2(self.activation(self.w1(x))))

# Typical dimensions: d_ff = 4 * d_model
ffn = FeedForward(d_model=256, d_ff=1024)
x = mx.random.normal((4, 32, 256))
output = ffn(x)
mx.eval(output)
print(f"FFN output shape: {output.shape}")  # (4, 32, 256)
```

### 8.8 Layer Normalization

Layer normalization stabilizes training by normalizing across the feature dimension:

```python
import mlx.nn as nn

# Standard LayerNorm
layer_norm = nn.LayerNorm(dims=256)

x = mx.random.normal((4, 32, 256))
normalized = layer_norm(x)
mx.eval(normalized)
print(f"Mean: {mx.mean(normalized).item():.4f}")  # Should be ~0
print(f"Std: {mx.std(normalized).item():.4f}")     # Should be ~1

# RMSNorm (used in LLaMA and many modern models faster than LayerNorm)
rms_norm = nn.RMSNorm(dims=256)
normalized_rms = rms_norm(x)
mx.eval(normalized_rms)
```

### 8.9 Residual Connections

Residual (skip) connections allow gradients to flow through deep networks:

```python
# Pre-norm Transformer block (used in GPT-2, LLaMA)
# x -> LayerNorm -> Attention -> + x -> LayerNorm -> FFN -> + x

# Post-norm Transformer block (used in original Transformer)
# x -> Attention -> + x -> LayerNorm -> FFN -> + x -> LayerNorm

# In code:
def transformer_block_pre_norm(x, attention, ffn, norm1, norm2):
    """Pre-norm Transformer block with residual connections."""
    # Attention sub-layer
    residual = x
    x = norm1(x)             # Normalize
    x = attention(x)         # Self-attention
    x = residual + x         # Residual connection

    # FFN sub-layer
    residual = x
    x = norm2(x)             # Normalize
    x = ffn(x)               # Feed-forward
    x = residual + x         # Residual connection

    return x
```

---

## Chapter 9: Building a Complete Transformer

### 9.1 Encoder-Only Architecture (BERT-style)

Encoder-only models process the entire input simultaneously and are used for understanding tasks (classification, NER, etc.):

```python
import mlx.core as mx
import mlx.nn as nn
import math


class TransformerEncoderBlock(nn.Module):
    """A single Transformer encoder block."""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x, mask=None):
        # Self-attention with residual
        residual = x
        x = self.norm1(x)
        x = self.attention(x, mask)
        x = self.dropout(x)
        x = residual + x

        # FFN with residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x


class TransformerEncoder(nn.Module):
    """Full Transformer encoder stack."""

    def __init__(self, vocab_size, d_model, num_heads, d_ff,
                 num_layers, max_seq_len, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.layers = [
            TransformerEncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ]
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, input_ids, mask=None):
        seq_len = input_ids.shape[1]
        positions = mx.arange(seq_len)

        # Embed tokens + positions
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)
        x = self.dropout(x)

        # Pass through encoder layers
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


# Build an encoder for classification
class BertClassifier(nn.Module):
    """BERT-style classifier for text classification."""

    def __init__(self, vocab_size, num_classes, d_model=256,
                 num_heads=8, num_layers=4, max_seq_len=128):
        super().__init__()
        self.encoder = TransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_model * 4,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_classes),
        )

    def __call__(self, input_ids):
        # Encode
        hidden = self.encoder(input_ids)  # (batch, seq_len, d_model)

        # Pool: use mean of all token representations
        pooled = mx.mean(hidden, axis=1)  # (batch, d_model)

        # Classify
        logits = self.classifier(pooled)  # (batch, num_classes)
        return logits
```

### 9.2 Decoder-Only Architecture (GPT-style)

Decoder-only models generate text autoregressively predicting one token at a time:

```python
import mlx.core as mx
import mlx.nn as nn
import math


class TransformerDecoderBlock(nn.Module):
    """A single Transformer decoder block with causal attention."""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def __call__(self, x, mask=None):
        # Causal self-attention with residual
        residual = x
        x = self.norm1(x)
        x = self.attention(x, mask)
        x = self.dropout(x)
        x = residual + x

        # FFN with residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x

        return x


class GPTModel(nn.Module):
    """GPT-style causal language model."""

    def __init__(self, vocab_size, d_model=256, num_heads=8,
                 num_layers=4, d_ff=1024, max_seq_len=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)

        # Transformer blocks
        self.layers = [
            TransformerDecoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ]

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

        # Output head (tied with token embedding weights)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def __call__(self, input_ids):
        """
        Forward pass.

        Args:
            input_ids: Token IDs, shape (batch, seq_len)

        Returns:
            Logits, shape (batch, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape

        # Create positions
        positions = mx.arange(seq_len)

        # Create causal mask
        causal_mask = mx.triu(
            mx.full((seq_len, seq_len), -1e9), k=1
        )

        # Embed tokens and positions
        x = self.token_embedding(input_ids) + self.pos_embedding(positions)
        x = self.dropout(x)

        # Pass through transformer blocks
        for layer in self.layers:
            x = layer(x, mask=causal_mask)

        # Final layer norm
        x = self.norm(x)

        # Project to vocabulary
        logits = self.output_proj(x)

        return logits

    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=None):
        """
        Generate text autoregressively.

        Args:
            input_ids: Starting token IDs, shape (batch, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
            top_k: If set, only sample from top k tokens

        Returns:
            Generated token IDs, shape (batch, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Truncate to max sequence length
            context = input_ids[:, -self.max_seq_len:]

            # Get logits for the last position
            logits = self(context)
            next_logits = logits[:, -1, :]  # (batch, vocab_size)

            # Apply temperature
            if temperature > 0:
                next_logits = next_logits / temperature

            # Apply top-k filtering
            if top_k is not None:
                top_values, top_indices = mx.topk(next_logits, top_k)
                probs = mx.softmax(top_values, axis=-1)
                # Sample from top-k
                idx = mx.random.categorical(mx.log(probs + 1e-10)[:, None, :])[:, None]
                next_token = mx.take_along_axis(top_indices, idx, axis=1)[:, 0]
            else:
                # Sample from full distribution
                probs = mx.softmax(next_logits, axis=-1)
                next_token = mx.random.categorical(
                    mx.log(probs + 1e-10)[:, None, :]
                )[:, 0]

            # Append to sequence
            input_ids = mx.concatenate(
                [input_ids, next_token[:, None]], axis=1
            )

        return input_ids
```

### 9.3 Using MLX's Built-in Transformer

MLX provides a built-in `nn.Transformer` for convenience:

```python
import mlx.nn as nn

# Built-in Transformer
transformer = nn.Transformer(
    dims=256,           # Model dimension
    num_heads=8,        # Number of attention heads
    num_encoder_layers=4,
    num_decoder_layers=4,
    d_ff=1024,          # Feed-forward dimension
    dropout=0.1,
)
```

### 9.4 RMSNorm and Modern Architecture Choices

Modern models like LLaMA use RMSNorm instead of LayerNorm, and SwiGLU instead of GELU in the FFN:

```python
import mlx.core as mx
import mlx.nn as nn


class LLaMABlock(nn.Module):
    """LLaMA-style transformer block with RMSNorm and SwiGLU."""

    def __init__(self, d_model, num_heads, d_ff, rope=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # RMSNorm instead of LayerNorm
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)

        # Attention projections (fused QKV)
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        # SwiGLU FFN
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)  # Gate
        self.w_up = nn.Linear(d_model, d_ff, bias=False)    # Up projection
        self.w_down = nn.Linear(d_ff, d_model, bias=False)  # Down projection

        # Rotary position embedding
        self.rope = rope or nn.RoPE(dims=self.d_k)

    def __call__(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        # Pre-norm attention
        residual = x
        x = self.norm1(x)

        # Compute Q, K, V
        Q = self.w_q(x).reshape(batch_size, seq_len, self.num_heads, self.d_k)
        K = self.w_k(x).reshape(batch_size, seq_len, self.num_heads, self.d_k)
        V = self.w_v(x).reshape(batch_size, seq_len, self.num_heads, self.d_k)

        # Transpose to (batch, heads, seq, d_k)
        Q = Q.transpose(0, 2, 1, 3)
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)

        # Apply RoPE to Q and K
        Q = self.rope(Q)
        K = self.rope(K)

        # Scaled dot-product attention
        scores = mx.matmul(Q, K.transpose(0, 1, 3, 2)) / mx.sqrt(
            mx.array(self.d_k)
        )
        if mask is not None:
            scores = scores + mask
        attention_weights = mx.softmax(scores, axis=-1)
        context = mx.matmul(attention_weights, V)

        # Reshape and project output
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        x = self.w_o(context)
        x = x + residual

        # Pre-norm FFN with SwiGLU
        residual = x
        x = self.norm2(x)
        gate = nn.silu(self.w_gate(x))   # SiLU (swish) activation on gate
        up = self.w_up(x)                # Up projection
        x = self.w_down(gate * up)       # Gated + down projection
        x = x + residual

        return x
```

### 9.5 Weight Tying

In GPT-style models, the output projection weights are often shared (tied) with the token embedding:

```python
class GPTWithWeightTying(nn.Module):
    """GPT model with weight tying between embedding and output projection."""

    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_seq_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        # ... other layers ...

    def __call__(self, input_ids):
        # ... transformer processing ...
        x = self.norm(x)

        # Use transposed embedding weights for output projection
        # This saves memory and can improve training
        logits = x @ self.token_embedding.weight.T

        return logits
```

### 9.6 Model Parameter Counting

Understanding how many parameters your model has is important:

```python
from mlx.utils import tree_flatten

def count_parameters(model):
    """Count total and trainable parameters in a model."""
    total = sum(v.size for _, v in tree_flatten(model.parameters()))
    trainable = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    return total, trainable

def format_params(n):
    """Format parameter count in human-readable form."""
    if n >= 1e9:
        return f"{n/1e9:.1f}B"
    elif n >= 1e6:
        return f"{n/1e6:.1f}M"
    elif n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)

# Example
model = GPTModel(vocab_size=50000, d_model=256, num_heads=8, num_layers=4)
total, trainable = count_parameters(model)
print(f"Total parameters: {format_params(total)}")
print(f"Trainable parameters: {format_params(trainable)}")
```

### 9.7 Memory Estimation

```python
def estimate_memory(model, batch_size=1, seq_len=128, dtype_bytes=4):
    """Estimate memory usage of a model."""
    # Parameter memory
    from mlx.utils import tree_flatten
    total_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    param_memory = total_params * dtype_bytes

    # Activation memory (rough estimate)
    # Each transformer layer stores Q, K, V, attention weights, FFN intermediates
    # Rough: ~5 * batch_size * seq_len * d_model * num_layers
    d_model = model.d_model
    num_layers = len(model.layers)
    activation_memory = (
        5 * batch_size * seq_len * d_model * num_layers * dtype_bytes
    )

    # Gradient memory (same as parameters for full fine-tuning)
    gradient_memory = param_memory

    # Optimizer state (Adam: 2x parameters for momentum and variance)
    optimizer_memory = 2 * param_memory

    total_memory = param_memory + activation_memory + gradient_memory + optimizer_memory

    print(f"Parameters: {param_memory / 1e9:.2f} GB")
    print(f"Activations: {activation_memory / 1e9:.2f} GB")
    print(f"Gradients: {gradient_memory / 1e9:.2f} GB")
    print(f"Optimizer: {optimizer_memory / 1e9:.2f} GB")
    print(f"Total: {total_memory / 1e9:.2f} GB")

    return total_memory
```

### 9.8 Model Configuration Patterns

Pre-defining configurations for different model sizes and purposes.  

```python
# Small model (for learning/testing)
small_config = {
    "vocab_size": 50000,
    "d_model": 256,
    "num_heads": 8,
    "num_layers": 4,
    "d_ff": 1024,
    "max_seq_len": 256,
    "dropout": 0.1,
}

# Medium model (for real tasks)
medium_config = {
    "vocab_size": 50000,
    "d_model": 512,
    "num_heads": 8,
    "num_layers": 8,
    "d_ff": 2048,
    "max_seq_len": 512,
    "dropout": 0.1,
}

# Large model (needs significant memory)
large_config = {
    "vocab_size": 50000,
    "d_model": 1024,
    "num_heads": 16,
    "num_layers": 16,
    "d_ff": 4096,
    "max_seq_len": 1024,
    "dropout": 0.1,
}
```

---

**Next**:  
[In Part 5](https://github.com/rutkat/MLX_training/blob/main/guide/part5_training_language_models.md), training and fine-tuning large pre-trained models with LoRA.
