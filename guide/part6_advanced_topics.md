# Part 6: Advanced Topics

---

## Chapter 13: Model Optimization and Performance

### 13.1 Memory Management in MLX

Understanding memory is crucial for training large models. MLX provides tools to monitor and control memory usage:

```python
import mlx.core as mx

# Monitor memory usage
print(f"Active memory: {mx.get_active_memory() / 1e9:.2f} GB")
print(f"Peak memory: {mx.get_peak_memory() / 1e9:.2f} GB")
print(f"Cache memory: {mx.get_cache_memory() / 1e9:.2f} GB")

# Reset peak memory tracking
mx.reset_peak_memory()

# Set memory limits
mx.set_memory_limit(8 * 1024 * 1024 * 1024)   # 8 GB limit
mx.set_cache_limit(2 * 1024 * 1024 * 1024)     # 2 GB cache limit

# Clear the cache manually
mx.clear_cache()
```

### 13.2 Metal GPU Information

```python
import mlx.core as mx

if mx.metal.is_available():
    info = mx.metal.device_info()
    print(f"GPU available: True")
    print(f"Device name: {info.get('device_name', 'Unknown')}")
    print(f"Memory: {info.get('memory_size', 0) / 1e9:.1f} GB")
    print(f"Compute cores: {info.get('num_compute_cores', 'Unknown')}")
    print(f"Max threadgroup memory: {info.get('max_threadgroup_memory', 'Unknown')}")
```

### 13.3 Compilation for Speed

MLX's `compile()` function can significantly speed up repetitive computations:

```python
import mlx.core as mx
import time

@mx.compile
def compiled_forward(model, x):
    return model(x)

@mx.compile
def compiled_loss_and_grad(model, x, y):
    logits = model(x)
    loss = mx.mean((logits - y) ** 2)
    return loss

# First call: compiles the graph (slower)
# Subsequent calls: uses compiled graph (faster)
# The compilation is cached based on input shapes and types
```

**When to use `compile()`:**
- Training loops where shapes don't change between iterations
- Inference with fixed batch sizes
- Repetitive computations in data processing pipelines

**When NOT to use `compile()`:**
- Rapidly changing input shapes
- One-off computations
- Debugging (compiled functions are harder to inspect)

### 13.4 Gradient Checkpointing

For models too large to fit in memory, gradient checkpointing trades compute for memory by recomputing activations during the backward pass:

```python
import mlx.core as mx
import mlx.nn as nn

class CheckpointedTransformerBlock(nn.Module):
    """Transformer block with gradient checkpointing."""

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def _attention_forward(self, x):
        """Attention sub-layer (will be checkpointed)."""
        residual = x
        x = self.norm1(x)
        x = self.attention(x)
        return residual + self.dropout(x)

    def _ffn_forward(self, x):
        """FFN sub-layer (will be checkpointed)."""
        residual = x
        x = self.norm2(x)
        return residual + self.ffn(x)

    def __call__(self, x):
        # Use checkpointing to save memory
        x = mx.checkpoint(self._attention_forward)(x)
        x = mx.checkpoint(self._ffn_forward)(x)
        return x
```

### 13.5 Quantization

Quantization reduces model size by storing weights in lower precision:

```python
import mlx.core as mx
import mlx.nn as nn

# Quantize a model to 4-bit
model = MyLargeModel()

# Method 1: Use nn.quantize
nn.quantize(model, bits=4, group_size=64)

# Method 2: Manual quantization
weights = model.layers[0].attention.w_q.weight
quantized, scales, biases = mx.quantize(weights, bits=4, group_size=64)
# Dequantize when needed for computation:
dequantized = mx.dequantize(quantized, scales, biases, bits=4, group_size=64)

# QuantizedLinear layer (automatic quantization for linear layers)
ql = nn.QuantizedLinear(
    input_dims=4096,
    output_dims=4096,
    bias=False,
    bits=4,
    group_size=64,
)
```

### 13.6 KV-Cache for Efficient Generation

During autoregressive generation, caching previous key-value pairs avoids recomputing attention for all previous positions:

```python
import mlx.core as mx

class KVCache:
    """Key-Value cache for efficient autoregressive generation."""

    def __init__(self):
        self.keys = None
        self.values = None

    def update(self, new_keys, new_values):
        """Append new KV pairs to cache."""
        if self.keys is None:
            self.keys = new_keys
            self.values = new_values
        else:
            self.keys = mx.concatenate([self.keys, new_keys], axis=2)
            self.values = mx.concatenate([self.values, new_values], axis=2)
        return self.keys, self.values

    def reset(self):
        """Clear the cache."""
        self.keys = None
        self.values = None


def generate_with_cache(model, input_ids, max_tokens=100, temperature=0.8):
    """Generate text using KV-cache for efficiency."""
    model.eval()
    batch_size = input_ids.shape[0]

    # Initialize KV caches for each layer
    caches = [KVCache() for _ in range(len(model.layers))]

    # Prefill: process the entire input at once
    logits = model(input_ids, caches=caches)
    next_logits = logits[:, -1, :]

    generated = [input_ids]

    for _ in range(max_tokens):
        # Sample next token
        next_logits = next_logits / temperature
        probs = mx.softmax(next_logits, axis=-1)
        next_token = mx.random.categorical(mx.log(probs + 1e-10)[:, None, :])[:, 0]

        generated.append(next_token[:, None])

        # Process only the new token (not the whole sequence again!)
        logits = model(next_token[:, None], caches=caches)
        next_logits = logits[:, -1, :]
        mx.eval(next_logits)

    return mx.concatenate(generated, axis=1)
```

**Without KV-cache**: Each generation step processes all previous tokens -> O(n^2) compute
**With KV-cache**: Each step processes only the new token -> O(n) compute

### 13.7 Speculative Decoding

For even faster generation, speculative decoding uses a small "draft" model to predict multiple tokens, then validates them with the large model:

```python
# Conceptual outline (not full implementation)
def speculative_decode(draft_model, target_model, input_ids,
                       max_tokens=100, num_speculative=4):
    """
    Use a small draft model to speculate multiple tokens,
    then validate with the target model.
    """
    for _ in range(max_tokens // num_speculative):
        # 1. Draft model generates num_speculative tokens quickly
        draft_tokens = draft_model.generate(input_ids, max_tokens=num_speculative)

        # 2. Target model validates all draft tokens in parallel
        target_logits = target_model(draft_tokens)

        # 3. Accept tokens that match, reject and resample from first mismatch
        accepted = 0
        for i in range(num_speculative):
            target_prob = mx.softmax(target_logits[:, i, :])
            draft_prob = mx.softmax(draft_model(draft_tokens[:, :i+1])[:, -1, :])

            # Accept if target probability >= draft probability
            if True:  # Simplified; real implementation compares distributions
                accepted += 1
            else:
                break

        input_ids = draft_tokens[:, :accepted + 1]

    return input_ids
```

---

## Chapter 14: Distributed Training and Advanced Patterns

### 14.1 Data Parallelism

MLX supports distributed training across multiple GPUs (on machines with multiple, like Mac Pro):

```python
import mlx.core as mx

# Check distributed availability
if mx.distributed.is_available():
    # Initialize distributed
    group = mx.distributed.init()

    # Get world size and rank
    world_size = group.size()
    rank = group.rank()
    print(f"Process {rank}/{world_size}")

    # All-reduce: sum gradients across all processes
    gradients = compute_gradients()
    summed = mx.distributed.all_sum(gradients)
    averaged = summed / world_size
```

### 14.2 Tensor Parallelism

Split large model layers across multiple devices:

```python
import mlx.nn as nn

# Shard a linear layer across devices
# Instead of one (4096, 4096) matrix, use two (4096, 2048) matrices
sharded_linear = nn.ShardedToAllLinear(
    input_dims=4096,
    output_dims=4096,
    bias=False,
)

# Or use the shard_linear utility
linear = nn.Linear(4096, 4096)
sharded = nn.layers.distributed.shard_linear(linear, group)
```

### 14.3 Stream-Based Parallelism

On a single Mac, you can use streams to overlap CPU and GPU computation:

```python
import mlx.core as mx

# Create separate streams for CPU and GPU
cpu_stream = mx.new_stream(mx.cpu)
gpu_stream = mx.new_stream(mx.gpu)

# Run data preprocessing on CPU while GPU processes previous batch
with mx.stream(cpu_stream):
    # Data preprocessing
    batch = preprocess_data(raw_data)

with mx.stream(gpu_stream):
    # Model forward pass
    output = model(batch)

mx.synchronize()  # Wait for both to complete
```

### 14.4 Custom Metal Kernels

For performance-critical operations, you can write custom GPU kernels:

```python
import mlx.core as mx

# Define a custom Metal kernel
kernel = mx.fast.metal_kernel(
    name="my_kernel",
    input_names=["input"],
    output_names=["output"],
    source="""
        uint elem = thread_position_in_grid.x;
        if (elem < input.shape[0]) {
            output[elem] = input[elem] * 2.0f;
        }
    """,
    grid=((1024,),),
    threads=(256,),
)

# Use the kernel
x = mx.random.normal((1024,))
result = kernel(inputs=[x])
mx.eval(result)
```

### 14.5 Profiling and Debugging

```python
import mlx.core as mx
import time

# Simple timing
def time_operation(fn, *args, warmup=5, repeats=20):
    """Time an MLX operation."""
    # Warmup
    for _ in range(warmup):
        result = fn(*args)
        mx.eval(result)

    # Time
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn(*args)
        mx.eval(result)
        end = time.perf_counter()
        times.append(end - start)

    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time)**2 for t in times) / len(times)) ** 0.5
    print(f"Average time: {avg_time*1000:.2f} ms (±{std_time*1000:.2f} ms)")
    return avg_time

# Metal capture for profiling
if mx.metal.is_available():
    mx.metal.start_capture()
    # ... run operations ...
    mx.metal.stop_capture()
    # Opens Xcode Instruments with the trace
```

### 14.6 Exporting Models

```python
import mlx.core as mx

# Export a function as a computation graph
@mx.export_function
def my_model(x):
    return model(x)

# Export and save
mx.export_to_dot(my_model, "model_graph.dot")  # GraphViz visualization

# Import an exported function
imported_fn = mx.import_function("exported_model.mlxfn")
```

---

## Chapter 15: Putting It All Together -- Building ML-Powered Web Applications

### 15.1 Architecture for ML-Powered Web Apps

As a web developer, you'll want to integrate MLX models into web applications. Here's a practical architecture:

```
┌─────────────────────────────────────────────┐
│                  Frontend                    │
│  (React, Vue, Svelte, etc.)                 │
│  ┌─────────┐  ┌──────────┐  ┌────────────┐ │
│  │  Chat UI │  │ Text Area │  │ Settings   │ │
│  └────┬─────┘  └─────┬────┘  └──────┬─────┘ │
└───────┼──────────────┼──────────────┼────────┘
        │    HTTP/WebSocket     │
┌───────┼──────────────┼──────────────┼────────┐
│       ▼              ▼              ▼        │
│            Python Backend (FastAPI/Flask)    │
│  ┌─────────────────────────────────────┐     │
│  │         MLX Model Server             │     │
│  │  ┌──────────┐  ┌─────────────────┐  │     │
│  │  │ Tokenizer │  │ Model (MLX)     │  │     │
│  │  └──────────┘  └─────────────────┘  │     │
│  │  ┌──────────┐  ┌─────────────────┐  │     │
│  │  │ KV Cache  │  │ Response Stream  │  │     │
│  │  └──────────┘  └─────────────────┘  │     │
│  └─────────────────────────────────────┘     │
│                   Mac Server (Apple Silicon)  │
└───────────────────────────────────────────────┘
```

### 15.2 Building a Model Server with FastAPI

```python
"""
FastAPI server for serving an MLX language model.

Install: pip install fastapi uvicorn
Run: uvicorn server:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import mlx.core as mx
import mlx.nn as nn
from transformers import AutoTokenizer
import json

app = FastAPI(title="MLX Model Server")

# Global model and tokenizer
model = None
tokenizer = None


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.8
    top_k: int = 50
    stream: bool = False


class GenerateResponse(BaseModel):
    text: str
    tokens_generated: int


@app.on_event("startup")
async def load_model():
    """Load model and tokenizer on startup."""
    global model, tokenizer

    model_path = "mlx-community/Qwen3-4B-Instruct-2507-4bit"

    # Use mlx-lm for easy loading
    from mlx_lm import load
    model, tokenizer = load(model_path)
    print("Model loaded successfully!")


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate text from a prompt."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    from mlx_lm import generate as mlx_generate

    response = mlx_generate(
        model,
        tokenizer,
        prompt=request.prompt,
        max_tokens=request.max_tokens,
        temp=request.temperature,
    )

    return GenerateResponse(
        text=response,
        tokens_generated=len(tokenizer.encode(response)),
    )


@app.post("/generate/stream")
async def generate_stream(request: GenerateRequest):
    """Stream generated text tokens."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    def token_generator():
        from mlx_lm import stream_generate

        for token in stream_generate(
            model,
            tokenizer,
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temp=request.temperature,
        ):
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        token_generator(),
        media_type="text/event-stream",
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "gpu_available": mx.metal.is_available(),
    }
```

### 15.3 Frontend Integration (JavaScript/TypeScript)

```typescript
// chat-client.ts -- Frontend client for the MLX model server

interface GenerateOptions {
  prompt: string;
  maxTokens?: number;
  temperature?: number;
  stream?: boolean;
}

class MLXClient {
  private baseUrl: string;

  constructor(baseUrl: string = "http://localhost:8000") {
    this.baseUrl = baseUrl;
  }

  async generate(options: GenerateOptions): Promise<string> {
    const response = await fetch(`${this.baseUrl}/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: options.prompt,
        max_tokens: options.maxTokens ?? 100,
        temperature: options.temperature ?? 0.8,
        stream: false,
      }),
    });

    if (!response.ok) {
      throw new Error(`Server error: ${response.statusText}`);
    }

    const data = await response.json();
    return data.text;
  }

  async *generateStream(options: GenerateOptions): AsyncGenerator<string> {
    const response = await fetch(`${this.baseUrl}/generate/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: options.prompt,
        max_tokens: options.maxTokens ?? 100,
        temperature: options.temperature ?? 0.8,
        stream: true,
      }),
    });

    if (!response.ok) {
      throw new Error(`Server error: ${response.statusText}`);
    }

    const reader = response.body!.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const text = decoder.decode(value);
      const lines = text.split("\n");

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          const data = line.slice(6);
          if (data === "[DONE]") return;
          try {
            const parsed = JSON.parse(data);
            yield parsed.token;
          } catch {
            // Skip malformed lines
          }
        }
      }
    }
  }
}

// Usage example
const client = new MLXClient();

// Non-streaming
const result = await client.generate({
  prompt: "Explain machine learning in simple terms.",
  maxTokens: 200,
});
console.log(result);

// Streaming (for chat-like UIs)
for await (const token of client.generateStream({
  prompt: "Write a poem about coding.",
  maxTokens: 100,
  stream: true,
})) {
  process.stdout.write(token); // Display each token as it arrives
}
```

### 15.4 Running MLX in Production

```bash
# Using gunicorn with uvicorn workers for production
pip install gunicorn uvicorn

# Run with 4 workers
gunicorn server:app \
    --workers 4 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 120
```

### 15.5 Batched Inference

```python
def batch_generate(model, tokenizer, prompts, max_tokens=100, temperature=0.8):
    """Generate text for multiple prompts in parallel."""
    # Tokenize all prompts
    input_ids = [tokenizer.encode(p) for p in prompts]

    # Pad to same length
    max_len = max(len(ids) for ids in input_ids)
    padded = [ids + [0] * (max_len - len(ids)) for ids in input_ids]

    # Batch inference
    input_tensor = mx.array(padded)
    logits = model(input_tensor)

    # Generate for each
    results = []
    for i in range(len(prompts)):
        generated = model.generate(
            mx.array([padded[i]]),
            max_new_tokens=max_tokens,
        )
        results.append(tokenizer.decode(generated[0].tolist()))

    return results
```

### 15.6 Building a RAG (Retrieval-Augmented Generation) System

```python
"""
Simple RAG implementation using MLX and sentence embeddings.
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from transformers import AutoTokenizer


class SimpleRAG:
    """Retrieval-Augmented Generation with MLX."""

    def __init__(self, model, tokenizer, embedding_dim=256):
        self.model = model
        self.tokenizer = tokenizer
        self.embedding_dim = embedding_dim
        self.documents = []
        self.embeddings = []

    def add_documents(self, documents):
        """Add documents to the knowledge base."""
        self.documents.extend(documents)

        # Create embeddings for each document
        for doc in documents:
            tokens = self.tokenizer.encode(doc, truncation=True, max_length=256)
            embedding = self._embed(tokens)
            self.embeddings.append(embedding)

    def _embed(self, token_ids):
        """Create a simple embedding for a document."""
        # In practice, use a proper embedding model
        input_ids = mx.array([token_ids])
        hidden = self.model(input_ids)
        # Mean pooling
        return mx.mean(hidden, axis=1)

    def retrieve(self, query, top_k=3):
        """Retrieve the most relevant documents for a query."""
        query_tokens = self.tokenizer.encode(query, truncation=True, max_length=256)
        query_embedding = self._embed(query_tokens)

        # Compute similarities
        similarities = []
        for doc_embedding in self.embeddings:
            sim = mx.sum(query_embedding * doc_embedding) / (
                mx.sqrt(mx.sum(query_embedding ** 2)) *
                mx.sqrt(mx.sum(doc_embedding ** 2))
            )
            mx.eval(sim)
            similarities.append(sim.item())

        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [self.documents[i] for i in top_indices]

    def generate(self, query, max_tokens=200, top_k=3):
        """Generate a response with retrieval augmentation."""
        # 1. Retrieve relevant documents
        relevant_docs = self.retrieve(query, top_k=top_k)

        # 2. Build augmented prompt
        context = "\n".join(relevant_docs)
        augmented_prompt = (
            f"Context:\n{context}\n\n"
            f"Based on the context above, answer the following:\n"
            f"Question: {query}\n"
            f"Answer:"
        )

        # 3. Generate response
        tokens = self.tokenizer.encode(augmented_prompt)
        input_ids = mx.array([tokens])
        output_ids = self.model.generate(input_ids, max_new_tokens=max_tokens)
        response = self.tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)

        return response
```

### 15.7 Continuous Learning and Next Steps

**What you've learned in this guide:**
1. MLX fundamentals: arrays, operations, lazy evaluation
2. Function transformations: grad, vmap, compile
3. NLP basics: tokenization, embeddings, positional encoding
4. Transformer architecture: attention, encoders, decoders
5. Training: from character models to LoRA fine-tuning
6. Optimization: quantization, KV-cache, compilation
7. Deployment: building web APIs with MLX models

**Where to go next:**
- **MLX Examples**: https://github.com/ml-explore/mlx-examples -- Study production-quality implementations
- **MLX LM**: The `mlx-lm` package for more advanced LLM usage
- **HuggingFace MLX Community**: https://huggingface.co/mlx-community -- Pre-converted models
- **MLX Documentation**: https://ml-explore.github.io/mlx/build/html/index.html -- Full API reference

**Advanced topics to explore:**
- Vision-Language Models (LLaVA, CLIP) with `mlx-vlm`
- Speech models (Whisper) with MLX
- Diffusion models (Stable Diffusion, FLUX) with MLX
- Custom Metal kernels for performance-critical operations
- Multi-GPU distributed training

**Community resources:**
- MLX GitHub Discussions: https://github.com/ml-explore/mlx/discussions
- MLX Examples Issues: https://github.com/ml-explore/mlx-examples/issues
- Apple Machine Learning Research: https://machinelearning.apple.com

---

## Appendix A: Quick Reference

### A.1 MLX Core Operations Cheat Sheet

```python
import mlx.core as mx

# Arrays
a = mx.array([1, 2, 3])              # Create array
b = mx.zeros((3, 4))                  # Zeros
c = mx.ones((3, 4))                   # Ones
d = mx.random.normal((3, 4))          # Random normal
e = mx.arange(0, 10, 2)              # Range
f = mx.linspace(0, 1, 5)             # Linspace

# Operations
x + y       # Add
x @ y       # Matrix multiply
x * y       # Element-wise multiply
x.T         # Transpose
x.reshape(2, 3)  # Reshape

# Reductions
mx.sum(x)          # Sum
mx.mean(x)         # Mean
mx.max(x)          # Max
mx.softmax(x)      # Softmax

# Eval
mx.eval(x)         # Force computation

# Gradients
grad_fn = mx.grad(f)                    # Gradient
vgrad_fn = mx.value_and_grad(f)         # Value + gradient
vmap_fn = mx.vmap(f, in_axes=(0,))      # Vectorize
```

### A.2 MLX Neural Network Cheat Sheet

```python
import mlx.nn as nn

# Layers
linear = nn.Linear(in_features, out_features)
embed = nn.Embedding(vocab_size, embed_dim)
mha = nn.MultiHeadAttention(dims, num_heads)
ln = nn.LayerNorm(dims)
rms = nn.RMSNorm(dims)
drop = nn.Dropout(p=0.1)
seq = nn.Sequential(layer1, layer2, ...)

# Activations
nn.relu(x)
nn.gelu(x)
nn.silu(x)
nn.sigmoid(x)
nn.softmax(x, axis=-1)
nn.tanh(x)

# Loss functions
nn.losses.cross_entropy(logits, targets)
nn.losses.mse_loss(pred, target)
nn.losses.l1_loss(pred, target)
nn.losses.binary_cross_entropy(logits, targets)

# Module
model.parameters()           # Get all parameters (nested dict)
model.train()                # Set training mode
model.eval()                 # Set eval mode
model.freeze()               # Freeze all parameters
model.unfreeze()             # Unfreeze all parameters
model.update(new_params)     # Update parameters
model.load_weights(path)     # Load weights
nn.value_and_grad(model, fn) # Gradient with respect to model

# Counting parameters (use tree_flatten for nested dicts)
from mlx.utils import tree_flatten
total = sum(v.size for _, v in tree_flatten(model.parameters()))
```

### A.3 Optimizers Cheat Sheet

```python
import mlx.optimizers as optim

# Common optimizers
sgd = optim.SGD(learning_rate=0.01)
adam = optim.Adam(learning_rate=1e-3)
adamw = optim.AdamW(learning_rate=1e-3, weight_decay=0.01)
lion = optim.Lion(learning_rate=1e-4)

# Schedules
cosine = optim.cosine_decay(init_lr, steps, end_lr)
linear = optim.linear_schedule(start_lr, end_lr, steps)
exponential = optim.exponential_decay(init_lr, decay_rate)

# Gradient utilities
grads = optim.clip_grad_norm(grads, max_norm=1.0)
```

### A.4 Common Training Loop Template

```python
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# 1. Define model
model = MyModel()

# 2. Define loss
def loss_fn(model, batch):
    x, y = batch["input"], batch["target"]
    pred = model(x)
    return nn.losses.cross_entropy(pred, y, reduction="mean")

# 3. Get gradient function
loss_and_grad = nn.value_and_grad(model, loss_fn)

# 4. Set up optimizer
optimizer = optim.AdamW(learning_rate=1e-3)

# 5. Training loop
model.train()
for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward + backward
        loss, grads = loss_and_grad(model, batch)
        # Update
        optimizer.update(model, grads)
        # Evaluate (trigger computation)
        mx.eval(model.parameters(), optimizer.state, loss)
        # Log
        print(f"Loss: {loss.item():.4f}")
```

---

## Appendix B: Glossary

| Term | Definition |
|------|-----------|
| **Array** | A multi-dimensional container of numerical data; the core data structure in MLX |
| **Autodiff** | Automatic differentiation; computing gradients without manual calculus |
| **Attention** | Mechanism that allows a model to focus on relevant parts of the input |
| **Batch** | A group of training examples processed together |
| **Causal Masking** | Preventing a model from seeing future tokens during generation |
| **Checkpointing** | Saving model state to disk during training; also a technique to trade memory for compute |
| **Compilation** | Optimizing a computation graph for faster execution |
| **Cross-Entropy** | Loss function measuring the difference between predicted and actual probability distributions |
| **Embedding** | Dense vector representation of discrete tokens |
| **Epoch** | One complete pass through the training dataset |
| **FFN** | Feed-Forward Network; two linear layers with an activation function |
| **Fine-Tuning** | Adapting a pre-trained model to a specific task |
| **Gradient** | The derivative of a loss function with respect to model parameters |
| **KV-Cache** | Caching key-value pairs in attention to speed up generation |
| **Lazy Evaluation** | Deferring computation until results are needed |
| **LayerNorm** | Layer Normalization; normalizes activations across features |
| **Learning Rate** | Step size for parameter updates during training |
| **LoRA** | Low-Rank Adaptation; parameter-efficient fine-tuning |
| **Loss** | A number measuring how wrong the model's predictions are |
| **Multi-Head Attention** | Running multiple attention computations in parallel |
| **Perplexity** | Metric measuring how well a language model predicts text |
| **Quantization** | Reducing the precision of model weights to save memory |
| **Residual Connection** | Adding input directly to output: `output = f(x) + x` |
| **RMSNorm** | Root Mean Square Normalization; faster alternative to LayerNorm |
| **RoPE** | Rotary Position Embedding; positional encoding used in modern models |
| **Self-Attention** | Attention where Q, K, V all come from the same input |
| **Token** | A piece of text (character, word, or subword) |
| **Tokenization** | Breaking text into tokens |
| **Transformer** | Neural network architecture based on attention mechanisms |
| **Unified Memory** | CPU and GPU sharing the same memory pool (Apple silicon) |
| **vmap** | Vectorized map; automatically batching a function |
