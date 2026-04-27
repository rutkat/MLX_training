# Part 1: Introduction and Setup

---

## Chapter 1: What is MLX?

### 1.1 MLX in a Nutshell

MLX is an array framework for machine learning on Apple silicon, created by Apple's machine learning research team. If you've ever used NumPy for numerical computing or PyTorch for deep learning, MLX will feel familiar. It provides the same kind of tensor (multi-dimensional array) operations but is specifically designed to take advantage of the unified memory architecture in Apple's M-series chips (M1, M2, M3, M4, and beyond).

For web developers, you can think of MLX as the "Node.js of machine learning frameworks" -- it's designed to be approachable, efficient, and flexible, removing many of the traditional pain points of ML development.

### 1.2 Why MLX Matters for Web Developers

As a web developer, you might wonder why you should care about yet another ML framework. Here's why MLX is particularly relevant:

**Familiar API Design.** MLX's Python API closely follows NumPy, which means the learning curve is gentle if you've done any scientific Python work. The higher-level neural network API (`mlx.nn`) and optimizer API (`mlx.optimizers`) follow PyTorch conventions.

**Runs on Your Mac.** If you're developing on a Mac with Apple silicon, you already have the hardware you need. No need for expensive GPUs or cloud instances. Your M-series Mac has a powerful GPU built right in.

**Unified Memory.** Unlike traditional setups where you have to shuttle data between CPU memory and GPU memory (a major source of complexity and bugs), Apple silicon has a unified memory architecture. The CPU and GPU share the same memory pool. MLX is designed to exploit this, meaning you don't need to think about device placement the way you do in PyTorch or TensorFlow.

**Lazy Evaluation.** Operations in MLX are lazy -- they build a computation graph but don't execute until you need the results. This enables automatic optimizations and makes it easier to reason about complex computations.

**Composable Transformations.** MLX provides powerful function transformations for automatic differentiation (computing gradients), automatic vectorization (batching), and computation graph optimization. These can be freely composed, so `grad(vmap(grad(fn)))` is perfectly valid.

### 1.3 Key Features at a Glance

| Feature | Description |
|---------|-------------|
| **NumPy-like API** | Familiar array operations with `mlx.core` |
| **PyTorch-like NN** | Neural network layers via `mlx.nn` |
| **Lazy Computation** | Operations deferred until results needed |
| **Unified Memory** | No data copies between CPU and GPU |
| **Auto-differentiation** | `grad()`, `value_and_grad()` for training |
| **Auto-vectorization** | `vmap()` for automatic batching |
| **Compilation** | `compile()` for graph optimization |
| **GPU Acceleration** | Metal backend for Apple silicon GPU |
| **Multi-device** | CPU and GPU support with stream control |

### 1.4 MLX vs Other Frameworks

| Aspect | MLX | PyTorch | JAX | TensorFlow |
|--------|-----|---------|-----|------------|
| **Memory Model** | Unified | Separate CPU/GPU | Separate CPU/GPU | Separate CPU/GPU |
| **Evaluation** | Lazy | Eager (default) | Lazy | Eager (default) |
| **Graph** | Dynamic | Dynamic | Static (via jit) | Static (via tf.function) |
| **Apple Silicon** | Native | Via MPS | Limited | Limited |
| **API Style** | NumPy-like | Unique | NumPy-like | Unique |
| **Transformations** | Composable | Limited | Composable | Limited |

### 1.5 The MLX Ecosystem

MLX isn't just a single library. It has a growing ecosystem:

- **mlx** -- The core array framework (what this guide covers)
- **mlx-lm** -- A package for LLM text generation and fine-tuning
- **mlx-data** -- Efficient data loading and preprocessing
- **mlx-vlm** -- Vision Language Model support
- **mlx-whisper** -- Speech recognition
- **mlx-examples** -- A rich collection of example implementations
- **mlx-community** (HuggingFace) -- Pre-converted model weights

The official resources you'll want to bookmark:

- Documentation: https://ml-explore.github.io/mlx/build/html/index.html
- GitHub: https://github.com/ml-explore/mlx
- Examples: https://github.com/ml-explore/mlx-examples
- HuggingFace Community: https://huggingface.co/mlx-community

---

## Chapter 2: Setting Up Your Development Environment

### 2.1 System Requirements

Before installing MLX, verify your system meets these requirements:

- **Hardware**: Mac with Apple silicon (M1, M2, M3, M4, or later)
- **OS**: macOS 14.0 (Sonoma) or later
- **Python**: 3.10 or later (must be a native ARM64 Python, not x86 via Rosetta)

To check your Python architecture:

```bash
python -c "import platform; print(platform.processor())"
# Should output: arm
# If it outputs: i386 -- you're using Rosetta, switch to native Python
```

To check your macOS version:

```bash
sw_vers
# ProductVersion should be 14.0 or higher
```

### 2.2 Installing MLX

The simplest installation is via pip:

```bash
pip install mlx
```

That's it. MLX will automatically use your Mac's GPU via the Metal framework.

### 2.3 Setting Up a Virtual Environment

As with any Python project, it's best practice to use a virtual environment:

```bash
# Using venv (built into Python)
python -m venv mlx_env
source mlx_env/bin/activate

# Install MLX
pip install mlx
```

Or using [uv](https://docs.astral.sh/uv/) (fast Python package manager):

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment with uv
uv python venv mlx_env

# Activate the environment
source mlx_env/bin/activate

# Install MLX
uv pip install mlx
```

### 2.4 Verifying Your Installation

Create a file called `verify_install.py`:

```python
import mlx.core as mx

# Check MLX version
print(f"MLX version: {mx.__version__}")

# Check Metal GPU availability
print(f"Metal available: {mx.metal.is_available()}")

# Get device info
if mx.metal.is_available():
    info = mx.metal.device_info()
    print(f"GPU: {info.get('device_name', 'Unknown')}")

# Quick computation test
a = mx.array([1.0, 2.0, 3.0])
b = mx.array([4.0, 5.0, 6.0])
c = a + b
mx.eval(c)  # Force evaluation
print(f"Computation test: {a} + {b} = {c}")

# Test GPU computation
d = mx.add(a, b, stream=mx.gpu)
mx.eval(d)
print(f"GPU computation test passed: {d}")

print("\nMLX is installed and working correctly!")
```

Run it:

```bash
python verify_install.py
```

Expected output:

```
MLX version: 0.x.x
Metal available: True
GPU: Apple M1 (or M2, M3, M4, etc.)
Computation test: array([1, 2, 3], dtype=float32) + array([4, 5, 6], dtype=float32) = array([5, 7, 9], dtype=float32)
GPU computation test passed: array([5, 7, 9], dtype=float32)

MLX is installed and working correctly!
```

### 2.5 Installing Supporting Packages

For this training guide, you'll also want some supporting packages:

```bash
pip install numpy          # For data interop
pip install matplotlib     # For visualization
pip install datasets       # HuggingFace datasets
pip install transformers   # Tokenizers and model configs
pip install sentencepiece  # Tokenization
pip install tqdm           # Progress bars
pip install mlx-lm         # LLM package (later chapters)
```

You can install them all at once:

```bash
pip install numpy matplotlib datasets transformers sentencepiece tqdm mlx-lm
```

### 2.6 Your First MLX Program

Let's write a "Hello World" of ML -- linear regression. This will give you a feel for the MLX workflow:

```python
import mlx.core as mx

# Generate some synthetic data: y = 3x + 2 + noise
mx.random.seed(42)
X = mx.random.normal((100, 1))
noise = mx.random.normal((100, 1)) * 0.5
y = 3.0 * X + 2.0 + noise

# Initialize parameters as scalars
weight = mx.array(0.0)
bias = mx.array(0.0)

# Define the model
def predict(x, w, b):
    return x * w + b

# Define the loss function (mean squared error)
def loss_fn(w, b, x, y):
    pred = predict(x, w, b)
    return mx.mean((pred - y) ** 2)

# Get the gradient function -- differentiate w.r.t. w AND b
loss_and_grad = mx.value_and_grad(loss_fn, argnums=[0, 1])

# Training loop
for step in range(200):
    loss, (grad_w, grad_b) = loss_and_grad(weight, bias, X, y)
    weight = weight - 0.1 * grad_w
    bias = bias - 0.1 * grad_b
    mx.eval(loss, weight, bias)

    if step % 50 == 0:
        print(f"Step {step}: loss = {loss.item():.4f}, "
              f"weight = {weight.item():.4f}, bias = {bias.item():.4f}")

print(f"\nFinal: weight = {weight.item():.4f} (target: 3.0), "
      f"bias = {bias.item():.4f} (target: 2.0)")
```

When you run this, you should see the weight converge toward 3.0 and the bias toward 2.0. Don't worry about understanding every line yet -- we'll cover each concept in detail in the coming chapters.

### 2.7 Development Tools

Here are some tools that will enhance your MLX development experience:

**Jupyter Notebooks** -- Great for interactive experimentation:

```bash
pip install jupyter
jupyter notebook
```

**VS Code** -- Excellent Python support with the Python extension. The Jupyter extension also works well for notebook-style development.

**Terminal** -- Since MLX scripts are just Python, any terminal works. Make sure you're running a native ARM terminal (not Rosetta).

### 2.8 Common Installation Issues

**"No matching distribution found"**

This usually means you're running x86 Python via Rosetta. Fix it:

```bash
# Check your architecture
python -c "import platform; print(platform.processor())"
# If it says i386, you need native Python

# Install native Python via uv
uv python venv mlx_env
source mlx_env/bin/activate
uv pip install mlx
```

**"MLX requires macOS 14.0 or later"**

Update your macOS to Sonoma (14.0) or later.

**Import errors after installation**

Make sure you're in the correct virtual environment:

```bash
which python
# Should point to your virtual environment
pip list | grep mlx
# Should show mlx with version number
```

### 2.9 Project Structure

Throughout this guide, we'll use the following project structure:

```
mlx_training/
├── guide/                    # This guide (markdown files)
│   ├── part1_introduction_setup.md
│   ├── part2_mlx_fundamentals.md
│   ├── part3_nlp_foundations.md
│   ├── part4_transformer_architecture.md
│   ├── part5_training_language_models.md
│   └── part6_advanced_topics.md
├── demos/                    # Demo scripts
│   ├── 01_array_basics.py
│   ├── 02_lazy_eval.py
│   ├── 03_autodiff.py
│   ├── 04_neural_network.py
│   ├── 05_transformer_block.py
│   └── ...
├── projects/                 # Full project implementations
│   ├── sentiment_classifier/
│   ├── text_generator/
│   └── lora_finetune/
├── mlx_links.txt
└── requirements.txt
```

Let's create the `requirements.txt`:

```txt
mlx>=0.21
numpy>=1.24
matplotlib>=3.7
datasets>=2.14
transformers>=4.35
sentencepiece>=0.1.99
tqdm>=4.66
mlx-lm>=0.19
safetensors>=0.4
```

---

**Next**: In Part 2, we'll dive deep into MLX fundamentals -- arrays, operations, lazy evaluation, and the core concepts that underpin everything you'll build with MLX.
