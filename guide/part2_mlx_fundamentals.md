# Part 2: MLX Fundamentals

---

## Chapter 3: Arrays - The Foundation of Everything

### 3.1 What is an Array?

In MLX, the `array` is the fundamental data structure similar to how JavaScript has `Array` and Python has `list`, but designed for numerical computing. An MLX array is a multi-dimensional container of numerical data that lives in unified memory and can be operated on by both the CPU and GPU without copying.

If you think of a spreadsheet as a 2D grid of numbers, an MLX array generalizes this concept to any number of dimensions.

```python
import mlx.core as mx

# A scalar (0-dimensional)
scalar = mx.array(42)
print(scalar)        # array(42, dtype=int32)
print(scalar.ndim)   # 0
print(scalar.shape)  # ()

# A vector (1-dimensional)
vector = mx.array([1, 2, 3, 4])
print(vector)        # array([1, 2, 3, 4], dtype=int32)
print(vector.ndim)   # 1
print(vector.shape)  # (4,)

# A matrix (2-dimensional)
matrix = mx.array([[1, 2], [3, 4], [5, 6]])
print(matrix)        # array([[1, 2], [3, 4], [5, 6]], dtype=int32)
print(matrix.ndim)   # 2
print(matrix.shape)  # (3, 2)

# A 3D tensor (think: batch of matrices)
tensor3d = mx.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(tensor3d.ndim)   # 3
print(tensor3d.shape)  # (2, 2, 2)
```

### 3.2 Data Types

MLX arrays are typed. The type determines how much memory each element uses and what operations are valid:

```python
# Integer types
a_int32 = mx.array([1, 2, 3])          # int32 (default for integers)
a_int8 = mx.array([1, 2, 3], dtype=mx.int8)
a_int16 = mx.array([1, 2, 3], dtype=mx.int16)
a_int64 = mx.array([1, 2, 3], dtype=mx.int64)
a_uint8 = mx.array([1, 2, 3], dtype=mx.uint8)
a_uint16 = mx.array([1, 2, 3], dtype=mx.uint16)
a_uint32 = mx.array([1, 2, 3], dtype=mx.uint32)

# Floating point types
b_float32 = mx.array([1.0, 2.0])        # float32 (default for floats)
b_float16 = mx.array([1.0, 2.0], dtype=mx.float16)
b_bfloat16 = mx.array([1.0, 2.0], dtype=mx.bfloat16)

# Boolean
c_bool = mx.array([True, False, True])  # bool

# Complex
d_complex = mx.array([1+2j, 3+4j])      # complex64

# Check the dtype
print(a_int32.dtype)    # int32
print(b_float32.dtype)  # float32
```

**Web Developer Analogy**: Think of dtypes like choosing between `int`, `float`, and `boolean` in JavaScript, but with more granular control over precision and memory usage.

**Why dtypes matter**: Using lower precision (like `float16` instead of `float32`) halves memory usage and can speed up computation. For training language models, `float16` or `bfloat16` are common choices. For inference, quantized types (4-bit, 8-bit) can further reduce memory.

### 3.3 Creating Arrays

MLX provides many ways to create arrays:

```python
import mlx.core as mx

# From Python lists
a = mx.array([1, 2, 3])

# From nested lists (multi-dimensional)
b = mx.array([[1.0, 2.0], [3.0, 4.0]])

# Zeros and ones
zeros = mx.zeros((3, 4))           # 3x4 matrix of zeros
ones = mx.ones((2, 3, 4))          # 2x3x4 tensor of ones

# Full (constant value)
full = mx.full((3, 3), 7.0)        # 3x3 matrix filled with 7.0

# Identity matrix
eye = mx.eye(4)                    # 4x4 identity matrix

# Sequences
arange = mx.arange(0, 10, 2)       # [0, 2, 4, 6, 8]
linspace = mx.linspace(0, 1, 5)    # [0.0, 0.25, 0.5, 0.75, 1.0]

# Random arrays
mx.random.seed(42)
rand_normal = mx.random.normal((3, 3))        # Normal distribution
rand_uniform = mx.random.uniform(shape=(3, 3))       # Uniform [0, 1)
rand_bernoulli = mx.random.bernoulli(shape=(3, 3))   # Bernoulli (0 or 1)
rand_int = mx.random.randint(0, 10, (3, 3))   # Random integers

# Like existing arrays (same shape/dtype)
x = mx.array([1.0, 2.0, 3.0])
zeros_like = mx.zeros_like(x)       # [0.0, 0.0, 0.0]
ones_like = mx.ones_like(x)         # [1.0, 1.0, 1.0]
```

### 3.4 Array Properties

Every array has useful properties for inspecting its shape and metadata:

```python
a = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

print(a.shape)     # (2, 3) dimensions
print(a.ndim)      # 2  number of dimensions
print(a.size)      # 6  total number of elements
print(a.dtype)     # float32  data type
print(a.nbytes)    # 24  bytes used (6 elements * 4 bytes/float32)
print(a.itemsize)  # 4  bytes per element
print(a.T)         # Transposed: shape (3, 2)
```

### 3.5 Indexing and Slicing

MLX indexing follows NumPy conventions, which should feel familiar:

```python
a = mx.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Single element
print(a[0, 0])       # 1
print(a[1, 2])       # 6

# Row and column slicing
print(a[0])          # First row: [1, 2, 3]
print(a[:, 0])       # First column: [1, 4, 7]
print(a[0:2])        # First two rows
print(a[0:2, 1:3])   # Sub-matrix

# Negative indexing
print(a[-1])         # Last row: [7, 8, 9]
print(a[-1, -1])     # Last element: 9

# Step slicing
b = mx.arange(10)
print(b[::2])        # [0, 2, 4, 6, 8]  every other element
print(b[1::2])       # [1, 3, 5, 7, 9]  odd indices
print(b[::-1])       # [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]  reversed

# Boolean indexing
mask = a > 5
print(mask)          # [[False, False, False], [False, False, True], [True, True, True]]

# Integer array indexing
indices = mx.array([0, 2])
print(a[indices])    # First and third rows
```

### 3.6 Reshaping and Broadcasting

Reshaping changes the view of an array without changing its data. Broadcasting allows operations between arrays of different shapes:

```python
# Reshaping
a = mx.arange(12)             # shape: (12,)
b = a.reshape(3, 4)           # shape: (3, 4)
c = a.reshape(2, 2, 3)        # shape: (2, 2, 3)

# Use -1 to auto-infer a dimension
d = a.reshape(3, -1)          # shape: (3, 4)  MLX infers the 4

# Flatten
e = c.reshape(-1)             # Back to shape (12,)
f = c.flatten()               # Same as reshape(-1)

# Squeeze and expand dims
g = mx.array([[1, 2, 3]])     # shape: (1, 3)
h = g.squeeze()               # shape: (3,)
i = mx.expand_dims(h, 0)      # shape: (1, 3)
j = mx.expand_dims(h, 1)      # shape: (3, 1)

# Broadcasting  operations on different shapes
x = mx.array([[1, 2, 3]])     # shape: (1, 3)
y = mx.array([[1], [2], [3]]) # shape: (3, 1)
z = x + y                     # shape: (3, 3) via broadcasting

# Broadcast explicitly
k = mx.broadcast_to(mx.array([1, 2, 3]), (3, 3))  # shape: (3, 3)
```

**Web Developer Analogy**: Broadcasting is like CSS flexbox  it automatically handles size mismatches in predictable ways. A `(1, 3)` array plus a `(3, 1)` array "stretches" to produce a `(3, 3)` result.

### 3.7 Mathematical Operations

MLX provides a comprehensive set of mathematical operations:

```python
import mlx.core as mx

a = mx.array([1.0, 2.0, 3.0])
b = mx.array([4.0, 5.0, 6.0])

# Element-wise arithmetic
print(a + b)      # [5.0, 7.0, 9.0]
print(a - b)      # [-3.0, -3.0, -3.0]
print(a * b)      # [4.0, 10.0, 18.0]
print(a / b)      # [0.25, 0.4, 0.5]
print(a ** 2)     # [1.0, 4.0, 9.0]

# Using explicit functions
print(mx.add(a, b))
print(mx.multiply(a, b))
print(mx.divide(a, b))
print(mx.power(a, 2))

# Unary operations
print(mx.sqrt(a))     # [1.0, 1.414, 1.732]
print(mx.exp(a))      # Exponential
print(mx.log(a))      # Natural logarithm
print(mx.abs(mx.array([-1, -2])))  # [1, 2]
print(mx.sign(mx.array([-3, 0, 5])))  # [-1, 0, 1]

# Trigonometric
print(mx.sin(a))
print(mx.cos(a))
print(mx.tan(a))

# Rounding
print(mx.round(mx.array([1.4, 2.5, 3.6])))  # [1, 2, 4]
print(mx.ceil(mx.array([1.1, 2.9])))          # [2, 3]
print(mx.floor(mx.array([1.1, 2.9])))         # [1, 2]

# Reductions (aggregations)
c = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(mx.sum(c))           # 21.0  sum of all elements
print(mx.sum(c, axis=0))   # [5.0, 7.0, 9.0]  sum along rows
print(mx.sum(c, axis=1))   # [6.0, 15.0]  sum along columns
print(mx.mean(c))          # 3.5
print(mx.max(c))           # 6.0
print(mx.min(c))           # 1.0
print(mx.prod(c))          # 720.0
print(mx.var(c))           # Variance
print(mx.std(c))           # Standard deviation

# Sorting
d = mx.array([3, 1, 4, 1, 5, 9, 2, 6])
print(mx.sort(d))          # [1, 1, 2, 3, 4, 5, 6, 9]
print(mx.argsort(d))       # Indices that would sort the array

# Comparison
print(mx.equal(a, b))        # Element-wise equality
print(mx.greater(a, b))      # Element-wise >
print(mx.less(a, b))         # Element-wise <
print(mx.maximum(a, b))      # Element-wise max
print(mx.minimum(a, b))      # Element-wise min
```

### 3.8 Matrix Operations

Matrix operations are central to neural networks. MLX provides optimized implementations:

```python
import mlx.core as mx

# Matrix multiplication
A = mx.random.normal((3, 4))
B = mx.random.normal((4, 5))
C = mx.matmul(A, B)          # shape: (3, 5)
# Or using the @ operator
C = A @ B                    # Same as matmul

# Dot product
v1 = mx.array([1.0, 2.0, 3.0])
v2 = mx.array([4.0, 5.0, 6.0])
print(mx.inner(v1, v2))      # 32.0

# Outer product
print(mx.outer(v1, v2))      # shape: (3, 3)

# Transpose
M = mx.random.normal((3, 4))
print(M.T)                   # shape: (4, 3)
print(mx.transpose(M))       # Same

# Matrix with specific dimensions
print(mx.transpose(M, (1, 0)))  # Explicit axis order

# Einsum (Einstein summation)  powerful for complex tensor operations
a = mx.random.normal((2, 3))
b = mx.random.normal((3, 4))
c = mx.einsum("ij,jk->ik", a, b)  # Matrix multiplication via einsum

# Trace, diagonal
square = mx.array([[1, 2], [3, 4]])
print(mx.trace(square))       # 5.0
print(mx.diag(mx.array([1, 2, 3])))  # Diagonal matrix
```

### 3.9 Concatenation, Stacking, and Splitting

```python
import mlx.core as mx

a = mx.array([[1, 2], [3, 4]])
b = mx.array([[5, 6], [7, 8]])

# Concatenate along existing axis
print(mx.concatenate([a, b], axis=0))  # Vertical: shape (4, 2)
print(mx.concatenate([a, b], axis=1))  # Horizontal: shape (2, 4)

# Stack along new axis
print(mx.stack([a, b], axis=0))  # shape: (2, 2, 2)

# Split
c = mx.arange(12)
print(mx.split(c, 3))          # Split into 3 equal parts
print(mx.split(c, [3, 7]))     # Split at indices 3 and 7

# Tile (repeat)
d = mx.array([[1, 2], [3, 4]])
print(mx.tile(d, (2, 3)))      # Repeat 2x vertically, 3x horizontally
```

### 3.10 Saving and Loading Arrays

```python
import mlx.core as mx

a = mx.random.normal((100, 50))
b = mx.random.normal((100,))

# Save a single array
mx.save("weights_a.npy", a)

# Save multiple arrays
mx.savez("weights.npz", {"a": a, "b": b})

# Save compressed
mx.savez_compressed("weights_compressed.npz", {"a": a, "b": b})

# Save in safetensors format (recommended for ML models)
mx.save_safetensors("model.safetensors", {"weight": a, "bias": b})

# Load arrays
loaded_a = mx.load("weights_a.npy")          # Single array
loaded_dict = mx.load("weights.npz")          # Dict of arrays

# Interop with NumPy
import numpy as np
np_array = np.array(a)           # MLX -> NumPy (triggers eval)
mlx_array = mx.array(np_array)  # NumPy -> MLX
```

---

## Chapter 4: Lazy Evaluation and Computation Graphs

### 4.1 Understanding Lazy Evaluation

This is one of the most important concepts in MLX, and it differs significantly from how PyTorch works (by default). In MLX, operations are **lazy**  when you write `c = a + b`, no actual addition happens. Instead, MLX records the operation in a computation graph and defers the actual computation until you explicitly request the result.

Think of it like a recipe: writing `c = a + b` adds a step to the recipe, but doesn't cook anything. You only start cooking when you call `mx.eval()`.

```python
import mlx.core as mx

a = mx.array([1.0, 2.0, 3.0])
b = mx.array([4.0, 5.0, 6.0])

# No computation happens here!
c = a + b

# The computation only happens when we evaluate
mx.eval(c)
print(c)  # Now the result exists: array([5, 7, 9], dtype=float32)
```

### 4.2 When Does Evaluation Happen Automatically?

While `mx.eval()` is the explicit way to trigger computation, there are implicit triggers you should be aware of:

```python
import mlx.core as mx

a = mx.array([1.0, 2.0])
b = mx.array([3.0, 4.0])
c = a + b

# These all trigger implicit evaluation:

# 1. Printing
print(c)  # Evaluates, then prints

# 2. Converting to NumPy
import numpy as np
np_array = np.array(c)  # Evaluates, then converts

# 3. Using .item() on a scalar
d = mx.sum(c)
print(d.item())  # Evaluates, then returns Python float

# 4. Saving to disk
mx.save("output.npy", c)  # Evaluates, then saves

# 5. Using in control flow (be careful!)
if d > 5:  # Evaluates d to check the condition
    print("Greater than 5")
```

### 4.3 Why Lazy Evaluation Matters

Lazy evaluation provides three key benefits:

**1. Graph Optimization.** Since MLX sees the full computation graph before executing, it can optimize it (e.g., fusing operations, eliminating dead code).

```python
def process(x):
    a = expensive_op(x)     # This might be eliminated if unused
    b = cheap_op(x)
    return b                 # Only b is returned, a is never computed

result = process(input_data)
mx.eval(result)  # expensive_op is never executed!
```

**2. Memory Efficiency.** No intermediate results are stored until needed. You can create large models without immediately using memory:

```python
model = BigModel()  # Graph is built, but no weights allocated yet
model.load_weights("weights_fp16.safetensors")  # Only fp16 memory used
```

**3. Composable Transformations.** Lazy evaluation makes it possible to transform the computation graph before executing it. This is how `grad()` (automatic differentiation) and `vmap()` (vectorization) work  they rewrite the graph.

### 4.4 Best Practices for Evaluation

The right granularity for `mx.eval()` matters:

```python
# BAD: Too many small evaluations (high overhead)
for i in range(1000):
    result = mx.add(a, b)
    mx.eval(result)  # 1000 separate graph evaluations!

# BAD: Too few evaluations (huge graph, growing overhead)
results = []
for i in range(1000):
    results.append(mx.add(a, b))
mx.eval(results)  # One huge graph

# GOOD: Evaluate at natural boundaries (e.g., per training step)
for batch in dataset:
    loss, grads = loss_and_grad_fn(model, batch)
    optimizer.update(model, grads)
    mx.eval(loss, model.parameters())  # One eval per step
```

**Rule of thumb**: Evaluate at each iteration of your outer training loop. This gives MLX enough graph to optimize while keeping overhead manageable.

### 4.5 Computation Graphs Explained

A computation graph is a directed acyclic graph (DAG) where:
- **Nodes** are operations (add, multiply, etc.) or array values
- **Edges** represent data flow

```
Input: a, b
  |
  +-- [add] --> c = a + b
  |                |
  +-- [multiply] --> d = a * b
                    |
              [multiply] --> e = c * d
                    |
              [sum] --> f = sum(e)
```

When you call `mx.eval(f)`, MLX traverses this graph from `f` backward, computing only what's needed to produce `f`.

### 4.6 Gradient Computation and Graphs

The lazy evaluation model is what makes MLX's `grad()` transformation work. When you apply `grad()` to a function, MLX:
1. Traces the forward computation graph
2. Reverses it to build a gradient computation graph
3. Returns a new function that, when called, builds both graphs

```python
import mlx.core as mx

def f(x):
    return mx.sum(x ** 2)

# Get gradient function
grad_fn = mx.grad(f)

x = mx.array([1.0, 2.0, 3.0])
gradient = grad_fn(x)
mx.eval(gradient)
print(gradient)  # array([2, 4, 6], dtype=float32) which is 2*x
```

### 4.7 Checkpointing for Memory

For very deep networks, the computation graph can consume a lot of memory storing intermediate activations. MLX provides `mx.checkpoint()` to trade computation for memory:

```python
import mlx.core as mx

def expensive_block(x):
    # Many operations, large intermediate tensors
    x = mx.matmul(x, w1)
    x = mx.relu(x)
    x = mx.matmul(x, w2)
    x = mx.relu(x)
    return x

# Without checkpointing: all intermediates stored
def forward(x):
    return expensive_block(x)

# With checkpointing: intermediates discarded, recomputed during backward
def forward_checkpointed(x):
    return mx.checkpoint(expensive_block)(x)
```

---

## Chapter 5: Function Transformations

### 5.1 Automatic Differentiation with `grad()`

Automatic differentiation (autodiff) is the backbone of neural network training. It computes the gradient (derivative) of a function automatically, without you having to derive calculus formulas by hand.

```python
import mlx.core as mx

# Simple example: derivative of sin(x) is cos(x)
x = mx.array(0.0)

# Forward: sin(0) = 0
print(mx.sin(x))               # array(0, dtype=float32)

# Gradient: cos(0) = 1
print(mx.grad(mx.sin)(x))      # array(1, dtype=float32)

# Second derivative: -sin(0) = 0
print(mx.grad(mx.grad(mx.sin))(x))  # array(-0, dtype=float32)
```

### 5.2 `grad()` with Multiple Arguments

When your function has multiple arguments, you can control which ones get differentiated:

```python
import mlx.core as mx

def loss_fn(weights, bias, inputs, targets):
    predictions = inputs @ weights + bias
    return mx.mean((predictions - targets) ** 2)

# By default, grad differentiates the FIRST argument
grad_fn = mx.grad(loss_fn)
# This gives d(loss)/d(weights)

# To differentiate specific arguments, use argnums
grad_fn = mx.grad(loss_fn, argnums=[0, 1])
# This returns a tuple: (d(loss)/d(weights), d(loss)/d(bias))
```

### 5.3 `value_and_grad()` Get Both

Often you want both the function value and the gradient (e.g., to track loss during training):

```python
import mlx.core as mx

def loss_fn(weights, inputs, targets):
    predictions = inputs @ weights
    return mx.mean((predictions - targets) ** 2)

# Get both value and gradient simultaneously
loss_and_grad = mx.value_and_grad(loss_fn)

weights = mx.random.normal((3, 1))
inputs = mx.random.normal((10, 3))
targets = mx.random.normal((10, 1))

loss, grads = loss_and_grad(weights, inputs, targets)
mx.eval(loss, grads)
print(f"Loss: {loss.item():.4f}")
print(f"Gradients shape: {grads.shape}")
```

### 5.4 Higher-Order Gradients

MLX supports nested `grad()` calls for higher-order derivatives:

```python
import mlx.core as mx

# f(x) = x^3
def f(x):
    return x ** 3

# f'(x) = 3x^2
df = mx.grad(f)

# f''(x) = 6x
d2f = mx.grad(df)

# f'''(x) = 6
d3f = mx.grad(d2f)

x = mx.array(2.0)
print(f"f(2) = {f(x).item()}")           # 8
print(f"f'(2) = {df(x).item()}")         # 12
print(f"f''(2) = {d2f(x).item()}")       # 24
print(f"f'''(2) = {d3f(x).item()}")      # 6
```

### 5.5 Vectorization with `vmap()`

`vmap()` (vectorized map) automatically transforms a function that operates on a single example into one that operates on a batch. This is essential for efficient training:

```python
import mlx.core as mx

# A function that works on a single vector
def predict(x, weight, bias):
    return mx.matmul(x, weight) + bias

x = mx.array([1.0, 2.0, 3.0])    # Single input
w = mx.array([0.5, 0.3, 0.2])
b = mx.array(0.1)

# Apply to one example
result = predict(x, w, b)

# Now batch it with vmap and apply to a batch of inputs
batch_x = mx.random.normal((32, 3))  # 32 inputs, each of size 3

# vmap over the first (0th) argument
batched_predict = mx.vmap(predict, in_axes=(0, None, None))
batch_results = batched_predict(batch_x, w, b)
print(batch_results.shape)  # (32,)
```

The `in_axes` parameter specifies which axis to vectorize over for each argument:
- `0` means vectorize along the first dimension
- `None` means don't vectorize (use the same value for all)

### 5.6 Jacobian-Vector Products (`jvp`) and Vector-Jacobian Products (`vjp`)

For advanced use cases, MLX provides forward-mode (`jvp`) and reverse-mode (`vjp`) autodiff primitives:

```python
import mlx.core as mx

def f(x):
    return mx.stack([mx.sin(x[0]), mx.cos(x[1])])

x = mx.array([1.0, 2.0])
v = mx.array([1.0, 0.0])  # Direction to compute Jacobian-vector product

# Forward-mode: J * v (Jacobian times vector)
# Useful when output dimension > input dimension
result = mx.jvp(f, (x,), (v,))
print(result)  # (f(x), J(x) @ v)

# Reverse-mode: v^T * J (vector times Jacobian)
# This is what grad() uses internally
# Useful when input dimension > output dimension
result = mx.vjp(f, (x,))
print(result)  # (f(x), vjp_fn) where vjp_fn(cotangent) gives gradients
```

### 5.7 Compilation with `compile()`

MLX's `compile()` function optimizes the computation graph by fusing operations and reducing memory overhead:

```python
import mlx.core as mx

@mx.compile
def compiled_function(x, y):
    return mx.sum(mx.sin(x) * mx.cos(y))

x = mx.random.normal((1000, 1000))
y = mx.random.normal((1000, 1000))

# First call: compilation (slightly slower)
result = compiled_function(x, y)
mx.eval(result)

# Subsequent calls: uses compiled graph (faster)
result = compiled_function(x, y)
mx.eval(result)
```

**Important notes about `compile()`:**
- The first call compiles the graph (has overhead)
- Subsequent calls with the same shapes/dtypes reuse the compiled graph
- If shapes change, a recompilation is triggered
- Works well combined with training loops where shapes are consistent

### 5.8 Combining Transformations

The real power of MLX is that transformations compose freely:

```python
import mlx.core as mx

def model_fn(params, x):
    w, b = params
    return mx.sum(mx.relu(x @ w + b))

# Compose: vectorize, then differentiate
batched_grad = mx.vmap(mx.grad(model_fn), in_axes=(None, 0))

# Compose: differentiate, then differentiate again
hessian_diag = mx.grad(mx.grad(model_fn))

# Compose: compile the gradient function
compiled_grad = mx.compile(mx.grad(model_fn))
```

### 5.9 `nn.value_and_grad()` for Neural Networks

For the common pattern of training neural networks, MLX provides a specialized `value_and_grad` in the `nn` module that works directly with `nn.Module` objects:

```python
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

# Define a model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 1)

    def __call__(self, x):
        x = mx.relu(self.linear1(x))
        return self.linear2(x)

model = MyModel()

# Loss function takes model and batch
def loss_fn(model, X, y):
    pred = model(X)
    return mx.mean((pred - y) ** 2)

# Get value and grad with respect to model parameters
loss_and_grad = nn.value_and_grad(model, loss_fn)

# Training loop
optimizer = optim.Adam(learning_rate=1e-3)

for batch_x, batch_y in dataset:
    loss, grads = loss_and_grad(model, batch_x, batch_y)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state, loss)
```

This pattern `nn.value_and_grad` + `optimizer.update` + `mx.eval` is the core training loop you'll use throughout this guide. Think of it like a React component's lifecycle: `loss_and_grad` computes (render), `optimizer.update` applies changes (commit), and `mx.eval` triggers the actual computation (DOM update).

### 5.10 Custom Functions

For operations that need custom gradient computation:

```python
import mlx.core as mx

@mx.custom_function
def my_custom_op(x):
    # Forward pass
    return mx.exp(x)

@my_custom_op.vjp
def my_custom_op_vjp(primals, cotangent):
    x, = primals
    # The gradient of exp(x) is exp(x) * cotangent
    return mx.exp(x) * cotangent,
```

---

**Next**:  
[In Part 3](https://github.com/rutkat/MLX_training/blob/main/guide/part3_nlp_foundations.md), we'll apply these fundamentals to Natural Language Processing.
