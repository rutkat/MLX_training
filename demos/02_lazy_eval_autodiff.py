"""
Demo 02: Lazy Evaluation and Automatic Differentiation
Run: python demos/02_lazy_eval_autodiff.py

This demo shows how lazy evaluation works and how to use grad() and vmap().
"""

import mlx.core as mx
import math


def main():
    print("=" * 60)
    print("Lazy Evaluation & Auto-differentiation Demo")
    print("=" * 60)

    # --- Lazy Evaluation ---
    print("\n--- Lazy Evaluation ---")

    # Operations are deferred
    a = mx.array([1.0, 2.0, 3.0])
    b = mx.array([4.0, 5.0, 6.0])
    c = a + b           # Not computed
    d = c * 2           # Not computed
    e = mx.sum(d)       # Not computed

    # Only computed when we eval
    mx.eval(e)
    print(f"Result: {e}")

    # Implicit evaluation triggers
    c2 = a + b
    print(f"Print triggers eval: {c2}")  # Eval happens here

    # --- Automatic Differentiation ---
    print("\n--- Automatic Differentiation ---")

    # Derivative of sin(x) at x=0: cos(0) = 1
    x = mx.array(0.0)
    grad_sin = mx.grad(mx.sin)
    result = grad_sin(x)
    mx.eval(result)
    print(f"d/dx sin(0) = {result.item()} (should be 1.0)")

    # Second derivative of sin(x): -sin(0) = 0
    grad2_sin = mx.grad(grad_sin)
    result2 = grad2_sin(x)
    mx.eval(result2)
    print(f"d^2/dx^2 sin(0) = {result2.item()} (should be 0.0)")

    # --- Gradient of a function ---
    print("\n--- Gradient of f(x) = sum(x^2) ---")

    def f(x):
        return mx.sum(x ** 2)

    grad_f = mx.grad(f)
    x = mx.array([1.0, 2.0, 3.0])
    gradient = grad_f(x)
    mx.eval(gradient)
    print(f"f(x) = sum(x^2) at x = [1, 2, 3]")
    print(f"gradient = {gradient} (should be [2, 4, 6] = 2*x)")

    # --- value_and_grad ---
    print("\n--- value_and_grad ---")

    def loss_fn(w, x, y):
        pred = x @ w
        return mx.mean((pred - y) ** 2)

    w = mx.array([1.0, 2.0])
    x = mx.array([[1.0, 2.0]])
    y = mx.array([5.0])

    loss_and_grad = mx.value_and_grad(loss_fn)
    loss_val, grads = loss_and_grad(w, x, y)
    mx.eval(loss_val, grads)
    print(f"Loss: {loss_val.item():.4f}")
    print(f"Gradients: {grads}")

    # --- vmap (Vectorization) ---
    print("\n--- vmap (Auto-vectorization) ---")

    def predict(x, w, b):
        return mx.matmul(x, w) + b

    w = mx.array([0.5, 0.3, 0.2])
    b = mx.array(0.1)

    # Single example
    single_x = mx.array([1.0, 2.0, 3.0])
    single_result = predict(single_x, w, b)
    mx.eval(single_result)
    print(f"Single prediction: {single_result.item():.4f}")

    # Batch with vmap
    batch_x = mx.random.normal((5, 3))  # 5 examples
    batched_predict = mx.vmap(predict, in_axes=(0, None, None))
    batch_results = batched_predict(batch_x, w, b)
    mx.eval(batch_results)
    print(f"Batch predictions shape: {batch_results.shape}")
    print(f"Batch predictions: {batch_results}")

    # --- Composing transformations ---
    print("\n--- Composing Transformations ---")

    # Scalar example: f(x) = x^3, all derivatives are scalars
    def f(x):
        return x ** 3

    f1 = mx.grad(f)       # 3x^2
    f2 = mx.grad(f1)      # 6x
    f3 = mx.grad(f2)      # 6

    x = mx.array(2.0)
    r0 = f(x)
    r1, r2, r3 = f1(x), f2(x), f3(x)
    mx.eval(r0, r1, r2, r3)
    print(f"f(x) = x^3 at x = 2.0")
    print(f"  f(2)  = {r0.item()} (2^3 = 8)")
    print(f"  f'(2) = {r1.item()} (3*4 = 12)")
    print(f"  f''(2)= {r2.item()} (6*2 = 12)")
    print(f"  f'''(2)= {r3.item()} (6)")

    # Vector example: grad of sum(x^3) gives element-wise 3x^2
    def g(x):
        return mx.sum(x ** 3)

    g1 = mx.grad(g)       # 3x^2 (element-wise)

    x_vec = mx.array([1.0, 2.0, 3.0])
    r_vec = g1(x_vec)
    mx.eval(r_vec)
    print(f"\ngrad of sum(x^3) at x = [1, 2, 3]")
    print(f"  gradient = {r_vec} (3x^2 = [3, 12, 27])")

    print("\n" + "=" * 60)
    print("Demo complete!")


if __name__ == "__main__":
    main()
