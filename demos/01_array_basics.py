"""
Demo 01: MLX Array Basics
Run: python demos/01_array_basics.py

This demo covers array creation, properties, indexing, and basic operations.
"""

import mlx.core as mx


def main():
    print("=" * 60)
    print("MLX Array Basics Demo")
    print("=" * 60)

    # --- Creating Arrays ---
    print("\n--- Creating Arrays ---")
    a = mx.array([1, 2, 3, 4])
    print(f"From list: {a}, dtype: {a.dtype}")

    b = mx.array([1.0, 2.0, 3.0])
    print(f"Float array: {b}, dtype: {b.dtype}")

    c = mx.array([[1, 2], [3, 4], [5, 6]])
    print(f"2D array:\n{c}")

    # --- Array Creation Functions ---
    print("\n--- Array Creation Functions ---")
    print(f"zeros(2,3): {mx.zeros((2, 3))}")
    print(f"ones(2,3): {mx.ones((2, 3))}")
    print(f"full(2,3,7): {mx.full((2, 3), 7)}")
    print(f"eye(3):\n{mx.eye(3)}")
    print(f"arange(0,10,2): {mx.arange(0, 10, 2)}")
    print(f"linspace(0,1,5): {mx.linspace(0, 1, 5)}")

    # --- Random Arrays ---
    print("\n--- Random Arrays ---")
    mx.random.seed(42)
    print(f"normal(2,3): {mx.random.normal((2, 3))}")
    print(f"uniform(2,3): {mx.random.uniform(shape=(2, 3))}")
    print(f"randint(0,10,(2,3)): {mx.random.randint(0, 10, (2, 3))}")

    # --- Array Properties ---
    print("\n--- Array Properties ---")
    a = mx.random.normal((3, 4, 5))
    print(f"Array shape: {a.shape}")
    print(f"Number of dims: {a.ndim}")
    print(f"Total elements: {a.size}")
    print(f"Data type: {a.dtype}")
    print(f"Bytes: {a.nbytes}")

    # --- Indexing ---
    print("\n--- Indexing ---")
    a = mx.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(f"Array:\n{a}")
    print(f"a[0, 0] = {a[0, 0]}")
    print(f"a[1] = {a[1]}")
    print(f"a[:, 0] = {a[:, 0]}")
    print(f"a[0:2] = {a[0:2]}")
    print(f"a[-1] = {a[-1]}")

    # --- Reshaping ---
    print("\n--- Reshaping ---")
    a = mx.arange(12)
    print(f"Original: {a}, shape: {a.shape}")
    b = a.reshape(3, 4)
    print(f"Reshaped (3,4):\n{b}")
    c = a.reshape(2, 2, 3)
    print(f"Reshaped (2,2,3), shape: {c.shape}")
    d = b.flatten()
    print(f"Flattened: {d}, shape: {d.shape}")

    # --- Mathematical Operations ---
    print("\n--- Mathematical Operations ---")
    a = mx.array([1.0, 2.0, 3.0])
    b = mx.array([4.0, 5.0, 6.0])
    print(f"a = {a}")
    print(f"b = {b}")
    print(f"a + b = {a + b}")
    print(f"a * b = {a * b}")
    print(f"a ** 2 = {a ** 2}")
    print(f"mx.sqrt(a) = {mx.sqrt(a)}")
    print(f"mx.exp(a) = {mx.exp(a)}")
    print(f"mx.log(a) = {mx.log(a)}")

    # --- Reductions ---
    print("\n--- Reductions ---")
    a = mx.array([[1.0, 2.0], [3.0, 4.0]])
    print(f"Array:\n{a}")
    print(f"sum() = {mx.sum(a)}")
    print(f"mean() = {mx.mean(a)}")
    print(f"max() = {mx.max(a)}")
    print(f"sum(axis=0) = {mx.sum(a, axis=0)}")
    print(f"sum(axis=1) = {mx.sum(a, axis=1)}")

    # --- Lazy Evaluation ---
    print("\n--- Lazy Evaluation ---")
    a = mx.array([1.0, 2.0])
    b = mx.array([3.0, 4.0])
    c = a + b  # NOT computed yet!
    print(f"c = a + b (lazy, not computed)")
    mx.eval(c)  # NOW computed
    print(f"After eval: c = {c}")

    print("\n" + "=" * 60)
    print("Demo complete!")


if __name__ == "__main__":
    main()
