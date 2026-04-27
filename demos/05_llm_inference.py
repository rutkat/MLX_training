"""
Demo 05: LLM Inference with MLX-LM
Run: python demos/05_llm_inference.py

This demo shows how to load and run inference with pre-trained
language models using the mlx-lm package.

Install: pip install mlx-lm
"""

import mlx.core as mx


def main():
    print("=" * 60)
    print("LLM Inference with MLX-LM Demo")
    print("=" * 60)

    # --- Check GPU ---
    print("\n--- System Info ---")
    print(f"Metal GPU available: {mx.metal.is_available()}")
    if mx.metal.is_available():
        info = mx.metal.device_info()
        print(f"GPU: {info.get('device_name', 'Unknown')}")
        mem = info.get('memory_size', 0)
        print(f"Memory: {mem / 1e9:.1f} GB")

    # --- Load a model ---
    print("\n--- Loading Model ---")
    print("Loading google/gemma-4-E2B-it...")
    print("(This will download the model on first run)")

    try:
        from mlx_lm import load, generate, stream_generate

        model, tokenizer = load("google/gemma-4-E2B-it")
        print("Model loaded successfully!")

        # --- Basic generation ---
        print("\n--- Basic Generation ---")
        prompt = "Explain machine learning in one paragraph."
        print(f"Prompt: {prompt}")
        print("-" * 40)

        response = generate(
            model, tokenizer,
            prompt=prompt,
            max_tokens=200,
            temp=0.7,
            verbose=True,
        )

        # --- Chat format ---
        print("\n--- Chat Format ---")
        messages = [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "Write a Python function that checks if a number is prime."},
        ]

        # Apply chat template
        if hasattr(tokenizer, 'apply_chat_template'):
            chat_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            chat_prompt = messages[-1]["content"]

        print(f"Chat prompt: {chat_prompt[:100]}...")
        print("-" * 40)

        response = generate(
            model, tokenizer,
            prompt=chat_prompt,
            max_tokens=300,
            temp=0.7,
            verbose=True,
        )

        # --- Streaming generation ---
        print("\n--- Streaming Generation ---")
        prompt = "Write a haiku about programming."
        print(f"Prompt: {prompt}")
        print("-" * 40)

        for token in stream_generate(
            model, tokenizer,
            prompt=prompt,
            max_tokens=100,
            temp=0.7,
        ):
            print(token, end="", flush=True)
        print()

        # --- Memory usage ---
        print("\n--- Memory Usage ---")
        print(f"Active memory: {mx.get_active_memory() / 1e9:.2f} GB")
        print(f"Peak memory: {mx.get_peak_memory() / 1e9:.2f} GB")

    except ImportError:
        print("mlx-lm not installed. Install with: pip install mlx-lm")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have enough memory and the model is available.")

    print("\n" + "=" * 60)
    print("Demo complete!")


if __name__ == "__main__":
    main()
