# MLX Training Guide

A comprehensive training guide for building with the [MLX](https://github.com/ml-explore/mlx) machine learning framework, written for web developers entering the ML space. Covers the Python API exclusively with a focus on NLP and Transformer language model training.

## What You Will Learn

This guide progresses from fundamentals to advanced topics across 15 chapters organized in 6 parts:

## Table of Contents

| Part | Chapter | Title | Guide File |
|------|---------|-------|------------|
| **1. Introduction & Setup** | 1 | What is MLX? | `guide/part1_introduction_setup.md` |
| | 2 | Setting Up Your Development Environment | `guide/part1_introduction_setup.md` |
| **2. MLX Fundamentals** | 3 | Arrays -- The Foundation of Everything | `guide/part2_mlx_fundamentals.md` |
| | 4 | Lazy Evaluation and Computation Graphs | `guide/part2_mlx_fundamentals.md` |
| | 5 | Function Transformations | `guide/part2_mlx_fundamentals.md` |
| **3. NLP Foundations** | 6 | Text as Numbers -- Tokenization and Embeddings | `guide/part3_nlp_foundations.md` |
| | 7 | NLP Tasks and Data Preparation | `guide/part3_nlp_foundations.md` |
| **4. Transformer Architecture** | 8 | Attention Is All You Need -- Understanding the Transformer | `guide/part4_transformer_architecture.md` |
| | 9 | Building a Complete Transformer | `guide/part4_transformer_architecture.md` |
| **5. Training Language Models** | 10 | Training a Character-Level Language Model | `guide/part5_training_language_models.md` |
| | 11 | Training a BERT-Style Classifier | `guide/part5_training_language_models.md` |
| | 12 | Fine-Tuning Large Language Models with LoRA | `guide/part5_training_language_models.md` |
| **6. Advanced Topics** | 13 | Model Optimization and Performance | `guide/part6_advanced_topics.md` |
| | 14 | Distributed Training and Advanced Patterns | `guide/part6_advanced_topics.md` |
| | 15 | Putting It All Together -- Building ML-Powered Web Applications | `guide/part6_advanced_topics.md` |

The guide also includes cheat sheets for MLX core operations, neural network layers, optimizers, and a full glossary of ML terminology.

## System Requirements

### Hardware

| Requirement | Details |
|-------------|---------|
| **Processor** | Mac with Apple silicon (M1, M2, M3, M4, or later) |
| **Memory** | 8 GB minimum; 16 GB+ recommended for larger models and LoRA fine-tuning |
| **Storage** | ~2 GB for MLX and dependencies; additional space for datasets and model weights |

> MLX relies on Apple's Metal GPU framework, which is only available on Apple silicon. It does **not** run on Intel-based Macs. Linux support is available with CUDA or CPU-only backends, but this guide targets Apple silicon.

### Software

| Requirement | Minimum Version |
|-------------|-----------------|
| **macOS** | 14.0 (Sonoma) or later |
| **Python** | 3.10 or later (must be native ARM64, not x86 via Rosetta) |
| **Xcode Command Line Tools** | Latest (`xcode-select --install`) |

### Verifying Your Setup

```bash
# Check you have native ARM Python (should print "arm")
python -c "import platform; print(platform.processor())"

# Check macOS version (should be 14.0+)
sw_vers
```

## Installation

```bash
# Clone or download this repository
cd mlx_training

# Create a virtual environment
python -m venv mlx_env
source mlx_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Verify MLX is working
python demos/01_array_basics.py
```

## Project Structure

```
mlx_training/
├── guide/                          # Training guide (markdown)
│   ├── part1_introduction_setup.md       Ch 1-2: Introduction, setup, first program
│   ├── part2_mlx_fundamentals.md         Ch 3-5: Arrays, lazy eval, transforms
│   ├── part3_nlp_foundations.md          Ch 6-7: Tokenization, embeddings, datasets
│   ├── part4_transformer_architecture.md Ch 8-9: Attention, encoders, decoders
│   ├── part5_training_language_models.md Ch 10-12: Training, fine-tuning, LoRA
│   └── part6_advanced_topics.md          Ch 13-15: Optimization, deployment, RAG
│
├── demos/                          # Standalone demo scripts
│   ├── 01_array_basics.py                Array creation, indexing, operations
│   ├── 02_lazy_eval_autodiff.py          Lazy evaluation, grad, vmap
│   ├── 03_neural_network.py              Building and training a neural network
│   ├── 04_transformer_block.py           Transformer components step by step
│   ├── 05_llm_inference.py               Running pre-trained LLMs with mlx-lm
│   └── 06_lora_finetuning.py             LoRA implementation and training
│
├── projects/                       # Full project implementations
│   ├── text_generator/                   Character-level transformer LM
│   └── sentiment_classifier/             Transformer encoder on IMDB reviews
│
├── requirements.txt                # Python dependencies
└── mlx_links.txt                   # Reference links used in this guide
```

## Running the Demos

Each demo is self-contained and can be run independently:

```bash
# Verify installation
python demos/01_array_basics.py

# Lazy evaluation and autodiff
python demos/02_lazy_eval_autodiff.py

# Neural network training on synthetic data
python demos/03_neural_network.py

# Transformer block construction
python demos/04_transformer_block.py

# LLM inference (requires mlx-lm, downloads model on first run)
python demos/05_llm_inference.py

# LoRA fine-tuning demonstration
python demos/06_lora_finetuning.py
```

## Running the Projects

### Text Generator

```bash
# Download a training corpus (e.g., Shakespeare)
curl -O https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# Train the character-level model
python projects/text_generator/train.py --data input.txt --epochs 20
```

### Sentiment Classifier

```bash
# Train on IMDB reviews (downloads dataset automatically)
python projects/sentiment_classifier/train.py
```

## Reading the Guide

The guide is designed to be read sequentially. Start with Part 1 and work through each part in order, as later chapters build on concepts introduced earlier.

Each guide file includes inline code snippets for illustration. Complete runnable programs are separated into the `demos/` and `projects/` directories.

## Additional Resources

### Official MLX Resources

- [MLX Documentation](https://ml-explore.github.io/mlx/build/html/index.html) -- Full API reference, usage guides, and examples
- [MLX GitHub Repository](https://github.com/ml-explore/mlx) -- Source code, issues, and discussions
- [MLX Examples Repository](https://github.com/ml-explore/mlx-examples) -- Official example implementations covering LLMs, image generation, speech, and more
- [MLX Community on HuggingFace](https://huggingface.co/mlx-community) -- Pre-converted model weights ready for use with MLX

### MLX Ecosystem Packages

- [mlx-lm](https://github.com/ml-explore/mlx-lm) -- LLM text generation and fine-tuning package (the primary tool for working with large language models)
- [mlx-data](https://github.com/ml-explore/mlx-data) -- Efficient data loading and preprocessing utilities
- [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) -- Vision Language Model inference and fine-tuning
- [mlx-whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper) -- Speech recognition with OpenAI's Whisper
- [mlx-lora](https://github.com/ml-explore/mlx-examples/tree/main/lora) -- Parameter-efficient fine-tuning with LoRA and QLoRA

### Learning Resources

- [Apple Machine Learning Research](https://machinelearning.apple.com) -- Research papers and blog posts from the MLX team
- [MLX GitHub Discussions](https://github.com/ml-explore/mlx/discussions) -- Community Q&A and feature requests
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) -- The original Transformer paper
- [LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)](https://arxiv.org/abs/2106.09685) -- The LoRA paper

### Pre-Trained Models (HuggingFace)

Popular models available in MLX format from the `mlx-community` organization:

| Model | Size | Link |
|-------|------|------|
| Qwen3 4B Instruct (4-bit) | ~2.5 GB | `mlx-community/Qwen3-4B-Instruct-2507-4bit` |
| Llama 3.1 8B Instruct (4-bit) | ~5 GB | `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` |
| Mistral 7B Instruct (4-bit) | ~4 GB | `mlx-community/Mistral-7B-Instruct-v0.3-4bit` |
| Phi-3.5 Mini Instruct (4-bit) | ~2 GB | `mlx-community/Phi-3.5-mini-instruct-4bit` |

Use any of these with `mlx-lm`:

```bash
pip install mlx-lm
mlx_lm.chat --model mlx-community/Qwen3-4B-Instruct-2507-4bit
```

Browse the full catalog at [huggingface.co/mlx-community](https://huggingface.co/mlx-community).

## License

This training guide is provided for educational purposes. The MLX framework itself is licensed under the MIT License by Apple machine learning research.
