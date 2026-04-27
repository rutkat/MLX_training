# Text Generator Project

A character-level language model trained with MLX.

## Quick Start

```bash
# Download Shakespeare dataset
curl -O https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# Train the model
python train.py --data input.txt --epochs 20
```

## Configuration

You can adjust model size and training parameters:

```bash
# Larger model
python train.py --data input.txt --d-model 256 --epochs 30
```

## Architecture

- Transformer decoder with causal masking
- Character-level tokenization
- Pre-norm (LayerNorm before attention and FFN)
- AdamW optimizer with cosine decay
