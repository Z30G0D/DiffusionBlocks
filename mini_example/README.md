# DiffusionBlocks, Explained Simply (MNIST mini-example)

A tiny, laptop-friendly explainer of the [DiffusionBlocks](https://arxiv.org/abs/2506.14202) idea:
train a residual network **one block at a time** by treating block-wise updates as the reverse
(denoising) process of a diffusion model, cutting training memory by ~1/B while keeping accuracy.

This is a teaching toy on MNIST, decoupled from the heavy official repo (no CUDA / flash-attn /
Lightning / wandb). It auto-uses MPS or CPU and runs in a few minutes.

## Run

```bash
pip install -r requirements.txt
jupyter notebook diffusionblocks_mnist.ipynb
```

## Tests

```bash
python -m pytest tests/ -v
```
