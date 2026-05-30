"""DiffusionBlocks MNIST mini-example: all testable logic in one focused module."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm


# --------------------------------------------------------------------------- #
# Config & device
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    H: int = 128
    num_layers: int = 6
    num_blocks: int = 3
    num_classes: int = 10
    image_dim: int = 28 * 28
    sigma_data: float = 0.5
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    p_mean: float = -1.2
    p_std: float = 1.2
    gamma: float = 0.05


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def block_layer_indices(block_idx: int, cfg: Config) -> list[int]:
    step = cfg.num_layers // cfg.num_blocks
    return list(range(block_idx * step, (block_idx + 1) * step))


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_mnist(n_train: int = 8000, n_test: int = 2000, batch_size: int = 128, seed: int = 0):
    """Return (train_loader, test_loader) over flattened, normalized MNIST subsets."""
    from torchvision import datasets, transforms

    tfm = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_full = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test_full = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)

    g = torch.Generator().manual_seed(seed)
    train_idx = torch.randperm(len(train_full), generator=g)[:n_train]
    test_idx = torch.randperm(len(test_full), generator=g)[:n_test]
    train = torch.utils.data.Subset(train_full, train_idx.tolist())
    test = torch.utils.data.Subset(test_full, test_idx.tolist())

    def collate(batch):
        xs = torch.stack([b[0].view(-1) for b in batch])
        ys = torch.tensor([b[1] for b in batch], dtype=torch.long)
        return xs, ys

    train_loader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True, collate_fn=collate
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=False, collate_fn=collate
    )
    return train_loader, test_loader
