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


# --------------------------------------------------------------------------- #
# EDM utilities (Karras et al. 2022) — preconditioning, sigma schedules, weights
# --------------------------------------------------------------------------- #
def get_block_sigmas(cfg: Config) -> list[float]:
    """Partition the sigma axis into num_blocks ranges via the log-normal CDF."""
    cdf_min = norm.cdf((np.log(cfg.sigma_min) - cfg.p_mean) / cfg.p_std)
    cdf_max = norm.cdf((np.log(cfg.sigma_max) - cfg.p_mean) / cfg.p_std)
    sigmas = []
    for i in range(cfg.num_blocks + 1):
        p = cdf_min + (cdf_max - cdf_min) * (i / cfg.num_blocks)
        sigmas.append(float(np.exp(cfg.p_mean + cfg.p_std * norm.ppf(p))))
    return sigmas


def edm_scalings(sigma: torch.Tensor, sigma_data: float):
    """Return (c_skip, c_out, c_in, c_noise) per EDM preconditioning."""
    c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)
    c_out = sigma * sigma_data / (sigma**2 + sigma_data**2) ** 0.5
    c_in = 1.0 / (sigma**2 + sigma_data**2) ** 0.5
    c_noise = 0.25 * sigma.log()
    return c_skip, c_out, c_in, c_noise


def edm_loss_weight(sigma: torch.Tensor, sigma_data: float) -> torch.Tensor:
    return (sigma**2 + sigma_data**2) / (sigma * sigma_data) ** 2


def sample_block_sigmas(block_idx: int, n: int, block_sigmas: list[float], cfg: Config) -> torch.Tensor:
    """Sample n sigmas inside block_idx's range (optionally widened by gamma)."""
    lo, hi = block_sigmas[block_idx], block_sigmas[block_idx + 1]
    if cfg.gamma > 0.0:
        log_lo, log_hi = np.log(lo), np.log(hi)
        span = log_hi - log_lo
        lo = max(np.exp(log_lo - cfg.gamma * span), block_sigmas[0])
        hi = min(np.exp(log_hi + cfg.gamma * span), block_sigmas[-1])
    cdf_lo = norm.cdf((np.log(lo) - cfg.p_mean) / cfg.p_std)
    cdf_hi = norm.cdf((np.log(hi) - cfg.p_mean) / cfg.p_std)
    u = np.random.uniform(cdf_lo, cdf_hi, n)
    sigma = np.exp(cfg.p_mean + cfg.p_std * norm.ppf(u))
    return torch.from_numpy(sigma).float()


def get_inference_sigmas(num_steps: int, cfg: Config) -> torch.Tensor:
    """Descending sigma schedule (high noise -> low noise) for the inference ODE walk."""
    cdf_min = norm.cdf((np.log(cfg.sigma_min) - cfg.p_mean) / cfg.p_std)
    cdf_max = norm.cdf((np.log(cfg.sigma_max) - cfg.p_mean) / cfg.p_std)
    cdf_points = np.linspace(cdf_min, cdf_max, num_steps)
    sigmas = np.exp(cfg.p_mean + cfg.p_std * norm.ppf(cdf_points))
    sigmas = torch.from_numpy(sigmas).float()
    return torch.flip(sigmas, dims=[0])  # descending
