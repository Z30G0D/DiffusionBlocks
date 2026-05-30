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


# --------------------------------------------------------------------------- #
# Shared residual-MLP backbone
# --------------------------------------------------------------------------- #
class FiLMResidualLayer(nn.Module):
    """z <- z + gate * MLP(AdaLN(z | cond + sigma_emb)). One discretized ODE step."""

    def __init__(self, H: int):
        super().__init__()
        self.norm = nn.LayerNorm(H, elementwise_affine=False)
        self.film = nn.Linear(H, 2 * H)
        self.mlp = nn.Sequential(nn.Linear(H, H), nn.GELU(), nn.Linear(H, H))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, z: torch.Tensor, cond: torch.Tensor, sigma_emb: torch.Tensor) -> torch.Tensor:
        scale, shift = self.film(cond + sigma_emb).chunk(2, dim=-1)
        h = self.norm(z) * (1 + scale) + shift
        return z + self.mlp(h)


class ResidualBackbone(nn.Module):
    """Image encoder + sigma embedder + L FiLM residual layers (grouped into blocks)."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = nn.Sequential(
            nn.Linear(cfg.image_dim, 256), nn.GELU(), nn.Linear(256, cfg.H)
        )
        self.sigma_mlp = nn.Sequential(
            nn.Linear(1, cfg.H), nn.GELU(), nn.Linear(cfg.H, cfg.H)
        )
        self.layers = nn.ModuleList([FiLMResidualLayer(cfg.H) for _ in range(cfg.num_layers)])

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def sigma_embed(self, c_noise: torch.Tensor) -> torch.Tensor:
        return self.sigma_mlp(c_noise.unsqueeze(-1))

    def run_layers(self, z, cond, sigma_emb, layer_indices: list[int]) -> torch.Tensor:
        for i in layer_indices:
            z = self.layers[i](z, cond, sigma_emb)
        return z


# --------------------------------------------------------------------------- #
# Diffusion classifier: denoise a label embedding, conditioned on the image
# --------------------------------------------------------------------------- #
class DiffusionClassifier(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.backbone = ResidualBackbone(cfg)
        self.label_embed = nn.Embedding(cfg.num_classes, cfg.H)
        self.block_sigmas = get_block_sigmas(cfg)

    def class_embeddings(self) -> torch.Tensor:
        """L2-normalized target embeddings z0 for every class."""
        return F.normalize(self.label_embed.weight, p=2, dim=-1)

    def estimate_block(self, sigma: torch.Tensor) -> int:
        """Map a sigma (or batch of sigmas) to the responsible block index."""
        edges = torch.tensor(self.block_sigmas, device=sigma.device)
        b = torch.bucketize(sigma, edges, right=True) - 1
        b = torch.clamp(b, 0, self.cfg.num_blocks - 1).long()
        vals, counts = b.unique(return_counts=True)
        return int(vals[counts.argmax()].item())

    def denoise(self, images, z, sigma, block_idx=None):
        """Run ONE block to denoise z, return class logits. No other block is touched."""
        if block_idx is None:
            block_idx = self.estimate_block(sigma)
        c_skip, c_out, c_in, c_noise = edm_scalings(sigma, self.cfg.sigma_data)
        cond = self.backbone.encode_image(images)
        sigma_emb = self.backbone.sigma_embed(c_noise)
        layer_indices = block_layer_indices(block_idx, self.cfg)
        h = self.backbone.run_layers(z * c_in[:, None], cond, sigma_emb, layer_indices)
        denoised = h * c_out[:, None] + z * c_skip[:, None]
        logits = F.linear(denoised, self.class_embeddings())
        return logits

    @torch.no_grad()
    def predict(self, images, num_steps=None):
        """Inference: start from noise, walk DOWN the sigma ladder one block per Euler step."""
        cfg = self.cfg
        num_steps = num_steps or cfg.num_blocks
        sigmas = get_inference_sigmas(num_steps, cfg).to(images.device)
        z = torch.randn(images.shape[0], cfg.H, device=images.device)
        z = z * (1.0 + sigmas[0] ** 2) ** 0.5
        embed = self.class_embeddings()
        for i in range(num_steps - 1):
            sigma = sigmas[i].expand(images.shape[0])
            logits = self.denoise(images, z, sigma)
            denoised = F.softmax(logits, dim=-1) @ embed
            d = (z - denoised) / sigma[:, None]
            z = z + (sigmas[i + 1] - sigmas[i]) * d
        last = sigmas[-1].expand(images.shape[0])
        return self.denoise(images, z, last)


# --------------------------------------------------------------------------- #
# Plain classifier baseline: SAME backbone, standard cross-entropy, no diffusion
# --------------------------------------------------------------------------- #
class PlainClassifier(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.backbone = ResidualBackbone(cfg)
        self.head = nn.Linear(cfg.H, cfg.num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        cond = self.backbone.encode_image(images)
        sigma_emb = torch.zeros_like(cond)
        z = cond
        z = self.backbone.run_layers(z, cond, sigma_emb, list(range(self.cfg.num_layers)))
        return self.head(z)


# --------------------------------------------------------------------------- #
# Training & evaluation
# --------------------------------------------------------------------------- #
def train_baseline(model, loader, epochs, lr, device):
    """Standard end-to-end training: forward all layers, backprop through all."""
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    history = {"loss": []}
    for _ in range(epochs):
        running, n = 0.0, 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * images.shape[0]
            n += images.shape[0]
        history["loss"].append(running / n)
    return history


def train_diffusionblocks(model, loader, epochs, lr, device):
    """Block-wise training: each step samples ONE block; only that block is in the graph."""
    import random

    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    cfg = model.cfg
    history = {"loss": [], "loss_per_block": {b: [] for b in range(cfg.num_blocks)}}
    embed_table = model.label_embed
    for _ in range(epochs * cfg.num_blocks):  # B x more steps to match total per-block updates
        running, n = 0.0, 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            block_idx = random.randrange(cfg.num_blocks)
            z0 = F.normalize(embed_table(labels), p=2, dim=-1)
            sigma = sample_block_sigmas(block_idx, z0.shape[0], model.block_sigmas, cfg).to(device)
            zt = z0 + sigma[:, None] * torch.randn_like(z0)
            logits = model.denoise(images, zt, sigma, block_idx=block_idx)
            ce = F.cross_entropy(logits, labels, reduction="none")
            w = edm_loss_weight(sigma, cfg.sigma_data)
            loss = (ce * w).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * images.shape[0]
            n += images.shape[0]
            history["loss_per_block"][block_idx].append(loss.item())
        history["loss"].append(running / n)
    return history


@torch.no_grad()
def evaluate(model, loader, device, diffusion: bool, num_steps: int | None = None):
    model.to(device).eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model.predict(images, num_steps=num_steps) if diffusion else model(images)
        correct += (logits.argmax(-1) == labels).sum().item()
        total += labels.shape[0]
    return correct / total
