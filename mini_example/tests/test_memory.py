import torch

import mini_db


def test_activation_bytes_scales_with_blocks_in_graph():
    cfg = mini_db.Config(num_layers=6, num_blocks=3)
    model = mini_db.DiffusionClassifier(cfg)
    images = torch.randn(16, cfg.image_dim)
    z = torch.randn(16, cfg.H)
    sigma = torch.full((16,), 1.0)

    full = mini_db.activation_bytes(model, images, z, sigma, blocks_in_graph=cfg.num_blocks)
    one = mini_db.activation_bytes(model, images, z, sigma, blocks_in_graph=1)
    assert full > one
    # roughly B x more activations when all blocks are in the graph
    assert abs(full / one - cfg.num_blocks) < 1.0


def test_measure_peak_memory_returns_nonnegative():
    peak = mini_db.measure_peak_memory(lambda: torch.randn(1000, 1000).sum())
    assert peak >= 0
