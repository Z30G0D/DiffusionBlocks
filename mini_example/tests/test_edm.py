import numpy as np
import torch

import mini_db


def test_block_sigmas_monotonic_and_length():
    cfg = mini_db.Config(num_blocks=3)
    bs = mini_db.get_block_sigmas(cfg)
    assert len(bs) == cfg.num_blocks + 1
    assert all(bs[i] < bs[i + 1] for i in range(len(bs) - 1))


def test_edm_scalings_shapes_and_skip_limit():
    cfg = mini_db.Config()
    sigma = torch.tensor([cfg.sigma_min, 1.0, cfg.sigma_max])
    c_skip, c_out, c_in, c_noise = mini_db.edm_scalings(sigma, cfg.sigma_data)
    for t in (c_skip, c_out, c_in, c_noise):
        assert t.shape == sigma.shape
    # at tiny sigma, c_skip -> ~1 (model mostly copies the clean signal through)
    assert c_skip[0] > 0.99


def test_loss_weight_positive():
    cfg = mini_db.Config()
    sigma = torch.tensor([0.01, 1.0, 50.0])
    w = mini_db.edm_loss_weight(sigma, cfg.sigma_data)
    assert torch.all(w > 0)


def test_sample_block_sigmas_in_range():
    cfg = mini_db.Config(num_blocks=3, gamma=0.0)
    bs = mini_db.get_block_sigmas(cfg)
    sig = mini_db.sample_block_sigmas(1, 256, bs, cfg)
    assert sig.shape == (256,)
    assert torch.all(sig >= bs[1] - 1e-4) and torch.all(sig <= bs[2] + 1e-4)


def test_inference_sigmas_descending():
    cfg = mini_db.Config()
    s = mini_db.get_inference_sigmas(num_steps=3, cfg=cfg)
    assert s.shape == (3,)
    assert s[0] > s[-1]
