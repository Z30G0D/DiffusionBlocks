import torch
import mini_db


def test_config_defaults_consistent():
    cfg = mini_db.Config()
    assert cfg.num_layers % cfg.num_blocks == 0
    assert cfg.H > 0


def test_get_device_returns_torch_device():
    dev = mini_db.get_device()
    assert isinstance(dev, torch.device)


def test_block_layer_indices_partition_all_layers():
    cfg = mini_db.Config(num_layers=6, num_blocks=3)
    idx = [mini_db.block_layer_indices(b, cfg) for b in range(cfg.num_blocks)]
    assert idx == [[0, 1], [2, 3], [4, 5]]
    flat = [i for block in idx for i in block]
    assert flat == list(range(cfg.num_layers))
