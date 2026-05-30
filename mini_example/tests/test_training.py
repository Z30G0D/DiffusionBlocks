import torch

import mini_db


def _toy_loader(cfg, n=128, batch_size=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    # two linearly separable-ish clusters in pixel space, 2 active classes
    x = torch.randn(n, cfg.image_dim, generator=g) * 0.1
    y = torch.randint(0, cfg.num_classes, (n,), generator=g)
    x += torch.nn.functional.one_hot(y, cfg.num_classes).float().repeat(
        1, cfg.image_dim // cfg.num_classes + 1
    )[:, : cfg.image_dim]
    ds = torch.utils.data.TensorDataset(x, y)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)


def test_train_baseline_loss_decreases():
    torch.manual_seed(0)
    cfg = mini_db.Config()
    model = mini_db.PlainClassifier(cfg)
    loader = _toy_loader(cfg)
    hist = mini_db.train_baseline(model, loader, epochs=5, lr=1e-3, device=torch.device("cpu"))
    assert hist["loss"][-1] < hist["loss"][0]


def test_train_diffusionblocks_learns():
    # The block-wise loss is EDM-weighted and block-randomized, so its raw value is noisy.
    # Instead of asserting the loss curve, we assert the model actually LEARNS the toy task:
    # accuracy ends up well above the 10% chance level. `seed` makes it deterministic.
    cfg = mini_db.Config()
    model = mini_db.DiffusionClassifier(cfg)
    loader = _toy_loader(cfg)
    mini_db.train_diffusionblocks(
        model, loader, epochs=8, lr=1e-3, device=torch.device("cpu"), seed=0
    )
    acc = mini_db.evaluate(model, loader, device=torch.device("cpu"), diffusion=True)
    assert acc > 0.3


def test_evaluate_returns_fraction():
    cfg = mini_db.Config()
    model = mini_db.PlainClassifier(cfg)
    loader = _toy_loader(cfg)
    acc = mini_db.evaluate(model, loader, device=torch.device("cpu"), diffusion=False)
    assert 0.0 <= acc <= 1.0
