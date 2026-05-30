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


def test_backbone_run_layers_preserves_shape():
    cfg = mini_db.Config()
    bb = mini_db.ResidualBackbone(cfg)
    z = torch.randn(4, cfg.H)
    cond = torch.randn(4, cfg.H)
    sig_emb = torch.randn(4, cfg.H)
    out = bb.run_layers(z, cond, sig_emb, layer_indices=[0, 1])
    assert out.shape == (4, cfg.H)


def test_backbone_run_layers_only_runs_requested_block():
    cfg = mini_db.Config(num_layers=6, num_blocks=3)
    bb = mini_db.ResidualBackbone(cfg)
    z = torch.randn(2, cfg.H)
    cond = torch.zeros(2, cfg.H)
    sig_emb = torch.zeros(2, cfg.H)
    # running only block 0's layers must equal manually running layers 0 and 1
    out_block = bb.run_layers(z, cond, sig_emb, layer_indices=[0, 1])
    manual = z
    for i in [0, 1]:
        manual = bb.layers[i](manual, cond, sig_emb)
    assert torch.allclose(out_block, manual, atol=1e-6)


def test_encode_image_and_sigma_embed_shapes():
    cfg = mini_db.Config()
    bb = mini_db.ResidualBackbone(cfg)
    img = torch.randn(5, cfg.image_dim)
    assert bb.encode_image(img).shape == (5, cfg.H)
    c_noise = torch.randn(5)
    assert bb.sigma_embed(c_noise).shape == (5, cfg.H)


def test_diffusion_denoise_logits_shape():
    cfg = mini_db.Config()
    model = mini_db.DiffusionClassifier(cfg)
    images = torch.randn(4, cfg.image_dim)
    z = torch.randn(4, cfg.H)
    sigma = torch.full((4,), 1.0)
    logits = model.denoise(images, z, sigma, block_idx=1)
    assert logits.shape == (4, cfg.num_classes)


def test_diffusion_denoise_block_independent_of_other_blocks():
    cfg = mini_db.Config()
    model = mini_db.DiffusionClassifier(cfg)
    images = torch.randn(2, cfg.image_dim)
    z = torch.randn(2, cfg.H)
    sigma = torch.full((2,), 2.0)
    # mutating a layer NOT in block 0 must not change block 0's output
    out_before = model.denoise(images, z, sigma, block_idx=0)
    with torch.no_grad():
        for p in model.backbone.layers[4].parameters():
            p.add_(1.0)
    out_after = model.denoise(images, z, sigma, block_idx=0)
    assert torch.allclose(out_before, out_after, atol=1e-6)


def test_diffusion_predict_returns_class_logits():
    cfg = mini_db.Config()
    model = mini_db.DiffusionClassifier(cfg)
    images = torch.randn(3, cfg.image_dim)
    logits = model.predict(images, num_steps=cfg.num_blocks)
    assert logits.shape == (3, cfg.num_classes)
