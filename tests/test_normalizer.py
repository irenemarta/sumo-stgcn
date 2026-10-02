import torch

from src.models.normalizer import MapNormalizer


def test_zscore_has_zero_mean_unit_std_per_feature():
    x = torch.randn(50, 10, 3) * torch.tensor([100.0, 10.0, 0.1]) + 7.0
    norm = MapNormalizer(x)
    z = norm.zscore(x)
    flat = z.reshape(-1, 3)
    assert torch.allclose(flat.mean(0), torch.zeros(3), atol=1e-4)
    assert torch.allclose(flat.std(0, unbiased=False), torch.ones(3), atol=1e-3)


def test_inverse_undoes_zscore():
    x = torch.randn(20, 3) * 50 + 300
    norm = MapNormalizer(x)
    assert torch.allclose(norm.inverse(norm.zscore(x)), x, atol=1e-3)


def test_constant_feature_does_not_produce_nan():
    # e.g. occupancy always 0 on an unused edge
    x = torch.zeros(10, 3)
    x[:, 0] = torch.arange(10.0)
    z = MapNormalizer(x).zscore(x)
    assert torch.isfinite(z).all()


def test_save_load_roundtrip(tmp_path):
    x = torch.randn(30, 3)
    norm = MapNormalizer(x)
    path = tmp_path / "dynamic_norm_stats.pt"
    norm.save(str(path))
    loaded = MapNormalizer.load(str(path))
    assert torch.equal(loaded.mu, norm.mu)
    assert torch.equal(loaded.sigma, norm.sigma)
    assert torch.allclose(loaded.inverse(loaded.zscore(x)), x, atol=1e-4)
