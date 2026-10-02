from functools import partial

import matplotlib

matplotlib.use("Agg")  # no display needed

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from conftest import WINDOW
from src.models.normalizer import MapNormalizer
from src.output_eval import collect_params_stats, evaluate_mae, plot_loss_curves, plot_param_evolution
from src.utils import collate_batch


def test_collect_params_stats_columns(model):
    df = collect_params_stats(model, epoch=3)
    assert {"layer", "mean", "std", "epoch"} <= set(df.columns)
    assert (df["epoch"] == 3).all()
    assert len(df) == len(list(model.parameters()))


def test_plot_param_evolution_accepts_tracker_output(model):
    history = pd.concat([collect_params_stats(model, e) for e in range(3)])
    fig = plot_param_evolution(history)
    assert len(fig.axes) == 2


def test_plot_loss_curves_accepts_str_out_path(tmp_path):
    """Regression: out_path passed as str used to crash on `out_path / "loss.png"`."""
    plot_loss_curves({"train_loss": [1.0, 0.5], "val_loss": [1.1, 0.6]}, out_path=str(tmp_path))
    assert (tmp_path / "loss.png").is_file()


def test_evaluate_mae_returns_one_value_per_feature(dataset, model):
    loader = DataLoader(Subset(dataset, range(4)), batch_size=2,
                        collate_fn=partial(collate_batch, window=WINDOW))
    normalizer = MapNormalizer(torch.randn(100, 3) * 10 + 50)
    mae = evaluate_mae(model, loader, normalizer, n_integration_steps=3, n_samples=2)
    assert mae.shape == (3,)  # flow, speed, occupancy
    assert (mae >= 0).all() and torch.isfinite(mae).all()
