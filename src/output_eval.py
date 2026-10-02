import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List
from pathlib import Path

from functools import partial
from torch.utils.data import DataLoader, Subset
from src.flowMatching import FlowMatchingModel, predict


## about training and validation loss
# why is the validation loss lower than trainin loss?
# https://towardsdatascience.com/what-your-validation-loss-is-lower-than-your-training-loss-this-is-why-5e92e0b1747e/

def collect_params_stats(model: nn.Module, epoch: int):
    """Collect avg and std values for each layer, for each epoch."""
    records = []
    for name, param in model.named_parameters():
        records.append(
            {
                "layer": name,
                "mean": param.detach().mean().item(),
                "std": param.detach().std().item(),
                "epoch": epoch,
            }
        )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # .numel(): Returns the total number of elements in the input tensor.
    print(f"Total number of parameters: {total_params}")

    return pd.DataFrame(records)


def plot_parameters(df_params: pd.DataFrame, out_path: Path = "."):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("Parameters distribution")

    for n in range(len(axs)):
        axs[n].set_xlabel("index")
        axs[n].set_ylabel("value")
        axs[n].grid(True, alpha=0.3)

    # weights
    axs[0].set_title("weights")
    sns.scatterplot(
        data=df_params,
        x=df_params.index,
        y="value",
        marker="o",
        color="blue",
        ax=axs[0],
    )
    # biases
    axs[1].set_title("bias")
    sns.scatterplot(
        data=df_params,
        x=df_params.index,
        y="value",
        marker="o",
        color="orange",
        ax=axs[1],
    )

    plt.savefig(Path(out_path) / "parameters.png", dpi=fig.dpi)


def plot_dataloaders(dl_train: DataLoader, dl_test: DataLoader, out_path: Path = "."):
    # Rappresentazione dai batch
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))
    fig.suptitle("Dataloaders")

    axs[0].set_title("training dataloader")
    axs[1].set_title("test dataloader")
    axs[0].grid(True, alpha=0.3)
    axs[1].grid(True, alpha=0.3)

    for batch, (x, y) in enumerate(dl_train):
        axs[0].plot(x.squeeze(), y.squeeze(), "o")  # 8 batch
    for batch, (x, y) in enumerate(dl_test):
        axs[1].plot(x.squeeze(), y.squeeze(), "o")  # 2 batch

    plt.savefig(Path(out_path) / "dataloaders.png", dpi=fig.dpi)


def plot_loss_curves(
    history: Dict[str, List[float]], log_scale: bool = False, out_path: Path = "."
):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["train_loss"], label="Train loss")
    ax.plot(history["val_loss"], label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and validation loss")
    if log_scale:
        ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.savefig(Path(out_path) / "loss.png", dpi=fig.dpi)


def print_test_predictions(
    epoch: int, model: FlowMatchingModel, dl: DataLoader, df_params: pd.DataFrame, out_path: Path = '.'
):
    fig, axs = plt.subplots(3, 1, figsize=(10, 14))
    fig.suptitle(f"Loop results: epoch number {epoch}")

    # Loop
    for _, (features, labels) in enumerate(dl):
        axs[0].scatter(features, labels, marker="s", color="blue", alpha=0.8)
        axs[0].scatter(
            features,
            model(features).squeeze().detach().numpy(),
            marker="o",
            color="orange",
        )
        axs[0].grid(True, alpha=0.3)

    axs[0].scatter([], [], marker="s", color="blue", label="Original data")
    axs[0].scatter([], [], marker="o", color="orange", label="Predictions")
    axs[0].set_title("Model training and testing")
    axs[0].set_xlabel("features")
    axs[0].set_ylabel("labels")
    axs[0].legend()

    plt.savefig(Path(out_path) / "dataloaders.png", dpi=fig.dpi)

    for n in range(1, 3):
        axs[n].set_xlabel("index")
        axs[n].set_ylabel("values")
        axs[n].hlines(
            y=0, xmin=0, xmax=8, linestyles="dashed", color="black", alpha=0.2
        )
        axs[n].grid(True, alpha=0.3)

    # Weights
    sns.lineplot(
        data=df_params[df_params["type"] == "weight"],
        x=df_params[df_params["type"] == "weight"].index,
        y="value",
        marker="o",
        color="blue",
        ax=axs[1],
    )
    axs[1].set_title("Mean weights")

    # Bias
    sns.lineplot(
        data=df_params[df_params["type"] == "bias"],
        x=df_params[df_params["type"] == "bias"].index,
        y="value",
        marker="o",
        color="violet",
        ax=axs[2],
    )
    axs[2].set_title("Mean bias")

    plt.subplots_adjust(hspace=0.5)
    plt.show()


def plot_param_evolution(param_history: pd.DataFrame, out_path='.'):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    sns.lineplot(
        data=param_history, x="epoch", y="mean", hue="layer", ax=axs[0], legend=False
    )
    axs[0].set_title("Avg Weigths per layer")
    axs[0].grid(True, alpha=0.3)

    sns.lineplot(
        data=param_history, x="epoch", y="std", hue="layer", ax=axs[1], legend=False
    )
    axs[1].set_title("Standard deviaiton of weigths per layer")
    axs[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Path(out_path)/ "parameters.png", dpi=fig.dpi)


@torch.no_grad()
def evaluate_mae_per_scenario(
    model: FlowMatchingModel, dataset: "SUMODataset", normalizer, window: int,
    device="cpu", n_integration_steps: int=50, n_samples:int=10, batch_size: int=16,
    feature_names: List[str] = ("flow", "speed", "occupancy"),
) -> pd.DataFrame:
    """MAE per feature (flow, speed, occupancy), in unità reali, per ogni scenario separatamente."""
    model.eval()
    collate_fn = partial(_collate_for_eval, window=window)
    records = []

    for sid, (start, end) in dataset.scenario_index_ranges().items():
        indices = list(range(start, end))
        if not indices:
            continue

        loader = DataLoader(
            Subset(dataset, indices=indices),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        scenario_mae = []
        for graphs, target in loader:
            graphs = [g.to(device) for g in graphs]
            target = target.to(device)
            preds = predict(
                model, graphs, feat_dyn_dim=target.shape[-1],
                n_integration_steps=n_integration_steps, n_samples=n_samples,
            )
            pred_mean_real = normalizer.inverse(preds.mean(dim=0))
            target_real = normalizer.inverse(target)
            scenario_mae.append((pred_mean_real - target_real).abs().mean(dim=(0, 1)))
        if not scenario_mae:
            continue

        mae = torch.stack(scenario_mae).mean(dim=0)  # [F]
        record = {"scenario": sid, "n_samples": len(indices)}
        for i, fname in enumerate(feature_names[: mae.shape[0]]):
            record[f"mae_{fname}"] = mae[i].item()
        records.append(record)

    return pd.DataFrame(records)


def _collate_for_eval(batch: list, window: int):
    """Stessa logica di src.utils.collate_batch, ridefinita qui per evitare
    un import circolare (src.utils importa gia' da src.output_eval)."""
    from torch_geometric.data import Batch

    batched_graphs = []
    for w in range(window):
        graphs_at_w = [sample[0][w] for sample in batch]
        batched_graphs.append(Batch.from_data_list(graphs_at_w))

    targets = torch.stack([sample[1] for sample in batch], dim=0)
    return batched_graphs, targets


def plot_scenario_comparison(df: pd.DataFrame, out_path: Path = Path(".")) -> plt.Figure:
    """Bar chart della MAE per scenario, una barra per feature (flow/speed/occupancy)."""
    feature_cols = [c for c in df.columns if c.startswith("mae_")]
    fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.6), 5))
    df.set_index("scenario")[feature_cols].plot(kind="bar", ax=ax)
    ax.set_ylabel("MAE (unita' reali)")
    ax.set_title("MAE per scenario, a confronto")
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(Path(out_path) / "scenario_comparison.png", dpi=fig.dpi)
    return fig