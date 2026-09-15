import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List
from pathlib import Path
from torch.utils.data import DataLoader
from src.flowMatching import FlowMatchingModel


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

    plt.savefig(Path(out_path / "parameters.png"), dpi=fig.dpi)


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

    plt.savefig(Path(out_path / "dataloaders.png"), dpi=fig.dpi)


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

    plt.savefig(Path(out_path / "loss.png"), dpi=fig.dpi)


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

    plt.savefig(Path(out_path / "dataloaders.png"), dpi=fig.dpi)

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


def plot_param_evolution(param_history: pd.DataFrame):
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
    return fig
