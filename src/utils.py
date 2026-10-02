import os, json
from tqdm import tqdm
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_geometric.profile import profileit
from torch_geometric.data import Batch

import inputs.config as cfg
import importlib
importlib.reload(cfg) # refresh root dir

from data.dataset import SUMODataset
# from src.models.normalizer import MapNormalizer
# from src.blocks.gcn import GCN
from src.flowMatching import FlowMatchingModel, cfm_loss
from src.output_eval import print_test_predictions


def check_gpu():
    # Setup device-agnostic code
    if torch.cuda.is_available():
        device = "cuda"  # NVIDIA GPU
    elif torch.backends.mps.is_available():
        device = "mps"  # Apple GPU
    else:
        device = "cpu"  # Defaults to CPU if NVIDIA GPU/Apple GPU aren't available

    print(f"Using device: {device}")

    return device


from torch_geometric.data import Batch


def collate_batch(batch: list, window: int) -> Tuple[list, torch.Tensor]:
    """
    Gathers `batch_size` indipendent samples  (each sample is made of `window`
    graphs + target) in a unique batch. Useful to process each sample in a unique forward pass.

    For each position w in the window, `batch_size` graphs (one per sample) are aggregated
    in a single Batch (same topology of the original graph).
    Targets are stacked in a unique tensor of size [batch_size, N_nodes, F].
    """
    # batch: list of samples = [([graphs_1], t1), ([g2], t2), ..., ([g_nsample], t_nsample)]
    # with g_n = list of W graphs (Data objects)
    # t_n = target at
    batched_graphs = []
    for w in range(window):
        graphs_at_w = [sample[0][w] for sample in batch]
        batched_graphs.append(Batch.from_data_list(graphs_at_w))

    targets = torch.stack([sample[1] for sample in batch], dim=0)
    return batched_graphs, targets


# collate fn is useful ot parallelize work and exploit GPU potential


def dataset_split(
    dataset: SUMODataset,
    window: int,
    train_fraction: float = cfg.TRAIN_FRACTION,
    val_fraction: float = cfg.VAL_FRACTION,
    test_fraction: float = cfg.TEST_FRACTION,
) -> Subset:
    train_idx, val_idx, test_idx = [], [], []
    
    for sid, (start, end) in dataset.scenario_index_ranges().items():
        n_total = end - start
        train_end = int(n_total * train_fraction)
        validation_end = train_end + int(n_total * val_fraction)

        train_idx += list(range(start, start + max(0, train_end - window)))
        val_idx += list(range(start + train_end, start + max(train_end, validation_end - window)))
        test_idx += list(range(start + validation_end, end))

    return (
        Subset(dataset, indices=train_idx),
        Subset(dataset, indices=val_idx),
        Subset(dataset, indices=test_idx),
    )


# @profileit()
def train(model: FlowMatchingModel, loader: DataLoader, optimizer, device: str = "cpu"):
    model.train()
    total_loss = 0.0
    for graphs, target in loader:  # one sample = (Window_dimension graphs, target)
        graphs = [g.to(device) for g in graphs]
        target = target.to(device)

        cond = model.conditioner(graphs)
        # print("target.shape:", target.shape)
        # print("cond.shape:", cond.shape)
        loss = cfm_loss(model.velocity_field, target, cond)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def eval(model: FlowMatchingModel, loader: DataLoader, device: str = "cpu"):
    model.eval()
    total_loss = 0.0
    for graphs, target in loader:
        graphs = [g.to(device) for g in graphs]
        target = target.to(device)

        cond = model.conditioner(graphs)
        loss = cfm_loss(model.velocity_field, target, cond)
        total_loss += loss.item()
        # .item() converts tensor cuda to float to avoid device mismatch

    return total_loss / len(loader)


# split in training, validation and evakuation DataLoader
# training set = 60%
# validation set = 20%
# test set = 20%
import json  # in testa al file, se non già importato

def run_training(
    model, train_loader, val_loader, optimizer, num_epochs, checkpoint_path,
    print_every=100, device="cpu", params_tracker=None,
    history_path: Optional[str] = None,
):
    model.to(device)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    best_val_loss = float("inf")
    loss_history = {"train_loss": [], "val_loss": []}

    for e in tqdm(range(num_epochs)):
        train_loss = train(model, train_loader, optimizer, device)
        val_loss = eval(model, val_loader, device)

        loss_history["train_loss"].append(train_loss)
        loss_history["val_loss"].append(val_loss)

        if params_tracker is not None:
            params_tracker(model, e)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)

        if history_path is not None:
            with open(history_path, "w") as f:
                json.dump(loss_history, f)

        if e % print_every == 0:
            print(f"Epoch: {e} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    return loss_history


def run_testing(model, test_loader, num_repeats, device="cpu", params_tracker=None):
    model.to(device)
    test_losses = []
    for e in tqdm(range(num_repeats)):
        test_loss = eval(model, test_loader, device)
        test_losses.append(test_loss.item())
        if params_tracker is not None:
            params_tracker(model, e)
    return {"test_loss": test_losses}
