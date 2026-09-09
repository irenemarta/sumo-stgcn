import torch
import torch.nn as nn
from torch_geometric.profile import profileit
from tqdm import tqdm
from typing import Dict

import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from data.dataset import SUMODataset
from src.models.normalizer import MapNormalizer
from src.blocks.gcn import GCN
from src.flowMatching import FlowMatchingModel, cfm_loss


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


def dataset_split(dataset, window:int, train_fraction:float=0.6, val_fraction:float=0.20, test_fraction:float=0.20) -> Subset:
    n_total = len(dataset)
    train_end = int(n_total*train_fraction)
    validation_end = train_end + int(n_total*val_fraction)
    
    train_idx = list(range(0, train_end - window))
    val_idx = list(range(train_end, validation_end - window))
    test_idx = list(range(validation_end, n_total))
    
    return Subset(dataset, indeces=train_idx), Subset(dataset, indices=val_idx), Subset(dataset, indices=test_idx)


# @profileit()
def train(model: FlowMatchingModel, loader: DataLoader, optimizer, device: str = "cpu"):
    model.train()
    total_loss = 0.0
    for graphs, target in loader:  # one sample = (Window_dimension graphs, target)
        graphs = [g.to(device) for g in graphs]
        target = target.to(device)

        cond = model.conditioner(graphs)
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
        total_loss += loss

    return total_loss / len(loader)


# split in training, validation and evakuation DataLoader
# training set = 60%
# validation set = 20%
# test set = 20%
def run_training(
    model: FlowMatchingModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer,
    num_epochs: int,
    checkpoint_path: str,
    print_every: int = 100,
    device: str = "cpu",
) -> Dict[list]:

    model.to(device)
    best_val_loss = float("inf")
    loss_history = {"train_loss": [], "val_loss": []}

    for e in tqdm(range(num_epochs)):
        train_loss = train(model, train_loader, optimizer, device)
        val_loss = eval(model, val_loader, device)

        loss_history["train_loss"].append(train_loss)
        loss_history["val_loss"].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)

        if e % print_every == 0:
            print(
                f"Epoch: {e} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )

    return loss_history