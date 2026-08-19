import torch
import torch.nn as nn
from  torch_geometric.profile import profileit
from tqdm import tqdm

import torch
from torch_geometric.loader import DataLoader
from src.dataset import SUMODataset
from src.models.normalizer import MapNormalizer
from src.blocks.gcn import GCN


def check_gpu():    
    # Setup device-agnostic code 
    if torch.cuda.is_available():
        device = "cuda" # NVIDIA GPU
    elif torch.backends.mps.is_available():
        device = "mps" # Apple GPU
    else:
        device = "cpu" # Defaults to CPU if NVIDIA GPU/Apple GPU aren't available

    return print(f"Using device: {device}")



@profileit()
def train(loader: DataLoader, model: GCN, loss_fn, optimizer, device:str='cpu'):
    model.train()
    for batch in loader:
        batch = batch.to(device) # in training mode
        pred = model(batch) # forward pass
        loss = loss_fn(pred, batch) # calculate the loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() # gradient descent
    
    return loss


@torch.no_grad()
def eval(model: GCN, loader: DataLoader, loss_fn, device: str = 'cpu'):
    model.eval()
    for batch in loader:
        batch = batch.to(device)
        pred_eval = model(batch) # in evaluation mode
        loss = loss_fn(pred_eval, batch) # test loss
    
    return loss



## TODO: in the right module
EPOCHS = 1000
for e in tqdm(range(EPOCHS)):
    train_loss = train()
    test_loss = eval()
    
    if e % 100 == 0:
        print(f"Epoch: {e} | Train Loss: {train_loss} | Test Loss: {test_loss}")