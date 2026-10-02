import os, sys
import pandas as pd
from pathlib import Path
from functools import partial

import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from data.dataset import SUMODataset
from src.blocks.gcn import GCN
from src.blocks.gat import GAT
from src.models.mapEncoder import SpatialEncoder
from src.blocks.gru import TemporalEncoder
from src.models.mapEncoder import SpatioTemporalConditioner
from src.flowMatching import VelocityVectorField, FlowMatchingModel, predict
from src.utils import check_gpu, run_training, dataset_split, collate_batch
from src.models.normalizer import MapNormalizer 
from src.output_eval import (
    collect_params_stats, plot_loss_curves, plot_param_evolution,
    evaluate_mae_per_scenario, plot_scenario_comparison, plot_dataloaders
)

# setup
ROOT = Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

checkpoint_dir = Path("checkpoints")
out_path = Path("output")
checkpoint_dir.mkdir(exist_ok=True)
out_path.mkdir(exist_ok=True)

print(f"project root: {ROOT}")

WINDOW = 6
EPOCHS = 500
device = check_gpu()

# Dataset split
dataset = SUMODataset(root="data/built_dataset", window=WINDOW)
train_set, val_set, test_set = dataset_split(dataset, window=WINDOW, train_fraction=0.6, val_fraction=0.2, test_fraction=0.2)

print("len(dataset):", len(dataset))
print("window:", WINDOW)
print(f"Found {len(os.listdir(os.path.join(f'{ROOT}/data/built_dataset', 'processed')))} .pt files")

# collate fn is useful ot parallelize work and exploit GPU potential
# Build dataloaders
# collate custom to deactivate automatic batching - DataLoader can't batch (list[Data], Tensor)
collate_fn = partial(collate_batch, window=WINDOW) 
train_loader = DataLoader(train_set, batch_size=16, shuffle=True, collate_fn=collate_fn) # shuffle only training set
val_loader = DataLoader(val_set, batch_size=16, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_set, batch_size=8, shuffle=False, collate_fn=collate_fn)

# dimensions
node_in_dim = 4 + 3 # x_static + x_dynamic
gcn_out_dim = 16 # edge_attributes
gat_out_dim = 16 # per-node spatial embedding
temporal_hidden = 32 # cond dimension
feat_dyn_dim = 3 # target: flow, speed, occupancy

gcn = GCN(hidden_dims=[node_in_dim, 32, gcn_out_dim], dropout_prob=0.1)
gat = GAT(hidden_dims=[gcn_out_dim, gat_out_dim], heads=2, edge_dim=6, dropout_prob=0.1)
spatial_encoder = SpatialEncoder(gcn, gat)
temporal_encoder = TemporalEncoder(in_dim=gat_out_dim, hidden_dim=temporal_hidden, num_layers=1)
conditioner = SpatioTemporalConditioner(spatial_encoder, temporal_encoder)

velocity_field = VelocityVectorField(feat_dyn_dim=feat_dyn_dim, cond_dim=temporal_hidden, hidden_dim=64)

# Instantiate the model
model = FlowMatchingModel(conditioner, velocity_field).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


normalizer = MapNormalizer.load(f"{ROOT}/data/built_dataset/normalized/dynamic_norm_stats.pt").to(device)
sample = train_set[0]   # passa per Subset -> SUMODataset.__getitem__ -> get()
graphs, target = sample
graphs = [g.to(device) for g in graphs]
target = target.to(device)
preds = predict(model, graphs, feat_dyn_dim=3, n_integration_steps=50, n_samples=10)
print(f"preds shape:{preds.shape}")
# preds.shape: [10, N_nodes, 3]
pred_mean = preds.mean(dim=0) # [N_nodes, 3]
pred_std = preds.std(dim=0) # [N_nodes, 3]
pred_mean_real = normalizer.inverse(pred_mean)   # torna in veh/h
target_real = normalizer.inverse(target)
# print("target.shape (accesso diretto):", target.shape)

# print("len(dataset):", len(dataset))
# print("window:", WINDOW)
# print("file .pt trovati:", len(os.listdir(os.path.join("data/built_dataset", "processed"))))

param_history = []
def param_tracker(model, epoch):
    if epoch % 10 == 0:
        param_history.append(collect_params_stats(model, epoch))

# Train model
history = run_training(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=device,
    num_epochs=EPOCHS,
    checkpoint_path="checkpoints/first_run.pt",
    history_path="checkpoints/history.json",
    params_tracker=param_tracker,
    print_every=20,
)

# plot_dataloaders rimosso per ora: assume batch (tensor, tensor) semplici, ma i nostri
# batch sono (list[Batch PyG], target) - richiederebbe una riscrittura, non un fix veloce.
# plot_dataloaders(dl_train=train_loader, dl_test=test_loader, out_path=out_path)

plot_param_evolution(param_history=pd.concat(param_history, ignore_index=True), out_path=out_path)
plot_loss_curves(history=history, log_scale=False, out_path=out_path)

mae_df = evaluate_mae_per_scenario(
    model=model,
    dataset=dataset,
    normalizer=normalizer,
    window=WINDOW,
    device=device,
    n_samples=10,
)
mae_df.to_csv(out_path / "mae_per_scenario.csv", index=False)
plot_scenario_comparison(mae_df, out_path=out_path)