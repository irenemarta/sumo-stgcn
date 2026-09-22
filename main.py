import os, sys
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

# setup
ROOT = Path.cwd().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

print(f"project root: {ROOT}")

WINDOW = 6
EPOCHS = 3000
device = check_gpu()

# Dataset split
dataset = SUMODataset(root="data/built_dataset", window=WINDOW)
train_set, val_set, test_set = dataset_split(dataset, window=WINDOW, train_fraction=0.6, val_fraction=0.2, test_fraction=0.2)

print("len(dataset):", len(dataset))
print("window:", WINDOW)
print("Found" + len(os.listdir(os.path.join(f"{ROOT}/data/built_dataset", "processed"))) + ".pt files")

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


normalizer = MapNormalizer.load(f"{ROOT}/data/built_dataset/normalized/norm_stats.pt").to(device)
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

# Train model
history = run_training(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=device,
    num_epochs=EPOCHS,
    checkpoint_path="checkpoints/first_run.pt",
    print_every=5,
)