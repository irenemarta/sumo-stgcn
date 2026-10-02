import json

import pytest
import torch
from torch_geometric.data import Data

N_NODES = 5
N_EDGES = 4
X_DIM = 7
EDGE_DIM = 6
FEAT_DYN = 3
WINDOW = 3

# scenario_id -> number of snapshots saved on disk
MANIFEST = {"baseline": 40, "pos9_sev07_seed1": 30}


def make_graph(seed: int) -> Data:
    g = torch.Generator().manual_seed(seed)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]], dtype=torch.long)
    edge_attr = torch.rand(N_EDGES, EDGE_DIM, generator=g)  # col 4 = severity (GCN edge weight)
    x = torch.randn(N_NODES, X_DIM, generator=g)
    y = torch.randn(N_NODES, FEAT_DYN, generator=g)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.fixture
def dataset_root(tmp_path):
    processed = tmp_path / "processed"
    processed.mkdir()
    (processed / "manifest.json").write_text(json.dumps(MANIFEST))
    seed = 0
    for sid, n in MANIFEST.items():
        for i in range(n):
            g = make_graph(seed)
            g.sid, g.local_i = sid, i  # tag to check windows never cross scenarios
            torch.save(g, processed / f"data_{sid}_{i}.pt")
            seed += 1
    return tmp_path


@pytest.fixture
def dataset(dataset_root):
    from data.dataset import SUMODataset

    return SUMODataset(root=str(dataset_root), window=WINDOW)


@pytest.fixture
def model():
    """Same architecture as main.py, smaller hidden sizes."""
    from src.blocks.gat import GAT
    from src.blocks.gcn import GCN
    from src.blocks.gru import TemporalEncoder
    from src.flowMatching import FlowMatchingModel, VelocityVectorField
    from src.models.mapEncoder import SpatialEncoder, SpatioTemporalConditioner

    gcn = GCN(hidden_dims=[X_DIM, 8, 8], dropout_prob=0.0)
    gat = GAT(hidden_dims=[8, 8], heads=2, edge_dim=EDGE_DIM, dropout_prob=0.0)
    conditioner = SpatioTemporalConditioner(
        SpatialEncoder(gcn, gat), TemporalEncoder(in_dim=8, hidden_dim=16)
    )
    velocity = VelocityVectorField(feat_dyn_dim=FEAT_DYN, cond_dim=16, hidden_dim=16)
    return FlowMatchingModel(conditioner, velocity)
