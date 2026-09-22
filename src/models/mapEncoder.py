import torch
import torch.nn as nn
from src.blocks.gcn import GCN
from src.blocks.gru import TemporalEncoder
from src.blocks.gat import GAT

# concatenate GCN conv result with GAT to obtain spatio-temporal elaboration

class SpatialEncoder(nn.Module):
    """Concatenates, successively, the results from GCN and GAT modules"""
    def __init__(self, gcn: GCN, gat: GAT):
        super().__init__()
        self.gcn = gcn
        self.gat = gat
    
    def forward(self, data):
        x = self.gcn(data)
        x = self.gat(data, x)
        return x
    
    
class SpatioTemporalConditioner(nn.Module):
    """Encodes the temporal window of per-node embeddings into a single node representation."""
    def __init__(self, spatial_encoder: nn.Module, temporal_encoder: TemporalEncoder):
        super().__init__()
        self.spatial_encoder = spatial_encoder
        self.temporal_encoder = temporal_encoder

    def forward(self, graphs: list) -> torch.Tensor:
        # graphs: list of W Data objects (window returned by SUMODataset.get() with traffic state features)
        embeddings = []
        for g in graphs:
            emb = self.spatial_encoder(g) # [B*N_nodi, hidden]
            B = getattr(g, "num_graphs", 1) # Batch of num_graphs
            emb = emb.view(B, -1, emb.shape[-1]) # tensor reshaping to [B, N_nodi, hidden]
            embeddings.append(emb) # GCN + GAT
        x_seq = torch.stack(embeddings, dim=0) # input to GRU, [W, B, N_nodes, hidden]
        return self.temporal_encoder(x_seq) # [B, N_nodes, hidden]