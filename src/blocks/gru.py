import torch
import torch.nn as nn

"""
Module to perfrom temporal aggregation of spatial embedding windows per node, via GRU.
"""


class TemporalEncoder(nn.Module):
    """
    Input: x_seq [W, N_nodes, in_dim] (per node)
    Output: cond [N_nodes, hidden_dim]
    """
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 1, dropout_prob: float = 0.0):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout_prob if num_layers > 1 else 0.0,
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # why do we use batches and dataloaders? 
        # https://www.diariodiunanalista.it/posts/addestramento-efficiente-di-modelli-di-deep-learning-in-pytorch/
        W, B, N, H = x_seq.shape
        # print(f"Window, Batch, Nodes, Hidden dims: {W, B, N, H}")
        x_seq_flat = x_seq.view(W, B * N, H) # GRU input: B*N
        _, h_n = self.gru(x_seq_flat)  # h_n: [num_layers, N_nodes, hidden_dim]
        return h_n[-1].view(B, N, -1)  # last layer state: [N_nodes, hidden_dim]