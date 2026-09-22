import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as tgn

# DROPOUT penalizes model variance by randomly freezing neurons in a layer during model training.


# https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html#data-handling-of-graphs

SEVERITY_IDX = 4 # positional idx in the features vector

class GCN(nn.Module):
    def __init__(self, hidden_dims: list[int], dropout_prob: float):
        super().__init__()
        assert len(hidden_dims) >= 2, "WARN: at least 2 dimensions (I/O)"
        conv_layers = [tgn.GCNConv(
            in_channels=hidden_dims[i],
            out_channels=hidden_dims[i+1]
            )
        for i in range(len(hidden_dims) - 1)
        ]
        propagations = [
            nn.Sequential(
                nn.LayerNorm(hidden_dims[i+1]),
                nn.ReLU(),
                nn.Dropout(p=dropout_prob),
            )
        for i in range(len(hidden_dims) - 1)
        ]

        self.convs = nn.ModuleList(conv_layers)
        self.propagations = nn.ModuleList(propagations)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_weight = data.edge_attr[:,SEVERITY_IDX]
        for i, (conv, prop) in enumerate(zip(self.convs, self.propagations)):
            x = conv(x, edge_index, edge_weight=edge_weight)
            if i < len(self.convs) - 1:
                x = prop(x)
        return x