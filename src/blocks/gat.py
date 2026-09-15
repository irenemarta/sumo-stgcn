import torch.nn as nn
import torch_geometric.nn as tgn

# check: https://www.youtube.com/watch?v=AWkPjrZshug
# and: https://pytorch-geometric.readthedocs.io/en/2.8.0/generated/torch_geometric.nn.conv.GATv2Conv.html

class GAT(nn.Module):
    def __init__(self, hidden_dims: list[int], heads: int, edge_dim: int, dropout_prob: float):
        super().__init__()
        assert len(hidden_dims) >= 2, "WARNING: hidden layers should be at least 2"
        conv_layers = [tgn.GATv2Conv(
            in_channels=hidden_dims[i],
            out_channels=hidden_dims[i-1],
            heads=heads,
            concat=False, # either concatenation or averaging of the feature vectors
            edge_dim=edge_dim,
            dropout=dropout_prob,
            )
        for i in range(len(hidden_dims) -1)
        ]
        props = [
            nn.Sequential(
                nn.LayerNorm(hidden_dims[i+1]),
                nn.ReLU(),
                nn.Dropout(p=dropout_prob),
            )
        for i in range(len(hidden_dims) - 1)
        ]
        
        self.convs = nn.ModuleList(conv_layers)
        self.propagations = nn.ModuleList(props)
        
    def forward(self, data, x):
        edge_index, edge_attr = data.edge_index, data.edge_attr
        for i, (conv, prop) in enumerate(zip(self.convs, self.propagations)):
            x = conv(x, edge_index, edge_attr)
            if i < len(self.convs) - 1:
                x = prop(x)
        return x