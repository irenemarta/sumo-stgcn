import torch.nn as nn

# concatenate GCN conv result with GAT to obtain spatio-temporal elaboration

class SpatialEncoder(nn.Module):
    def __init__(self, gcn: GCN, gat: GAT):
        super().__init__()
        self.gecn = gcn
        self.gat = gat
    
    def forward(self, data):
        x = self.gcn(data)
        x = self.gat(data, x)
        return x