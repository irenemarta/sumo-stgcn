import torch
import torch.nn as nn
from typing import OrderedDict


"""
For more about Convolutional Layers: https://pytorch-geometric.readthedocs.io/en/2.5.1/modules/nn.html#convolutional-layers
"""

class GraphConv(nn.Module):
    def __init__(self, in_channles:int, out_channles:int, attention:bool=False):
        super().__init__()
        self.convs = nn.ModuleList(
            nn.Sequential(
            OrderedDict[
                ("l1", nn.)
            ]
        )
    
    def forward(self, x):
        return self.l1(self.l2(x))
        

model = nn.Sequential()