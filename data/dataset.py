import torch
import numpy as np
import xml.etree.ElementTree as ET
from torch_geometric.data import Data
from pathlib import Path


# class SUMOGraph(Data):
#     """
#     Nodes = SUMO edges (streets)
#     Edges = SUMO junctions (intersections)
#     """
#     def __init__(self, input_dir: Path, device: str='cpu'):
#         self.graphs_dir = input_dir.resolve()
#         self.paths = list(sorted(self.graphs_dir.glob(".pt"))) # PyTorch estension
        
#         self.device = device 
#         self.edge_index_map = {} # key 0 edge_id, value = connected nodes
#         self.junction_map = {} # key = junction id, value = involved edges
        
#         self._parse_net()
        
#     def _parse_net(self):
#         tree = ET.parse(self.net)
#         root_edge = tree.getroot("edge")
        

# data/sumo_dataset.py
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from .parser import XMLBuilder


class GraphBuilder(XMLBuilder):
    """     
    Nodes = SUMO edges (streets)
    Edges = SUMO junctions (intersections)
    """
    
    def build(self, net) -> dict:
        edge_id_to_idx = {}
        edge_static_features = []
        
        for idx, edge in enumerate(net.getEdges()):
            if edge.getID().startswith(":"):
                continue
            edge_id_to_idx[edge.getID()] = idx
            
            lanes = edge.getLanes()
            length = self.calculate_edge_length(edge.getShape())
            speed_limit = edge.getSpeed()
            num_lanes = len(lanes)
            priority = edge.getPriority()
            capacity = ...
            
            edge_static_features.append([length, speed_limit, float(num_lanes), float(priority), capacity])
            x_static = torch.tensor(edge_static_features, dtype=torch.long) # shape = torch.tensor
        for edge in net.