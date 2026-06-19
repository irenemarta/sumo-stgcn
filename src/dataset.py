import torch
import os
import sumolib
from torch_geometric.data import Data, Dataset
from typing import List, Dict, Optional, Tuple
from ..data.parser import XMLBuilder

from collections import defaultdict
import xml.etree.ElementTree as ET
from pathlib import Path

""" 
Module used to create the tensors to be elaborated by the GCN

As a reference, from torch-geometrics, one needs to compute:
    data.x: Node feature matrix with shape [num_nodes, num_node_features]
    data.edge_index: Graph connectivity in COO format with shape [2, num_edges] and type torch.long
    data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
    data.y: Target to train against (may have arbitrary shape), e.g., node-level targets of shape [num_nodes, *] or graph-level targets of shape [1, *]
    data.pos: Node position matrix with shape [num_nodes, num_dimensions]
"""


class GraphBuilder(XMLBuilder):
    """
    Function to build the tensors representing static features of the graph.
    Nodes = SUMO edges (streets)
    Edges = SUMO junctions (intersections)
    """

    JUNCTION_TYPE_MAP = {
        "traffic_light": 0,
        "priority": 1,
        "right_before_left": 2,
        "unregulated": 3,
        "dead_end": 4,
        "unknown": 5,
        "internal": 6,
    }

    def build(self, net_path: Path, tls_states: dict) -> dict:
        """tls phases to be evaluated as dynamic features of the graph"""
        net = sumolib.net.readNet(net_path)
        edge_id_to_idx = {}
        edge_static_features = []

        # BUILD NODES (SUMO edges)
        idx = 0
        for edge in net.getEdges():
            if edge.getID().startswith(":"):
                continue
            edge_id_to_idx[edge.getID()] = idx
            idx += 1

            lanes = edge.getLanes()
            length = self.calculate_edge_length(edge.getShape())
            speed_limit = edge.getSpeed()
            num_lanes = len(lanes)
            priority = edge.getPriority()

            edge_static_features.append(
                [length, speed_limit, float(num_lanes), float(priority)]
            )
        x_static = torch.tensor(edge_static_features, dtype=torch.float)

        # BUILD EDGES (SUMO junctions)
        source_nodes: List[int] = []
        dest_nodes: List[int] = [] 
        attribute_list: List[int] = []
        seen_conn = set()

        for node in net.getNodes():
            incoming = [e for e in node.getIncoming() if not e.getID().startswith(":")]
            outgoing = [e for e in node.getOutgoing() if not e.getID().startswith(":")]

            # print(incoming)
            jid = node.getID()
            jtype = self.JUNCTION_TYPE_MAP.get(node.getType(), 5)
            lon, lat = self._transform_coord(
                node.getCoord()[0], node.getCoord()[1]
            )
            has_tls = float(node.getType() == "traffic_light")
            # tl_link_idx = ...
            if has_tls == 1:
                tls = tls_states.get(jid, {})

            for e_in in incoming:
                for e_out in outgoing:
                    source_id = edge_id_to_idx.get(e_in.getID())
                    dest_id = edge_id_to_idx.get(e_out.getID())
                    if source_id is None or dest_id is None or source_id == dest_id:
                        continue
                    connection_pair = (source_id, dest_id)
                    if connection_pair in seen_conn:
                        continue
                    else:
                        seen_conn.add(connection_pair) # avoid duplicates
                        source_nodes.append(source_id)
                        dest_nodes.append(dest_id)
                        attribute_list.append([float(jtype), lon, lat, has_tls])

        # source_nodes = where does the connection start
        # dest_nodes = where does the connection sink
        # connections_idx -> to map connections all over the junction
        connections_idx = torch.tensor([source_nodes, dest_nodes], dtype=torch.long)
        jun_attrib = torch.tensor(attribute_list, dtype=torch.float)
        

        return {
            "edge_id_to_idx": edge_id_to_idx,  # dict -> key = edge_id, value = idx in the graph
            "x_static": x_static,  # shape = torch.tensor([N_nodes, N_features]) = [2350, 4] -> length, speed_limit, num_lanes, priority
            "edge_index": connections_idx,  # shape -> torch.Size([2, 4292]) = [2, N_edges]
            "edge_attr": jun_attrib,  # shape -> torch.Size([5292, 4]) = [N_edges, N_features] -> jtype, lon, lat, has_tls 
        }



class DetectorParser:
    """Dynamic features definition from detectors output"""
    FEATURES_MAP = {"flow": 0,
                    "speed": 1,
                    "occupancy": 2
                    }
    N_FEATURES = len(FEATURES_MAP)
    
    def __init__(self, det_dir: Path, edge_id_to_idx: dict):
        self.det_dir = det_dir
        self.edge_id_to_idx = edge_id_to_idx
        self.n_nodes = len(edge_id_to_idx)
        
        
    def _parse_one_det(self, eid: str) -> Dict:
        path = os.path.join(self.det_dir, f"det_{eid}.xml")
        if not os.path.isfile(path):
            return {}
        
        accumulator = defaultdict(lambda: defaultdict(list))
        """
        accumulates values from each lane to aggregate records in a second step
        {
        "begin": {
            "flow": ["val1", "val2", ...],
            "speed": ["val1", "val2", ...],
            "occupancy": ["val1", "val2", ...]
        },
        ...
        }
        """
        tree = ET.parse(path)
        root = tree.getroot()
        
        for interval in root.findall("interval"):
            begin = float(interval.get("begin"))
            fl = float(interval.get("flow"))
            speed = float(interval.get("speed", -1))
            occ = float(interval.get("occupancy"))
            
            # accumulator takes a dict with all detectors features for each timestep
            accumulator[begin]["flow"].append(float(fl)) if fl is not None else 0.0
            if speed > 0:
                accumulator[begin]["speed"].append(float(speed))
            accumulator[begin]["occupancy"].append(float(occ)) if occ is not None else 0.0
            
        # aggregate by edge 
        records: Dict[float, Dict[str, float]] = {} # record = Dict['begin', features]
        for begin, lane_data in accumulator.items():
            # each accum dict entry is an edge --> aggregate duplicated value
            feats: Dict[str, float] = {}
            feats['flow'] = sum(lane_data['flow']) # total flow over the street
            feats['speed'] = sum(lane_data['speed']) / len(lane_data['speed']) if lane_data['speed'] else 0.0
            feats['occupancy'] = max(lane_data['occupancy']) # worst case
            records[begin] = feats
            
        return records

    def parse(self) -> torch.Tensor:
        raw_data = {}
        for eid in self.edge_id_to_idx:
            records = self._parse_one_det(eid)
            if records:
                raw_data[eid] = records
        
        if not raw_data:
            raise ValueError(f"No data found: check {self.det_dir}")
        
        all_begins = sorted(set(
            float(begin) for records in raw_data.values() for begin in records.keys()
        ), key=float)
        # build tensor [Time, Det_edge, Features]
        duration = len(all_begins)
        begins_to_idx = {b: i for i, b in enumerate(all_begins)}
        x_dynamic = torch.full(size=[duration, self.n_nodes, self.N_FEATURES], fill_value=0.0, dtype=torch.float32)

        # fill matrix with data
        for det, rec in raw_data.items():
            node_idx = self.edge_id_to_idx[det]
            for b, feats in rec.items():
                t = begins_to_idx[b]
                for fname, fidx in self.FEATURES_MAP.items():
                    if fname in feats:
                        x_dynamic[t, node_idx, fidx] = feats[fname]
                        
        return x_dynamic
        


class Snapshot:
    def __init__(self, 
        x_static: torch.Tensor, # [N_nodes, F_static]
        x_dynamic: torch.Tensor,   # [T, N_nodes, F_dynamic]
        t: int, # timeline index
        edge_index: torch.Tensor, # [2, N_edges]
        edge_attr: torch.Tensor, # [N_edges, F_edge]
        ):
        self.x_static = x_static
        self.x_dynamic = x_dynamic
        self.t = t
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        
    def to_data(self) -> Tuple[Data, torch.Tensor]:
        x = torch.concat(tensors=[self.x_static, self.x_dynamic[self.t]], dim=1)
        # target
        y = self.x_dynamic[self.t+1]
        return x, y



class SUMODataset(Dataset):
    """PyTorch Geometric Dataset over SUMO traffic pre-processed snapshots."""
    def __init__(self, root: str, transform=None, pre_transform=None, pre_filter=None):
        self._available_snapshots_idx: List[int] = self._scan_data_idx(os.path.join(root, "processed"))
        super().__init__(root, transform, pre_transform, pre_filter)
    
    # PyTorch Geometrics protocol for Dataset creation
    @property
    def processed_file_names(self) -> List[str]:
        return [f'data_{i}.pt' for i in self._available_snapshots_idx]

    def download(self):
        pass # no automatic download
        
    def len(self) -> int:
        return len(self._available_snapshots_idx)

    def get(self, idx: int) -> Data:
        i = self._available_snapshots_idx[idx]
        return torch.load(
            os.path.join(self.processed_dir, f'data_{i}.pt'), weights_only=False
        )
    
    @staticmethod
    def _scan_data_idx(processed_dir: str) -> List[int]:
        if not os.path.isdir(processed_dir):
            return []
        idxs = []
        for filename in os.listdir(processed_dir):
            if filename.startswith("data_") and filename.endswith(".pt"):
                try:
                    idxs.append(int(filename[5:-3])) # index value
                except ValueError:
                    pass
        return sorted(idxs)

    @staticmethod
    def build_and_save(
        root: str,
        x_static: torch.Tensor,
        x_dynamic: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> None:
        out_dir = os.path.join(root, "processed")
        os.makedirs(out_dir, exist_ok=True)
        
        T = x_dynamic.shape[0]
        for t in range(T-1): # no target for last step
            snap = Snapshot(x_static, x_dynamic, t, edge_index, edge_attr)
            data, target = snap.to_data()
            data.y = target

            torch.save(obj=data, f=os.path.join(out_dir, f"data_{t}.pt"))
        print(f"Saved {T-1} SUMO snapshots in {out_dir}")


if __name__ == "__main__":
    net_path = "/home/marta/tesi-5t/sumo-stgcn/data/raw/francia_peschiera_passenger.net.xml"
    det_dir = "/home/marta/tesi-5t/sumo-stgcn/data/raw/DetOut_Morning"
    graph = GraphBuilder().build(net, tls_states)
    parser = DetectorParser(det_dir, graph["edge_id_to_idx"])
    x_dyn, timesteps = parser.parse()

    SUMODataset.build_and_save(
        root = "data/built_dataset",
        x_static = graph["x_static"],
        x_dynamic = x_dyn,
        edge_index = graph["edge_index"],
        edge_attr = graph["edge_attr"],
    )

    dataset = SUMODataset(root="data/built_dataset")