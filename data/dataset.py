import torch
import os, sys, json, yaml, bisect
# bisect: to insert and divide lists mantaining their sorting order
import sumolib
from pathlib import Path

ROOT = Path.cwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

print(ROOT)

from torch_geometric.data import Data, Dataset
from typing import List, Dict, Optional, Tuple
from data.parser import XMLBuilder

from collections import defaultdict
import xml.etree.ElementTree as ET

import inputs.config as cfg
from src.models.normalizer import MapNormalizer

import importlib
importlib.reload(cfg) # refresh root dir

""" 
Module used to create the tensors to be elaborated by the GCN

As a reference, from torch-geometrics, one needs to compute:
    data.x: Node feature matrix with shape [num_nodes, num_node_features]
    data.edge_index: Graph connectivity in COO format with shape [2, num_edges] and type torch.long
    data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
    data.y: Target to train against (may have arbitrary shape), e.g., node-level targets of shape [num_nodes, *] or graph-level targets of shape [1, *]
    data.pos: Node position matrix with shape [num_nodes, num_dimensions]
"""


def load_scenarios(path: str) -> list[dict]:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    scenarios = []
    for sc in raw["scenarios"]:
        disturbances = {
            (d["from"], d["to"]): d["severity"] for d in sc.get("disturbances", [])
        }
        scenarios.append(
            {"id": sc["id"], "det_dir": sc["det_dir"], "disturbances": disturbances}
        )
    return scenarios


class GraphBuilder(XMLBuilder):
    """
    Function to build the tensors representing static features of the graph.
    Nodes = SUMO edges (streets)
    Edges = SUMO junctions (intersections)

    Returns:
            edge_id_to_idx: dict -> key = edge_id, value = idx in the graph
            x_static: node static features (length, speed_limit, num_lanes, priority), shape = torch.tensor([N_nodes, N_features])
            edge_index: adjiacency matrix which defines soruce and sink pairs for connections,  shape -> torch.Size([2, N_edges])
            edge_attr: edge features (jtype, lon, lat, has_tls, severity ),  # shape -> torch.Size([N_edges, N_features] )
            "conn_to_edge_idx": conn_to_edge_idx,
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

    def build(
        self,
        net_path: Path,
        tls_states: dict,
        disturbances: Dict[Tuple[str, str], float] = None,
    ) -> dict:
        """tls phases to be evaluated as dynamic features of the graph"""
        net = sumolib.net.readNet(net_path)
        """disturbances: map for WHAT-IF scenario management 
        (to admit non-binary disturbances. e.g. capacity reduction)
        map (from_edge_id, to_edge_id) -> severity in [0.0, 1.0]
        where 1.0 = normal connection, 0.0 = total closure.
        """
        disturbances = disturbances or {}
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
        conn_to_edge_idx: Dict[Tuple[str, str], int] = {}

        for node in net.getNodes():
            incoming = [e for e in node.getIncoming() if not e.getID().startswith(":")]
            outgoing = [e for e in node.getOutgoing() if not e.getID().startswith(":")]

            # print(incoming)
            jid = node.getID()
            jtype = self.JUNCTION_TYPE_MAP.get(node.getType(), 5)
            lon, lat = self._transform_coord(node.getCoord()[0], node.getCoord()[1])
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
                        seen_conn.add(connection_pair)
                        source_nodes.append(source_id)
                        dest_nodes.append(dest_id)
                        severity = disturbances.get((e_in.getID(), e_out.getID()), 1.0)
                        attribute_list.append(
                            [float(jtype), lon, lat, has_tls, severity]
                        )
                        conn_to_edge_idx[(e_in.getID(), e_out.getID())] = (
                            len(source_nodes) - 1
                        )

        # source_nodes = where does the connection start
        # dest_nodes = where does the connection sink
        # connections_idx -> to map connections all over the junction
        connections_idx = torch.tensor([source_nodes, dest_nodes], dtype=torch.long)
        jun_attrib = torch.tensor(attribute_list, dtype=torch.float)

        return {
            "edge_id_to_idx": edge_id_to_idx,  # dict -> key = edge_id, value = idx in the graph
            "x_static": x_static,  # shape = torch.tensor([N_nodes, N_features]) = [2350, 4] -> length, speed_limit, num_lanes, priority
            "edge_index": connections_idx,  # shape -> torch.Size([2, 4292]) = [2, N_edges]
            "edge_attr": jun_attrib,  # shape -> torch.Size([5292, 5]) = [N_edges, N_features] -> jtype, lon, lat, has_tls, severity
            "conn_to_edge_idx": conn_to_edge_idx,
        }


class DetectorParser:
    """
    Dynamic features extraction from detectors output.
    Returns:
            x_dynamic: tensor containing nodes dynamic features (shape: torch.Size([N_nodes, N_dyn_features]))
    """

    FEATURES_MAP = {"flow": 0, "speed": 1, "occupancy": 2}
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
            fl = float(interval.get("flow", 0.0))
            speed = float(interval.get("speed", -1))
            occ = float(interval.get("occupancy", 0.0))

            # accumulator takes a dict with all detectors features for each timestep
            accumulator[begin]["flow"].append(float(fl)) if fl is not None else 0.0
            if speed > 0:
                accumulator[begin]["speed"].append(float(speed))
            (
                accumulator[begin]["occupancy"].append(float(occ))
                if occ is not None
                else 0.0
            )

        # aggregate by edge
        records: Dict[float, Dict[str, float]] = {}  # record = Dict['begin', features]
        for begin, lane_data in accumulator.items():
            # each accum dict entry is an edge --> aggregate duplicated value
            feats: Dict[str, float] = {}
            feats["flow"] = sum(lane_data["flow"])  # total flow over the street
            feats["speed"] = (
                sum(lane_data["speed"]) / len(lane_data["speed"])
                if lane_data["speed"]
                else 0.0
            )
            feats["occupancy"] = max(lane_data["occupancy"])  # worst case
            records[begin] = feats

        return records

    def parse(self) -> torch.Tensor:
        """To parse detectors outputs from SUMO simulation."""
        raw_data = {}
        for eid in self.edge_id_to_idx:
            records = self._parse_one_det(eid)
            if records:
                raw_data[eid] = records

        if not raw_data:
            raise ValueError(f"No data found: check {self.det_dir}")

        all_begins = sorted(
            set(
                float(begin)
                for records in raw_data.values()
                for begin in records.keys()
            ),
            key=float,
        )
        # build tensor [Time, Det_edge, Features]
        duration = len(all_begins)
        begins_to_idx = {b: i for i, b in enumerate(all_begins)}
        x_dynamic = torch.full(
            size=[duration, self.n_nodes, self.N_FEATURES],
            fill_value=0.0,
            dtype=torch.float32,
        )

        # fill matrix with data
        for det, rec in raw_data.items():
            node_idx = self.edge_id_to_idx[det]
            for b, feats in rec.items():
                t = begins_to_idx[b]
                for fname, fidx in self.FEATURES_MAP.items():
                    if fname in feats:
                        x_dynamic[t, node_idx, fidx] = feats[fname]

        return x_dynamic


class TLSStateParser:
    _STATE_VALUES = {"G": 1.0, "g": 0.8, "y": 0.4, "Y": 0.4, "r": 0.0, "R": 0.0}

    def __init__(
        self,
        net_path: Path,
        conn_to_edge_idx: Dict[Tuple[str, str], int],
        n_edges: int,
        step_len: float,
        active_tls_ids: Optional[set] = None,
    ):
        self.net = sumolib.net.readNet(
            net_path, withPrograms=True, withLatestPrograms=True
        )
        self.conn_to_edge_idx = conn_to_edge_idx
        self.n_edges = n_edges
        self.step_len = step_len
        self.active_tls_ids = active_tls_ids
        self.tls_link_map = self._build_tls_link_map()

    @staticmethod
    def _state_at_time(
        phases: list[Tuple[float, str]], t: float, offset: float = 0.0
    ) -> str:
        # for static logics, otherwise parse --tls-state-output
        """Define in which state the connection is at time t."""
        cycle_len = sum(duration for duration, _ in phases)
        t_in_cycle = (t + offset) % cycle_len
        accumulator = 0.0
        for duration, state in phases:
            accumulator += duration
            if t_in_cycle < accumulator:
                return state
        return phases[-1][1]

    def _build_tls_link_map(self) -> Dict[str, Dict[int, Tuple[str, str]]]:
        """Creates a map that determines which connections are affected by each tls."""
        tls_link_map = {}
        for edge in self.net.getEdges():
            for lane in edge.getLanes():
                for conn in lane.getOutgoing():
                    tls_id = conn.getTLSID()
                    if not tls_id:
                        continue
                    link_idx = conn.getTLLinkIndex()
                    from_edge = conn.getFromLane().getEdge().getID()
                    to_edge = conn.getToLane().getEdge().getID()
                    tls_link_map.setdefault(tls_id, {})[link_idx] = (from_edge, to_edge)
        return tls_link_map

    def _parse_programs(self) -> Dict[str, Dict[str, object]]:
        """Extract logic programs from the tls file."""
        programs = {}
        for tls in self.net.getTrafficLights():
            program = list(tls.getPrograms().values())[
                0
            ]  # withLatestPrograms=True -> uno solo
            phases = [
                (float(phase.duration), phase.state) for phase in program.getPhases()
            ]
            programs[tls.getID()] = {
                "phases": phases,
                "offset": float(program.getOffset()),
            }
        return programs

    def parse(self, n_steps: int) -> torch.Tensor:
        """
        For each tls, for each timestep, it finds the tl phase and defines which edges and connections are involved.
        """
        programs = self._parse_programs()
        if self.active_tls_ids is not None:
            matched = programs.keys() & self.active_tls_ids
            unmatched = self.active_tls_ids - programs.keys()
            if unmatched:
                print(f"[TLSStateParser] {len(unmatched)} id in active_tls_ids non trovati nella mappa: {unmatched}")
            print(f"[TLSStateParser] {len(matched)}/{len(self.active_tls_ids)} TLS attivi trovati nella mappa.")
            programs = {tid: p for tid, p in programs.items() if tid in self.active_tls_ids}

        tls_dynamic = torch.zeros(n_steps, self.n_edges)
        for tls_id, program in programs.items():
            link_map = self.tls_link_map.get(tls_id, {})
            for t in range(n_steps):
                state = self._state_at_time(program["phases"], t * self.step_len, program["offset"])
                for link_idx, conn_pair in link_map.items():
                    edge_idx = self.conn_to_edge_idx.get(conn_pair)
                    if edge_idx is not None and link_idx < len(state):
                        tls_dynamic[t, edge_idx] = self._STATE_VALUES.get(state[link_idx], 0.0)
        return tls_dynamic


class Snapshot:
    def __init__(
        self,
        x_static: torch.Tensor,  # [N_nodes, F_static]
        x_dynamic: torch.Tensor,  # [T, N_nodes, F_dynamic]
        t: int,  # timeline index
        edge_index: torch.Tensor,  # [2, N_edges]
        edge_attr_static: torch.Tensor,  # [N_edges, F_static]  (jtype, lon, lat, has_tls, severity)
        edge_attr_dynamic: torch.Tensor,  # [T, N_edges] (TLS state)
    ):
        self.x_static = x_static
        self.x_dynamic = x_dynamic
        self.t = t
        self.edge_index = edge_index
        self.edge_attr_static = edge_attr_static
        self.edge_attr_dynamic = edge_attr_dynamic

    def to_data(self) -> Data:
        x = torch.concat(tensors=[self.x_static, self.x_dynamic[self.t]], dim=1)
        edge_attr = torch.cat(
            [self.edge_attr_static, self.edge_attr_dynamic[self.t].unsqueeze(-1)],
            dim=-1,
        )
        # target
        y = self.x_dynamic[self.t + 1]

        return Data(x=x, edge_index=self.edge_index, edge_attr=edge_attr, y=y)


class SUMODataset(Dataset):
    """PyTorch Geometric Dataset over SUMO traffic pre-processed snapshots."""

    def __init__(
        self,
        root: str,
        window: int,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.window = window
        manifest_path = os.path.join(root, "processed", "manifest.json")
        with open(manifest_path) as f:
            self._manifest = json.load(f)  # {scenario_id: n_snapshots}
        self.scenario_ids = list(self._manifest.keys())

        self._scenario_lengths = {
            sid: max(0, n - window + 1) for sid, n in self._manifest.items()
        }
        self._cum_lengths = []
        acc = 0
        for sid in self.scenario_ids:
            acc += self._scenario_lengths[sid]
            self._cum_lengths.append(acc)

        super().__init__(root, transform, pre_transform, pre_filter)
        
    # PyTorch Geometrics protocol for Dataset creation
    @property
    def processed_file_names(self) -> List[str]:
        names = []
        for sid, n in self._manifest.items():
            names += [f"data_{sid}_{i}.pt" for i in range(n)]
        return names

    @property
    def raw_file_names(self) -> List[str]:
        return []

    def download(self):
        pass  # no automatic download

    def len(self) -> int:
        return self._cum_lengths[-1] if self._cum_lengths else 0

    def get(self, idx: int):
        scenario_i = bisect.bisect_right(self._cum_lengths, idx)
        sid = self.scenario_ids[scenario_i]
        prev_cum = self._cum_lengths[scenario_i - 1] if scenario_i > 0 else 0
        local_idx = idx - prev_cum

        graphs = [
            torch.load(os.path.join(self.processed_dir, f"data_{sid}_{i}.pt"), weights_only=False)
            for i in range(local_idx, local_idx + self.window)
        ]
        target = graphs[-1].y  # x_dynamic al timestep window_idxs[-1] + 1
        # print("dentro get() — target.shape:", target.shape)
        return graphs, target  # list of Data, to process more timestamps

    def scenario_index_ranges(self) -> Dict[str, Tuple[int, int]]:
        """{scenario_id: (global_start, global_end_esclusive)} of valid window indeces."""
        ranges, prev = {}, 0
        for sid, cum in zip(self.scenario_ids, self._cum_lengths):
            ranges[sid] = (prev, cum)
            prev = cum
        return ranges

    @staticmethod
    def build_and_save(
        root: str,
        scenario_id: str,
        x_static: torch.Tensor,
        x_dynamic: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr_static: torch.Tensor,
        edge_attr_dynamic: torch.Tensor,
    ) -> None:

        out_dir = os.path.join(root, "processed")
        os.makedirs(out_dir, exist_ok=True)

        T = x_dynamic.shape[0]
        for t in range(T - 1):  # no target for last step
            snap = Snapshot(
                x_static, x_dynamic, t, edge_index, edge_attr_static, edge_attr_dynamic
            )
            data = snap.to_data()
            torch.save(obj=data, f=os.path.join(out_dir, f"data_{scenario_id}_{t}.pt"))
        
        manifest_path = os.path.join(out_dir, "manifest.json")
        manifest = {}
        if os.path.isfile(manifest_path):
            with open(manifest_path) as f:
                manifest = json.load(f)
        manifest[scenario_id] = T - 1
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"Saved {T-1} snapshots for scenario '{scenario_id}' in {out_dir}")


if __name__ == "__main__":
    cfg._ensure_folder_existence()

    net_path = os.path.join(cfg.INPUTS_DIR, "map_restricted.net.xml")
    scenarios = load_scenarios("inputs/scenarios.yaml")
    net = sumolib.net.readNet(net_path)
    active_tls_ids = cfg.TLS_ON_SET  # active TLS IDs (to edit in config.py)
    
    graph_topology = GraphBuilder().build(net_path, tls_states={}, disturbances=None)
    x_static = graph_topology["x_static"]
    
    tls_parser = TLSStateParser(
        net_path,
        graph_topology["conn_to_edge_idx"],
        n_edges=graph_topology["edge_attr"].shape[0],
        step_len=300.0,
        active_tls_ids=active_tls_ids,
    ) # step_len = 300s = 5 min (step code sensors for validation)
    
    
    raw_x_dyn: Dict[str, torch.Tensor] = {}
    raw_edge_attr: Dict[str, torch.Tensor] = {}
    for sc in scenarios:
        parser = DetectorParser(sc["det_dir"], graph_topology["edge_id_to_idx"])
        raw_x_dyn[sc["id"]] = parser.parse()
        raw_edge_attr[sc["id"]] = GraphBuilder().build(
            net_path, tls_states={}, disturbances=sc["disturbances"]
        )["edge_attr"]
    
    # HACK: assuming same duration for all scenarios
    n_steps = raw_x_dyn[scenarios[0]["id"]].shape[0]
    tls_dynamic = tls_parser.parse(n_steps=n_steps)
    
    """
    normalise over all the dataset, but estimate normalisation metrics (avg/std) only over training:
    see: https://datascience.stackexchange.com/questions/27615/should-we-apply-normalization-to-test-data-as-well
    """
    # normalization of static features -> same for every scenario
    static_normalizer = MapNormalizer(map_features=x_static)
    x_static_norm = static_normalizer.zscore(x_static)
    static_norm_path = Path(cfg.DATA_DIR, "built_dataset", "normalized", "static_norm_stats.pt")
    os.makedirs(os.path.dirname(static_norm_path), exist_ok=True)
    static_normalizer.save(static_norm_path)
    # normalization of dynamic features: over all the scenario (training only)
    train_slices = [
        raw_x_dyn[sc["id"]][: int(n_steps * cfg.TRAIN_FRACTION)] for sc in scenarios
    ]
    combined_train = torch.cat(train_slices, dim=0)
    dynamic_normalizer = MapNormalizer(map_features=combined_train)
    dynamic_norm_path = Path(cfg.DATA_DIR, "built_dataset", "normalized", "dynamic_norm_stats.pt")
    os.makedirs(os.path.dirname(dynamic_norm_path), exist_ok=True)
    dynamic_normalizer.save(dynamic_norm_path)
    
    # save Data after normalization (one save per scenario)
    for sc in scenarios:
        x_dyn_norm = dynamic_normalizer.zscore(raw_x_dyn[sc["id"]])
        SUMODataset.build_and_save(
            root=os.path.join(cfg.DATA_DIR, "built_dataset"),
            scenario_id=sc["id"],
            x_static=x_static_norm,
            x_dynamic=x_dyn_norm,
            edge_index=graph_topology["edge_index"],
            edge_attr_static=raw_edge_attr[sc["id"]],
            edge_attr_dynamic=tls_dynamic,
        )