import pytest
import torch
import yaml

from conftest import EDGE_DIM, FEAT_DYN, MANIFEST, N_NODES, WINDOW, X_DIM
from data.dataset import DetectorParser, Snapshot, load_scenarios


# SUMODataset

def test_len_counts_valid_windows_per_scenario(dataset):
    expected = sum(n - WINDOW + 1 for n in MANIFEST.values())
    assert len(dataset) == expected


def test_scenario_ranges_are_contiguous_and_cover_dataset(dataset):
    ranges = list(dataset.scenario_index_ranges().values())
    assert ranges[0][0] == 0
    for (_, end), (start, _) in zip(ranges, ranges[1:]):
        assert end == start
    assert ranges[-1][1] == len(dataset)


def test_get_returns_window_and_target_of_last_graph(dataset):
    graphs, target = dataset[0]
    assert len(graphs) == WINDOW
    assert torch.equal(target, graphs[-1].y)
    assert target.shape == (N_NODES, FEAT_DYN)


def test_windows_never_cross_scenarios(dataset):
    for sid, (start, end) in dataset.scenario_index_ranges().items():
        for idx in range(start, end):
            graphs, _ = dataset[idx]
            assert {g.sid for g in graphs} == {sid}
            locs = [g.local_i for g in graphs]
            assert locs == list(range(locs[0], locs[0] + WINDOW))


# Snapshot

def test_snapshot_to_data_shapes_and_target_is_next_step():
    T, N, E = 4, N_NODES, 4
    x_dyn = torch.randn(T, N, FEAT_DYN)
    snap = Snapshot(
        x_static=torch.randn(N, X_DIM - FEAT_DYN),
        x_dynamic=x_dyn,
        t=1,
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]),
        edge_attr_static=torch.randn(E, EDGE_DIM - 1),
        edge_attr_dynamic=torch.randn(T, E),
    )
    d = snap.to_data()
    assert d.x.shape == (N, X_DIM)
    assert d.edge_attr.shape == (E, EDGE_DIM)
    assert torch.equal(d.y, x_dyn[2])


# DetectorParser

DET_XML = """<detector>
  <interval begin="0.00" end="300.00" id="d_lane0" flow="120" speed="10.0" occupancy="5.0"/>
  <interval begin="0.00" end="300.00" id="d_lane1" flow="60"  speed="20.0" occupancy="9.0"/>
  <interval begin="300.00" end="600.00" id="d_lane0" flow="30" speed="-1.00" occupancy="1.0"/>
</detector>"""


def test_detector_parser_aggregates_lanes(tmp_path):
    (tmp_path / "det_e1.xml").write_text(DET_XML)
    x = DetectorParser(tmp_path, {"e0": 0, "e1": 1}).parse()

    assert x.shape == (2, 2, 3)  # [T, N_nodes, (flow, speed, occupancy)]
    # t=0: flow summed, speed averaged, occupancy max
    assert x[0, 1].tolist() == pytest.approx([180.0, 15.0, 9.0])
    # t=1: speed=-1 (no vehicles) is ignored -> 0
    assert x[1, 1].tolist() == pytest.approx([30.0, 0.0, 1.0])
    # edge without a detector file stays at zero
    assert x[:, 0].abs().sum() == 0


def test_detector_parser_fails_loudly_on_wrong_det_dir(tmp_path):
    # det_dir without the inputs/ prefix -> no file found: it must raise, not return zeros
    with pytest.raises(ValueError, match="No data found"):
        DetectorParser(tmp_path / "does_not_exist", {"e0": 0}).parse()


# ---------- load_scenarios ----------

def test_load_scenarios_maps_disturbances_to_tuples(tmp_path):
    cfg = {
        "scenarios": [
            {"id": "baseline", "det_dir": "inputs/DetOut_Day/baseline"},
            {
                "id": "pos9_sev07",
                "det_dir": "inputs/DetOut_Day/pos9_sev07",
                "disturbances": [{"from": "J1", "to": "J2", "severity": 0.7}],
            },
        ]
    }
    path = tmp_path / "scenarios.yaml"
    path.write_text(yaml.safe_dump(cfg))
    sc = load_scenarios(str(path))
    assert sc[0]["disturbances"] == {}
    assert sc[1]["disturbances"] == {("J1", "J2"): 0.7}


def test_scenarios_yaml_det_dir_has_inputs_prefix():
    """Regression for the recurring det_dir bug: every det_dir must start with inputs/."""
    from pathlib import Path

    candidates = [p for p in Path("inputs").glob("*.yaml") if "scenarios" in (yaml.safe_load(p.read_text()) or {})]
    if not candidates:
        pytest.skip("no scenarios yaml in inputs/")
    for p in candidates:
        for sc in load_scenarios(str(p)):
            assert sc["det_dir"].startswith("inputs/"), f"{p.name}: {sc['id']} -> {sc['det_dir']}"
