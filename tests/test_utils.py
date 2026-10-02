import json

import torch

from conftest import FEAT_DYN, N_NODES, WINDOW
from src.utils import collate_batch, dataset_split, run_training


# dataset_split

def _split_indices(dataset):
    train, val, test = dataset_split(dataset, window=WINDOW, train_fraction=0.6, val_fraction=0.2, test_fraction=0.2)
    return set(train.indices), set(val.indices), set(test.indices)


def test_split_is_disjoint_and_in_range(dataset):
    tr, va, te = _split_indices(dataset)
    assert not (tr & va) and not (tr & te) and not (va & te)
    assert all(0 <= i < len(dataset) for i in tr | va | te)


def test_split_indices_stay_inside_their_scenario(dataset):
    ranges = dataset.scenario_index_ranges()
    tr, va, te = _split_indices(dataset)
    for i in tr | va | te:
        assert any(s <= i < e for s, e in ranges.values())


def test_every_scenario_appears_in_every_split(dataset):
    """Regression: n_total must be the scenario length (end - start), not len(dataset)."""
    tr, va, te = _split_indices(dataset)
    for sid, (s, e) in dataset.scenario_index_ranges().items():
        scen = set(range(s, e))
        assert scen & tr, f"{sid} missing from train"
        assert scen & va, f"{sid} missing from val"
        assert scen & te, f"{sid} missing from test"


def test_no_temporal_leakage_between_splits(dataset):
    """Inside a scenario, train windows must end before val starts, val before test."""
    tr, va, te = _split_indices(dataset)
    for s, e in dataset.scenario_index_ranges().values():
        t_ = sorted(i for i in tr if s <= i < e)
        v_ = sorted(i for i in va if s <= i < e)
        x_ = sorted(i for i in te if s <= i < e)
        # a window starting at i covers snapshots i .. i+WINDOW-1 (target at i+WINDOW)
        if t_ and v_:
            assert t_[-1] + WINDOW <= v_[0]
        if v_ and x_:
            assert v_[-1] + WINDOW <= x_[0]


# collate_batch

def test_collate_batch_shapes(dataset):
    samples = [dataset[i] for i in range(4)]
    graphs, targets = collate_batch(samples, window=WINDOW)
    assert len(graphs) == WINDOW
    assert all(g.num_graphs == 4 for g in graphs)
    assert graphs[0].x.shape[0] == 4 * N_NODES
    assert targets.shape == (4, N_NODES, FEAT_DYN)


# run_training

def _loaders(dataset):
    from functools import partial

    from torch.utils.data import DataLoader, Subset

    collate = partial(collate_batch, window=WINDOW)
    tr = DataLoader(Subset(dataset, range(8)), batch_size=4, collate_fn=collate)
    va = DataLoader(Subset(dataset, range(8, 12)), batch_size=4, collate_fn=collate)
    return tr, va


def test_run_training_saves_checkpoint_and_calls_tracker(dataset, model, tmp_path):
    tr, va = _loaders(dataset)
    calls = []
    ckpt = tmp_path / "ckpt" / "best.pt"
    history = run_training(
        model=model,
        train_loader=tr,
        val_loader=va,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        num_epochs=2,
        checkpoint_path=str(ckpt),
        print_every=1,
        params_tracker=lambda m, e: calls.append(e),
    )
    assert calls == [0, 1]
    assert len(history["train_loss"]) == len(history["val_loss"]) == 2
    assert all(torch.isfinite(torch.tensor(history["train_loss"])))
    assert ckpt.is_file()
    model.load_state_dict(torch.load(ckpt))  # checkpoint matches the architecture


def test_run_training_writes_history_json_every_epoch(dataset, model, tmp_path):
    """history.json must be on disk so a crash mid-training does not lose the losses."""
    tr, va = _loaders(dataset)
    hist_path = tmp_path / "history.json"
    run_training(
        model=model,
        train_loader=tr,
        val_loader=va,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        num_epochs=2,
        checkpoint_path=str(tmp_path / "best.pt"),
        history_path=str(hist_path),
    )
    saved = json.loads(hist_path.read_text())
    assert len(saved["train_loss"]) == 2
