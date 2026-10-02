import torch

from conftest import FEAT_DYN, N_NODES, WINDOW
from src.flowMatching import cfm_loss, predict
from src.utils import collate_batch


def _batch(dataset, b=2):
    return collate_batch([dataset[i] for i in range(b)], window=WINDOW)


def test_conditioner_output_shape(dataset, model):
    graphs, _ = _batch(dataset, b=2)
    cond = model.conditioner(graphs)
    assert cond.shape == (2, N_NODES, 16)


def test_cfm_loss_is_finite_and_backpropagates_to_all_blocks(dataset, model):
    graphs, target = _batch(dataset)
    loss = cfm_loss(model.velocity_field, target, model.conditioner(graphs))
    assert loss.dim() == 0 and torch.isfinite(loss)
    loss.backward()
    for block in ("conditioner.spatial_encoder.gcn", "conditioner.spatial_encoder.gat",
                "conditioner.temporal_encoder", "velocity_field"):
        grads = [p.grad for n, p in model.named_parameters() if n.startswith(block)]
        assert any(g is not None and g.abs().sum() > 0 for g in grads), f"no gradient in {block}"


def test_predict_shape_and_sample_diversity(dataset, model):
    graphs, _ = _batch(dataset, b=1)
    preds = predict(model, graphs, feat_dyn_dim=FEAT_DYN, n_integration_steps=5, n_samples=4)
    assert preds.shape == (4, 1, N_NODES, FEAT_DYN)  # [n_samples, B, N_nodes, F]
    assert torch.isfinite(preds).all()
    assert not torch.allclose(preds[0], preds[1])  # different noise -> different samples


def test_model_can_overfit_one_batch(dataset, model):
    """Sanity check: if the loss cannot go down on one fixed batch, the pipeline is broken."""
    graphs, target = _batch(dataset, b=2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)

    def avg_loss(n=20):
        with torch.no_grad():
            return sum(cfm_loss(model.velocity_field, target, model.conditioner(graphs)).item() for _ in range(n)) / n

    before = avg_loss()
    model.train()
    for _ in range(150):
        loss = cfm_loss(model.velocity_field, target, model.conditioner(graphs))
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    assert avg_loss() < 0.7 * before
