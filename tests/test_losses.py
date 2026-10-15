import numpy as np
import pytest
import torch

from kgr4vrd.losses import compute_loss, update_active_mask
from kgr4vrd.training import select_checkpoint


def test_cached_classification_is_applied_per_image_and_persists():
    incompatible = torch.tensor([[True, True, False]])
    previous = torch.zeros_like(incompatible)
    logits = torch.tensor([[1.0, 0.0, 100.0]])
    assert not update_active_mask(logits, incompatible, previous, 1).any()
    assert not update_active_mask(logits, incompatible, previous, 2).any()
    for _image in ("first", "later"):
        active = update_active_mask(logits, incompatible, previous, 3)
        assert active.tolist() == [[True, False, False]]
        assert update_active_mask(-torch.ones_like(logits), incompatible, active, 4).equal(active)
    assert not previous.any()


def test_extreme_loss_semantic_gradient_and_target_immutability():
    logits = torch.tensor([[1000.0, -1000.0, 1000.0]], requires_grad=True)
    targets = torch.tensor([[0.0, 0.0, 1.0]])
    original = targets.clone()
    rows, active = torch.tensor([False]), torch.tensor([[True, False]])
    loss, ordinary, semantic = compute_loss(logits, targets, rows, active, semantic_lambda=1)
    loss.backward()
    assert ordinary == 0 and semantic == 1000
    assert torch.isfinite(logits.grad).all() and logits.grad[0, 0] > 0
    assert (logits - 0.1 * logits.grad)[0, 0] < logits[0, 0]
    assert targets.equal(original)
    assert compute_loss(logits, targets, rows, active, semantic_lambda=0) is None


def test_positive_weights_only_change_positive_predicate_terms():
    logits = torch.zeros((2, 3), requires_grad=True)
    targets = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rows = torch.tensor([True, False])
    active = torch.zeros((2, 2), dtype=torch.bool)
    loss, _, _ = compute_loss(
        logits, targets, rows, active, positive_weights=torch.tensor([3.0, 9.0])
    )
    assert loss.item() == pytest.approx(5 * np.log(2))
    loss.backward()
    assert logits.grad.tolist() == [[-1.5, 0.5, 0.5], [0.0, 0.0, 0.0]]


def test_earliest_unrounded_validation_selection_ignores_test():
    history = [
        {"epoch": 1, "selection_score": 0.5, "test": 0},
        {"epoch": 2, "selection_score": 0.5000000001, "test": 0.1},
        {"epoch": 3, "selection_score": 0.5000000001, "test": 1},
    ]
    assert select_checkpoint(history)["epoch"] == 2
    with pytest.raises(ValueError, match="non-finite"):
        select_checkpoint([{"selection_score": float("nan")}])
