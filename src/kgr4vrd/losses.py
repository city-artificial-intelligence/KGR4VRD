"""Stable image-sum BCE and persistent confidence-gated semantic regularisation."""

import torch
from torch import Tensor
from torch.nn import functional as F


def update_active_mask(
    logits: Tensor, incompatible: Tensor, previous: Tensor, epoch: int
) -> Tensor:
    """Return a detached Boolean [pairs,predicates] mask, never mutating inputs."""
    if epoch < 1 or incompatible.shape != logits.shape or previous.shape != logits.shape:
        raise ValueError("Invalid semantic activation shape or epoch")
    if incompatible.dtype != torch.bool or previous.dtype != torch.bool:
        raise ValueError("Activation masks must be Boolean")
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Non-finite activation logits")
    if epoch < 3:
        return torch.zeros_like(previous)
    return (previous | ((logits.detach() > 0) & incompatible)).detach()


def compute_loss(
    logits: Tensor,
    targets: Tensor,
    ordinary_rows: Tensor,
    active: Tensor,
    *,
    semantic_lambda: float = 0.0,
    positive_weights: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor] | None:
    """Compute one image's total/ordinary/semantic sums; None means no update.

    Logits/targets have P+1 columns; active and positive_weights exclude auxiliary P.
    Batch callers average these sums across images. Unannotated rows have no BCE.
    """
    if (
        targets.shape != logits.shape
        or ordinary_rows.shape != logits.shape[:1]
        or active.shape != (logits.shape[0], logits.shape[1] - 1)
    ):
        raise ValueError("Loss shapes disagree")
    if ordinary_rows.dtype != torch.bool or active.dtype != torch.bool:
        raise ValueError("Loss masks must be Boolean")
    if not torch.isfinite(logits).all() or not torch.isfinite(targets).all():
        raise FloatingPointError("Non-finite loss input")
    if semantic_lambda < 0:
        raise ValueError("Negative semantic lambda")
    if not ordinary_rows.any() and not (semantic_lambda and active.any()):
        return None
    weights = torch.ones(logits.shape[1], dtype=logits.dtype, device=logits.device)
    if positive_weights is not None:
        if (
            positive_weights.shape != (logits.shape[1] - 1,)
            or not torch.isfinite(positive_weights).all()
        ):
            raise ValueError("Invalid positive weights")
        if (positive_weights < 0).any():
            raise ValueError("Negative positive weights")
        weights[:-1] = positive_weights
    ordinary = F.binary_cross_entropy_with_logits(
        logits, targets, reduction="none", pos_weight=weights
    )[ordinary_rows].sum()
    semantic = F.softplus(logits[:, :-1])[active].sum() * semantic_lambda
    total = ordinary + semantic
    if not torch.isfinite(total):
        raise FloatingPointError("Non-finite loss")
    return total, ordinary, semantic
