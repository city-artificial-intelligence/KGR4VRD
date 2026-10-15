"""The two predeclared supervision controls, independent of training RNGs."""

from collections import Counter
from typing import Any

import numpy as np

from .types import Relation


def frequency_weights(raw_counts: Any, added_counts: Any) -> tuple[np.ndarray, dict]:
    """Return positive-only weights [P] and disclose unmatched added mass."""
    raw, added = np.asarray(raw_counts), np.asarray(added_counts)
    if raw.shape != added.shape or raw.ndim != 1:
        raise ValueError("Raw and added counts must be equally sized predicate vectors")
    if not np.isfinite(raw).all() or not np.isfinite(added).all():
        raise ValueError("Predicate counts must be finite")
    if ((raw < 0) | (added < 0) | (raw != raw.astype(int)) | (added != added.astype(int))).any():
        raise ValueError("Predicate counts must be nonnegative integers")
    weights = np.ones(len(raw), dtype=np.float64)
    np.divide(raw + added, raw, out=weights, where=raw > 0)
    unmatched = int(added[raw == 0].sum())
    total = int(added.sum())
    return weights, {
        "policy": "positive_frequency_v1",
        "raw_counts": raw.astype(int).tolist(),
        "added_counts": added.astype(int).tolist(),
        "weights": weights.tolist(),
        "unmatched_count": unmatched,
        "unmatched_fraction": unmatched / total if total else 0.0,
        "requires_shuffle": bool(total and unmatched / total > 0.05),
        "degenerate": total == 0,
    }


def shuffle_added_labels(
    raw: dict[str, tuple[Relation, ...]],
    augmented: dict[str, tuple[Relation, ...]],
    *,
    seed: int = 2027,
) -> tuple[dict[str, tuple[Relation, ...]], dict]:
    """Bounded predicate swaps preserve added counts per pair and per predicate."""
    if set(raw) != set(augmented):
        raise ValueError("Raw and augmented image IDs differ")
    raw_cells = {
        (i, r.subject_id, r.object_id, r.predicate_id) for i, rs in raw.items() for r in rs
    }
    full_cells = {
        (i, r.subject_id, r.object_id, r.predicate_id) for i, rs in augmented.items() for r in rs
    }
    if not raw_cells <= full_cells:
        raise ValueError("Augmentation removed raw labels")
    original = full_cells - raw_cells
    cells = sorted(original)
    present = set(cells)
    rng = np.random.default_rng(seed)
    attempts = 100 * max(1, len(cells))
    accepted = 0
    for _ in range(attempts):
        if len(cells) < 2:
            continue
        a, b = map(int, rng.integers(len(cells), size=2))
        left, right = cells[a], cells[b]
        new_left, new_right = left[:3] + (right[3],), right[:3] + (left[3],)
        if left[3] == right[3] or left[:3] == right[:3]:
            continue
        if any(c in present or c in raw_cells for c in (new_left, new_right)):
            continue
        present.difference_update((left, right))
        present.update((new_left, new_right))
        cells[a], cells[b] = new_left, new_right
        accepted += 1
    result: dict[str, list[Relation]] = {i: [] for i in sorted(raw)}
    for image, subject, obj, predicate in sorted(raw_cells | present):
        result[image].append(Relation(subject, predicate, obj))
    coverage = len(present - original) / len(original) if original else 0.0
    return {i: tuple(rs) for i, rs in result.items()}, {
        "policy": "added_label_shuffle_v1",
        "seed": seed,
        "attempts": attempts,
        "accepted_swaps": accepted,
        "added_count": len(original),
        "changed_cell_fraction": coverage,
        "adequate": bool(original and coverage >= 0.8),
        "degenerate": not original,
        "predicate_counts": dict(sorted(Counter(c[3] for c in cells).items())),
        "pair_counts": [[*pair, n] for pair, n in sorted(Counter(c[:3] for c in cells).items())],
    }
