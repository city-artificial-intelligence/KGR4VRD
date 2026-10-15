"""Pure exact-instance ranking and metrics for the registered multi-label view."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def rank_predictions(
    pairs: Sequence[tuple[str, str]],
    logits: np.ndarray,
    predicate_count: int,
    *,
    decisions: np.ndarray | None = None,
    filter_mode: str = "none",
    simple_mask: np.ndarray | None = None,
    threshold: float | None = None,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Rank raw logits, filter, then cap; auxiliary column P is never emitted.

    Decisions are [pairs, predicates], with -1 unknown, 0 compatible and
    1 incompatible. Missing decisions mean explicitly unknown, never compatible.
    Exact ties use subject, object, predicate IDs; repeated triples count once.
    """
    logits = np.asarray(logits)
    if predicate_count < 1 or logits.shape != (len(pairs), predicate_count + 1):
        raise ValueError("logits must have shape [pairs, predicate_count + 1]")
    if not np.isfinite(logits).all():
        raise ValueError("non-finite prediction logits")
    shape = (len(pairs), predicate_count)
    states = np.full(shape, -1, dtype=np.int8) if decisions is None else np.asarray(decisions)
    if states.shape != shape or not np.isin(states, [-1, 0, 1]).all():
        raise ValueError("decisions must be resolved -1/0/1 cells with shape [pairs, predicates]")
    if filter_mode not in {"none", "ontology", "simple_asserted"}:
        raise ValueError(f"unsupported filter mode: {filter_mode}")
    if filter_mode == "ontology" and decisions is None:
        raise ValueError("ontology filtering requires the full checker decisions")
    if filter_mode == "simple_asserted":
        if simple_mask is None or np.asarray(simple_mask).shape != shape:
            raise ValueError("simple_asserted filtering requires a [pairs, predicates] mask")
        if np.asarray(simple_mask).dtype != np.bool_:
            raise ValueError("simple_asserted mask must be Boolean")
    if threshold is not None and not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    if top_k is not None and (not isinstance(top_k, int) or top_k < 0):
        raise ValueError("top_k must be a non-negative integer or None")
    cutoff = -math.inf
    if threshold is not None and threshold > 0:
        cutoff = math.inf if threshold == 1 else math.log(threshold / (1 - threshold))
    candidates: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row, (subject_id, object_id) in enumerate(pairs):
        if subject_id == object_id:
            raise ValueError("candidate pairs cannot contain self relations")
        for predicate_id in range(predicate_count):
            value = float(logits[row, predicate_id])
            decision = int(states[row, predicate_id])
            if value <= cutoff or (filter_mode == "ontology" and decision == 1):
                continue
            if (
                filter_mode == "simple_asserted"
                and simple_mask is not None
                and simple_mask[row, predicate_id]
            ):
                continue
            triple = (subject_id, predicate_id, object_id)
            score = (
                1 / (1 + math.exp(-value))
                if value >= 0
                else math.exp(value) / (1 + math.exp(value))
            )
            candidate = {
                "subject_id": subject_id,
                "object_id": object_id,
                "predicate_id": predicate_id,
                "logit": value,
                "score": score,
                "decision": decision,
            }
            if triple not in candidates or value > candidates[triple]["logit"]:
                candidates[triple] = candidate
    ranked = sorted(
        candidates.values(),
        key=lambda r: (-r["logit"], r["subject_id"], r["object_id"], r["predicate_id"]),
    )
    return [dict(row, rank=rank) for rank, row in enumerate(ranked[:top_k], 1)]


def _triple(relation: Any) -> tuple[str, int, str]:
    if isinstance(relation, Mapping):
        return relation["subject_id"], relation["predicate_id"], relation["object_id"]
    return relation.subject_id, relation.predicate_id, relation.object_id


def _counts(predictions: Sequence[dict[str, Any]]) -> dict[str, int]:
    return {
        "prediction_count": len(predictions),
        **{
            name: sum(row["decision"] == code for row in predictions)
            for name, code in (
                ("compatible_count", 0),
                ("incompatible_count", 1),
                ("unknown_count", -1),
            )
        },
    }


def _aggregate(
    per_image: list[dict[str, Any]], per_predicate: list[dict[str, Any]], top_ks: Sequence[int]
) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics: dict[str, Any] = {}
    details: dict[str, Any] = {}

    def ratio(name: str, numerator: float, denominator: int, reason: str) -> None:
        metrics[name] = numerator / denominator if denominator else None
        details[name] = {
            "numerator": numerator,
            "denominator": denominator,
            "null_reason": None if denominator else reason,
        }

    eligible = [row for row in per_image if row["ground_truth_count"]]
    supported = [row for row in per_predicate if row["ground_truth_count"]]
    support = sum(row["ground_truth_count"] for row in eligible)
    for k in top_ks:
        hit_key = f"hits_at_{k}"
        ratio(
            f"recall_image_mean_at_{k}",
            sum(r[hit_key] / r["ground_truth_count"] for r in eligible),
            len(eligible),
            "no annotated images",
        )
        ratio(f"recall_global_at_{k}", sum(r[hit_key] for r in eligible), support, "no annotations")
        ratio(
            f"recall_predicate_mean_at_{k}",
            sum(r[hit_key] / r["ground_truth_count"] for r in supported),
            len(supported),
            "no supported predicates",
        )
        ratio(
            f"annotation_ap_image_mean_at_{k}",
            sum(r[f"ap_sum_at_{k}"] / r["ground_truth_count"] for r in eligible),
            len(eligible),
            "no annotated images",
        )
    for key in (
        "prediction_count",
        "compatible_count",
        "incompatible_count",
        "unknown_count",
        "eligible_pair_count",
        "removed_candidate_count",
        "empty_slot_count",
    ):
        metrics[key] = sum(r[key] for r in per_image)
    metrics["image_count"] = len(per_image)
    metrics["empty_prediction_images"] = sum(r["prediction_count"] == 0 for r in per_image)
    metrics["excluded_unannotated_images"] = len(per_image) - len(eligible)
    metrics["omitted_zero_support_predicates"] = [
        r["predicate_id"] for r in per_predicate if not r["ground_truth_count"]
    ]
    emitted = metrics["prediction_count"]
    decided = metrics["compatible_count"] + metrics["incompatible_count"]
    ratio("incompatible_rate_all", metrics["incompatible_count"], emitted, "no emitted predictions")
    ratio(
        "incompatible_rate_decided",
        metrics["incompatible_count"],
        decided,
        "no decided predictions",
    )
    ratio("reasoning_coverage", decided, emitted, "no emitted predictions")
    ratio("predictions_per_image", emitted, len(per_image), "no images")
    return metrics, details


def evaluate(
    expected_image_ids: Sequence[str],
    targets: Mapping[str, Sequence[Any]],
    scores: Mapping[str, np.ndarray],
    scenes: Mapping[str, Any],
    predicate_count: int,
    *,
    decisions: Mapping[str, np.ndarray] | None = None,
    filter_mode: str = "none",
    simple_masks: Mapping[str, np.ndarray] | None = None,
    threshold: float | None = None,
    top_ks: Sequence[int] = (25, 50, 100),
) -> dict[str, Any]:
    """Evaluate exactly the expected image set against raw target relations.

    Counts refer to the largest primary K. The historical strict >0.5 view has
    separate capped-999 and uncapped counts. Empty records are valid; absent or
    extra records fail, preventing denominator dilution across partitions.
    """
    from .data import build_pairs

    expected = set(expected_image_ids)
    if len(expected) != len(expected_image_ids):
        raise ValueError("duplicate expected image IDs")
    for name, records in (
        ("targets", targets),
        ("scores", scores),
        ("scenes", scenes),
        ("decisions", decisions),
        ("simple_masks", simple_masks),
    ):
        if records is not None and set(records) != expected:
            raise ValueError(
                f"{name} image IDs differ: missing={sorted(expected - set(records))}, extra={sorted(set(records) - expected)}"
            )
    ks = tuple(sorted(set(top_ks)))
    if not ks or any(not isinstance(k, int) or k <= 0 for k in ks):
        raise ValueError("top_ks must contain positive integers")
    per_predicate = [
        {"predicate_id": p, "ground_truth_count": 0, **{f"hits_at_{k}": 0 for k in ks}}
        for p in range(predicate_count)
    ]
    per_image: list[dict[str, Any]] = []
    ranked_rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {
        "threshold": 0.5,
        "comparison": "strictly_greater",
        "top_k": 999,
        "capped": {key: 0 for key in _counts([])},
        "uncapped": {key: 0 for key in _counts([])},
    }
    for image_id in expected_image_ids:
        pairs = build_pairs(scenes[image_id])
        pair_set = set(pairs)
        truth = {_triple(r) for r in targets[image_id]}
        if any((s, o) not in pair_set or not 0 <= p < predicate_count for s, p, o in truth):
            raise ValueError(f"invalid target triple in image {image_id}")
        for _, p, _ in truth:
            per_predicate[p]["ground_truth_count"] += 1
        options: dict[str, Any] = dict(
            decisions=None if decisions is None else decisions[image_id],
            filter_mode=filter_mode,
            simple_mask=None if simple_masks is None else simple_masks[image_id],
        )
        ranked = rank_predictions(
            pairs, scores[image_id], predicate_count, threshold=threshold, **options
        )
        unfiltered = rank_predictions(
            pairs,
            scores[image_id],
            predicate_count,
            threshold=threshold,
            decisions=options["decisions"],
        )
        emitted = ranked[: max(ks)]
        row = {
            "image_id": image_id,
            "ground_truth_count": len(truth),
            "eligible_pair_count": len(pairs),
            **_counts(emitted),
            "removed_candidate_count": len(unfiltered) - len(ranked),
            "empty_slot_count": max(ks) - len(emitted),
        }
        for k in ks:
            hits, ap_sum = 0, 0.0
            for rank, candidate in enumerate(ranked[:k], 1):
                if _triple(candidate) in truth:
                    hits += 1
                    ap_sum += hits / rank
                    per_predicate[candidate["predicate_id"]][f"hits_at_{k}"] += 1
            row[f"hits_at_{k}"], row[f"ap_sum_at_{k}"] = hits, ap_sum
            for name, count in _counts(ranked[:k]).items():
                row[f"{name}_at_{k}"] = count
        historical = rank_predictions(
            pairs, scores[image_id], predicate_count, threshold=0.5, **options
        )
        for label, selected in (("capped", historical[:999]), ("uncapped", historical)):
            for name, count in _counts(selected).items():
                diagnostics[label][name] += count
        per_image.append(row)
        ranked_rows.extend(dict(r, image_id=image_id) for r in emitted)
    metrics, details = _aggregate(per_image, per_predicate, ks)
    return {
        "metrics": metrics,
        "metric_details": details,
        "per_image": per_image,
        "per_predicate": per_predicate,
        "ranked_predictions": ranked_rows,
        "diagnostics": diagnostics,
        "policy": {
            "ranking": "multi_label",
            "matching": "exact_instance",
            "top_ks": list(ks),
            "threshold": threshold,
            "filter_mode": filter_mode,
            "count_top_k": max(ks),
            "policy_id": "multi_label_unthresholded_v1"
            if threshold is None
            else f"multi_label_strict_threshold_{threshold}_v1",
        },
    }
