import numpy as np
import pytest

from kgr4vrd.data import build_pairs
from kgr4vrd.metrics import evaluate, rank_predictions
from kgr4vrd.types import Object, Relation, Scene


def scene(image_id, objects=2):
    return Scene(
        image_id,
        "unused.jpg",
        20,
        20,
        tuple(Object(f"o{i}", 0, (0.0, 0.0, 1.0, 1.0)) for i in range(objects)),
    )


def test_raw_logit_ranking_ties_multiple_labels_auxiliary_filter_and_backfill():
    pairs = [("b", "a"), ("a", "b"), ("a", "b")]
    scores = np.array([[1000.0, 999.0, 9999.0], [1000.0, 0.0, 9999.0], [998.0, 0.0, 9999.0]])
    states = np.array([[1, -1], [0, 0], [0, 0]])
    ranked = rank_predictions(pairs, scores, 2, decisions=states, filter_mode="ontology", top_k=3)
    assert [(r["subject_id"], r["predicate_id"]) for r in ranked] == [("a", 0), ("b", 1), ("a", 1)]
    assert ranked[1]["decision"] == -1
    assert len(rank_predictions(pairs, scores, 2, threshold=0.5)) == 3
    assert rank_predictions(pairs, scores, 2, top_k=0) == []
    with pytest.raises(ValueError, match="non-finite"):
        rank_predictions(pairs, scores * np.nan, 2)


def test_distinct_recall_denominators_and_duplicate_annotation_deduplication():
    scenes = {"one": scene("one"), "nine": scene("nine", 4)}
    targets = {
        "one": [Relation("o0", 0, "o1")] * 2,
        "nine": [Relation(s, 0, o) for s, o in build_pairs(scenes["nine"])[:9]],
    }
    scores = {"one": np.array([[2.0, 0.0], [-2.0, 0.0]]), "nine": np.tile([-2.0, 0.0], (12, 1))}
    result = evaluate(["one", "nine"], targets, scores, scenes, 1, threshold=0.5, top_ks=(1,))
    assert result["metrics"]["recall_image_mean_at_1"] == 0.5
    assert result["metrics"]["recall_global_at_1"] == 0.1
    assert result["metrics"]["recall_predicate_mean_at_1"] == 0.1
    assert result["metrics"]["annotation_ap_image_mean_at_1"] == 0.5
    with pytest.raises(ValueError, match="targets image IDs differ"):
        evaluate(["one"], targets, {"one": scores["one"]}, {"one": scenes["one"]}, 1)
    with pytest.raises(ValueError, match="scores image IDs differ"):
        evaluate(["one", "nine"], targets, {"one": scores["one"]}, scenes, 1)


def test_predicate_mean_ap_and_exact_matching():
    s = scene("image", 3)
    # Predicate 0 has three annotations and one hit; predicate 1 has one and one.
    targets = {
        "image": [
            Relation("o0", 0, "o1"),
            Relation("o0", 0, "o2"),
            Relation("o1", 0, "o0"),
            Relation("o1", 1, "o2"),
        ]
    }
    scores = np.full((6, 3), -4.0)
    scores[0, 1], scores[0, 0], scores[3, 1] = 5, 4, 3
    result = evaluate(["image"], targets, {"image": scores}, {"image": s}, 2, top_ks=(3,))
    assert result["metrics"]["recall_predicate_mean_at_3"] == pytest.approx((1 / 3 + 1) / 2)
    assert result["metrics"]["annotation_ap_image_mean_at_3"] == pytest.approx((1 / 2 + 2 / 3) / 4)


def test_absolute_consistency_and_null_denominators():
    s = scene("image")
    states = np.array([[0, 0, 0, 0], [1, 1, -1, -1]])
    result = evaluate(
        ["image"],
        {"image": []},
        {"image": np.ones((2, 5))},
        {"image": s},
        4,
        decisions={"image": states},
        top_ks=(8,),
    )
    m = result["metrics"]
    assert m["incompatible_rate_all"] == 0.25
    assert m["incompatible_rate_decided"] == pytest.approx(1 / 3)
    assert m["reasoning_coverage"] == 0.75
    assert m["recall_image_mean_at_8"] is None
    empty = evaluate(
        ["image"], {"image": []}, {"image": np.zeros((2, 5))}, {"image": s}, 4, threshold=0.5
    )
    assert empty["metrics"]["reasoning_coverage"] is None
    assert empty["metric_details"]["reasoning_coverage"]["null_reason"] == "no emitted predictions"


def test_999_is_capped_and_unresolved_decisions_fail():
    s = scene("many", 33)
    result = evaluate(["many"], {"many": []}, {"many": np.ones((1056, 2))}, {"many": s}, 1)
    assert result["diagnostics"]["capped"]["prediction_count"] == 999
    assert result["diagnostics"]["uncapped"]["prediction_count"] == 1056
    with pytest.raises(ValueError, match="resolved"):
        rank_predictions([("a", "b")], np.zeros((1, 2)), 1, decisions=np.array([[-2]]))
