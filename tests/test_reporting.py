import csv

import numpy as np
import pytest

from kgr4vrd.artifacts import read_json, write_json
from kgr4vrd.reporting import _summary, paired_bootstrap, report_evaluations


def record(seed, hits, counts=(1, 9), *, rate=None):
    rows = [
        {
            "image_id": str(i),
            "ground_truth_count": count,
            "hits_at_50": hit,
            "ap_sum_at_50": float(hit),
            "prediction_count": 1,
            "compatible_count": 1,
            "incompatible_count": 0,
            "unknown_count": 0,
        }
        for i, (hit, count) in enumerate(zip(hits, counts, strict=True))
    ]
    return {
        "manifest": {
            "seed": seed,
            "split": "test",
            "target_digest": "same",
            "vocabulary_digest": "vocab",
            "evaluation_id": f"e{seed}",
        },
        "metrics": {
            "recall_image_mean_at_50": np.mean([h / c for h, c in zip(hits, counts, strict=True)]),
            "recall_global_at_50": sum(hits) / sum(counts),
            "recall_predicate_mean_at_50": rate,
        },
        "per_image": rows,
    }


def test_crossed_bootstrap_reuses_one_image_draw_for_all_seeds():
    # Opposite errors across seeds cancel only when one image draw is shared.
    left = [record(11, [1, 0], (1, 1)), record(23, [0, 1], (1, 1))]
    right = [record(11, [0, 1], (1, 1)), record(23, [1, 0], (1, 1))]
    result = paired_bootstrap(left, right, "recall_image_mean_at_50", draws=10000, seed=2028)
    assert result["effect"] == 0
    assert result["lower"] == -1
    assert result["upper"] == 1
    identical = paired_bootstrap(left, left, "recall_global_at_50", draws=10000)
    assert identical["lower"] == identical["upper"] == identical["effect"] == 0
    assert identical["defined_draws"] == 10000
    left = [
        record(11, [1, 0, 1], (1, 1, 1)),
        record(23, [0, 1, 1], (1, 1, 1)),
        record(37, [1, 1, 0], (1, 1, 1)),
    ]
    right = [
        record(11, [0, 0, 1], (1, 1, 1)),
        record(23, [1, 1, 0], (1, 1, 1)),
        record(37, [0, 1, 1], (1, 1, 1)),
    ]
    rng = np.random.default_rng(44)
    seed_samples = rng.integers(3, size=(7, 3))
    image_samples = rng.integers(3, size=(7, 3))
    manual = []
    for seeds, images in zip(seed_samples, image_samples, strict=True):
        differences = []
        for seed_index in seeds:
            differences.append(
                sum(
                    left[seed_index]["per_image"][i]["hits_at_50"]
                    - right[seed_index]["per_image"][i]["hits_at_50"]
                    for i in images
                )
                / 3
            )
        manual.append(sum(differences) / 3)
    actual = paired_bootstrap(left, right, "recall_image_mean_at_50", draws=7, seed=44)
    assert (actual["lower"], actual["upper"]) == pytest.approx(np.quantile(manual, [0.025, 0.975]))


def test_seed_only_predicate_intervals_and_undefined_draws():
    left = [record(11, [1, 0], rate=0.7), record(23, [0, 1], rate=0.6)]
    right = [record(11, [0, 1], rate=0.5), record(23, [1, 0], rate=0.4)]
    result = paired_bootstrap(left, right, "recall_predicate_mean_at_50")
    assert result["interval_method"] == "paired_seed_only"
    assert result["lower"] == pytest.approx(0.2)
    assert result["upper"] == pytest.approx(0.2)
    for item in left + right:
        item["metrics"]["incompatible_rate_all"] = None
        for row in item["per_image"]:
            row["prediction_count"] = 0
    undefined = paired_bootstrap(left, right, "incompatible_rate_all")
    assert undefined["undefined_draws"] == 10000
    assert undefined["lower"] is None


def test_missing_seed_prevents_favourable_only_aggregate():
    expected = [
        {
            "key": str(seed),
            "seed": seed,
            "model": "geometry_mlp",
            "condition": "baseline",
            "filter_mode": "none",
        }
        for seed in (11, 23)
    ]
    rows = _summary(expected, {"11": record(11, [1, 9])}, {"11": "completed", "23": "failed"})
    assert any(r["seed"] == 23 and r["status"] == "failed" for r in rows)
    means = [r for r in rows if r["statistic"] == "mean"]
    assert all(r["value"] is None and r["status"] == "incomplete" for r in means)


def test_saved_report_regenerates_without_training(tmp_path):
    value = record(11, [1, 0])
    directory = tmp_path / "evaluation"
    directory.mkdir()
    manifest = {
        **value["manifest"],
        "schema_version": "kgr4vrd.evaluation/1",
        "status": "completed",
        "model": "geometry_mlp",
        "condition": "baseline",
        "filter_mode": "none",
        "run_id": "run",
    }
    write_json(directory / "evaluation.json", manifest)
    write_json(directory / "metrics.json", {"metrics": value["metrics"]})
    with (directory / "per_image.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(value["per_image"][0]))
        writer.writeheader()
        writer.writerows(value["per_image"])
    output = report_evaluations([directory / "evaluation.json"], tmp_path / "report")
    assert read_json(output)["completed_evaluations"] == 1
    first = (output.parent / "summary.csv").read_bytes()
    report_evaluations([directory / "evaluation.json"], tmp_path / "report")
    assert first == (output.parent / "summary.csv").read_bytes()
    for extension in ("pdf", "svg", "png"):
        assert (output.parent / f"absolute-recall.{extension}").is_file()
