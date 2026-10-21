"""The pilot observers must leave the registered trainer's CUDA arithmetic exact."""

import copy
import os

import pytest
import torch
from resource_pilot import CudaMeter, measure_training, projections, tree_bytes
from test_vision import visual_fixture

from kgr4vrd import training
from kgr4vrd.adapters import vision
from kgr4vrd.artifacts import read_json, read_jsonl, verify_run
from kgr4vrd.config import data_path
from kgr4vrd.data import load_objects
from kgr4vrd.smoke import prepare_smoke


@pytest.mark.integration
@pytest.mark.skipif(
    os.environ.get("KGR4VRD_CUDA_TEST") != "1",
    reason="KGR4VRD_CUDA_TEST=1 and an actual CUDA device are required",
)
@pytest.mark.parametrize("model", ["geometry_mlp", "visual_mlp_v1"])
def test_meter_preserves_exact_cuda_training_and_gate(tmp_path, monkeypatch, model):
    assert os.environ.get("KGR4VRD_CUDA_TEST") == "1" and torch.cuda.is_available()
    torch.set_num_threads(1)
    if model == "visual_mlp_v1":
        config, _, _, _, _ = visual_fixture(tmp_path / "inputs", monkeypatch)
        vision.prepare_features(config)
        config["model"]["name"] = model
    else:
        config = prepare_smoke(tmp_path / "inputs")
    config["training"].update(device="cuda", seed=23, max_epochs=3)
    config["intervention"]["condition"] = "combined"
    dataset = data_path(config, config["dataset"]["manifest"])
    # Invalid test-target bytes make any accidental target read fail immediately.
    (dataset.parent / "relations/test.jsonl").write_text("PILOT MUST NOT READ TEST TARGETS")
    ordinary = training.train(config)
    measured_config = copy.deepcopy(config)
    measured_config["paths"]["artifact_root"] = str(tmp_path / "measured")
    meter = CudaMeter(load_objects(dataset, "train"))
    original_get, original_build = training.SceneFeatures.__getitem__, training.build_model
    with measure_training(meter):
        measured = training.train(measured_config)
    assert training.SceneFeatures.__getitem__ is original_get
    assert training.build_model is original_build
    for directory in (ordinary, measured):
        verify_run(directory)
    a, b = [
        torch.load(p / "checkpoints/latest.pt", weights_only=False) for p in (ordinary, measured)
    ]
    for section in ("model", "active", "best_model"):
        assert a[section].keys() == b[section].keys()
        for key in a[section]:
            assert torch.equal(a[section][key], b[section][key])
    for left, right in zip(
        read_jsonl(ordinary / "epochs.jsonl"), read_jsonl(measured / "epochs.jsonl"), strict=True
    ):
        assert {k: v for k, v in left.items() if not k.endswith("_seconds")} == {
            k: v for k, v in right.items() if not k.endswith("_seconds")
        }
    assert (
        read_json(ordinary / "selected.json")["epoch"]
        == read_json(measured / "selected.json")["epoch"]
    )
    summary = meter.summary()
    for split in ("train", "validation"):
        assert summary["splits"][split]["image_visits"] == 3 * len(load_objects(dataset, split))
        assert summary["splits"][split]["largest_image_allocated_bytes"] > 0
    assert [h["active_cells"] for h in a["history"][:2]] == [0, 0]


def test_meter_restores_observers_after_failure_and_storage_deduplicates_links(tmp_path):
    original_get, original_build = training.SceneFeatures.__getitem__, training.build_model
    with pytest.raises(RuntimeError, match="injected"):
        with measure_training(CudaMeter([])):
            raise RuntimeError("injected")
    assert training.SceneFeatures.__getitem__ is original_get
    assert training.build_model is original_build
    path = tmp_path / "checkpoint"
    path.write_bytes(b"12345")
    (tmp_path / "alias").hardlink_to(path)
    assert tree_bytes(tmp_path) == 5


def test_projection_respects_patience_and_registered_view_counts():
    inventory = {
        "counts": {"test": {"pairs": 20}, "validation": {"pairs": 10}},
        "storage_bytes": {"inputs": 100},
        "cumulative_preparation_seconds": 3600,
    }
    records = []
    for model in ("geometry_mlp", "visual_mlp_v1"):
        for condition in ("baseline", "combined"):
            records.append(
                {
                    "model": model,
                    "condition": condition,
                    "epochs": [{"training_seconds": 1, "validation_seconds": 1}],
                    "training_wall_seconds": 2,
                    "run_bytes": 10,
                    "score_bytes": 10,
                    "prediction_wall_seconds": 1,
                    "evaluations": [
                        {"mode": mode, "seconds": 1, "bytes": 1}
                        for mode in ("none", "ontology", "simple_asserted")
                    ],
                }
            )
    result = projections(inventory, records)
    # 58 registered runs cannot stop before 26 epochs under patience 25.
    # Exactly 50 raw scores and 110 views, each scaled by 20/10 candidate pairs.
    assert result["runtime_hours_range"] == [
        (3600 + 58 * 26 * 2 + (50 + 110) * 2) / 3600,
        (3600 + 58 * 100 * 2 + (50 + 110) * 2) / 3600,
    ]
