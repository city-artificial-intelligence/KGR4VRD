import copy
import shutil

import numpy as np
import pytest
import torch

from kgr4vrd.artifacts import read_json, source_snapshot, verify_run, write_json
from kgr4vrd.config import load_config
from kgr4vrd.data import load_objects
from kgr4vrd.prediction import ScoreShards, evaluate_predictions, predict
from kgr4vrd.smoke import prepare_smoke
from kgr4vrd.training import train


def test_resume_and_saved_score_re_evaluation(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    snapshot = source_snapshot()
    monkeypatch.setattr("kgr4vrd.training.source_snapshot", lambda: snapshot)
    config = prepare_smoke(tmp_path / "complete")
    config["intervention"]["condition"] = "combined"
    complete = train(config)
    resumed_config = copy.deepcopy(config)
    resumed_config["paths"]["artifact_root"] = str(tmp_path / "resumed")
    interrupted = train(resumed_config, stop_after_epochs=3)
    assert read_json(interrupted / "run.json")["status"] == "interrupted"
    resumed = train(resumed_config, resume=True)
    assert complete.name == resumed.name
    a = torch.load(complete / "checkpoints/latest.pt", weights_only=False)
    b = torch.load(resumed / "checkpoints/latest.pt", weights_only=False)
    for key in a["model"]:
        torch.testing.assert_close(a["model"][key], b["model"][key], atol=1e-6, rtol=1e-5)
    for key in a["active"]:
        assert torch.equal(a["active"][key], b["active"][key])
    assert a["best_epoch"] == b["best_epoch"] and a["updates"] == b["updates"]
    for key, state in a["optimiser"]["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, b["optimiser"]["state"][key][name])
    assert verify_run(resumed)["status"] == "verified"
    with pytest.raises(ValueError, match="overwritten"):
        train(resumed_config)
    scores_a = predict(complete / "selected.json", split="validation")
    scores_b = predict(resumed / "selected.json", split="validation")
    scenes = load_objects(tmp_path / "complete/data/prepared/synthetic/dataset.json", "validation")
    sa, sb = ScoreShards(scores_a, scenes), ScoreShards(scores_b, scenes)
    for key in sa:
        np.testing.assert_allclose(sa[key], sb[key], atol=1e-6, rtol=1e-5)
    result = evaluate_predictions(scores_a)
    saved = read_json(result.parent / "metrics.json")
    shutil.rmtree(result.parent)
    regenerated = evaluate_predictions(scores_a)
    assert read_json(regenerated.parent / "metrics.json") == saved
    with pytest.raises(ValueError, match="frozen campaign"):
        predict(complete / "selected.json", split="test")


def test_training_and_prediction_never_open_test_targets_and_detect_corruption(tmp_path):
    torch.set_num_threads(1)
    config = prepare_smoke(tmp_path)
    test_targets = tmp_path / "data/prepared/synthetic/relations/test.jsonl"
    test_targets.write_text("TEST TARGET SENTINEL - NOT VALID JSON", encoding="utf-8")
    config["training"]["max_epochs"] = 1
    run = train(config)
    scores_path = predict(run / "selected.json", split="validation")
    assert evaluate_predictions(scores_path).exists()
    relocated_root = tmp_path / "relocated-artifacts"
    shutil.copytree(run.parent.parent, relocated_root)
    relocated_run = relocated_root / "runs" / run.name
    assert verify_run(relocated_run)["status"] == "verified"
    relocated_scores = predict(relocated_run / "selected.json", split="validation")
    assert relocated_scores.is_relative_to(relocated_root)
    assert evaluate_predictions(relocated_scores).is_relative_to(relocated_root)
    scores_manifest = read_json(scores_path)
    shard = scores_path.parent / scores_manifest["shards"][0]["path"]
    shard.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Checksum"):
        evaluate_predictions(scores_path)


def test_config_roots_relocate_and_unknown_fields_fail(tmp_path):
    config = prepare_smoke(tmp_path / "source")
    destination = tmp_path / "relocated"
    shutil.copytree(tmp_path / "source/data", destination)
    before = load_objects(tmp_path / "source/data/prepared/synthetic/dataset.json", "train")
    after = load_objects(destination / "prepared/synthetic/dataset.json", "train")
    assert before == after
    path = tmp_path / "source/smoke-config.json"
    relocated = load_config(path, data_root=destination)
    assert relocated["paths"]["data_root"] == str(destination)
    config["training"]["test_selection"] = True
    write_json(path, config)
    with pytest.raises(ValueError, match="Unknown training"):
        load_config(path)


def test_offline_four_condition_smoke(tmp_path):
    from kgr4vrd.smoke import smoke

    report = smoke(tmp_path / "smoke")
    assert report.exists()
    runs = list((tmp_path / "smoke/artifacts/runs").glob("*/run.json"))
    assert len(runs) == 4
    for path in runs:
        assert read_json(path)["completed_epochs"] == 4
        verify_run(path.parent)


@pytest.mark.parametrize("condition", ["augment", "combined"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", "kgr4vrd.augmentation/99"),
        ("profile", "all"),
        ("profile_digest", "0" * 64),
        ("training_objects_digest", "0" * 64),
    ],
)
def test_training_rejects_mismatched_augmentation_identity(tmp_path, condition, field, value):
    from kgr4vrd.config import data_path
    from kgr4vrd.data import load_vocabulary
    from kgr4vrd.training import _training_targets

    config = prepare_smoke(tmp_path)
    config["intervention"]["condition"] = condition
    path = data_path(config, config["intervention"]["augmentation_manifest"])
    artifact = read_json(path)
    artifact[field] = value
    write_json(path, artifact)
    dataset_path = data_path(config, config["dataset"]["manifest"])
    with pytest.raises(ValueError):
        _training_targets(
            config, dataset_path, load_objects(dataset_path, "train"), load_vocabulary(dataset_path)
        )


def test_control_rejects_schema_and_changed_object_classes(tmp_path):
    from kgr4vrd.artifacts import file_digest, read_jsonl, write_jsonl
    from kgr4vrd.config import data_path
    from kgr4vrd.data import load_vocabulary
    from kgr4vrd.training import _training_targets

    config = prepare_smoke(tmp_path)
    config["intervention"]["condition"] = "frequency_control"
    dataset_path = data_path(config, config["dataset"]["manifest"])
    dataset = read_json(dataset_path)
    path = data_path(config, config["intervention"]["control_manifest"])
    control = {
        "schema_version": "kgr4vrd.control/1",
        "status": "completed",
        "vocabulary_digest": dataset["vocabulary_digest"],
        "training_relations_digest": dataset["partitions"]["train"]["relations"]["sha256"],
        "training_objects_digest": dataset["partitions"]["train"]["objects"]["sha256"],
        "selected_condition": "frequency_control",
        "unmatched_fraction": 0,
        "weights": [2, 1, 1, 1],
    }
    write_json(path, control)
    vocabulary = load_vocabulary(dataset_path)
    _training_targets(config, dataset_path, load_objects(dataset_path, "train"), vocabulary)
    write_json(path, {**control, "schema_version": "kgr4vrd.control/99"})
    with pytest.raises(ValueError, match="schema"):
        _training_targets(config, dataset_path, load_objects(dataset_path, "train"), vocabulary)
    write_json(path, control)
    objects_path = dataset_path.parent / dataset["partitions"]["train"]["objects"]["path"]
    records = read_jsonl(objects_path)
    records[0]["objects"][0]["class_id"] = 2
    write_jsonl(objects_path, records)
    dataset["partitions"]["train"]["objects"]["sha256"] = file_digest(objects_path)
    write_json(dataset_path, dataset)
    with pytest.raises(ValueError, match="objects identity"):
        _training_targets(config, dataset_path, load_objects(dataset_path, "train"), vocabulary)
