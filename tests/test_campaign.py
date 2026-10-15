from pathlib import Path

import pytest

from kgr4vrd.artifacts import canonical_digest, file_digest, read_json, write_json, write_jsonl
from kgr4vrd.campaign import _verified_inputs, assert_test_selection, freeze, plan
from kgr4vrd.smoke import prepare_smoke


def campaign_fixture(tmp_path):
    config = prepare_smoke(tmp_path / "fixture")
    data = Path(config["paths"]["data_root"])
    dataset = read_json(data / config["dataset"]["manifest"])
    dataset["image_manifest"] = []  # Identity-only campaign fixture, not encoder evidence.
    write_json(data / config["dataset"]["manifest"], dataset)
    config["preparation"] = {"mapping": "mapping.json"}
    config["visual"] = {
        "weights_manifest": "weights.json",
        "features_manifest": "features.json",
        "crop_batch_size": 4,
    }
    write_json(data / "mapping.json", {"scope": "synthetic fixture"})
    write_json(data / "weights.json", {"scope": "fake encoder test fixture"})
    write_json(
        data / "features.json",
        {
            "schema_version": "kgr4vrd.features/1",
            "status": "completed",
            "vocabulary_digest": dataset["vocabulary_digest"],
            "weights_manifest": {
                "path": "weights.json",
                "sha256": file_digest(data / "weights.json"),
            },
            "identity": {"image_manifest_digest": canonical_digest(dataset["image_manifest"])},
        },
    )
    write_json(
        data / config["intervention"]["control_manifest"],
        {
            "schema_version": "kgr4vrd.control/1",
            "status": "completed",
            "vocabulary_digest": dataset["vocabulary_digest"],
            "training_relations_digest": dataset["partitions"]["train"]["relations"]["sha256"],
            "training_objects_digest": dataset["partitions"]["train"]["objects"]["sha256"],
            "unmatched_fraction": 0.0,
            "selected_condition": "frequency_control",
        },
    )
    write_json(tmp_path / "base.json", config)
    source = tmp_path / "campaign.toml"
    source.write_text(
        'schema_version="kgr4vrd.campaign-config/1"\nbase_config="base.json"\n', encoding="utf-8"
    )
    return plan(source, tmp_path / "plan")


def fake_selected(directory, config, evidence, value=0.5, predicate=0.5):
    directory.mkdir(parents=True)
    (directory / "checkpoints").mkdir()
    (directory / "checkpoints/best.pt").write_bytes(b"independent-checkpoint-fixture")
    (directory / "checkpoints/latest.pt").write_bytes(b"independent-checkpoint-fixture")
    condition = config["intervention"]["condition"]
    keys = []
    if condition in {"augment", "combined"}:
        keys.append("augmentation_manifest")
    if condition in {"semantic_loss", "combined"}:
        keys.append("compatibility")
    if config["model"]["name"] == "visual_mlp_v1":
        keys.extend(("visual_features", "visual_weights"))
    inputs = {**evidence["common"], "knowledge": {key: evidence["knowledge"][key] for key in keys}}
    identity = {
        "config": config,
        "inputs": inputs,
        "source_digest": "source-fixture",
        "environment": {"fixture": True},
    }
    run_id = canonical_digest(identity)
    write_json(directory / "resolved-config.json", config)
    write_json(directory / "environment.json", {"fixture": True})
    write_json(directory / "source.json", {"fixture": True})
    write_jsonl(directory / "events.jsonl", [])
    write_jsonl(
        directory / "epochs.jsonl",
        [
            {
                "epoch": 1,
                "selection_score": value,
                "validation_metrics": {"recall_predicate_mean_at_50": predicate},
            }
        ],
    )
    selection = {
        "schema_version": "kgr4vrd.selection/1",
        "run_id": run_id,
        "checkpoint": "checkpoints/best.pt",
        "checkpoint_digest": file_digest(directory / "checkpoints/best.pt"),
        "metric": "recall_image_mean_at_50",
        "value": value,
        "epoch": 1,
        "config": "resolved-config.json",
        "identity": identity,
        "tie_rule": "earliest",
    }
    write_json(directory / "selected.json", selection)
    names = (
        "checkpoints/best.pt",
        "checkpoints/latest.pt",
        "epochs.jsonl",
        "selected.json",
        "resolved-config.json",
        "source.json",
        "environment.json",
        "events.jsonl",
    )
    write_json(
        directory / "run.json",
        {
            "schema_version": "kgr4vrd.run/1",
            "run_id": run_id,
            "identity": identity,
            "status": "completed",
            "completed_epochs": 1,
            "outputs": {name: file_digest(directory / name) for name in names},
        },
    )
    return directory / "selected.json"


def ready_campaign(path):
    campaign = read_json(path)
    evidence = _verified_inputs(campaign, path.parent)
    campaign["inputs"] = evidence["digests"]
    campaign["pilot"] = {"runtime_hours_range": [10, 20], "storage_gb_range": [2, 4]}
    campaign["control"] = {
        "record": {
            "unmatched_fraction": 0.0,
            "selected_condition": "frequency_control",
            "artifact_digest": evidence["digests"]["control"],
        }
    }
    records = []
    for index, row in enumerate(campaign["development_runs"]):
        config = read_json(path.parent / row["config"])
        score = 0.6 if row["model"] == "visual_mlp_v1" else 0.5
        selected = fake_selected(path.parent / "development" / str(index), config, evidence, score)
        records.append(
            {
                "model": row["model"],
                "seed": row["seed"],
                "learning_rate": row["learning_rate"],
                "selection": str(selected),
            }
        )
    campaign["development"] = {"records": records}
    write_json(path, campaign)


def test_real_campaign_rejects_synthetic_checker_conformance(tmp_path):
    path = campaign_fixture(tmp_path)
    campaign = read_json(path)
    config = read_json(path.parent / campaign["runs"][0]["config"])
    dataset_path = Path(config["paths"]["data_root"]) / config["dataset"]["manifest"]
    dataset = read_json(dataset_path)
    dataset["dataset_id"] = "nesy4vrd"
    write_json(dataset_path, dataset)
    with pytest.raises(ValueError, match="200 fresh queries"):
        _verified_inputs(campaign, path.parent)


def test_exact_finite_matrix_and_absolute_execution_roots(tmp_path):
    path = campaign_fixture(tmp_path)
    campaign = read_json(path)
    assert len(campaign["runs"]) == 50
    assert sum(r["condition"] != "frequency_control" for r in campaign["runs"]) == 40
    assert len(campaign["development_runs"]) == 8
    assert len(campaign["evaluations"]) == 110
    assert sum(r["filter_mode"] == "simple_asserted" for r in campaign["evaluations"]) == 10
    assert len({r["job_id"] for r in campaign["runs"] + campaign["development_runs"]}) == 58
    config = read_json(path.parent / campaign["runs"][0]["config"])
    assert Path(config["paths"]["data_root"]).is_absolute()
    with pytest.raises(FileExistsError):
        plan(tmp_path / "campaign.toml", path.parent)


def test_freeze_requires_development_and_complete_groups_and_detects_tampering(tmp_path):
    path = campaign_fixture(tmp_path)
    with pytest.raises(ValueError, match="input digests"):
        freeze(path)
    ready_campaign(path)
    freeze(path)
    campaign = read_json(path)
    assert campaign["development"]["visual_qualification"]["qualified"]
    assert campaign["effective_recipes"]["geometry_mlp"]["learning_rate"] == 0.0001
    with pytest.raises(ValueError, match="incomplete group"):
        freeze(path, "geometry-core")
    status = read_json(path.parent / "status.json")
    selected = None
    for row in campaign["runs"]:
        if row["group"] != "geometry-core":
            continue
        selected = fake_selected(
            path.parent / "runs" / row["job_id"],
            read_json(path.parent / row["config"]),
            campaign["input_evidence"],
        )
        entry = next(r for r in status["runs"] if r["key"] == row["key"])
        entry.update(status="completed", selection=str(selected))
    write_json(path.parent / "status.json", status)
    wrong_evidence = {
        **campaign["input_evidence"],
        "common": {
            **campaign["input_evidence"]["common"],
            "vocabulary_digest": "different-vocabulary",
        },
    }
    original = entry["selection"]
    wrong_selection = fake_selected(
        path.parent / "runs" / "wrong-input",
        read_json(selected.parent / "resolved-config.json"),
        wrong_evidence,
    )
    entry["selection"] = str(wrong_selection)
    write_json(path.parent / "status.json", status)
    with pytest.raises(ValueError, match="data, vocabulary, features or knowledge"):
        freeze(path, "geometry-core")
    entry["selection"] = original
    write_json(path.parent / "status.json", status)
    group_path = freeze(path, "geometry-core")
    assert len(read_json(group_path)["selections"]) == 20
    assert assert_test_selection(path, selected)["plan_digest"] == campaign["plan_digest"]
    selection = read_json(selected)
    selection["epoch"] = 2
    write_json(selected, selection)
    with pytest.raises(ValueError, match="absent or changed"):
        assert_test_selection(path, selected)
    campaign["metric_policy"]["top_ks"] = [1]
    write_json(path, campaign)
    with pytest.raises(ValueError, match="content changed"):
        freeze(path)


def test_control_threshold_and_failed_visual_qualification_are_enforced(tmp_path):
    path = campaign_fixture(tmp_path)
    ready_campaign(path)
    campaign = read_json(path)
    campaign["control"]["record"]["unmatched_fraction"] = 0.051
    write_json(path, campaign)
    with pytest.raises(ValueError, match="5%"):
        freeze(path)
    campaign["control"]["record"].update(
        selected_condition="shuffled_augmentation_control", changed_cell_fraction=0.7
    )
    write_json(path, campaign)
    with pytest.raises(ValueError, match="weak/infeasible"):
        freeze(path)
    campaign["control"]["record"].update(changed_cell_fraction=0.9)
    campaign["development"]["records"] = campaign["development"]["records"][:-1]
    write_json(path, campaign)
    with pytest.raises(ValueError, match="eight"):
        freeze(path)
