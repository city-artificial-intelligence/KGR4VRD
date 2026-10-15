"""Finite registered experiment groups and immutable test-selection manifests."""

from __future__ import annotations

import copy
import math
import tomllib
from pathlib import Path
from typing import Any

from .artifacts import (
    canonical_digest,
    file_digest,
    read_json,
    read_jsonl,
    verify_file,
    verify_run,
    write_json,
)

FINAL_SEEDS = (11, 23, 37, 53, 71)
DEVELOPMENT_SEEDS = (101, 103)
MODELS = ("geometry_mlp", "visual_mlp_v1")
CONDITIONS = ("baseline", "augment", "semantic_loss", "combined", "frequency_control")
REQUIRED_INPUTS = (
    "dataset",
    "split",
    "vocabulary",
    "augmentation",
    "compatibility",
    "mapping",
    "geometry_features",
    "visual_features",
)


def _resolved(base: Path, value: str) -> Path:
    return (base / value).resolve()


def _config_identity(config: dict[str, Any]) -> str:
    from .config import effective_config

    # Prepared input hashes are bound by the common campaign, independently of
    # execution roots. Filtering and inactive intervention defaults are not jobs.
    return canonical_digest(effective_config(config))


def _contrasts() -> list[dict[str, str]]:
    comparisons = [
        ("augment", "baseline", "none", "none", "primary"),
        ("semantic_loss", "baseline", "none", "none", "primary"),
        ("combined", "augment", "none", "none", "primary"),
        ("combined", "semantic_loss", "none", "none", "primary"),
        ("baseline", "baseline", "simple_asserted", "none", "control"),
        ("baseline", "baseline", "ontology", "none", "control"),
        ("baseline", "baseline", "ontology", "simple_asserted", "control"),
        ("augment", "frequency_control", "none", "none", "control"),
        ("frequency_control", "baseline", "none", "none", "control"),
    ]
    return [
        {
            "name": f"{model}:{left}:{lf}-minus-{right}:{rf}",
            "model": model,
            "left_condition": left,
            "right_condition": right,
            "left_filter": lf,
            "right_filter": rf,
            "family": family,
        }
        for model in MODELS
        for left, right, lf, rf, family in comparisons
    ] + [
        {
            "name": "visual-minus-geometry-baseline",
            "model": "visual_mlp_v1",
            "right_model": "geometry_mlp",
            "left_condition": "baseline",
            "right_condition": "baseline",
            "left_filter": "none",
            "right_filter": "none",
            "family": "predictor",
        }
    ]


def plan(campaign_path: str | Path, output: str | Path) -> Path:
    """Write 50 final jobs, eight bounded development jobs and 110 named views."""
    from .config import load_config

    source, destination = Path(campaign_path).resolve(), Path(output).resolve()
    if source.suffix != ".toml":
        raise ValueError("Campaign input must be TOML")
    raw = tomllib.loads(source.read_text(encoding="utf-8"))
    unknown = set(raw) - {
        "schema_version",
        "base_config",
        "pilot",
        "development",
        "control",
        "inputs",
        "metadata",
    }
    if unknown or raw.get("schema_version") != "kgr4vrd.campaign-config/1":
        raise ValueError(f"Unsupported campaign schema/fields: {sorted(unknown)}")
    if (destination / "campaign.json").exists():
        raise FileExistsError("Campaign already exists; use a new output directory")
    base = load_config(_resolved(source.parent, raw["base_config"]))
    final, development, evaluations, groups = [], [], [], []
    jobs: dict[str, str] = {}
    for model in MODELS:
        prefix = "geometry" if model == "geometry_mlp" else "visual"
        for condition in CONDITIONS:
            group = f"{prefix}-control" if condition == "frequency_control" else f"{prefix}-core"
            for seed in FINAL_SEEDS:
                key = f"{model}:{condition}:{seed}"
                config = copy.deepcopy(base)
                config["model"]["name"] = model
                config["intervention"]["condition"] = condition
                config["training"]["seed"] = seed
                digest = _config_identity(config)
                config_name = jobs.setdefault(digest, f"configs/{digest}.json")
                write_json(destination / config_name, config)
                final.append(
                    {
                        "key": key,
                        "job_id": digest,
                        "model": model,
                        "condition": condition,
                        "seed": seed,
                        "group": group,
                        "config": config_name,
                    }
                )
                for mode in (
                    ("none", "ontology", "simple_asserted")
                    if condition == "baseline"
                    else ("none", "ontology")
                ):
                    evaluations.append(
                        {
                            "key": f"{key}:{mode}",
                            "run_key": key,
                            "model": model,
                            "dataset": base["dataset"]["name"],
                            "condition": condition,
                            "seed": seed,
                            "filter_mode": mode,
                            "group": group,
                        }
                    )
        for seed in DEVELOPMENT_SEEDS:
            for learning_rate in (1e-4, 3e-4):
                config = copy.deepcopy(base)
                config["model"]["name"] = model
                config["intervention"]["condition"] = "baseline"
                config["training"].update(seed=seed, learning_rate=learning_rate)
                digest = _config_identity(config)
                name = jobs.setdefault(digest, f"configs/{digest}.json")
                write_json(destination / name, config)
                development.append(
                    {
                        "key": f"{model}:development:{seed}:{learning_rate}",
                        "job_id": digest,
                        "model": model,
                        "condition": "baseline",
                        "seed": seed,
                        "learning_rate": learning_rate,
                        "config": name,
                    }
                )
    for name in ("geometry-core", "visual-core", "geometry-control", "visual-control"):
        groups.append({"name": name, "run_keys": [r["key"] for r in final if r["group"] == name]})
    value = {
        "schema_version": "kgr4vrd.campaign/1",
        "status": "draft",
        "protocol_id": base["protocol_id"],
        "protocol_digest": canonical_digest(
            {
                "protocol_id": base["protocol_id"],
                "contrasts": _contrasts(),
                "seeds": FINAL_SEEDS,
                "ranking": "multi_label_unthresholded_v1",
            }
        ),
        "final_seeds": list(FINAL_SEEDS),
        "development_seeds": list(DEVELOPMENT_SEEDS),
        "runs": final,
        "development_runs": development,
        "evaluations": evaluations,
        "groups": groups,
        "contrasts": _contrasts(),
        "inputs": raw.get("inputs", {}),
        "pilot": raw.get("pilot", {}),
        "development": raw.get("development", {}),
        "control": raw.get("control", {}),
        "metadata": raw.get("metadata", {}),
        "effective_recipes": {},
        "metric_policy": {
            "top_ks": [25, 50, 100],
            "selection": "recall_image_mean_at_50",
            "bootstrap_draws": 10000,
            "uncertainty_seed": 2028,
        },
    }
    for section, field in (("development", "records"), ("control", "record")):
        if isinstance(value[section].get(field), str):
            value[section][field] = str(_resolved(source.parent, value[section][field]))
    write_json(destination / "campaign.json", value)
    write_json(
        destination / "status.json",
        {
            "runs": [{"key": r["key"], "status": "not_started"} for r in final + development],
            "evaluations": [{"key": r["key"], "status": "not_started"} for r in evaluations],
        },
    )
    return destination / "campaign.json"


def load_campaign(path: str | Path, *, frozen: bool = False) -> dict[str, Any]:
    value = read_json(path)
    if value.get("schema_version") != "kgr4vrd.campaign/1":
        raise ValueError("Unsupported campaign schema")
    if value.get("status") == "frozen":
        contents = {k: v for k, v in value.items() if k != "plan_digest"}
        if canonical_digest(contents) != value.get("plan_digest"):
            raise ValueError("Frozen campaign content changed")
    elif frozen:
        raise ValueError("Test use requires a frozen scientific analysis plan")
    _validate_plan_shape(value)
    return value


def _validate_plan_shape(value: dict[str, Any]) -> None:
    """A missing registered seed cannot be repaired by deleting its plan row."""
    if value["final_seeds"] != list(FINAL_SEEDS) or value["development_seeds"] != list(
        DEVELOPMENT_SEEDS
    ):
        raise ValueError("Campaign seeds differ from the registered finite protocol")
    expected = {
        f"{model}:{condition}:{seed}": (model, condition, seed)
        for model in MODELS
        for condition in CONDITIONS
        for seed in FINAL_SEEDS
    }
    if len(value["runs"]) != 50 or {r["key"] for r in value["runs"]} != set(expected):
        raise ValueError("Campaign must retain all 50 registered final run rows")
    for row in value["runs"]:
        model, condition, seed = expected[row["key"]]
        if row["condition"] == "shuffled_augmentation_control" and condition == "frequency_control":
            condition = "shuffled_augmentation_control"
        if (row["model"], row["condition"], row["seed"]) != (model, condition, seed):
            raise ValueError("Campaign run identity differs from its registered row")
    development = {(r["model"], r["seed"], r["learning_rate"]) for r in value["development_runs"]}
    if len(value["development_runs"]) != 8 or development != {
        (model, seed, rate)
        for model in MODELS
        for seed in DEVELOPMENT_SEEDS
        for rate in (1e-4, 3e-4)
    }:
        raise ValueError("Campaign must retain the eight bounded development rows")
    views = {
        (key, mode)
        for key, (_, condition, _) in expected.items()
        for mode in (
            ("none", "ontology", "simple_asserted")
            if condition == "baseline"
            else ("none", "ontology")
        )
    }
    if (
        len(value["evaluations"]) != 110
        or {(r["run_key"], r["filter_mode"]) for r in value["evaluations"]} != views
    ):
        raise ValueError("Campaign must retain all 110 registered primary views")
    groups = {
        name: {r["key"] for r in value["runs"] if r["group"] == name}
        for name in ("geometry-core", "visual-core", "geometry-control", "visual-control")
    }
    if len(value["groups"]) != 4 or any(
        set(g["run_keys"]) != groups.get(g["name"]) for g in value["groups"]
    ):
        raise ValueError("Campaign group inventories differ from registered run rows")


def _verify_references(value: Any, base: Path) -> None:
    """Verify component records without opening relation labels or arrays."""
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            verify_file(base, value)
        else:
            for child in value.values():
                _verify_references(child, base)
    elif isinstance(value, list):
        for child in value:
            _verify_references(child, base)


def _verified_inputs(campaign: dict[str, Any], base: Path) -> dict[str, Any]:
    from . import features
    from .config import data_path
    from .data import load_manifest, load_split_ids, load_vocabulary
    from .reasoning import load_knowledge, load_simple_filter, validate_profile_metadata

    configs = {
        model: read_json(base / next(r["config"] for r in campaign["runs"] if r["model"] == model))
        for model in MODELS
    }
    config, visual = configs[MODELS[0]], configs[MODELS[1]]
    dataset_path = data_path(config, config["dataset"]["manifest"])
    manifest, vocabulary = load_manifest(dataset_path), load_vocabulary(dataset_path)
    load_split_ids(dataset_path)
    for partition in manifest["partitions"].values():
        for kind in ("objects", "relations"):
            verify_file(dataset_path.parent, partition[kind])
    common = {
        "dataset": manifest["dataset_id"],
        "dataset_version": manifest["dataset_version"],
        "split_digest": manifest["split_digest"],
        "vocabulary_digest": vocabulary.digest,
        "partitions": {
            split: {
                kind: manifest["partitions"][split][kind]["sha256"]
                for kind in ("objects", "relations")
            }
            for split in ("train", "validation")
        },
        "feature_profile": config["model"]["feature_profile"],
    }
    paths = {
        "augmentation": data_path(config, config["intervention"]["augmentation_manifest"]),
        "compatibility": data_path(config, config["intervention"]["knowledge_manifest"]),
        "control": data_path(config, config["intervention"]["control_manifest"]),
        "simple_filter": data_path(config, config["evaluation"]["simple_filter_manifest"]),
        "visual_features": data_path(visual, visual["visual"]["features_manifest"]),
        "visual_weights": data_path(visual, visual["visual"]["weights_manifest"]),
        "mapping": data_path(config, config["preparation"]["mapping"]),
    }
    records = {name: read_json(path) for name, path in paths.items() if name != "mapping"}
    if records["visual_features"]["weights_manifest"]["sha256"] != file_digest(
        paths["visual_weights"]
    ):
        raise ValueError("Configured visual weights differ from the feature cache encoder")
    if records["visual_features"]["identity"]["image_manifest_digest"] != canonical_digest(
        manifest["image_manifest"]
    ):
        raise ValueError("Visual feature cache uses a different source image inventory")
    for name, record in records.items():
        _verify_references(record, paths[name].parent)
        if name != "visual_weights" and (
            record.get("status") != "completed"
            or record.get("vocabulary_digest") != vocabulary.digest
        ):
            raise ValueError(f"Incomplete or mismatched prepared {name} artifact")
    for name in ("augmentation", "control"):
        schema = f"kgr4vrd.{name}/1"
        if records[name].get("schema_version") != schema:
            raise ValueError(f"Unsupported prepared {name} schema")
        for kind in ("relations", "objects"):
            if records[name].get(f"training_{kind}_digest") != common["partitions"]["train"][kind]:
                raise ValueError(f"Prepared {name} artifact uses different training {kind}")
    validate_profile_metadata(
        records["augmentation"], config["intervention"]["augmentation_profile"]
    )
    table = load_knowledge(paths["compatibility"], vocabulary)
    conformance = table.metadata.get("conformance", {})
    if manifest["dataset_id"] != "synthetic" and (
        table.metadata.get("backend") != "GraphDB"
        or conformance.get("status") != "passed"
        or conformance.get("sample_count", 0) < 200
    ):
        raise ValueError(
            "Real campaign requires verified GraphDB conformance with at least 200 fresh queries"
        )
    load_simple_filter(paths["simple_filter"], vocabulary)
    digests = {name: file_digest(path) for name, path in paths.items()}
    digests.update(
        dataset=file_digest(dataset_path),
        split=manifest["split_digest"],
        vocabulary=vocabulary.digest,
        geometry_features=canonical_digest(
            {"profile": common["feature_profile"], "source": file_digest(features.__file__)}
        ),
    )
    for name, digest in campaign["inputs"].items():
        if digests.get(name) != digest:
            raise ValueError(f"Declared campaign input differs from verified files: {name}")
    return {
        "digests": digests,
        "common": common,
        "knowledge": {
            "augmentation_manifest": digests["augmentation"],
            "control_manifest": digests["control"],
            "compatibility": table.digest,
            "visual_features": digests["visual_features"],
            "visual_weights": digests["visual_weights"],
        },
        "control": records["control"],
    }


def _match_run_inputs(
    selection: dict[str, Any], planned: dict[str, Any], evidence: dict[str, Any]
) -> dict[str, Any]:
    identity = selection["identity"]
    expected = dict(evidence["common"])
    keys = []
    condition = planned["condition"]
    if condition in {"augment", "combined"}:
        keys.append("augmentation_manifest")
    if condition in {"frequency_control", "shuffled_augmentation_control"}:
        keys.append("control_manifest")
    if condition in {"semantic_loss", "combined"}:
        keys.append("compatibility")
    if planned["model"] == "visual_mlp_v1":
        keys.extend(("visual_features", "visual_weights"))
    expected["knowledge"] = {key: evidence["knowledge"][key] for key in keys}
    if identity["inputs"] != expected:
        raise ValueError(
            "Selected run data, vocabulary, features or knowledge differs from frozen inputs"
        )
    return {"source_digest": identity["source_digest"], "environment": identity["environment"]}


def _development_decisions(
    campaign: dict[str, Any], base: Path, evidence: dict[str, Any]
) -> dict[str, Any]:
    records = campaign["development"].get("records")
    if isinstance(records, str):
        records = read_json(_resolved(base, records))
    if not isinstance(records, list) or len(records) != 8:
        raise ValueError("Freeze requires all eight bounded development results")
    expected = {
        (model, seed, rate)
        for model in MODELS
        for seed in DEVELOPMENT_SEEDS
        for rate in (1e-4, 3e-4)
    }
    measured: dict[tuple[str, int, float], dict[str, float]] = {}
    selections, executions = [], []
    for row in records:
        key = row["model"], row["seed"], row["learning_rate"]
        if key not in expected or key in measured:
            raise ValueError("Unexpected or duplicate development run")
        path = _resolved(base, row["selection"])
        verify_run(path.parent)
        selection = read_json(path)
        executions.append(
            _match_run_inputs(selection, {"model": row["model"], "condition": "baseline"}, evidence)
        )
        config = read_json(path.parent / selection["config"])
        if (
            config["model"]["name"],
            config["training"]["seed"],
            config["training"]["learning_rate"],
        ) != key or config["intervention"]["condition"] != "baseline":
            raise ValueError("Development selection does not match its baseline recipe")
        if selection["metric"] != "recall_image_mean_at_50":
            raise ValueError("Wrong development selection metric")
        history = read_jsonl(path.parent / "epochs.jsonl")
        epoch = next(r for r in history if r["epoch"] == selection["epoch"])
        image, predicate = (
            selection["value"],
            epoch["validation_metrics"]["recall_predicate_mean_at_50"],
        )
        if "recall_predicate_mean_at_50" in row and row["recall_predicate_mean_at_50"] != predicate:
            raise ValueError("Development predicate metric differs from selected validation epoch")
        if any(
            not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1
            for v in (image, predicate)
        ):
            raise ValueError("Invalid development metric")
        measured[key] = {"image": image, "predicate": predicate}
        selections.append(dict(row, selection_digest=file_digest(path), selection_value=image))
    if any(execution != executions[0] for execution in executions):
        raise ValueError("Development runs must share the declared source snapshot and environment")
    recipes = {}
    for model in MODELS:
        rates = {
            rate: sum(measured[model, seed, rate]["image"] for seed in DEVELOPMENT_SEEDS) / 2
            for rate in (1e-4, 3e-4)
        }
        rate = max(sorted(rates), key=lambda rate: rates[rate])
        recipes[model] = {
            "learning_rate": rate,
            "image_mean": rates[rate],
            "predicate_mean": sum(
                measured[model, seed, rate]["predicate"] for seed in DEVELOPMENT_SEEDS
            )
            / 2,
        }
    geometry, visual = recipes[MODELS[0]], recipes[MODELS[1]]
    qualified = (
        visual["image_mean"] - geometry["image_mean"] >= 0.01
        and visual["predicate_mean"] - geometry["predicate_mean"] >= -0.01
    )
    fallback = campaign["development"].get("fallback_decision")
    if not qualified and not (
        isinstance(fallback, dict)
        and fallback.get("status") == "unavailable"
        and fallback.get("reason")
    ):
        raise ValueError(
            "Visual qualification failed; record the bounded fallback decision before test use"
        )
    return {
        "records": selections,
        "recipes": recipes,
        "execution_identity": executions[0],
        "visual_qualification": {
            "qualified": qualified,
            "image_mean_difference": visual["image_mean"] - geometry["image_mean"],
            "predicate_mean_difference": visual["predicate_mean"] - geometry["predicate_mean"],
            "fallback_decision": fallback,
            "stronger_predictor_claim_supported": qualified,
        },
    }


def _control_decision(
    campaign: dict[str, Any], base: Path, evidence: dict[str, Any]
) -> dict[str, Any]:
    value = campaign["control"].get("record")
    if isinstance(value, str):
        value = read_json(_resolved(base, value))
    if not isinstance(value, dict):
        raise ValueError("Freeze requires a C2 coverage/fallback decision record")
    fraction = value.get("unmatched_fraction")
    if (
        not isinstance(fraction, (int, float))
        or not math.isfinite(fraction)
        or not 0 <= fraction <= 1
    ):
        raise ValueError("Invalid unmatched added-label fraction")
    expected = "shuffled_augmentation_control" if fraction > 0.05 else "frequency_control"
    if value.get("selected_condition") != expected:
        raise ValueError("Control condition does not follow the registered 5% coverage trigger")
    if expected == "shuffled_augmentation_control" and value.get("changed_cell_fraction", -1) < 0.8:
        raise ValueError(
            "Shuffled control is weak/infeasible; the required control claim is incomplete"
        )
    if not value.get("artifact_digest"):
        raise ValueError("Control decision must bind its prepared artifact digest")
    if value["artifact_digest"] != evidence["digests"]["control"]:
        raise ValueError("Control decision does not bind the actual prepared artifact")
    for key in ("unmatched_fraction", "selected_condition", "changed_cell_fraction"):
        if key in value and value[key] != evidence["control"].get(key):
            raise ValueError(f"Control decision differs from prepared evidence: {key}")
    return value


def freeze(campaign_path: str | Path, group: str | None = None) -> Path:
    """Freeze scientific choices once, then bind complete groups' selected weights."""
    path = Path(campaign_path).resolve()
    campaign = load_campaign(path, frozen=group is not None)
    if group is not None:
        return _freeze_group(path, campaign, group)
    if campaign["status"] == "frozen":
        return path
    missing = [name for name in REQUIRED_INPUTS if not campaign["inputs"].get(name)]
    if missing:
        raise ValueError(f"Missing frozen scientific input digests: {missing}")
    for key, digest in campaign["inputs"].items():
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise ValueError(f"Invalid SHA-256 input identity: {key}")
    for key in ("runtime_hours_range", "storage_gb_range"):
        bounds = campaign["pilot"].get(key)
        if not isinstance(bounds, list) or len(bounds) != 2 or not 0 <= bounds[0] <= bounds[1]:
            raise ValueError(f"Missing or invalid measured pilot projection: {key}")
    evidence = _verified_inputs(campaign, path.parent)
    development = _development_decisions(campaign, path.parent, evidence)
    control = _control_decision(campaign, path.parent, evidence)
    campaign["inputs"] = evidence["digests"]
    campaign["input_evidence"] = {k: v for k, v in evidence.items() if k != "control"}
    campaign["development"] = development
    campaign["control"] = control
    campaign["effective_recipes"] = development["recipes"]
    for row in campaign["runs"]:
        config_path = path.parent / row["config"]
        config = read_json(config_path)
        config["training"]["learning_rate"] = development["recipes"][row["model"]]["learning_rate"]
        if row["condition"] == "frequency_control":
            row["condition"] = control["selected_condition"]
            config["intervention"]["condition"] = control["selected_condition"]
        row["job_id"] = _config_identity(config)
        write_json(config_path, config)
        row["config_digest"] = file_digest(config_path)
    if control["selected_condition"] != "frequency_control":
        for row in campaign["evaluations"]:
            if row["condition"] == "frequency_control":
                row["condition"] = control["selected_condition"]
        for contrast in campaign["contrasts"]:
            for key in ("left_condition", "right_condition"):
                if contrast[key] == "frequency_control":
                    contrast[key] = control["selected_condition"]
    campaign["status"] = "frozen"
    campaign["plan_digest"] = canonical_digest(campaign)
    write_json(path, campaign)
    return path


def _freeze_group(path: Path, campaign: dict[str, Any], group: str) -> Path:
    matches = [g for g in campaign["groups"] if g["name"] == group]
    if len(matches) != 1:
        raise ValueError(f"Unknown campaign group: {group}")
    status = read_json(path.parent / "status.json")
    rows = status.get("runs", [])
    if len({r["key"] for r in rows}) != len(rows):
        raise ValueError(
            "Duplicate run attempts require an explicit representative before freezing"
        )
    inventory = {r["key"]: r for r in rows}
    planned = {r["key"]: r for r in campaign["runs"]}
    selections = []
    for key in matches[0]["run_keys"]:
        item = inventory.get(key, {})
        if item.get("status") != "completed" or not item.get("selection"):
            raise ValueError(f"Cannot freeze incomplete group {group}: {key}")
        selection_path = _resolved(path.parent, item["selection"])
        verify_run(selection_path.parent)
        selection = read_json(selection_path)
        config = read_json(selection_path.parent / selection["config"])
        wanted = planned[key]
        if _config_identity(config) != wanted["job_id"]:
            raise ValueError(f"Selected run does not match frozen configuration: {key}")
        execution = _match_run_inputs(selection, wanted, campaign["input_evidence"])
        if execution != campaign["development"]["execution_identity"]:
            raise ValueError(
                "Selected run source snapshot/environment differs from frozen development"
            )
        selections.append(
            {
                "key": key,
                "run_id": selection["run_id"],
                "selection": str(selection_path),
                "selection_digest": file_digest(selection_path),
                "checkpoint_digest": selection["checkpoint_digest"],
            }
        )
    result = {
        "schema_version": "kgr4vrd.group-selection/1",
        "group": group,
        "plan_digest": campaign["plan_digest"],
        "selections": selections,
    }
    destination = path.parent / "selections" / f"{group}.json"
    if destination.exists() and read_json(destination) != result:
        raise ValueError("An immutable group selection cannot be changed")
    write_json(destination, result)
    return destination


def assert_test_selection(campaign_path: str | Path, selected_path: str | Path) -> dict[str, Any]:
    """Return the frozen binding for this exact selection, refusing later edits."""
    path, selected = Path(campaign_path).resolve(), Path(selected_path).resolve()
    campaign = load_campaign(path, frozen=True)
    selection = read_json(selected)
    matches: list[dict[str, Any]] = []
    for group in campaign["groups"]:
        manifest = path.parent / "selections" / f"{group['name']}.json"
        if not manifest.exists():
            continue
        value = read_json(manifest)
        if value["plan_digest"] != campaign["plan_digest"]:
            raise ValueError("Group selection belongs to another analysis plan")
        if {r["key"] for r in value["selections"]} != set(group["run_keys"]):
            raise ValueError("Frozen group does not contain its complete expected run list")
        matches.extend(r for r in value["selections"] if r["run_id"] == selection["run_id"])
    if len(matches) != 1 or matches[0]["selection_digest"] != file_digest(selected):
        raise ValueError("Test checkpoint is absent or changed from the frozen group selection")
    if file_digest(selected.parent / selection["checkpoint"]) != matches[0]["checkpoint_digest"]:
        raise ValueError("Frozen selected checkpoint checksum mismatch")
    return dict(matches[0], plan_digest=campaign["plan_digest"])
