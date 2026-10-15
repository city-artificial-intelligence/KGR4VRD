"""Small strict TOML/JSON configuration with location-independent scientific identity."""

import copy
import math
import tomllib
from pathlib import Path
from typing import Any

from .artifacts import read_json

DEFAULTS: dict[str, Any] = {
    "schema_version": "kgr4vrd.config/1",
    "protocol_id": "nesy4vrd-corrected-v1",
    "paths": {"data_root": "../data", "artifact_root": "../artifacts"},
    "dataset": {
        "name": "nesy4vrd",
        "manifest": "prepared/nesy4vrd/v1/dataset.json",
        "split": "prepared/nesy4vrd/v1/splits.json",
    },
    "model": {
        "name": "geometry_mlp",
        "feature_profile": "legacy_geometry_v1",
        "auxiliary_output": True,
    },
    "intervention": {
        "condition": "baseline",
        "augmentation_profile": "S_TEU",
        "augmentation_manifest": "knowledge/augmentation/S_TEU/manifest.json",
        "knowledge_manifest": "knowledge/compatibility/manifest.json",
        "control_manifest": "knowledge/control/manifest.json",
        "semantic_policy": "gated_type_mask",
        "semantic_lambda": 1.0,
        "semantic_start_epoch": 3,
        "activation_threshold": 0.5,
    },
    "training": {
        "seed": 11,
        "learning_rate": 1e-4,
        "weight_decay": 1e-3,
        "max_epochs": 100,
        "patience": 25,
        "batch_images": 1,
        "loss_reduction": "image_sum",
        "device": "cpu",
        "precision": "float32",
        "num_workers": 0,
        "selection_metric": "recall_image_mean_at_50",
    },
    "evaluation": {
        "ranking": "multi_label",
        "top_k": [25, 50, 100],
        "threshold_policy": "none",
        "matching": "exact_instance",
        "simple_filter_manifest": "knowledge/simple-filter/manifest.json",
        "diagnostic_threshold": 0.5,
        "diagnostic_top_k": 999,
    },
}
OPTIONAL_FIELDS = {
    "preparation": {
        "source_version",
        "sources",
        "output_directory",
        "augmentation_ontology",
        "compatibility_ontology",
        "augmentation_format",
        "compatibility_format",
        "mapping",
        "closure_timeout_seconds",
        "closure_max_triples",
        "license_provenance",
        "ontology_imports",
    },
    "graphdb": {
        "endpoint",
        "repository",
        "ruleset",
        "expected_version",
        "username_env",
        "password_env",
        "timeout_seconds",
        "max_attempts",
    },
    "visual": {"weights_manifest", "features_manifest", "crop_batch_size"},
    "metadata": None,
}
SOURCE_FIELDS = {
    "object_vocabulary",
    "predicate_vocabulary",
    "train_annotations",
    "test_annotations",
    "images",
    "historical_split",
}
CONDITIONS = {
    "baseline",
    "augment",
    "semantic_loss",
    "combined",
    "frequency_control",
    "shuffled_augmentation_control",
}


def load_config(
    path: str | Path,
    *,
    data_root: str | Path | None = None,
    artifact_root: str | Path | None = None,
    device: str | None = None,
) -> dict:
    path = Path(path).resolve()
    if path.suffix == ".toml":
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    elif path.suffix == ".json":
        raw = read_json(path)
    else:
        raise ValueError("Configuration must be .toml or planner-produced .json")
    config = validate_config(raw)
    for name, override in (("data_root", data_root), ("artifact_root", artifact_root)):
        value = Path(override) if override is not None else path.parent / config["paths"][name]
        config["paths"][name] = str(value.resolve())
    if device is not None:
        config["training"]["device"] = device
    validate_config(config)
    return config


def validate_config(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be an object")
    unknown = set(raw) - set(DEFAULTS) - set(OPTIONAL_FIELDS)
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    config = copy.deepcopy(DEFAULTS)
    for section, value in raw.items():
        if isinstance(config.get(section), dict) or section in OPTIONAL_FIELDS:
            if not isinstance(value, dict):
                raise ValueError(f"{section} must be a table")
            allowed = set(DEFAULTS[section]) if section in DEFAULTS else OPTIONAL_FIELDS[section]
            if allowed is not None and set(value) - allowed:
                raise ValueError(f"Unknown {section} fields: {sorted(set(value) - allowed)}")
            config.setdefault(section, {}).update(value)
        else:
            config[section] = value
    if config["schema_version"] != "kgr4vrd.config/1":
        raise ValueError("Unsupported configuration schema")
    fixed = {
        ("model", "feature_profile"): "legacy_geometry_v1",
        ("model", "auxiliary_output"): True,
        ("intervention", "augmentation_profile"): "S_TEU",
        ("intervention", "semantic_policy"): "gated_type_mask",
        ("intervention", "semantic_lambda"): 1.0,
        ("intervention", "semantic_start_epoch"): 3,
        ("intervention", "activation_threshold"): 0.5,
        ("training", "loss_reduction"): "image_sum",
        ("training", "precision"): "float32",
        ("training", "num_workers"): 0,
        ("training", "selection_metric"): "recall_image_mean_at_50",
        ("evaluation", "ranking"): "multi_label",
        ("evaluation", "matching"): "exact_instance",
        ("evaluation", "threshold_policy"): "none",
        ("evaluation", "diagnostic_threshold"): 0.5,
        ("evaluation", "diagnostic_top_k"): 999,
    }
    for (section, key), expected in fixed.items():
        if config[section][key] != expected:
            raise ValueError(f"Unregistered {section}.{key}: expected {expected!r}")
    if config["model"]["name"] not in {"geometry_mlp", "visual_mlp_v1"}:
        raise ValueError("Unknown model")
    if config["intervention"]["condition"] not in CONDITIONS:
        raise ValueError("Unknown intervention")
    if config["dataset"]["name"] not in {"nesy4vrd", "synthetic"}:
        raise ValueError("Unknown dataset")
    for key in ("seed", "max_epochs", "patience", "batch_images"):
        value = config["training"][key]
        if type(value) is not int or value < (0 if key == "seed" else 1):
            raise ValueError(f"training.{key} must be a positive integer")
    for key in ("learning_rate", "weight_decay"):
        value = config["training"][key]
        if not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid training.{key}")
    if config["training"]["learning_rate"] == 0:
        raise ValueError("Learning rate must be positive")
    if config["evaluation"]["top_k"] != [25, 50, 100]:
        raise ValueError("Primary evaluation K must be [25,50,100]")
    if config["training"]["device"] not in {"cpu", "cuda", "mps"}:
        raise ValueError("Unsupported device")
    sources = config.get("preparation", {}).get("sources", {})
    if not isinstance(sources, dict) or set(sources) - SOURCE_FIELDS:
        raise ValueError("Unknown preparation source fields")
    for key in ("augmentation_format", "compatibility_format"):
        if config.get("preparation", {}).get(key, "turtle") not in {"turtle", "xml", "nt"}:
            raise ValueError(f"preparation.{key} must be turtle, xml or nt")
    return config


def data_path(config: dict, value: str) -> Path:
    return Path(config["paths"]["data_root"]) / value


def effective_config(config: dict) -> dict:
    """Training choices only: roots, optional descriptions and filters are not factors."""
    intervention = config["intervention"]
    condition = intervention["condition"]
    return {
        "protocol_id": config["protocol_id"],
        "schema_version": config["schema_version"],
        "model": config["model"],
        "training": config["training"],
        "condition": condition,
        "augmentation_profile": "S_TEU" if condition in {"augment", "combined"} else None,
        "semantic_policy": "gated_type_mask"
        if condition in {"semantic_loss", "combined"}
        else None,
        "semantic_lambda": 1.0 if condition in {"semantic_loss", "combined"} else 0.0,
    }
