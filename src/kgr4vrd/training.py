"""One training loop, raw validation selection, and epoch-boundary resumption."""

import logging
import math
import random
import resource
import sys
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .artifacts import (
    atomic_write,
    canonical_digest,
    environment,
    file_digest,
    read_json,
    read_jsonl,
    source_snapshot,
    utc_now,
    verify_file,
    write_json,
    write_jsonl,
)
from .config import data_path, effective_config, validate_config
from .data import (
    build_pairs,
    build_targets,
    load_manifest,
    load_objects,
    load_relations,
    load_vocabulary,
)
from .features import geometry_features
from .losses import compute_loss, update_active_mask
from .metrics import evaluate
from .models import build_model
from .types import Relation

LOGGER = logging.getLogger(__name__)


def select_checkpoint(history: list[dict]) -> dict:
    """Choose earliest strictly best validation score; test fields are never consulted."""
    if not history or any(not math.isfinite(row["selection_score"]) for row in history):
        raise ValueError("Empty or non-finite validation history")
    return max(history, key=lambda row: row["selection_score"])


def feature_array(
    config: dict, scene: Any, vocabulary: Any, visual_manifest: dict | None = None
) -> np.ndarray:
    geometry = geometry_features(scene, vocabulary)
    if config["model"]["name"] == "visual_mlp_v1":
        from .adapters.vision import load_visual_features

        return load_visual_features(
            data_path(config, config["visual"]["features_manifest"]),
            scene,
            vocabulary,
            geometry,
            manifest=visual_manifest,
        )
    return geometry


class SceneFeatures(Mapping[str, torch.Tensor]):
    """Materialise one image at a time; visual pair vectors need not all fit in RAM."""

    def __init__(self, config: dict, scenes: dict, vocabulary: Any):
        self.config, self.scenes, self.vocabulary = config, scenes, vocabulary
        self.visual_manifest = (
            read_json(data_path(config, config["visual"]["features_manifest"]))
            if config["model"]["name"] == "visual_mlp_v1"
            else None
        )

    def __iter__(self) -> Iterator[str]:
        return iter(self.scenes)

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, key: str) -> torch.Tensor:
        return torch.from_numpy(
            feature_array(self.config, self.scenes[key], self.vocabulary, self.visual_manifest)
        )


def _training_targets(
    config: dict, manifest_path: Path, scenes: dict, vocabulary: Any
) -> tuple[dict, Any, dict]:
    targets = load_relations(manifest_path, "train")
    condition = config["intervention"]["condition"]
    identities: dict[str, Any] = {}
    weights = None
    if condition in {"augment", "combined", "shuffled_augmentation_control", "frequency_control"}:
        key = (
            "augmentation_manifest" if condition in {"augment", "combined"} else "control_manifest"
        )
        path = data_path(config, config["intervention"][key])
        artifact = read_json(path)
        expected_schema = (
            "kgr4vrd.augmentation/1"
            if condition in {"augment", "combined"}
            else "kgr4vrd.control/1"
        )
        if artifact.get("schema_version") != expected_schema:
            raise ValueError(f"Unsupported augmentation/control schema: {path}")
        if condition in {"augment", "combined"}:
            from .reasoning import validate_profile_metadata

            validate_profile_metadata(artifact, config["intervention"]["augmentation_profile"])
        if (
            artifact.get("status") != "completed"
            or artifact["vocabulary_digest"] != vocabulary.digest
        ):
            raise ValueError(f"Incomplete or mismatched targets: {path}")
        partition = load_manifest(manifest_path)["partitions"]["train"]
        for kind in ("relations", "objects"):
            if artifact.get(f"training_{kind}_digest") != partition[kind]["sha256"]:
                raise ValueError(f"Augmentation/control raw training {kind} identity mismatch")
        identities[key] = file_digest(path)
        if condition == "frequency_control":
            if (
                artifact.get("selected_condition") != condition
                or artifact["unmatched_fraction"] > 0.05
            ):
                raise ValueError("Frequency control requires the registered shuffle fallback")
            weights = torch.tensor(artifact["weights"], dtype=torch.float32)
        else:
            if condition == "shuffled_augmentation_control" and (
                artifact.get("selected_condition") != condition
                or artifact["unmatched_fraction"] <= 0.05
                or artifact.get("changed_cell_fraction", 0) < 0.8
            ):
                raise ValueError("Shuffle control lacks triggered, feasible frozen evidence")
            relation_path = verify_file(path.parent, artifact["relations"])
            records = read_jsonl(relation_path)
            if len(records) != len(scenes) or {r["image_id"] for r in records} != set(scenes):
                raise ValueError("Augmentation changes training image membership")
            expanded = {
                r["image_id"]: tuple(
                    Relation(x["subject_id"], x["predicate_id"], x["object_id"])
                    for x in r["relations"]
                )
                for r in records
            }
            if any(not set(targets[key]).issubset(expanded[key]) for key in targets):
                raise ValueError("Augmentation dropped a raw label")
            targets = expanded
    return targets, weights, identities


def _inputs(config: dict) -> tuple[dict, dict, Any, dict, dict, Any, dict]:
    manifest_path = data_path(config, config["dataset"]["manifest"])
    manifest, vocabulary = load_manifest(manifest_path), load_vocabulary(manifest_path)
    if file_digest(data_path(config, config["dataset"]["split"])) != manifest["splits"]["sha256"]:
        raise ValueError("Configured split file differs from the dataset manifest")
    train_scenes = load_objects(manifest_path, "train")
    validation_scenes = load_objects(manifest_path, "validation")
    if not train_scenes or not validation_scenes:
        raise ValueError("Training and validation partitions must be nonempty")
    train_targets, weights, knowledge_ids = _training_targets(
        config, manifest_path, train_scenes, vocabulary
    )
    validation_targets = load_relations(manifest_path, "validation")
    table = None
    if config["intervention"]["condition"] in {"semantic_loss", "combined"}:
        from .reasoning import load_knowledge

        table = load_knowledge(
            data_path(config, config["intervention"]["knowledge_manifest"]), vocabulary
        )
        knowledge_ids["compatibility"] = table.digest
    if config["model"]["name"] == "visual_mlp_v1":
        knowledge_ids["visual_features"] = file_digest(
            data_path(config, config["visual"]["features_manifest"])
        )
        knowledge_ids["visual_weights"] = file_digest(
            data_path(config, config["visual"]["weights_manifest"])
        )
        feature_manifest = read_json(data_path(config, config["visual"]["features_manifest"]))
        if feature_manifest["weights_manifest"]["sha256"] != knowledge_ids["visual_weights"]:
            raise ValueError("Configured visual weights differ from the feature cache encoder")
        if feature_manifest["identity"]["image_manifest_digest"] != canonical_digest(
            manifest["image_manifest"]
        ):
            raise ValueError("Visual feature cache uses a different source image inventory")
    prepared = {}
    for split, scenes in (("train", train_scenes), ("validation", validation_scenes)):
        prepared[split] = SceneFeatures(config, scenes, vocabulary)
    target_arrays = {
        key: tuple(
            torch.from_numpy(x)
            for x in build_targets(
                build_pairs(train_scenes[key]), value, len(vocabulary.predicates)
            )
        )
        for key, value in train_targets.items()
    }
    identity = {
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
        "knowledge": knowledge_ids,
    }
    return (
        prepared,
        target_arrays,
        weights,
        train_scenes,
        validation_scenes,
        (validation_targets, table, vocabulary),
        identity,
    )


def _synchronise(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _rng_state(order: random.Random) -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "order": order.getstate(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "mps": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None,
    }


def _restore_rng(state: dict, order: random.Random) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    order.setstate(state["order"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])
    if state["mps"] is not None:
        torch.mps.set_rng_state(state["mps"])


def train(config: dict, *, resume: bool = False, stop_after_epochs: int | None = None) -> Path:
    """Train from train/raw-validation inputs; test relation files are never opened."""
    config = validate_config(config)
    arrays, targets, weights, train_scenes, val_scenes, context, inputs = _inputs(config)
    val_targets, table, vocabulary = context
    snapshot, env = source_snapshot(), environment()
    identity = {
        "effective_config": effective_config(config),
        "inputs": inputs,
        "source_digest": snapshot["digest"],
        "environment": env,
    }
    run_id = canonical_digest(identity)
    directory = Path(config["paths"]["artifact_root"]) / "runs" / run_id
    state_path = directory / "run.json"
    if state_path.exists() and (not resume or read_json(state_path)["status"] == "completed"):
        raise ValueError(
            f"Existing run cannot be overwritten; resume incomplete work only: {directory}"
        )
    if resume and not state_path.exists():
        raise ValueError(
            "No matching interrupted run; configuration, source or environment changed"
        )
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "resolved-config.json", config)
    write_json(directory / "environment.json", env)
    write_json(directory / "source.json", snapshot)
    run = {
        "schema_version": "kgr4vrd.run/1",
        "run_id": run_id,
        "identity": identity,
        "status": "running",
        "started_at": utc_now(),
        "model": config["model"]["name"],
        "condition": config["intervention"]["condition"],
        "seed": config["training"]["seed"],
    }
    write_json(state_path, run)
    try:
        _execute(
            config,
            directory,
            run,
            arrays,
            targets,
            weights,
            train_scenes,
            val_scenes,
            val_targets,
            table,
            vocabulary,
            resume,
            stop_after_epochs,
        )
    except BaseException as error:
        run.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            failure={"type": type(error).__name__, "message": str(error)},
            ended_at=utc_now(),
        )
        write_json(state_path, run)
        raise
    return directory


def _execute(
    config: dict,
    directory: Path,
    run: dict,
    arrays: dict,
    targets: dict,
    weights: Any,
    train_scenes: dict,
    val_scenes: dict,
    val_targets: dict,
    table: Any,
    vocabulary: Any,
    resume: bool,
    stop_after_epochs: int | None,
) -> None:
    settings = config["training"]
    device, seed = torch.device(settings["device"]), settings["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    order = random.Random(seed)
    dimension = next(iter(arrays["train"].values())).shape[1]
    model = build_model(config["model"]["name"], dimension, len(vocabulary.predicates)).to(device)
    optimiser = torch.optim.Adam(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )
    active = {
        key: torch.zeros(
            (len(build_pairs(scene)), len(vocabulary.predicates)), dtype=torch.bool, device=device
        )
        for key, scene in train_scenes.items()
    }
    incompatible = {
        key: torch.from_numpy(table.lookup(scene) == 1).to(device)
        if table is not None
        else torch.zeros_like(active[key])
        for key, scene in train_scenes.items()
    }
    history: list[dict] = []
    best_model, bad_epochs, updates = None, 0, 0
    if resume:
        saved_run = read_json(directory / "resume-boundary.json")
        if file_digest(directory / "checkpoints/latest.pt") != saved_run["checkpoint_digest"]:
            raise ValueError("Resume checkpoint checksum mismatch")
        checkpoint = torch.load(
            directory / "checkpoints/latest.pt", map_location="cpu", weights_only=False
        )
        if checkpoint["run_id"] != run["run_id"] or checkpoint["pair_ids"] != {
            key: build_pairs(scene) for key, scene in train_scenes.items()
        }:
            raise ValueError("Resume checkpoint identity mismatch")
        model.load_state_dict(checkpoint["model"])
        optimiser.load_state_dict(checkpoint["optimiser"])
        active = {key: mask.to(device) for key, mask in checkpoint["active"].items()}
        history, best_model, bad_epochs, updates = (
            checkpoint[key] for key in ("history", "best_model", "bad_epochs", "updates")
        )
        _restore_rng(checkpoint["rng"], order)
        write_jsonl(directory / "epochs.jsonl", history)
    if weights is not None:
        weights = weights.to(device)
    run.update(
        parameter_count=sum(p.numel() for p in model.parameters()), effective_device=str(device)
    )
    for epoch in range(len(history) + 1, settings["max_epochs"] + 1):
        if bad_epochs >= settings["patience"]:
            break
        model.train()
        image_order = sorted(train_scenes)
        order.shuffle(image_order)
        _synchronise(device)
        started = time.perf_counter()
        ordinary_sum, semantic_sum, empty = 0.0, 0.0, 0
        for start in range(0, len(image_order), settings["batch_images"]):
            batch = image_order[start : start + settings["batch_images"]]
            losses = []
            optimiser.zero_grad(set_to_none=True)
            for key in batch:
                logits = model(arrays["train"][key].to(device))
                active[key] = update_active_mask(
                    logits[:, :-1], incompatible[key], active[key], epoch
                )
                target, rows = (value.to(device) for value in targets[key])
                parts = compute_loss(
                    logits,
                    target,
                    rows,
                    active[key],
                    semantic_lambda=run["identity"]["effective_config"]["semantic_lambda"],
                    positive_weights=weights,
                )
                if parts is None:
                    empty += 1
                    continue
                total, ordinary, semantic = parts
                losses.append(total)
                ordinary_sum += ordinary.item()
                semantic_sum += semantic.item()
            if losses:
                (torch.stack(losses).sum() / len(batch)).backward()
                if any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in model.parameters()
                ):
                    raise FloatingPointError(f"Non-finite gradient in epoch {epoch}")
                optimiser.step()
                updates += 1
        _synchronise(device)
        training_seconds = time.perf_counter() - started
        started = time.perf_counter()
        model.eval()
        with torch.inference_mode():
            scores = {
                key: model(value.to(device)).cpu().numpy()
                for key, value in arrays["validation"].items()
            }
        validation = evaluate(
            tuple(val_scenes), val_targets, scores, val_scenes, len(vocabulary.predicates)
        )
        value = validation["metrics"]["recall_image_mean_at_50"]
        if value is None or not math.isfinite(value):
            raise ValueError("Validation selection metric is undefined or non-finite")
        improved = not history or value > select_checkpoint(history)["selection_score"]
        if improved:
            best_model = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
        _synchronise(device)
        history.append(
            {
                "epoch": epoch,
                "selection_score": value,
                "selected_best": improved,
                "validation_metrics": validation["metrics"],
                "ordinary_loss": ordinary_sum,
                "semantic_loss": semantic_sum,
                "empty_images": empty,
                "updates": updates,
                "active_cells": sum(mask.sum().item() for mask in active.values()),
                "learning_rate": settings["learning_rate"],
                "training_seconds": training_seconds,
                "validation_seconds": time.perf_counter() - started,
            }
        )
        checkpoint = {
            "schema_version": "kgr4vrd.checkpoint/1",
            "run_id": run["run_id"],
            "identity": run["identity"],
            "model": model.state_dict(),
            "optimiser": optimiser.state_dict(),
            "completed_epoch": epoch,
            "best_model": best_model,
            "best_epoch": select_checkpoint(history)["epoch"],
            "best_metric": select_checkpoint(history)["selection_score"],
            "bad_epochs": bad_epochs,
            "rng": _rng_state(order),
            "history": history,
            "updates": updates,
            "active": {key: mask.cpu() for key, mask in active.items()},
            "pair_ids": {key: build_pairs(scene) for key, scene in train_scenes.items()},
        }
        atomic_write(
            directory / "checkpoints/latest.pt", lambda stream: torch.save(checkpoint, stream)
        )
        write_json(
            directory / "resume-boundary.json",
            {
                "schema_version": "kgr4vrd.resume/1",
                "epoch": epoch,
                "checkpoint_digest": file_digest(directory / "checkpoints/latest.pt"),
            },
        )
        write_jsonl(directory / "epochs.jsonl", history)
        LOGGER.info("%s epoch=%d validation_recall50=%.9f", run["run_id"][:12], epoch, value)
        if (
            stop_after_epochs is not None
            and epoch >= stop_after_epochs
            and epoch < settings["max_epochs"]
        ):
            run.update(status="interrupted", completed_epochs=epoch, completed_updates=updates)
            write_json(directory / "run.json", run)
            return
    selected = select_checkpoint(history)
    best = {
        "schema_version": "kgr4vrd.checkpoint/1",
        "run_id": run["run_id"],
        "model": best_model,
        "identity": run["identity"],
        "input_dimension": dimension,
        "epoch": selected["epoch"],
    }
    atomic_write(directory / "checkpoints/best.pt", lambda stream: torch.save(best, stream))
    selection = {
        "schema_version": "kgr4vrd.selection/1",
        "run_id": run["run_id"],
        "checkpoint": "checkpoints/best.pt",
        "checkpoint_digest": file_digest(directory / "checkpoints/best.pt"),
        "metric": settings["selection_metric"],
        "value": selected["selection_score"],
        "epoch": selected["epoch"],
        "validation_split_digest": run["identity"]["inputs"]["split_digest"],
        "validation_target_digest": run["identity"]["inputs"]["partitions"]["validation"][
            "relations"
        ],
        "policy_digest": canonical_digest({"ranking": "multi_label", "filter": "none", "K": 50}),
        "tie_rule": "earliest",
        "config": "resolved-config.json",
        "identity": run["identity"],
    }
    write_json(directory / "selected.json", selection)
    write_jsonl(directory / "events.jsonl", [{"event": "completed", "at": utc_now()}])
    names = (
        "resolved-config.json",
        "environment.json",
        "source.json",
        "epochs.jsonl",
        "events.jsonl",
        "resume-boundary.json",
        "checkpoints/latest.pt",
        "checkpoints/best.pt",
        "selected.json",
    )
    run.update(
        status="completed",
        completed_epochs=len(history),
        completed_updates=updates,
        peak_process_memory_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        * (1 if sys.platform == "darwin" else 1024),
        memory_scope="process peak including preparation; CPU resident memory",
        ended_at=utc_now(),
        outputs={name: file_digest(directory / name) for name in names},
    )
    write_json(directory / "run.json", run)
