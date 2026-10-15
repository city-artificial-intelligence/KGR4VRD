"""Selected-checkpoint inference, bounded raw-score shards and offline evaluation."""

import csv
import os
import time
from collections.abc import Iterator, Mapping
from pathlib import Path

import numpy as np
import torch

from .artifacts import (
    atomic_write,
    canonical_digest,
    file_digest,
    read_json,
    read_jsonl,
    source_snapshot,
    verify_file,
    verify_run,
    write_json,
    write_jsonl,
)
from .config import data_path, load_config
from .data import build_pairs, load_manifest, load_objects, load_relations, load_vocabulary
from .metrics import evaluate
from .models import build_model
from .training import SceneFeatures, _synchronise


def predict(
    selection_path: str | Path,
    *,
    split: str = "test",
    campaign: str | Path | None = None,
    data_root: str | Path | None = None,
) -> Path:
    """Save all P+1 logits for every input pair; no target relations are read."""
    selection_path = Path(selection_path).resolve()
    selection = read_json(selection_path)
    verify_run(selection_path.parent)
    frozen_binding = None
    if split == "test":
        if campaign is None:
            raise ValueError("Test prediction requires a frozen campaign and group selection")
        from .campaign import assert_test_selection

        frozen_binding = assert_test_selection(Path(campaign), selection_path)
    config = load_config(selection_path.parent / selection["config"], data_root=data_root)
    dataset_path = data_path(config, config["dataset"]["manifest"])
    dataset, vocabulary = load_manifest(dataset_path), load_vocabulary(dataset_path)
    inputs = selection["identity"]["inputs"]
    if frozen_binding is not None:
        from .campaign import load_campaign

        assert campaign is not None
        frozen_plan = load_campaign(campaign, frozen=True)
        if file_digest(dataset_path) != frozen_plan["inputs"]["dataset"]:
            raise ValueError("Prediction dataset differs from the frozen analysis plan")
    if config["model"]["name"] == "visual_mlp_v1":
        for key, field in (
            ("visual_features", "features_manifest"),
            ("visual_weights", "weights_manifest"),
        ):
            if file_digest(data_path(config, config["visual"][field])) != inputs["knowledge"][key]:
                raise ValueError("Visual features or weights changed since checkpoint selection")
        feature_manifest = read_json(data_path(config, config["visual"]["features_manifest"]))
        if feature_manifest["identity"]["image_manifest_digest"] != canonical_digest(
            dataset["image_manifest"]
        ):
            raise ValueError("Visual feature cache uses a different source image inventory")
    if (
        vocabulary.digest != inputs["vocabulary_digest"]
        or dataset["split_digest"] != inputs["split_digest"]
    ):
        raise ValueError("Checkpoint vocabulary or split mismatch")
    scenes = load_objects(dataset_path, split)
    checkpoint_path = selection_path.parent / selection["checkpoint"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["identity"] != selection["identity"] or checkpoint["epoch"] != selection["epoch"]:
        raise ValueError("Selected checkpoint identity mismatch")
    identity = {
        "checkpoint_digest": selection["checkpoint_digest"],
        "split": split,
        "objects_digest": dataset["partitions"][split]["objects"]["sha256"],
        "vocabulary_digest": vocabulary.digest,
        "split_digest": dataset["split_digest"],
        "feature_profile": config["model"]["feature_profile"],
        "feature_digest": inputs["knowledge"].get("visual_features"),
        "source_digest": source_snapshot()["digest"],
    }
    if frozen_binding is not None:
        identity["plan_digest"] = frozen_binding["plan_digest"]
    prediction_id = canonical_digest(identity)
    # Selections travel with runs/<id>; keep new scores in a relocated artifact tree.
    directory = selection_path.parent.parent.parent / "predictions" / prediction_id
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        existing = ScoreShards(manifest_path, scenes)
        for key in existing:
            existing[key]
        return manifest_path
    device = torch.device(config["training"]["device"])
    model = build_model(
        config["model"]["name"], checkpoint["input_dimension"], len(vocabulary.predicates)
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    directory.mkdir(parents=True, exist_ok=True)
    shards, index = [], []
    _synchronise(device)
    started = time.perf_counter()
    inputs_by_image = SceneFeatures(config, scenes, vocabulary)
    with torch.inference_mode():
        for number, (image_id, scene) in enumerate(scenes.items()):
            logits = model(inputs_by_image[image_id].to(device)).cpu().numpy()
            if not np.isfinite(logits).all():
                raise FloatingPointError(f"Non-finite inference logits for {image_id}")
            pairs = build_pairs(scene)
            object_ids = sorted(obj.object_id for obj in scene.objects)
            positions = {value: i for i, value in enumerate(object_ids)}
            pair_indices = np.asarray(
                [(positions[s], positions[o]) for s, o in pairs], dtype=np.int64
            ).reshape(-1, 2)
            name = f"scores-{number:05d}.npz"
            atomic_write(
                directory / name,
                lambda stream: np.savez_compressed(
                    stream,
                    image_ids=np.array([image_id], dtype=np.str_),
                    pair_offsets=np.array([0, len(pairs)], dtype=np.int64),
                    pair_indices=pair_indices,
                    logits=logits.astype(np.float32),
                ),
            )
            shards.append(
                {"path": name, "sha256": file_digest(directory / name), "image_ids": [image_id]}
            )
            index.append({"image_id": image_id, "shard": name, "start": 0, "end": len(pairs)})
    _synchronise(device)
    write_jsonl(directory / "index.jsonl", index)
    manifest = {
        "schema_version": "kgr4vrd.predictions/1",
        "status": "completed",
        "prediction_id": prediction_id,
        "identity": identity,
        "run_id": selection["run_id"],
        "selection": os.path.relpath(selection_path, directory),
        "split": split,
        "expected_image_ids": list(scenes),
        "predicate_count": len(vocabulary.predicates),
        "logits_dtype": "float32",
        "pair_order": "subject_id,object_id lexicographic",
        "shards": shards,
        "index": {"path": "index.jsonl", "sha256": file_digest(directory / "index.jsonl")},
        "inference_seconds": time.perf_counter() - started,
    }
    write_json(manifest_path, manifest)
    return manifest_path


class ScoreShards(Mapping[str, np.ndarray]):
    """Read one verified image at a time, keeping score memory bounded."""

    def __init__(self, manifest_path: str | Path, scenes: dict):
        self.path = Path(manifest_path)
        self.manifest = read_json(self.path)
        manifest = self.manifest
        if (
            manifest.get("schema_version") != "kgr4vrd.predictions/1"
            or manifest.get("status") != "completed"
        ):
            raise ValueError("Incomplete or unsupported prediction artifact")
        if canonical_digest(manifest["identity"]) != manifest["prediction_id"]:
            raise ValueError("Prediction identity mismatch")
        ids = manifest["expected_image_ids"]
        if len(ids) != len(set(ids)) or set(ids) != set(scenes):
            raise ValueError("Prediction image membership mismatch")
        index = read_jsonl(verify_file(self.path.parent, manifest["index"]))
        if len(index) != len(ids) or {r["image_id"] for r in index} != set(ids):
            raise ValueError("Incomplete or duplicate prediction index")
        self.records = {row["image_id"]: row for row in index}
        self.shards = {row["path"]: row for row in manifest["shards"]}
        if (
            len(self.shards) != len(manifest["shards"])
            or len(self.shards) != len(ids)
            or {r["shard"] for r in index} != set(self.shards)
        ):
            raise ValueError("Missing or duplicate score shard")
        self.scenes = scenes

    def __iter__(self) -> Iterator[str]:
        return iter(self.manifest["expected_image_ids"])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, key: str) -> np.ndarray:
        row, scene = self.records[key], self.scenes[key]
        path = verify_file(self.path.parent, self.shards[row["shard"]])
        pairs = build_pairs(scene)
        ids = sorted(obj.object_id for obj in scene.objects)
        expected_pairs = np.array(
            [(ids.index(s), ids.index(o)) for s, o in pairs], dtype=np.int64
        ).reshape(-1, 2)
        with np.load(path, allow_pickle=False) as shard:
            logits = shard["logits"]
            if (
                shard["image_ids"].dtype.kind != "U"
                or shard["pair_offsets"].dtype != np.int64
                or shard["pair_indices"].dtype != np.int64
                or shard["image_ids"].tolist() != [key]
                or shard["pair_offsets"].tolist() != [0, len(pairs)]
                or not np.array_equal(shard["pair_indices"], expected_pairs)
                or logits.shape != (len(pairs), self.manifest["predicate_count"] + 1)
                or logits.dtype != np.float32
                or not np.isfinite(logits).all()
                or (row["start"], row["end"]) != (0, len(pairs))
            ):
                raise ValueError(f"Corrupt or misaligned score shard: {key}")
            return logits


def write_csv(path: Path, rows: list[dict]) -> None:
    import io

    fields = list(dict.fromkeys(key for row in rows for key in row))
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(path, lambda target: target.write(stream.getvalue().encode()))


def evaluate_predictions(
    predictions: str | Path,
    *,
    campaign: str | Path | None = None,
    filter_mode: str = "none",
    data_root: str | Path | None = None,
) -> Path:
    """Re-evaluate immutable scores without training or a live reasoner."""
    predictions = Path(predictions).resolve()
    manifest = read_json(predictions)
    selection_path = (predictions.parent / manifest["selection"]).resolve()
    selection = read_json(selection_path)
    verify_run(selection_path.parent)
    frozen_binding = None
    if manifest["split"] == "test":
        if campaign is None:
            raise ValueError("Test evaluation requires frozen campaign selection")
        from .campaign import assert_test_selection

        frozen_binding = assert_test_selection(Path(campaign), selection_path)
    config = load_config(selection_path.parent / selection["config"], data_root=data_root)
    if filter_mode == "simple_asserted" and config["intervention"]["condition"] != "baseline":
        raise ValueError("Simple filter is registered for baseline only")
    dataset_path = data_path(config, config["dataset"]["manifest"])
    dataset, vocabulary = load_manifest(dataset_path), load_vocabulary(dataset_path)
    if frozen_binding is not None:
        from .campaign import load_campaign

        assert campaign is not None
        frozen_plan = load_campaign(campaign, frozen=True)
        paths = {
            "dataset": dataset_path,
            "compatibility": data_path(config, config["intervention"]["knowledge_manifest"]),
        }
        if filter_mode == "simple_asserted":
            paths["simple_filter"] = data_path(
                config, config["evaluation"]["simple_filter_manifest"]
            )
        if any(file_digest(path) != frozen_plan["inputs"][key] for key, path in paths.items()):
            raise ValueError("Evaluation inputs differ from the frozen analysis plan")
        if manifest["identity"].get("plan_digest") != frozen_binding["plan_digest"]:
            raise ValueError("Predictions were not bound to this frozen analysis plan")
    split = manifest["split"]
    if (
        manifest["identity"]["vocabulary_digest"] != vocabulary.digest
        or manifest["identity"]["objects_digest"]
        != dataset["partitions"][split]["objects"]["sha256"]
        or manifest["identity"]["checkpoint_digest"] != selection["checkpoint_digest"]
    ):
        raise ValueError("Prediction input/checkpoint identity mismatch")
    scenes = load_objects(dataset_path, split)
    targets = load_relations(dataset_path, split)
    from .reasoning import load_knowledge, load_simple_filter

    table = load_knowledge(
        data_path(config, config["intervention"]["knowledge_manifest"]), vocabulary
    )
    decisions = {key: table.lookup(scene) for key, scene in scenes.items()}
    simple = None
    simple_digest = None
    if filter_mode == "simple_asserted":
        simple_table = load_simple_filter(
            data_path(config, config["evaluation"]["simple_filter_manifest"]), vocabulary
        )
        simple, simple_digest = (
            {key: simple_table.lookup(scene) for key, scene in scenes.items()},
            simple_table.digest,
        )
    scores = ScoreShards(predictions, scenes)
    result = evaluate(
        tuple(scenes),
        targets,
        scores,
        scenes,
        len(vocabulary.predicates),
        decisions=decisions,
        filter_mode=filter_mode,
        simple_masks=simple,
    )
    identity = {
        "prediction_id": manifest["prediction_id"],
        "target_digest": dataset["partitions"][split]["relations"]["sha256"],
        "knowledge_digest": table.digest,
        "simple_digest": simple_digest,
        "policy": result["policy"],
        "source_digest": source_snapshot()["digest"],
    }
    evaluation_id = canonical_digest(identity)
    directory = predictions.parent.parent.parent / "evaluations" / evaluation_id
    if (directory / "evaluation.json").exists():
        previous = read_json(directory / "evaluation.json")
        for name, digest in previous["outputs"].items():
            if file_digest(directory / name) != digest:
                raise ValueError(f"Corrupted accepted evaluation: {name}")
        return directory / "evaluation.json"
    directory.mkdir(parents=True, exist_ok=True)
    for table_name in ("per_image", "per_predicate"):
        rows = [
            {**row, "run_id": manifest["run_id"], "evaluation_id": evaluation_id}
            for row in result[table_name]
        ]
        write_csv(directory / f"{table_name}.csv", rows)
    ranked = result["ranked_predictions"]
    if isinstance(ranked, dict):
        ranked = [{"image_id": key, "predictions": values} for key, values in ranked.items()]
    write_jsonl(directory / "ranked_predictions.jsonl", ranked)
    write_json(
        directory / "metrics.json",
        {key: result[key] for key in ("metrics", "metric_details", "diagnostics")},
    )
    names = ("per_image.csv", "per_predicate.csv", "ranked_predictions.jsonl", "metrics.json")
    record = {
        "schema_version": "kgr4vrd.evaluation/1",
        "status": "completed",
        "evaluation_id": evaluation_id,
        "identity": identity,
        "run_id": manifest["run_id"],
        "checkpoint_digest": selection["checkpoint_digest"],
        "model": config["model"]["name"],
        "condition": config["intervention"]["condition"],
        "seed": config["training"]["seed"],
        "dataset": dataset["dataset_id"],
        "split": split,
        "target_digest": identity["target_digest"],
        "vocabulary_digest": vocabulary.digest,
        "split_digest": dataset["split_digest"],
        "knowledge_digest": table.digest,
        "filter_mode": filter_mode,
        "policy": result["policy"],
        "metrics": "metrics.json",
        "expected_image_count": len(scenes),
        "actual_image_count": len(result["per_image"]),
        "predictions": os.path.relpath(predictions, directory),
        "selection": os.path.relpath(selection_path, directory),
        "plan_digest": None if frozen_binding is None else frozen_binding["plan_digest"],
        "outputs": {name: file_digest(directory / name) for name in names},
    }
    write_json(directory / "evaluation.json", record)
    return directory / "evaluation.json"
