"""Measure isolated pilots around the frozen trainer; never select campaign runs.

Run each case in a fresh numeric Slurm step process. The meter wraps feature
access only: it synchronizes/resets CUDA counters without changing tensors, RNG,
losses, optimizer updates, or the package's training/validation implementation.
"""

import argparse
import fcntl
import os
import resource
import shutil
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from kgr4vrd import training
from kgr4vrd.adapters.vision import crop_layout
from kgr4vrd.artifacts import (
    file_digest,
    read_json,
    read_jsonl,
    source_snapshot,
    utc_now,
    verify_file,
    verify_run,
    write_json,
)
from kgr4vrd.config import data_path, load_config
from kgr4vrd.data import build_pairs, load_manifest, load_objects, load_vocabulary
from kgr4vrd.prediction import evaluate_predictions, predict
from kgr4vrd.reasoning import load_knowledge, load_simple_filter, validate_profile_metadata

MODELS = ("geometry_mlp", "visual_mlp_v1")


def tree_bytes(path):
    """Count allocated logical file bytes once per inode (checkpoint aliases exist)."""
    seen, total = set(), 0
    for item in Path(path).rglob("*"):
        if item.is_file():
            stat = item.stat()
            identity = (stat.st_dev, stat.st_ino)
            if identity not in seen:
                total += stat.st_size
                seen.add(identity)
    return total


class CudaMeter:
    """Attribute an image's forward/backward/update through the next input access."""

    def __init__(self, train_ids, device="cuda"):
        self.train_ids = set(train_ids)
        self.device = torch.device(device)
        self.enabled = False
        self.current = None
        self.rows = []
        self.setup = None

    def finish(self):
        torch.cuda.synchronize(self.device)
        value = {
            "allocated_bytes": torch.cuda.max_memory_allocated(self.device),
            "reserved_bytes": torch.cuda.max_memory_reserved(self.device),
        }
        if self.current is not None:
            self.rows.append({**self.current, **value})
        elif self.enabled:
            self.setup = value
        self.current = None

    def begin(self, features, key):
        self.finish()
        torch.cuda.reset_peak_memory_stats(self.device)
        scene = features.scenes[key]
        self.current = {
            "image_id": key,
            "split": "train" if key in self.train_ids else "validation",
            "pairs": len(build_pairs(scene)),
            "objects": len(scene.objects),
        }

    def summary(self):
        if not self.rows:
            raise ValueError("No CUDA image measurements")
        result = {}
        for split in ("train", "validation"):
            rows = [row for row in self.rows if row["split"] == split]
            largest = max(row["pairs"] for row in rows)
            candidates = [row for row in rows if row["pairs"] == largest]
            result[split] = {
                "image_visits": len(rows),
                "largest_image_pairs": largest,
                "largest_image_ids": sorted({r["image_id"] for r in candidates}),
                "largest_image_allocated_bytes": max(r["allocated_bytes"] for r in candidates),
                "largest_image_reserved_bytes": max(r["reserved_bytes"] for r in candidates),
                "peak_allocated_bytes": max(r["allocated_bytes"] for r in rows),
                "peak_reserved_bytes": max(r["reserved_bytes"] for r in rows),
            }
        return {"setup": self.setup, "splits": result}


@contextmanager
def measure_training(meter):
    """Install observers only for this isolated process and restore on every exit."""
    original_get = training.SceneFeatures.__getitem__
    original_build = training.build_model

    def measured_get(features, key):
        if meter.enabled:
            meter.begin(features, key)
        return original_get(features, key)

    def measured_build(*args, **kwargs):
        model = original_build(*args, **kwargs)
        meter.enabled = True
        return model

    training.SceneFeatures.__getitem__ = measured_get
    training.build_model = measured_build
    try:
        yield meter
    finally:
        training.SceneFeatures.__getitem__ = original_get
        training.build_model = original_build
        if meter.enabled:
            meter.finish()


def inventory(protocol):
    config = load_config(protocol["base_config"])
    dataset_path = data_path(config, config["dataset"]["manifest"])
    dataset, vocabulary = load_manifest(dataset_path), load_vocabulary(dataset_path)
    knowledge_path = data_path(config, config["intervention"]["knowledge_manifest"])
    table = load_knowledge(knowledge_path, vocabulary)
    conformance = read_json(knowledge_path.parent / "conformance.json")
    conflicts = read_json(knowledge_path.parent / "training-conflicts.json")
    if conformance["status"] != "passed" or conformance["sample_count"] < 200:
        raise ValueError("GraphDB conformance not accepted")
    if conflicts["conflicts"]:
        raise ValueError("Unresolved raw training conflicts")
    augmentation_path = data_path(config, config["intervention"]["augmentation_manifest"])
    augmentation = read_json(augmentation_path)
    validate_profile_metadata(augmentation, "S_TEU")
    control_path = data_path(config, config["intervention"]["control_manifest"])
    control = read_json(control_path)
    for artifact in (augmentation, control):
        if artifact["status"] != "completed":
            raise ValueError("Incomplete augmentation/control")
        for kind in ("objects", "relations"):
            if (
                artifact[f"training_{kind}_digest"]
                != dataset["partitions"]["train"][kind]["sha256"]
            ):
                raise ValueError("Prepared training provenance mismatch")
    for field in ("relations", "derivations"):
        verify_file(augmentation_path.parent, augmentation[field])
    simple_path = data_path(config, config["evaluation"]["simple_filter_manifest"])
    simple = load_simple_filter(simple_path, vocabulary)
    counts = {}
    for split in ("train", "validation", "test"):
        scenes = load_objects(dataset_path, split)  # Supplied inputs only; no test labels.
        rows = []
        decisions = {"unknown": 0, "compatible": 0, "incompatible": 0}
        for key, scene in scenes.items():
            values = table.lookup(scene)
            for name, code in (("unknown", -1), ("compatible", 0), ("incompatible", 1)):
                decisions[name] += int(np.count_nonzero(values == code))
            rows.append(
                {
                    "image_id": key,
                    "objects": len(scene.objects),
                    "pairs": len(build_pairs(scene)),
                    "crops": len(crop_layout(scene)[0]),
                }
            )
        counts[split] = {
            "images": len(rows),
            "objects": sum(r["objects"] for r in rows),
            "pairs": sum(r["pairs"] for r in rows),
            "unique_crops": sum(r["crops"] for r in rows),
            "largest_image": max(rows, key=lambda r: (r["pairs"], r["image_id"])),
            "decisions": decisions,
        }
        if counts[split]["pairs"] != dataset["partitions"][split]["pair_count"]:
            raise ValueError("Pair count differs from accepted manifest")
    feature_path = data_path(config, config["visual"]["features_manifest"])
    features = read_json(feature_path)
    if features["status"] != "completed":
        raise ValueError("Feature cache is incomplete")
    attempts = []
    for path in protocol["preparation_receipts"]:
        record = read_json(path)
        attempts.append({"path": path, "sha256": file_digest(path), **record})
    total_crops = sum(row["unique_crops"] for row in counts.values())
    cumulative_seconds = sum(row["elapsed_seconds"] for row in attempts)
    result = {
        "status": "completed",
        "recorded_at": utc_now(),
        "counts": counts,
        "class_count": len(vocabulary.object_classes),
        "predicate_count": len(vocabulary.predicates),
        "raw_score_payload_bytes_per_test_run": 4
        * counts["test"]["pairs"]
        * (len(vocabulary.predicates) + 1),
        "feature_extraction": {
            "seconds": features["seconds"],
            "unique_crops": total_crops,
            "crops_per_second": total_crops / features["seconds"],
        },
        "preparation_attempts": attempts,
        "cumulative_preparation_seconds": cumulative_seconds,
        "query_throughput": read_json(protocol["query_benchmark"])["comparison"],
        "production_query_window": read_json(protocol["query_window"]),
        "query_throughput_note": "Measured benchmark includes insertion/query/cleanup. Final preparation resumed all queries; its elapsed time is not fresh query throughput.",
        "reasoning": {
            "query_count": len(table.metadata["decision_evidence"]),
            "unknown_typed_queries": table.metadata["unknown_count"],
            "conformance_samples": conformance["sample_count"],
            "training_conflicts": 0,
            "scope": conformance["scope"],
            "S_TEU_profile_digest": augmentation["profile_digest"],
            "C1_blocked_typed_entries": int(simple.decisions.sum()),
            "C2": {key: control[key] for key in ("selected_condition", "unmatched_fraction")},
        },
        "storage_bytes": {
            name: tree_bytes(path) for name, path in protocol["storage_roots"].items()
        },
        "free_repository_bytes": shutil.disk_usage(protocol["directory"]).free,
        "input_bindings": protocol["input_bindings"],
    }
    write_json(Path(protocol["directory"]) / "inventory.json", result)
    return result


def run_case(protocol, key):
    case = next(row for row in protocol["cases"] if row["key"] == key)
    config = load_config(case["config"])
    output = Path(case["measurement"])
    if output.exists():
        raise FileExistsError("Pilot measurement already exists; preserve and review it")
    dataset_path = data_path(config, config["dataset"]["manifest"])
    train_ids = load_objects(dataset_path, "train")
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    meter = CudaMeter(train_ids)
    started = time.perf_counter()
    with measure_training(meter):
        directory = training.train(config)
    training_wall = time.perf_counter() - started
    verify_run(directory)
    history = read_jsonl(directory / "epochs.jsonl")
    if len(history) != case["epochs"]:
        raise ValueError("Pilot did not complete every requested train/validation epoch")
    memory = meter.summary()
    expected = read_json(Path(protocol["directory"]) / "inventory.json")["counts"]
    for split in ("train", "validation"):
        if memory["splits"][split]["image_visits"] != expected[split]["images"] * case["epochs"]:
            raise ValueError("Pilot meter missed train/validation images")
    if case["condition"] == "combined" and any(row["active_cells"] for row in history[:2]):
        raise ValueError("Semantic activation occurred before epoch 3")
    # Separate selected validation prediction measures the package's actual shard overhead.
    prediction_started = time.perf_counter()
    scores = predict(directory / "selected.json", split="validation")
    prediction_seconds = time.perf_counter() - prediction_started
    evaluations = []
    modes = ("none", "ontology", "simple_asserted") if case["condition"] == "baseline" else ()
    for mode in modes:
        evaluation_started = time.perf_counter()
        path = evaluate_predictions(scores, filter_mode=mode)
        evaluations.append(
            {
                "mode": mode,
                "path": str(path),
                "sha256": file_digest(path),
                "seconds": time.perf_counter() - evaluation_started,
                "bytes": tree_bytes(path.parent),
            }
        )
    record = {
        "status": "completed",
        "pilot_only": True,
        "key": key,
        "recorded_at": utc_now(),
        "model": config["model"]["name"],
        "condition": case["condition"],
        "config": case["config"],
        "config_sha256": file_digest(case["config"]),
        "run": str(directory),
        "run_sha256": file_digest(directory / "run.json"),
        "epochs": history,
        "cuda": memory,
        "host_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "training_wall_seconds": training_wall,
        "run_bytes": tree_bytes(directory),
        "checkpoint_bytes": tree_bytes(directory / "checkpoints"),
        "scores": str(scores),
        "scores_sha256": file_digest(scores),
        "score_bytes": tree_bytes(scores.parent),
        "prediction_wall_seconds": prediction_seconds,
        "evaluations": evaluations,
        "semantic_gate_epoch": 3 if case["condition"] == "combined" else None,
        "semantic_active_cells_epoch3": history[2]["active_cells"] if len(history) >= 3 else None,
        "measurement_scope": "CUDA peaks cover input transfer, forward, backward, optimizer and boundary work. Per-image synchronization adds overhead. RSS includes preparation and validation-only prediction/evaluation. No test relation labels or test scoring.",
        "source_digest": source_snapshot()["digest"],
        "hardware": {
            "device": torch.cuda.get_device_name(),
            "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        },
    }
    write_json(output, record)
    return record


def projections(inventory_record, records):
    """Measured scenarios, not a promise or an ETA-based termination policy."""
    lower = upper = 0.0
    run_storage = 0
    score_storage = evaluation_storage = 0.0
    ratio = (
        inventory_record["counts"]["test"]["pairs"]
        / inventory_record["counts"]["validation"]["pairs"]
    )
    for model in MODELS:
        cases = [row for row in records if row["model"] == model]
        epoch_times = [
            e["training_seconds"] + e["validation_seconds"] for c in cases for e in c["epochs"]
        ]
        startup = max(
            max(
                0,
                c["training_wall_seconds"]
                - sum(e["training_seconds"] + e["validation_seconds"] for e in c["epochs"]),
            )
            for c in cases
        )
        baseline = next(c for c in cases if c["condition"] == "baseline")
        # 4 development + 25 final runs per head; earliest stop is epoch 26.
        lower += 29 * (26 * min(epoch_times) + startup)
        upper += 29 * (100 * max(epoch_times) + startup)
        # 25 raw-score artifacts and 55 views per head; scale by actual candidate pairs.
        views = {r["mode"]: r for r in baseline["evaluations"]}
        prediction = baseline["prediction_wall_seconds"] * ratio * 25
        evaluation = (
            25 * views["none"]["seconds"]
            + 25 * views["ontology"]["seconds"]
            + 5 * views["simple_asserted"]["seconds"]
        ) * ratio
        lower += prediction + evaluation
        upper += prediction + evaluation
        run_storage += 29 * max(c["run_bytes"] for c in cases)
        score_storage += 25 * max(c["score_bytes"] for c in cases) * ratio
        evaluation_storage += (
            25 * views["none"]["bytes"]
            + 25 * views["ontology"]["bytes"]
            + 5 * views["simple_asserted"]["bytes"]
        ) * ratio
    retained = sum(inventory_record["storage_bytes"].values())
    storage = retained + run_storage + score_storage + evaluation_storage
    prep = inventory_record["cumulative_preparation_seconds"]
    return {
        "runtime_hours_range": [(prep + lower) / 3600, (prep + upper) / 3600],
        "remaining_runtime_hours_range": [lower / 3600, upper / 3600],
        "storage_gb_range": [
            storage / 1e9,
            (storage + run_storage + score_storage + evaluation_storage) / 1e9,
        ],
        "assumptions": "58 runs, 26-to-100 epochs (earliest stop: best epoch 1 plus patience 25). Epoch timing range from both heads' baseline and combined pilots, including gate epoch. Prediction/evaluation scale validation measurements by test/validation candidate pairs (50 scores, 110 views). Upper storage adds 100% run/score/evaluation reserve for longer logs and recovery. Retained preparation includes failed attempts' cumulative time. Report-generation/bootstrap runtime is unmeasured and excluded. Image complexity, cache warmth and stopping epoch affect cost; these are projections, never termination limits.",
    }


def freeze_pilot(protocol):
    root = Path(protocol["directory"])
    record_path = root / "frozen.json"
    if record_path.exists():
        raise FileExistsError("Measured pilot record is already frozen")
    inv = read_json(root / "inventory.json")
    records = []
    for case in protocol["cases"]:
        record = read_json(case["measurement"])
        if record["status"] != "completed" or record["source_digest"] != protocol["source_digest"]:
            raise ValueError("Incomplete or incompatible pilot measurement")
        verify_run(record["run"])
        if file_digest(Path(record["run"]) / "run.json") != record["run_sha256"]:
            raise ValueError("Pilot run changed")
        if file_digest(record["scores"]) != record["scores_sha256"]:
            raise ValueError("Pilot score manifest changed")
        records.append(record)
    projection = projections(inv, records)
    frozen = {
        "schema_version": "kgr4vrd.resource-pilot/1",
        "status": "frozen",
        "pilot_only": True,
        "recorded_at": utc_now(),
        "source_digest": protocol["source_digest"],
        "inventory": {
            "path": str(root / "inventory.json"),
            "sha256": file_digest(root / "inventory.json"),
        },
        "cases": [
            {"path": c["measurement"], "sha256": file_digest(c["measurement"])}
            for c in protocol["cases"]
        ],
        **projection,
    }
    campaign_path = Path(protocol["campaign"])
    with campaign_path.with_suffix(".json.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        campaign = read_json(campaign_path)
        if campaign["status"] != "draft" or campaign["pilot"]:
            raise ValueError("Pilot evidence must populate an unchanged draft plan")
        if (
            len(campaign["development_runs"]),
            len(campaign["runs"]),
            len(campaign["evaluations"]),
        ) != (8, 50, 110):
            raise ValueError("Campaign matrix changed")
        write_json(record_path, frozen)
        campaign["pilot"] = {
            "runtime_hours_range": projection["runtime_hours_range"],
            "storage_gb_range": projection["storage_gb_range"],
        }
        campaign.setdefault("metadata", {})["resource_pilot"] = {
            "path": str(record_path),
            "sha256": file_digest(record_path),
            "pilot_only": True,
        }
        write_json(campaign_path, campaign)
    return frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("inventory", "case", "freeze"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--key")
    args = parser.parse_args()
    protocol = read_json(args.protocol)
    if (
        os.environ.get("SLURM_JOB_ID") != protocol["allocation"]
        or not os.environ.get("SLURM_STEP_ID", "").isdigit()
    ):
        raise RuntimeError("Pilot requires its retained allocation and numeric Slurm step")
    if any(Path(p).exists() for p in protocol["pause_paths"]):
        raise RuntimeError("Campaign STOP/PAUSE prevents pilot work")
    if source_snapshot()["digest"] != protocol["source_digest"]:
        raise ValueError("Scientific source changed since acceptance")
    for binding in protocol["input_bindings"]:
        if file_digest(binding["path"]) != binding["sha256"]:
            raise ValueError("Pilot input binding changed: " + binding["path"])
    if args.stage == "inventory":
        inventory(protocol)
    elif args.stage == "case":
        run_case(protocol, args.key)
    else:
        freeze_pilot(protocol)


if __name__ == "__main__":
    main()
