"""Tiny deterministic synthetic data and the complete offline integration example."""

import copy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from rdflib import OWL, RDF, Graph, URIRef

from .artifacts import atomic_write, canonical_digest, file_digest, write_json, write_jsonl
from .config import validate_config
from .reasoning import CompatibilityTable, construct_profile, save_knowledge
from .types import Object, Relation, Scene, Vocabulary


def prepare_smoke(output: str | Path) -> dict:
    """Create clearly synthetic immutable inputs, including one empty-pair image."""
    root = Path(output).resolve()
    dataset_root = root / "data/prepared/synthetic"
    if (dataset_root / "dataset.json").exists():
        raise ValueError(f"Use a new smoke output directory: {root}")
    vocabulary = Vocabulary(
        tuple(
            {"id": i, "label": name, "source_id": str(i)}
            for i, name in enumerate(("Person", "Horse", "Car"))
        ),
        tuple(
            {"id": i, "label": name, "source_id": str(i)}
            for i, name in enumerate(("near", "rides", "left", "unmapped"))
        ),
    )
    write_json(
        dataset_root / "vocabulary.json",
        {
            "schema_version": "kgr4vrd.vocabulary/1",
            "object_classes": vocabulary.object_classes,
            "predicates": vocabulary.predicates,
        },
    )
    split_ids = {
        "train": ["train-a", "train-b", "train-empty"],
        "validation": ["validation-a", "validation-b"],
        "test": ["test-a"],
    }
    splits = {
        "schema_version": "kgr4vrd.splits/1",
        **split_ids,
        "provenance": "fixed synthetic integration fixture",
    }
    write_json(dataset_root / "splits.json", splits)
    partitions: dict = {}
    expanded: list[dict] = []
    for split, ids in split_ids.items():
        objects, relations = [], []
        for index, image_id in enumerate(ids):
            supplied = tuple(
                Object(f"o{i}", i % 3, (float(5 + 20 * i), 10.0, float(20 + 20 * i), 50.0))
                for i in range(4)
            )
            if image_id.endswith("empty"):
                supplied = supplied[:1]
            scene = Scene(image_id, f"synthetic/{image_id}.png", 100, 80, supplied)
            truth = (
                (Relation("o0", 0, "o1"), Relation("o0", 1, "o1"), Relation("o1", 2, "o2"))
                if len(supplied) > 1
                else ()
            )
            objects.append(asdict(scene))
            relations.append({"image_id": image_id, "relations": [asdict(r) for r in truth]})
            if split == "train":
                added = (Relation("o1", 0, "o0"),) if truth else ()
                expanded.append(
                    {"image_id": image_id, "relations": [asdict(r) for r in (*truth, *added)]}
                )
        write_jsonl(dataset_root / f"objects/{split}.jsonl", objects)
        write_jsonl(dataset_root / f"relations/{split}.jsonl", relations)
        partitions[split] = {
            kind: {
                "path": f"{kind}/{split}.jsonl",
                "sha256": file_digest(dataset_root / f"{kind}/{split}.jsonl"),
            }
            for kind in ("objects", "relations")
        }
        partitions[split].update(
            image_count=len(ids),
            object_count=sum(len(r["objects"]) for r in objects),
            pair_count=sum(len(r["objects"]) * (len(r["objects"]) - 1) for r in objects),
            relation_count=sum(len(r["relations"]) for r in relations),
        )
    manifest = {
        "schema_version": "kgr4vrd.dataset/1",
        "dataset_id": "synthetic",
        "dataset_version": "1",
        "coordinate_convention": "xyxy_pixel_continuous",
        "vocabulary_digest": vocabulary.digest,
        "split_digest": canonical_digest(splits),
        "partitions": partitions,
        "vocabulary": {
            "path": "vocabulary.json",
            "sha256": file_digest(dataset_root / "vocabulary.json"),
        },
        "splits": {"path": "splits.json", "sha256": file_digest(dataset_root / "splits.json")},
        "image_root": ".",
        "source_files": [],
        "adapter_version": "embedded-synthetic/1",
        "metadata": {"evidence": "synthetic integration only; no real images or judgments"},
    }
    write_json(dataset_root / "dataset.json", manifest)
    decisions = np.zeros((3, 4, 3), dtype=np.int8)
    decisions[:, 1, 2] = 1
    decisions[:, 3, :] = -1
    knowledge_root = root / "data/knowledge"
    save_knowledge(
        knowledge_root / "compatibility/manifest.json",
        CompatibilityTable(
            decisions,
            vocabulary.digest,
            {
                "policy": "embedded_synthetic",
                "query_scope": "independent_typed_relation",
                "unknown_reason": "unmapped synthetic predicate 3",
            },
        ),
        vocabulary,
    )
    save_knowledge(
        knowledge_root / "simple-filter/manifest.json",
        CompatibilityTable(
            decisions == 1,
            vocabulary.digest,
            {"policy": "simple_asserted_mask_v1", "evidence": "synthetic"},
        ),
        vocabulary,
    )
    augmentation = knowledge_root / "augmentation/S_TEU"
    write_jsonl(augmentation / "relations.jsonl", expanded)
    ontology = Graph()
    near = URIRef("urn:kgr4vrd:synthetic:near")
    ontology.add((near, RDF.type, OWL.SymmetricProperty))
    _, profile = construct_profile(ontology, [str(near)], "S_TEU")
    atomic_write(
        augmentation / "ontology.ttl",
        lambda stream: stream.write(ontology.serialize(format="turtle").encode()),
    )
    write_json(
        augmentation / "manifest.json",
        {
            "schema_version": "kgr4vrd.augmentation/1",
            "status": "completed",
            "vocabulary_digest": vocabulary.digest,
            "training_relations_digest": partitions["train"]["relations"]["sha256"],
            "training_objects_digest": partitions["train"]["objects"]["sha256"],
            **profile,
            "source_sha256": file_digest(augmentation / "ontology.ttl"),
            "relations": {
                "path": "relations.jsonl",
                "sha256": file_digest(augmentation / "relations.jsonl"),
            },
        },
    )
    config = validate_config(
        {
            "protocol_id": "synthetic-integration-v1",
            "paths": {"data_root": str(root / "data"), "artifact_root": str(root / "artifacts")},
            "dataset": {
                "name": "synthetic",
                "manifest": "prepared/synthetic/dataset.json",
                "split": "prepared/synthetic/splits.json",
            },
            "training": {"max_epochs": 4, "patience": 25},
        }
    )
    write_json(root / "smoke-config.json", config)
    return config


def smoke(output: str | Path) -> Path:
    from .prediction import evaluate_predictions, predict
    from .reporting import report_evaluations
    from .training import train

    config = prepare_smoke(output)
    torch.set_num_threads(1)
    evaluations: list[str | Path] = []
    for condition in ("baseline", "augment", "semantic_loss", "combined"):
        current = copy.deepcopy(config)
        current["intervention"]["condition"] = condition
        run = train(current)
        scores = predict(run / "selected.json", split="validation")
        for filter_mode in ("none", "ontology"):
            evaluations.append(evaluate_predictions(scores, filter_mode=filter_mode))
    return report_evaluations(evaluations, Path(output) / "report")
