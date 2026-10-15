"""Explicit partition loaders and deterministic pair/target construction."""

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .artifacts import canonical_digest, file_digest, read_json, read_jsonl
from .types import Object, Relation, Scene, Vocabulary

SPLITS = ("train", "validation", "test")


def build_pairs(scene: Scene) -> tuple[tuple[str, str], ...]:
    """Return every distinct ordered pair, lexicographically by object IDs."""
    if any(not isinstance(obj.object_id, str) or not obj.object_id for obj in scene.objects):
        raise ValueError(f"{scene.image_id}: object IDs must be nonempty strings")
    ids = sorted(obj.object_id for obj in scene.objects)
    if len(ids) != len(set(ids)):
        raise ValueError(f"{scene.image_id}: duplicate object IDs")
    return tuple((subject, obj) for subject in ids for obj in ids if subject != obj)


def build_targets(
    pairs: tuple[tuple[str, str], ...], relations: Iterable[Relation], predicate_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Multi-hot [N,P+1] float32 labels and [N] Boolean supervised-row mask."""
    if type(predicate_count) is not int or predicate_count < 1 or len(set(pairs)) != len(pairs):
        raise ValueError("Targets require a positive predicate count and unique pairs")
    if any(
        len(pair) != 2
        or pair[0] == pair[1]
        or any(not isinstance(value, str) or not value for value in pair)
        for pair in pairs
    ):
        raise ValueError("Pair IDs must contain distinct nonempty subject/object strings")
    targets = np.zeros((len(pairs), predicate_count + 1), dtype=np.float32)
    pair_rows = {pair: index for index, pair in enumerate(pairs)}
    for relation in relations:
        pair = (relation.subject_id, relation.object_id)
        if (
            any(not isinstance(value, str) or not value for value in pair)
            or type(relation.predicate_id) is not int
            or pair not in pair_rows
            or not 0 <= relation.predicate_id < predicate_count
        ):
            raise ValueError(f"Relation outside pair/vocabulary identity: {relation}")
        targets[pair_rows[pair], relation.predicate_id] = 1
    ordinary_rows = targets[:, :predicate_count].any(axis=1)
    targets[:, predicate_count] = ~ordinary_rows
    return targets, ordinary_rows


def component_path(manifest_path: Path, entry: dict[str, Any]) -> Path:
    """Resolve and verify one component without accessing other partitions."""
    path = manifest_path.parent / entry["path"]
    if file_digest(path) != entry["sha256"]:
        raise ValueError(f"Artifact checksum mismatch: {path}")
    return path


def load_manifest(manifest_path: str | Path) -> dict[str, Any]:
    manifest = read_json(Path(manifest_path))
    if manifest.get("schema_version") != "kgr4vrd.dataset/1":
        raise ValueError("Unsupported dataset schema")
    if manifest.get("coordinate_convention") != "xyxy_pixel_continuous":
        raise ValueError("Unsupported coordinate convention")
    if set(manifest["partitions"]) != set(SPLITS):
        raise ValueError("Dataset must declare train, validation and test partitions")
    return manifest


def load_split_ids(manifest_path: str | Path) -> dict[str, tuple[str, ...]]:
    path = Path(manifest_path)
    manifest = load_manifest(path)
    splits = read_json(component_path(path, manifest["splits"]))
    if splits.get("schema_version") != "kgr4vrd.splits/1":
        raise ValueError("Unsupported split schema")
    if canonical_digest(splits) != manifest["split_digest"]:
        raise ValueError("Split identity mismatch")
    result = {split: tuple(splits[split]) for split in SPLITS}
    all_ids = [image_id for ids in result.values() for image_id in ids]
    if any(not isinstance(image_id, str) or not image_id for image_id in all_ids):
        raise ValueError("Image IDs must be nonempty strings")
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Split lists contain duplicate or overlapping image IDs")
    return result


def load_vocabulary(manifest_path: str | Path) -> Vocabulary:
    path = Path(manifest_path)
    manifest = load_manifest(path)
    record = read_json(component_path(path, manifest["vocabulary"]))
    if record.get("schema_version") != "kgr4vrd.vocabulary/1":
        raise ValueError("Unsupported vocabulary schema")
    for kind in ("object_classes", "predicates"):
        entries = record[kind]
        if not entries or [entry["id"] for entry in entries] != list(range(len(entries))):
            raise ValueError(f"{kind}: vocabulary IDs must be dense and ordered")
        if any(
            not isinstance(entry["label"], str) or not isinstance(entry["source_id"], str)
            for entry in entries
        ):
            raise ValueError(f"{kind}: invalid vocabulary label or source ID")
    vocabulary = Vocabulary(tuple(record["object_classes"]), tuple(record["predicates"]))
    if vocabulary.digest != manifest["vocabulary_digest"]:
        raise ValueError("Vocabulary identity mismatch")
    return vocabulary


def validate_scene(scene: Scene, class_count: int) -> None:
    """Reject malformed canonical inputs before features or training see them."""
    if (
        not isinstance(scene.image_id, str)
        or not scene.image_id
        or type(scene.width) is not int
        or type(scene.height) is not int
        or scene.width <= 0
        or scene.height <= 0
    ):
        raise ValueError(f"{scene.image_id}: invalid image identity/dimensions")
    build_pairs(scene)
    for obj in scene.objects:
        if (
            not obj.object_id
            or type(obj.class_id) is not int
            or not 0 <= obj.class_id < class_count
        ):
            raise ValueError(f"{scene.image_id}/{obj.object_id}: invalid class ID")
        if (
            len(obj.bbox) != 4
            or any(type(value) not in (int, float) for value in obj.bbox)
            or not np.isfinite(obj.bbox).all()
        ):
            raise ValueError(f"{scene.image_id}/{obj.object_id}: non-finite or malformed box")
        xmin, ymin, xmax, ymax = obj.bbox
        if not (0 <= xmin < xmax <= scene.width and 0 <= ymin < ymax <= scene.height):
            raise ValueError(f"{scene.image_id}/{obj.object_id}: box outside image or empty")


def _partition_records(manifest_path: Path, split: str, kind: str) -> list[dict[str, Any]]:
    if split not in SPLITS:
        raise ValueError(f"Unknown explicit partition: {split}")
    manifest = load_manifest(manifest_path)
    expected = load_split_ids(manifest_path)[split]
    records = list(read_jsonl(component_path(manifest_path, manifest["partitions"][split][kind])))
    ids = [record["image_id"] for record in records]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError(f"{split}/{kind}: records do not match exact split membership")
    if len(ids) != manifest["partitions"][split]["image_count"]:
        raise ValueError(f"{split}/{kind}: image count mismatch")
    by_id = {record["image_id"]: record for record in records}
    return [by_id[image_id] for image_id in expected]


def load_objects(manifest_path: str | Path, split: str) -> dict[str, Scene]:
    """Read supplied inputs for exactly one partition; never open relation files."""
    path = Path(manifest_path)
    vocabulary = load_vocabulary(path)
    scenes = {}
    for record in _partition_records(path, split, "objects"):
        scene = Scene(
            record["image_id"],
            record["image_path"],
            record["width"],
            record["height"],
            tuple(
                Object(obj["object_id"], obj["class_id"], tuple(obj["bbox"]))
                for obj in record["objects"]
            ),
        )
        validate_scene(scene, len(vocabulary.object_classes))
        scenes[scene.image_id] = scene
    return scenes


def load_relations(manifest_path: str | Path, split: str) -> dict[str, tuple[Relation, ...]]:
    """Open targets only for the caller's explicit partition and verify references."""
    path = Path(manifest_path)
    scenes = load_objects(path, split)
    predicate_count = len(load_vocabulary(path).predicates)
    result = {}
    for record in _partition_records(path, split, "relations"):
        relations = tuple(Relation(**item) for item in record["relations"])
        build_targets(build_pairs(scenes[record["image_id"]]), relations, predicate_count)
        result[record["image_id"]] = tuple(sorted(set(relations)))
    return result
