"""Partition isolation, conversion and target invariants (T02/T08/T09/T17)."""

import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from kgr4vrd.adapters.nesy4vrd import prepare_dataset
from kgr4vrd.artifacts import canonical_digest, file_digest, read_json, write_json
from kgr4vrd.data import build_pairs, build_targets, load_objects, load_relations, load_vocabulary
from kgr4vrd.types import Object, Relation, Scene


def source_fixture(tmp_path):
    images = tmp_path / "raw" / "images"
    images.mkdir(parents=True)
    annotation = [
        {
            "subject": {"category": 0, "bbox": [0, 5, 0, 5]},
            "predicate": predicate,
            "object": {"category": 1, "bbox": [1, 8, 4, 9]},
        }
        for predicate in (0, 1)
    ]
    for index in range(6):
        Image.new("RGB", (10, 10), (index * 30, 0, 0)).save(images / f"{index}.png")
    write_json(tmp_path / "raw" / "objects.json", ["person", "horse"])
    write_json(tmp_path / "raw" / "predicates.json", ["ride", "near"])
    write_json(tmp_path / "raw" / "train.json", {f"{index}.png": annotation for index in range(5)})
    write_json(tmp_path / "raw" / "test.json", {"5.png": annotation})
    return {
        "paths": {"data_root": str(tmp_path)},
        "preparation": {
            "source_version": "synthetic",
            "output_directory": "prepared/v1",
            "sources": {
                "object_vocabulary": "raw/objects.json",
                "predicate_vocabulary": "raw/predicates.json",
                "train_annotations": "raw/train.json",
                "test_annotations": "raw/test.json",
                "images": "raw/images",
            },
        },
    }


def test_pair_order_multilabel_auxiliary_and_no_mutation():
    scene = Scene(
        "x",
        "x.png",
        10,
        10,
        (Object("b", 1, (4.0, 1.0, 9.0, 8.0)), Object("a", 0, (0.0, 0.0, 5.0, 5.0))),
    )
    pairs = build_pairs(scene)
    assert pairs == (("a", "b"), ("b", "a"))
    assert build_pairs(replace(scene, objects=tuple(reversed(scene.objects)))) == pairs
    targets, rows = build_targets(pairs, (Relation("a", 0, "b"), Relation("a", 1, "b")), 2)
    np.testing.assert_array_equal(targets, [[1, 1, 0], [0, 0, 1]])
    np.testing.assert_array_equal(rows, [True, False])
    assert targets.dtype == np.float32 and rows.dtype == np.bool_
    with pytest.raises(ValueError, match="outside"):
        build_targets(pairs, (Relation("a", 2, "b"),), 2)


def test_conversion_partition_isolation_hashes_and_relocation(tmp_path, monkeypatch):
    config = source_fixture(tmp_path)
    manifest_path = prepare_dataset(config)
    manifest = read_json(manifest_path)
    vocabulary = load_vocabulary(manifest_path)
    assert vocabulary.digest == canonical_digest(
        {"object_classes": vocabulary.object_classes, "predicates": vocabulary.predicates}
    )
    assert len(load_objects(manifest_path, "train")) == 4
    scenes = load_objects(manifest_path, "validation")
    targets = load_relations(manifest_path, "validation")
    assert len(scenes) == 1 and len(next(iter(targets.values()))) == 2
    assert next(iter(scenes.values())).objects[1].bbox == (4.0, 1.0, 9.0, 8.0)
    test_targets = manifest_path.parent / manifest["partitions"]["test"]["relations"]["path"]
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path == test_targets:
            raise AssertionError("TEST TARGET SENTINEL")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    load_relations(manifest_path, "train")
    load_objects(manifest_path, "test")
    with pytest.raises(FileExistsError):
        prepare_dataset(config)
    relocated = tmp_path / "relocated"
    shutil.copytree(manifest_path.parent, relocated)
    assert load_vocabulary(relocated / "dataset.json").digest == vocabulary.digest
    assert load_objects(relocated / "dataset.json", "validation") == scenes
    with (relocated / "objects" / "train.jsonl").open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="checksum"):
        load_objects(relocated / "dataset.json", "train")


def test_overlap_and_missing_records_fail(tmp_path):
    manifest_path = prepare_dataset(source_fixture(tmp_path))
    manifest = read_json(manifest_path)
    splits_path = manifest_path.parent / "splits.json"
    splits = read_json(splits_path)
    splits["validation"] = splits["train"][:1]
    write_json(splits_path, splits)
    manifest["splits"]["sha256"] = file_digest(splits_path)
    manifest["split_digest"] = canonical_digest(splits)
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="overlapping"):
        load_objects(manifest_path, "train")


def test_duplicate_test_images_excluded_and_train_groups_kept(tmp_path):
    config = source_fixture(tmp_path)
    images = tmp_path / "raw" / "images"
    shutil.copyfile(images / "5.png", images / "0.png")
    shutil.copyfile(images / "1.png", images / "2.png")
    manifest_path = prepare_dataset(config)
    splits = read_json(manifest_path.parent / "splits.json")
    assert "0.png" not in splits["train"] + splits["validation"]
    assert any(entry["image_id"] == "0.png" for entry in splits["exclusions"])
    assert ("1.png" in splits["train"]) == ("2.png" in splits["train"])


def test_malformed_boxes_and_predicates_fail_preparation(tmp_path):
    config = source_fixture(tmp_path)
    raw = read_json(tmp_path / "raw" / "train.json")
    raw["0.png"][0]["subject"]["bbox"] = [0, 0, 1, 2]
    write_json(tmp_path / "raw" / "train.json", raw)
    with pytest.raises(ValueError, match="box"):
        prepare_dataset(config)
    assert not (tmp_path / "prepared" / "v1").exists()


@pytest.mark.parametrize("predicate_id", [True, 0.5, "0"])
def test_noninteger_canonical_predicate_ids_are_rejected(predicate_id):
    with pytest.raises(ValueError, match="outside pair/vocabulary"):
        build_targets((("a", "b"),), (Relation("a", predicate_id, "b"),), 2)


@pytest.mark.parametrize("width", [True, 10.5, float("inf"), "10"])
def test_canonical_dimensions_require_positive_integers(width):
    from kgr4vrd.data import validate_scene

    with pytest.raises(ValueError, match="dimensions"):
        validate_scene(Scene("image", "image.png", width, 10, ()), 2)


def test_provided_license_provenance_is_preserved(tmp_path):
    config = source_fixture(tmp_path)
    provenance = {
        "reference": "https://example.test/source-terms",
        "redistribution_status": "synthetic fixture only",
    }
    config["preparation"]["license_provenance"] = provenance
    manifest_path = prepare_dataset(config)
    assert read_json(manifest_path)["license_provenance"] == provenance
