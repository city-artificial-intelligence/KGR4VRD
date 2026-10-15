"""Convert the NeSy4VRD/VRD relation JSON format into isolated partitions.

Raw boxes are [ymin,ymax,xmin,xmax]. They are permuted, with no +1, into
continuous xyxy coordinates, matching the registered legacy interpretation.
"""

import hashlib
import math
import os
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from PIL import Image

from ..artifacts import canonical_digest, file_digest, read_json, write_json, write_jsonl
from ..data import SPLITS, build_pairs, build_targets, validate_scene
from ..types import Object, Relation, Scene, Vocabulary


def _vocabulary(path: Path) -> tuple[dict[str, Any], ...]:
    raw = read_json(path)
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path}: expected a nonempty ordered vocabulary JSON array")
    result = []
    for index, entry in enumerate(raw):
        if isinstance(entry, str):
            entry = {"id": index, "label": entry, "source_id": str(index)}
        if not isinstance(entry, dict) or entry.get("id") != index:
            raise ValueError(f"{path}: vocabulary IDs must be dense, zero-based and ordered")
        if not isinstance(entry.get("label"), str):
            raise ValueError(f"{path}: vocabulary labels must be strings")
        result.append({**entry, "source_id": str(entry.get("source_id", index))})
    return tuple(result)


def _image_path(image_root: Path, image_id: str, by_name: dict[str, list[Path]]) -> Path:
    direct = image_root / image_id
    if direct.is_file():
        path = direct
    else:
        candidates = by_name.get(Path(image_id).name, [])
        if len(candidates) != 1:
            raise ValueError(
                f"{image_id}: expected exactly one source image, found {len(candidates)}"
            )
        path = candidates[0]
    if not path.resolve().is_relative_to(image_root.resolve()):
        raise ValueError(f"Image path escapes image root: {image_id}")
    return path


def _convert_image(
    image_id: str, raw: list, path: Path, image_root: Path, vocabulary: Vocabulary
) -> tuple[Scene, tuple[Relation, ...], int]:
    if not isinstance(raw, list):
        raise ValueError(f"{image_id}: expected a list of raw relation records")
    with Image.open(path) as image:
        width, height = image.size  # Do not apply EXIF rotation; boxes use stored pixels.
    objects: dict[tuple, tuple[int, tuple]] = {}
    triples = []
    for relation in raw:
        keys = []
        for role in ("subject", "object"):
            item = relation[role]
            bbox = item["bbox"]
            if len(bbox) != 4:
                raise ValueError(f"{image_id}: malformed raw box")
            box = (float(bbox[2]), float(bbox[0]), float(bbox[3]), float(bbox[1]))
            category = item["category"]
            if type(category) is not int:
                raise ValueError(f"{image_id}: category must be an integer source ID")
            native = item.get("object_id")
            key = ("native", str(native)) if native is not None else ("box", category, box)
            if key in objects and objects[key] != (category, box):
                raise ValueError(f"{image_id}: native object ID has conflicting class/box")
            objects[key] = category, box
            keys.append(key)
        predicate = relation["predicate"]
        if type(predicate) is not int or not 0 <= predicate < len(vocabulary.predicates):
            raise ValueError(f"{image_id}: predicate outside source vocabulary")
        triples.append((keys[0], predicate, keys[1]))
    ids = {
        key: (str(key[1]) if key[0] == "native" else f"o{index}")
        for index, key in enumerate(sorted(objects))
    }
    if len(set(ids.values())) != len(ids):
        raise ValueError(f"{image_id}: generated IDs collide with native object IDs")
    scene = Scene(
        image_id,
        path.relative_to(image_root).as_posix(),
        width,
        height,
        tuple(Object(ids[key], *objects[key]) for key in sorted(objects)),
    )
    validate_scene(scene, len(vocabulary.object_classes))
    self_count = sum(subject == obj for subject, _, obj in triples)
    relations = tuple(
        sorted(
            {
                Relation(ids[subject], predicate, ids[obj])
                for subject, predicate, obj in triples
                if subject != obj
            }
        )
    )
    build_targets(build_pairs(scene), relations, len(vocabulary.predicates))
    return scene, relations, self_count


def _split_images(
    train_ids: list[str], test_ids: list[str], hashes: dict[str, str], historical: Path | None
) -> tuple[dict, list[dict]]:
    """Group exact copies before the registered whole-group SHA-256 split."""
    test_hashes = {hashes[image_id] for image_id in test_ids}
    excluded = [
        {"image_id": image_id, "reason": "exact_duplicate_of_official_test"}
        for image_id in train_ids
        if hashes[image_id] in test_hashes
    ]
    excluded_ids = {entry["image_id"] for entry in excluded}
    pool = sorted(set(train_ids) - excluded_ids)
    groups: dict[str, list[str]] = {}
    for image_id in pool:
        groups.setdefault(hashes[image_id], []).append(image_id)
    if historical:
        recovered = read_json(historical)
        train, validation = list(recovered["train"]), list(recovered["validation"])
        if set(train) & set(validation) or set(train) | set(validation) != set(pool):
            raise ValueError(
                "Recovered split must partition the duplicate-audited training pool exactly"
            )
        if len(train) + len(validation) != len(pool):
            raise ValueError("Recovered split contains duplicate IDs")
        if "test" in recovered and set(recovered["test"]) != set(test_ids):
            raise ValueError("Recovered split changes official test membership")
        if any(
            set(group) & set(train) and set(group) & set(validation) for group in groups.values()
        ):
            raise ValueError("Recovered split crosses confirmed duplicate groups")
        provenance = {"historical_split_recovered": True, "source_sha256": file_digest(historical)}
    else:
        ordered = sorted(
            groups.values(),
            key=lambda group: hashlib.sha256(f"1729:{min(group)}".encode()).hexdigest(),
        )
        train = []
        validation = []
        limit = math.floor(0.8 * len(pool))
        for group in ordered:
            if not validation and len(train) + len(group) <= limit:
                train.extend(sorted(group))
            else:
                validation.extend(sorted(group))
        provenance = {
            "historical_split_recovered": False,
            "seed": 1729,
            "algorithm": "sha256_seeded_whole_group_prefix_80_20",
        }
    if not train or not validation:
        raise ValueError(
            "Duplicate-aware splitting produced an empty train or validation partition"
        )
    splits = {
        "schema_version": "kgr4vrd.splits/1",
        "train": train,
        "validation": validation,
        "test": sorted(test_ids),
        "provenance": provenance,
        "exclusions": excluded,
        "realised_train_fraction": len(train) / len(pool),
    }
    return splits, excluded


def prepare_dataset(config: dict[str, Any]) -> Path:
    """Build a new immutable local preparation from explicitly acquired inputs."""
    root = Path(config["paths"]["data_root"])
    preparation = config["preparation"]
    license_record = preparation.get("license_provenance")
    if license_record is not None and (
        not isinstance(license_record, dict)
        or not isinstance(license_record.get("reference"), str)
        or not isinstance(license_record.get("redistribution_status"), str)
    ):
        raise ValueError("license_provenance requires reference and redistribution_status strings")
    sources = {name: root / value for name, value in preparation["sources"].items()}
    required = (
        "object_vocabulary",
        "predicate_vocabulary",
        "train_annotations",
        "test_annotations",
        "images",
    )
    for name in required:
        if name not in sources or not sources[name].exists():
            raise FileNotFoundError(
                f"Missing preparation.sources.{name}; acquire the pinned resource locally"
            )
    output = root / preparation["output_directory"]
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite prepared dataset: {output}")
    vocabulary = Vocabulary(
        _vocabulary(sources["object_vocabulary"]), _vocabulary(sources["predicate_vocabulary"])
    )
    by_name: dict[str, list[Path]] = {}
    for path in sources["images"].rglob("*"):
        if path.is_file():
            by_name.setdefault(path.name, []).append(path)
    scenes: dict[str, Scene] = {}
    relations: dict[str, tuple[Relation, ...]] = {}
    hashes: dict[str, str] = {}
    membership: dict[str, list[str]] = {}
    self_count = 0
    for split in ("train", "test"):
        raw = read_json(sources[f"{split}_annotations"])
        if not isinstance(raw, dict):
            raise ValueError("Raw annotations must map image IDs to relation lists")
        membership[split] = sorted(raw)
        for image_id, record in raw.items():
            path = _image_path(sources["images"], image_id, by_name)
            image_hash = file_digest(path)
            if image_id in hashes and hashes[image_id] != image_hash:
                raise ValueError(f"{image_id}: cross-partition image ID collision")
            scene, targets, removed = _convert_image(
                image_id, record, path, sources["images"], vocabulary
            )
            scenes[image_id], relations[image_id], hashes[image_id] = scene, targets, image_hash
            self_count += removed
    splits, exclusions = _split_images(
        membership["train"], membership["test"], hashes, sources.get("historical_split")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent))
    try:
        manifest = _write_dataset(
            temporary,
            output,
            sources,
            scenes,
            relations,
            hashes,
            vocabulary,
            splits,
            exclusions,
            self_count,
            preparation,
        )
        write_json(temporary / "dataset.json", manifest)
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output / "dataset.json"


def _write_dataset(
    directory: Path,
    output: Path,
    sources: dict,
    scenes: dict,
    relations: dict,
    hashes: dict,
    vocabulary: Vocabulary,
    splits: dict,
    exclusions: list,
    self_count: int,
    preparation: dict,
) -> dict:
    vocabulary_record = {
        "schema_version": "kgr4vrd.vocabulary/1",
        "object_classes": vocabulary.object_classes,
        "predicates": vocabulary.predicates,
    }
    write_json(directory / "vocabulary.json", vocabulary_record)
    write_json(directory / "splits.json", splits)
    partitions: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        ids = splits[split]
        write_jsonl(
            directory / "objects" / f"{split}.jsonl", [asdict(scenes[image_id]) for image_id in ids]
        )
        write_jsonl(
            directory / "relations" / f"{split}.jsonl",
            [
                {
                    "image_id": image_id,
                    "relations": [asdict(relation) for relation in relations[image_id]],
                }
                for image_id in ids
            ],
        )
        partitions[split] = {
            kind: _reference(directory, f"{kind}/{split}.jsonl")
            for kind in ("objects", "relations")
        }
        partitions[split].update(
            image_count=len(ids),
            object_count=sum(len(scenes[i].objects) for i in ids),
            pair_count=sum(len(build_pairs(scenes[i])) for i in ids),
            relation_count=sum(len(relations[i]) for i in ids),
        )
    images = [
        {
            "image_id": image_id,
            "image_path": scenes[image_id].image_path,
            "width": scenes[image_id].width,
            "height": scenes[image_id].height,
            "sha256": hashes[image_id],
            "duplicate_group_id": hashes[image_id],
        }
        for image_id in sorted({i for split in SPLITS for i in splits[split]})
    ]
    discrepancy = len(vocabulary.object_classes) != 109 or len(vocabulary.predicates) != 71
    write_json(
        directory / "conversion-report.json",
        {
            "schema_version": "kgr4vrd.conversion/1",
            "excluded_self_relations": self_count,
            "exclusions": exclusions,
            "object_count": len(vocabulary.object_classes),
            "predicate_count": len(vocabulary.predicates),
            "historical_vocabulary_discrepancy": discrepancy,
            "raw_box_interpretation": "ymin,ymax,xmin,xmax permuted to continuous xyxy; no added pixel",
            "object_identity": "native object_id preserved; otherwise exact class/box deduplication",
            "duplicate_audit": "exact image SHA-256; raw VRD format has no native duplicate-group metadata",
            "perceptual_duplicate_audit": "not performed; exact hashes do not detect recompressed or edited copies",
        },
    )
    return {
        "schema_version": "kgr4vrd.dataset/1",
        "dataset_id": "nesy4vrd",
        "dataset_version": preparation["source_version"],
        "preparation_version": output.name,
        "adapter_version": "nesy4vrd-vrd-json/1",
        "code_digest": file_digest(Path(__file__)),
        "source_files": [
            {
                "identifier": name,
                "local_provenance": path.name,
                "source_url": "https://zenodo.org/records/17076303",
                "sha256": file_digest(path),
                "byte_count": path.stat().st_size,
                "publisher_checksum": None,
            }
            for name, path in sources.items()
            if path.is_file()
        ],
        "vocabulary": _reference(directory, "vocabulary.json"),
        "vocabulary_digest": vocabulary.digest,
        "splits": _reference(directory, "splits.json"),
        "split_digest": canonical_digest(splits),
        "partitions": partitions,
        "coordinate_convention": "xyxy_pixel_continuous",
        "image_root": os.path.relpath(sources["images"], output),
        "image_manifest": images,
        "exclusions": exclusions,
        "conversion_report": _reference(directory, "conversion-report.json"),
        "license_provenance": preparation.get(
            "license_provenance",
            {
                "reference": "https://zenodo.org/records/17076303",
                "redistribution_status": "unverified; inspect acquired source terms before redistribution",
            },
        ),
    }


def _reference(directory: Path, name: str) -> dict[str, str]:
    return {"path": name, "sha256": file_digest(directory / name)}
