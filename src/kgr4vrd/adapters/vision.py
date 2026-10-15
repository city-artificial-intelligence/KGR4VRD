"""Explicit local frozen ResNet features, cached once per unique image crop."""

import importlib.metadata
import os
import platform
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from ..artifacts import (
    atomic_write,
    canonical_digest,
    file_digest,
    read_json,
    verify_file,
    write_json,
)
from ..config import data_path
from ..data import SPLITS, build_pairs, load_manifest, load_objects, load_vocabulary
from ..types import Scene, Vocabulary

TRANSFORM = {
    "resize_size": 232,
    "crop_size": 224,
    "interpolation": "bilinear",
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "antialias": True,
}
CROP_RULE = "stored_rgb_floor_min_ceil_max_clip_union_v1"


def scene_digest(scene: Scene) -> str:
    record = asdict(scene)
    record["objects"] = sorted(record["objects"], key=lambda obj: obj["object_id"])
    return canonical_digest(record)


def crop_layout(scene: Scene) -> tuple[np.ndarray, np.ndarray]:
    """Deduplicate clipped pixel bounds; index each pair's subject/object/union."""
    boxes = {obj.object_id: obj.bbox for obj in scene.objects}
    unique: dict[tuple[int, ...], int] = {}
    rows = []
    for subject, obj in build_pairs(scene):
        left, right = boxes[subject], boxes[obj]
        union = (
            min(left[0], right[0]),
            min(left[1], right[1]),
            max(left[2], right[2]),
            max(left[3], right[3]),
        )
        row = []
        for box in (left, right, union):
            bounds = (
                max(0, int(np.floor(box[0]))),
                max(0, int(np.floor(box[1]))),
                min(scene.width, int(np.ceil(box[2]))),
                min(scene.height, int(np.ceil(box[3]))),
            )
            if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
                raise ValueError(f"{scene.image_id}: empty crop")
            row.append(unique.setdefault(bounds, len(unique)))
        rows.append(row)
    return (
        np.asarray(list(unique), dtype=np.int64).reshape(-1, 4),
        np.asarray(rows, dtype=np.int64).reshape(-1, 3),
    )


def extract_crops(
    scene: Scene,
    image: Image.Image,
    encoder: Any,
    transform: Any,
    batch_size: int,
    *,
    device: torch.device | str = "cpu",
) -> dict[str, np.ndarray]:
    """Encode bounded batches on the encoder's device; persist normalised CPU FP32."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("crop_batch_size must be a positive integer")
    if image.size != (scene.width, scene.height):
        raise ValueError(f"{scene.image_id}: image dimensions changed")
    boxes, indices = crop_layout(scene)
    vectors = np.empty((len(boxes), 2048), dtype=np.float32)
    image = image.convert("RGB")
    encoder.eval()
    with torch.inference_mode():
        for start in range(0, len(boxes), batch_size):
            batch = torch.stack(
                [transform(image.crop(tuple(box))) for box in boxes[start : start + batch_size]]
            ).to(device=device, dtype=torch.float32)
            encoded = encoder(batch).to(dtype=torch.float32, device="cpu")
            if encoded.shape != (len(batch), 2048) or not torch.isfinite(encoded).all():
                raise ValueError("Encoder must produce finite [N,2048] vectors")
            encoded = encoded / encoded.norm(dim=1, keepdim=True).clamp_min(1e-12)
            vectors[start : start + len(batch)] = encoded.numpy()
    return {
        "crop_boxes": boxes,
        "features": vectors,
        "pair_crops": indices,
        "pair_ids": np.asarray(build_pairs(scene), dtype=np.str_).reshape(-1, 2),
    }


def _extraction_device(selected: str) -> tuple[torch.device, dict[str, Any]]:
    """Resolve the requested device by allocation, never by silently falling back."""
    requested = torch.device(selected)
    if requested.type not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported visual extraction device: {selected}")
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested visual extraction device {selected!r} is unavailable")
    if requested.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(f"Requested visual extraction device {selected!r} is unavailable")
    try:
        device = torch.empty(1, device=requested).device
    except RuntimeError as exc:
        raise RuntimeError(f"Cannot allocate on visual extraction device {selected!r}") from exc
    hardware: dict[str, Any] = {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        hardware["accelerator"] = {
            "name": properties.name,
            "capability": [properties.major, properties.minor],
            "total_memory": properties.total_memory,
            "multiprocessors": properties.multi_processor_count,
        }
    return device, {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": importlib.metadata.version("torchvision"),
        "Pillow": importlib.metadata.version("Pillow"),
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "selected_device": selected,
        "device": str(device),
        "hardware": hardware,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
    }


def _encoder_manifest(path: Path) -> dict:
    record = read_json(path)
    if (
        record.get("schema_version") != "kgr4vrd.encoder/1"
        or record.get("encoder") != "resnet50"
        or record.get("weights") != "IMAGENET1K_V2"
        or record.get("torchvision_version") != "0.17.2"
        or record.get("transform") != TRANSFORM
    ):
        raise ValueError("Encoder requires pinned ResNet-50 IMAGENET1K_V2 and exact transform")
    digest = record["weights_file"]["sha256"]
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("Encoder requires the measured SHA-256 of local weights")
    return record


def load_encoder(path: Path) -> tuple[Any, Any]:
    """No download path: instantiate empty architecture and load verified local weights."""
    from torchvision.models import ResNet50_Weights, resnet50

    record = _encoder_manifest(path)
    if importlib.metadata.version("torchvision").split("+")[0] != record["torchvision_version"]:
        raise ValueError("TorchVision version does not match the encoder manifest")
    weights_path = verify_file(path.parent, record["weights_file"])
    official = ResNet50_Weights.IMAGENET1K_V2
    published_prefix = Path(official.url).stem.rsplit("-", 1)[1]
    if not record["weights_file"]["sha256"].startswith(published_prefix):
        raise ValueError("Local state dictionary does not match the named official weight revision")
    encoder = resnet50(weights=None)
    encoder.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    encoder.fc = torch.nn.Identity()
    encoder.requires_grad_(False)
    encoder.eval()
    return encoder, official.transforms()


def _encoder_identity(record: dict) -> str:
    return canonical_digest(
        {
            **{key: value for key, value in record.items() if key != "weights_file"},
            "weights_sha256": record["weights_file"]["sha256"],
            "crop_rule": CROP_RULE,
            "normalisation": "l2_eps_1e-12",
            "dtype": "float32",
        }
    )


def _read_shard(path: Path, item: dict, scene: Scene) -> dict[str, np.ndarray]:
    if item["scene_digest"] != scene_digest(scene):
        raise ValueError("Visual feature image/box identity mismatch")
    with np.load(verify_file(path.parent, item["shard"]), allow_pickle=False) as stored:
        arrays = {key: stored[key] for key in ("features", "crop_boxes", "pair_crops", "pair_ids")}
    boxes, indices = crop_layout(scene)
    if (
        arrays["features"].dtype != np.float32
        or arrays["features"].shape != (len(boxes), 2048)
        or not np.isfinite(arrays["features"]).all()
        or arrays["crop_boxes"].dtype != np.int64
        or not np.array_equal(arrays["crop_boxes"], boxes)
        or arrays["pair_crops"].dtype != np.int64
        or not np.array_equal(arrays["pair_crops"], indices)
        or arrays["pair_ids"].dtype.kind != "U"
        or not np.array_equal(
            arrays["pair_ids"], np.asarray(build_pairs(scene), dtype=np.str_).reshape(-1, 2)
        )
    ):
        raise ValueError("Invalid visual feature shard rows, bounds, dtype or values")
    norms = np.linalg.norm(arrays["features"], axis=1)
    if not np.all((norms == 0) | np.isclose(norms, 1, atol=1e-5)):
        raise ValueError("Visual feature vectors are not L2 normalised")
    return arrays


def load_visual_features(
    manifest_path: str | Path,
    scene: Scene,
    vocabulary: Vocabulary,
    geometry: np.ndarray,
    *,
    manifest: dict | None = None,
) -> np.ndarray:
    path = Path(manifest_path)
    record = read_json(path) if manifest is None else manifest
    if (
        record.get("schema_version") != "kgr4vrd.features/1"
        or record.get("status") != "completed"
        or record["vocabulary_digest"] != vocabulary.digest
        or record["crop_rule"] != CROP_RULE
    ):
        raise ValueError("Incomplete or mismatched visual feature manifest")
    weights = _encoder_manifest(verify_file(path.parent, record["weights_manifest"]))
    if record["encoder_identity"] != _encoder_identity(weights):
        raise ValueError("Visual feature encoder identity mismatch")
    if geometry.dtype != np.float32 or geometry.shape != (
        len(build_pairs(scene)),
        2 * len(vocabulary.object_classes) + 15,
    ):
        raise ValueError("Visual features require unchanged float32 legacy geometry")
    arrays = _read_shard(path, record["images"][scene.image_id], scene)
    visual = arrays["features"][arrays["pair_crops"]].reshape(len(geometry), 6144)
    return np.concatenate((visual, geometry), axis=1)


def prepare_features(config: dict) -> Path:
    """Prepare all supplied object crops; held-out relation targets are never opened."""
    visual = config["visual"]
    batch_size = visual.get("crop_batch_size", 32)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("crop_batch_size must be a positive integer")
    dataset_path = data_path(config, config["dataset"]["manifest"])
    dataset = load_manifest(dataset_path)
    vocabulary = load_vocabulary(dataset_path)
    weights_path = data_path(config, visual["weights_manifest"])
    weights = _encoder_manifest(weights_path)
    verify_file(weights_path.parent, weights["weights_file"])
    path = data_path(config, visual["features_manifest"])
    scenes = {
        key: scene for split in SPLITS for key, scene in load_objects(dataset_path, split).items()
    }
    image_records = {item["image_id"]: item for item in dataset["image_manifest"]}
    if set(image_records) != set(scenes):
        raise ValueError("Feature preparation requires exact source-image inventory")
    device, environment = _extraction_device(config["training"]["device"])
    identity: dict[str, Any] = {
        "encoder_identity": _encoder_identity(weights),
        "vocabulary_digest": vocabulary.digest,
        "image_manifest_digest": canonical_digest(dataset["image_manifest"]),
        "extractor_source": file_digest(__file__),
        "environment": environment,
        "images": {
            key: {"scene_digest": scene_digest(scene), "image_sha256": image_records[key]["sha256"]}
            for key, scene in scenes.items()
        },
    }
    for key, scene in scenes.items():
        image_path = dataset_path.parent / dataset["image_root"] / scene.image_path
        if file_digest(image_path) != image_records[key]["sha256"]:
            raise ValueError(f"{key}: source image checksum mismatch")
    if path.exists():
        existing = read_json(path)
        if existing.get("identity") != identity or existing.get("status") != "completed":
            raise ValueError("Feature cache exists for different inputs; use a new destination")
        from ..features import geometry_features

        for scene in scenes.values():
            load_visual_features(
                path, scene, vocabulary, geometry_features(scene, vocabulary), manifest=existing
            )
        return path
    encoder, transform = load_encoder(weights_path)
    encoder = encoder.to(device=device, dtype=torch.float32)
    encoder.requires_grad_(False)
    started = time.perf_counter()
    records = {}
    for key, scene in scenes.items():
        with Image.open(dataset_path.parent / dataset["image_root"] / scene.image_path) as image:
            arrays = extract_crops(scene, image, encoder, transform, batch_size, device=device)
        shard = path.parent / "shards" / f"{canonical_digest(key)}.npz"
        if shard.exists():
            raise ValueError("Unpublished feature shards exist; use a new destination")
        atomic_write(shard, lambda stream: np.savez_compressed(stream, **arrays))
        records[key] = {
            **identity["images"][key],
            "pairs": len(build_pairs(scene)),
            "shard": {"path": os.path.relpath(shard, path.parent), "sha256": file_digest(shard)},
        }
    write_json(
        path,
        {
            "schema_version": "kgr4vrd.features/1",
            "status": "completed",
            "identity": identity,
            "vocabulary_digest": vocabulary.digest,
            "encoder_identity": identity["encoder_identity"],
            "crop_rule": CROP_RULE,
            "normalisation": "l2_eps_1e-12",
            "dtype": "float32",
            "weights_manifest": {
                "path": os.path.relpath(weights_path, path.parent),
                "sha256": file_digest(weights_path),
            },
            "images": records,
            "seconds": time.perf_counter() - started,
            "cache_bytes": sum(
                (path.parent / row["shard"]["path"]).stat().st_size for row in records.values()
            ),
        },
    )
    return path
