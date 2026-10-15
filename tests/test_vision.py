"""Crop/cache semantics with a fake encoder; real weights are explicitly opt in."""

import copy
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from kgr4vrd.adapters import vision
from kgr4vrd.artifacts import file_digest, read_json, write_json
from kgr4vrd.config import data_path, load_config
from kgr4vrd.data import load_objects, load_vocabulary
from kgr4vrd.features import geometry_features
from kgr4vrd.prediction import ScoreShards, predict
from kgr4vrd.smoke import prepare_smoke
from kgr4vrd.training import train
from kgr4vrd.types import Object, Scene


class SizeEncoder(torch.nn.Module):
    """Inspectable two-coordinate vectors, including an intentional zero vector."""

    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    def forward(self, batch):
        assert not self.training and not torch.is_grad_enabled()
        self.batch_sizes.append(len(batch))
        vectors = torch.zeros((len(batch), 2048), dtype=torch.float32)
        vectors[:, :2] = batch
        vectors[(batch[:, 0] == 10) & (batch[:, 1] == 8)] = 0
        return vectors


def size_transform(crop):
    assert crop.mode == "RGB"
    return torch.tensor(crop.size, dtype=torch.float32)


def fractional_scene():
    # Direct crop-level fixture also tests clipping; normal dataset loading
    # independently rejects out-of-image canonical annotations.
    return Scene(
        "fractional",
        "unused.png",
        10,
        8,
        (Object("b", 0, (3.2, -0.3, 11.4, 8.2)), Object("a", 0, (-0.2, 1.7, 4.2, 6.1))),
    )


def test_crop_bounds_union_dedup_order_rgb_and_normalised_zero_vectors():
    scene = fractional_scene()
    boxes, indices = vision.crop_layout(scene)
    np.testing.assert_array_equal(boxes, [[0, 1, 5, 7], [3, 0, 10, 8], [0, 0, 10, 8]])
    np.testing.assert_array_equal(indices, [[0, 1, 2], [1, 0, 2]])
    encoder = SizeEncoder()
    encoded = vision.extract_crops(scene, Image.new("L", (10, 8)), encoder, size_transform, 2)
    assert encoder.batch_sizes == [2, 1]  # Three unique crops serve both ordered pairs.
    np.testing.assert_array_equal(encoded["pair_ids"], [["a", "b"], ["b", "a"]])
    np.testing.assert_allclose(
        encoded["features"][0, :2], np.array([5, 6]) / np.sqrt(61), rtol=1e-6
    )
    np.testing.assert_allclose(
        encoded["features"][1, :2], np.array([7, 8]) / np.sqrt(113), rtol=1e-6
    )
    np.testing.assert_array_equal(encoded["features"][2], np.zeros(2048))
    np.testing.assert_array_equal(encoded["features"][:, 2:], np.zeros((3, 2046)))
    duplicate = replace(scene, objects=scene.objects + (replace(scene.objects[1], object_id="c"),))
    assert len(vision.crop_layout(duplicate)[0]) == 3
    with pytest.raises(ValueError, match="empty crop"):
        vision.crop_layout(
            replace(scene, objects=(Object("a", 0, (10.0, 0.0, 11.0, 1.0)), scene.objects[0]))
        )
    with pytest.raises(ValueError, match="dimensions changed"):
        vision.extract_crops(scene, Image.new("RGB", (8, 10)), encoder, size_transform, 2)


def visual_fixture(tmp_path, monkeypatch):
    config = prepare_smoke(tmp_path)
    dataset_path = data_path(config, config["dataset"]["manifest"])
    dataset = read_json(dataset_path)
    all_scenes = {
        key: scene
        for split in ("train", "validation", "test")
        for key, scene in load_objects(dataset_path, split).items()
    }
    images = []
    for key, scene in all_scenes.items():
        image_path = dataset_path.parent / dataset["image_root"] / scene.image_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (scene.width, scene.height), (17, 43, 91)).save(image_path)
        images.append({"image_id": key, "sha256": file_digest(image_path)})
    dataset["image_manifest"] = images
    write_json(dataset_path, dataset)
    config["visual"] = {
        "weights_manifest": "visual/weights.json",
        "features_manifest": "visual/features/manifest.json",
        "crop_batch_size": 2,
    }
    weights_path = data_path(config, config["visual"]["weights_manifest"])
    weights_path.parent.mkdir(parents=True)
    weights_file = weights_path.parent / "fake-weights.pt"
    weights_file.write_bytes(b"fake encoder fixture; no real weights")
    write_json(
        weights_path,
        {
            "schema_version": "kgr4vrd.encoder/1",
            "encoder": "resnet50",
            "weights": "IMAGENET1K_V2",
            "torchvision_version": "0.17.2",
            "transform": copy.deepcopy(vision.TRANSFORM),
            "weights_file": {"path": weights_file.name, "sha256": file_digest(weights_file)},
        },
    )
    encoder = SizeEncoder()
    loads = []

    def fake_loader(path):
        loads.append(path)
        return encoder, size_transform

    monkeypatch.setattr(vision, "load_encoder", fake_loader)
    return config, dataset_path, all_scenes, encoder, loads


def test_preparation_reuses_verified_cache_without_reading_any_relation_labels(
    tmp_path, monkeypatch
):
    config, dataset_path, scenes, encoder, loads = visual_fixture(tmp_path, monkeypatch)
    targets = {
        dataset_path.parent / f"relations/{split}.jsonl"
        for split in ("train", "validation", "test")
    }
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path in targets:
            raise AssertionError("Feature extraction read relation targets")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    path = vision.prepare_features(config)
    first_digest, first_batches = file_digest(path), list(encoder.batch_sizes)
    assert vision.prepare_features(config) == path
    assert file_digest(path) == first_digest and encoder.batch_sizes == first_batches
    assert len(loads) == 1
    record = read_json(path)
    assert set(record["images"]) == set(scenes)
    assert record["crop_rule"] == vision.CROP_RULE
    vocabulary = load_vocabulary(dataset_path)
    scene = scenes["train-a"]
    geometry = geometry_features(scene, vocabulary)
    assembled = vision.load_visual_features(path, scene, vocabulary, geometry)
    with np.load(
        path.parent / record["images"][scene.image_id]["shard"]["path"], allow_pickle=False
    ) as saved:
        expected = saved["features"][saved["pair_crops"]].reshape(len(geometry), 6144)
    assert assembled.shape == (len(geometry), 6144 + 2 * len(vocabulary.object_classes) + 15)
    np.testing.assert_array_equal(assembled[:, :6144], expected)
    np.testing.assert_array_equal(assembled[:, 6144:], geometry)
    empty = scenes["train-empty"]
    assert vision.load_visual_features(
        path, empty, vocabulary, geometry_features(empty, vocabulary)
    ).shape == (0, assembled.shape[1])


def test_visual_cache_rejects_changed_weights_transform_boxes_and_partial_shards(
    tmp_path, monkeypatch
):
    config, dataset_path, scenes, _, _ = visual_fixture(tmp_path, monkeypatch)
    path = vision.prepare_features(config)
    record = read_json(path)
    scene, vocabulary = scenes["train-a"], load_vocabulary(dataset_path)
    geometry = geometry_features(scene, vocabulary)
    changed_scene = replace(
        scene, objects=(replace(scene.objects[0], bbox=(6.0, 10.0, 20.0, 50.0)), *scene.objects[1:])
    )
    with pytest.raises(ValueError, match="image/box identity"):
        vision.load_visual_features(path, changed_scene, vocabulary, geometry)
    weights_path = data_path(config, config["visual"]["weights_manifest"])
    original_weights_manifest = weights_path.read_bytes()
    weights = read_json(weights_path)
    weights["transform"]["resize_size"] = 233
    write_json(weights_path, weights)
    with pytest.raises(ValueError, match="Checksum mismatch"):
        vision.load_visual_features(path, scene, vocabulary, geometry)
    with pytest.raises(ValueError, match="exact transform"):
        vision.prepare_features(config)
    weights_path.write_bytes(original_weights_manifest)
    weight_file = weights_path.parent / weights["weights_file"]["path"]
    original_weights = weight_file.read_bytes()
    weight_file.write_bytes(b"different pretrained weights")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        vision.prepare_features(config)
    weight_file.write_bytes(original_weights)
    shard = path.parent / record["images"][scene.image_id]["shard"]["path"]
    shard.write_bytes(b"partial shard")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        vision.load_visual_features(path, scene, vocabulary, geometry)


def test_fake_visual_features_use_shared_training_checkpoint_and_prediction(tmp_path, monkeypatch):
    config, dataset_path, scenes, _, _ = visual_fixture(tmp_path, monkeypatch)
    vision.prepare_features(config)
    config["model"]["name"] = "visual_mlp_v1"
    config["training"]["max_epochs"] = 1
    directory = train(config)
    selection = read_json(directory / "selected.json")
    checkpoint = torch.load(
        directory / selection["checkpoint"], map_location="cpu", weights_only=False
    )
    vocabulary = load_vocabulary(dataset_path)
    dimension = 6144 + 2 * len(vocabulary.object_classes) + 15
    assert checkpoint["input_dimension"] == dimension
    assert checkpoint["model"]["0.weight"].shape == (512, dimension)
    assert checkpoint["model"]["2.weight"].shape == (256, 512)
    assert checkpoint["model"]["4.weight"].shape == (len(vocabulary.predicates) + 1, 256)
    prediction_path = predict(directory / "selected.json", split="validation")
    validation = {key: value for key, value in scenes.items() if key.startswith("validation")}
    scores = ScoreShards(prediction_path, validation)
    assert set(scores) == set(validation)
    for image_id in scores:
        assert scores[image_id].shape == (12, len(vocabulary.predicates) + 1)
        assert scores[image_id].dtype == np.float32 and np.isfinite(scores[image_id]).all()


def test_training_refuses_mismatched_weight_and_image_cache_identity(tmp_path, monkeypatch):
    config, dataset_path, _, _, _ = visual_fixture(tmp_path, monkeypatch)
    vision.prepare_features(config)
    original_path = data_path(config, config["visual"]["weights_manifest"])
    alternate = read_json(original_path)
    alternate_file = original_path.parent / "alternate-weights.pt"
    alternate_file.write_bytes(b"different but individually hashed weights")
    alternate["weights_file"] = {"path": alternate_file.name, "sha256": file_digest(alternate_file)}
    alternate_path = original_path.parent / "alternate-manifest.json"
    write_json(alternate_path, alternate)
    config["visual"]["weights_manifest"] = str(alternate_path)
    config["model"]["name"] = "visual_mlp_v1"
    with pytest.raises(ValueError, match="Configured visual weights differ"):
        train(config)
    config["visual"]["weights_manifest"] = str(original_path)
    dataset = read_json(dataset_path)
    dataset["image_manifest"][0]["sha256"] = "0" * 64
    write_json(dataset_path, dataset)
    with pytest.raises(ValueError, match="different source image inventory"):
        train(config)


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("KGR4VRD_ENCODER_CONFIG"),
    reason="KGR4VRD_ENCODER_CONFIG and pinned local real weights/images are required",
)
def test_real_pinned_encoder_cache_is_repeatable(tmp_path):
    config = load_config(os.environ["KGR4VRD_ENCODER_CONFIG"])
    config["visual"]["features_manifest"] = str(tmp_path / "real-features/manifest.json")
    first = vision.prepare_features(config)
    digest = file_digest(first)
    assert vision.prepare_features(config) == first
    assert file_digest(first) == digest
    manifest = read_json(first)
    assert manifest["crop_rule"] == vision.CROP_RULE and manifest["normalisation"] == "l2_eps_1e-12"
    config["visual"]["features_manifest"] = str(tmp_path / "repeat-features/manifest.json")
    second = vision.prepare_features(config)
    repeated = read_json(second)
    assert repeated["identity"] == manifest["identity"]
    dataset_path = data_path(config, config["dataset"]["manifest"])
    vocabulary = load_vocabulary(dataset_path)
    for split in ("train", "validation", "test"):
        for scene in load_objects(dataset_path, split).values():
            geometry = geometry_features(scene, vocabulary)
            np.testing.assert_allclose(
                vision.load_visual_features(first, scene, vocabulary, geometry),
                vision.load_visual_features(second, scene, vocabulary, geometry),
                atol=1e-6,
                rtol=1e-5,
            )
