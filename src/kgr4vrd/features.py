"""The versioned geometry feature contract, independent of training conditions."""

import math
import warnings

import numpy as np

from .data import build_pairs, validate_scene
from .types import Scene, Vocabulary

PROFILE = "legacy_geometry_v1"


def geometry_features(scene: Scene, vocabulary: Vocabulary) -> np.ndarray:
    """[N,2*C+15] float32; legacy rounding with downward-positive image y."""
    class_count = len(vocabulary.object_classes)
    validate_scene(scene, class_count)
    longest = max(scene.width, scene.height)
    resized_width = round(8 * scene.width / longest, 2)
    resized_height = round(8 * scene.height / longest, 2)
    sx = round(resized_width / scene.width, 5)
    sy = round(resized_height / scene.height, 5)
    boxes = {}
    by_id = {obj.object_id: obj for obj in scene.objects}
    for obj in scene.objects:
        boxes[obj.object_id] = tuple(
            round(float(value * scale), 5) for value, scale in zip(obj.bbox, (sx, sy, sx, sy))
        )
        box = boxes[obj.object_id]
        if box[2] <= box[0] or box[3] <= box[1]:
            warnings.warn(
                f"{scene.image_id}/{obj.object_id}: legacy scaling degenerates box",
                RuntimeWarning,
                stacklevel=2,
            )
    pairs = build_pairs(scene)
    result = np.zeros((len(pairs), 2 * class_count + 15), dtype=np.float32)
    for index, (subject_id, object_id) in enumerate(pairs):
        subject, obj = by_id[subject_id], by_id[object_id]
        subject_hot = [float(i == subject.class_id) for i in range(class_count)]
        object_hot = [float(i == obj.class_id) for i in range(class_count)]
        subject_box, object_box = boxes[subject_id], boxes[object_id]
        resized_dx, resized_dy = _displacement(subject_box, object_box)
        dx, dy = _displacement(subject.bbox, obj.bbox)
        distance = math.sqrt(dx**2 + dy**2)
        subject_area, object_area = _area(subject.bbox), _area(obj.bbox)
        intersection = max(
            0.0, min(subject.bbox[2], obj.bbox[2]) - max(subject.bbox[0], obj.bbox[0])
        )
        intersection *= max(
            0.0, min(subject.bbox[3], obj.bbox[3]) - max(subject.bbox[1], obj.bbox[1])
        )
        spatial = [
            math.sqrt(resized_dx**2 + resized_dy**2),
            subject_area / (scene.width * scene.height),
            object_area / (scene.width * scene.height),
            dy / distance if distance else 0.0,
            dx / distance if distance else 0.0,
            intersection / subject_area,
            intersection / object_area,
        ]
        result[index] = (
            subject_hot
            + list(subject_box)
            + object_hot
            + list(object_box)
            + [round(value, 3) for value in spatial]
        )
    return result


def _displacement(subject: tuple, obj: tuple) -> tuple[float, float]:
    return (
        (obj[0] + (obj[2] - obj[0]) / 2) - (subject[0] + (subject[2] - subject[0]) / 2),
        (obj[1] + (obj[3] - obj[1]) / 2) - (subject[1] + (subject[3] - subject[1]) / 2),
    )


def _area(box: tuple) -> float:
    return float((box[2] - box[0]) * (box[3] - box[1]))
