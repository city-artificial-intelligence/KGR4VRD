"""Independently specified geometry cases and historical function conformance."""

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from kgr4vrd.features import geometry_features
from kgr4vrd.types import Object, Scene, Vocabulary


def vocabulary():
    return Vocabulary(
        tuple({"id": index, "label": str(index), "source_id": str(index)} for index in range(2)),
        ({"id": 0, "label": "near", "source_id": "0"},),
    )


def test_hand_computed_containment_coincident_and_reversed():
    scene = Scene(
        "x",
        "x.png",
        100,
        50,
        (Object("a", 0, (0.0, 0.0, 20.0, 20.0)), Object("b", 1, (5.0, 5.0, 15.0, 15.0))),
    )
    features = geometry_features(scene, vocabulary())
    expected = [1, 0, 0, 0, 1.6, 1.6, 0, 1, 0.4, 0.4, 1.2, 1.2, 0, 0.08, 0.02, 0, 0, 0.25, 1]
    np.testing.assert_allclose(features[0], expected)
    np.testing.assert_allclose(features[1, -7:], [0, 0.02, 0.08, 0, 0, 1, 0.25])
    assert features.shape == (2, 19) and features.dtype == np.float32


def test_downward_positive_y_and_nonoverlap():
    scene = Scene(
        "x",
        "x.png",
        100,
        100,
        (Object("a", 0, (0.0, 0.0, 10.0, 10.0)), Object("b", 1, (30.0, 40.0, 40.0, 50.0))),
    )
    features = geometry_features(scene, vocabulary())
    np.testing.assert_allclose(features[0, -7:], [4, 0.01, 0.01, 0.8, 0.6, 0, 0])
    np.testing.assert_allclose(features[1, -7:], [4, 0.01, 0.01, -0.8, -0.6, 0, 0])


def test_against_selected_legacy_pure_functions():
    # Extract only historical pure functions; the active package never imports legacy modules.
    root = Path(__file__).resolve().parents[1] / "legacy" / "predicatePrediction"
    namespace = {"math": math}
    for filename in ("vrd_utils5.py", "vrd_utils8.py"):
        tree = ast.parse((root / filename).read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        exec(compile(ast.Module(body=functions, type_ignores=[]), filename, "exec"), namespace)
    legacy = SimpleNamespace(**namespace)
    subject, obj = (11.0, 17.0, 87.0, 124.0), (103.0, 77.0, 401.0, 307.0)
    scene = Scene("x", "x.png", 641, 479, (Object("a", 0, subject), Object("b", 1, obj)))
    _, sx, _, sy = legacy.get_new_image_dims_and_scaling_factors(641, 479, 8)
    first = legacy.resize_bbox(subject, sx, sy)
    second = legacy.resize_bbox(obj, sx, sy)
    spatial = [
        legacy.bb_euclidean_distance_centroids(first, second),
        *legacy.bb_area_to_image_area_ratios(subject, obj, 641, 479),
        *legacy.bb_sine_and_cosine_of_angle_between_centroids(subject, obj),
        *legacy.bb_inclusion_ratios(subject, obj),
    ]
    expected = [1, 0, *first, 0, 1, *second, *(round(value, 3) for value in spatial)]
    np.testing.assert_array_equal(
        geometry_features(scene, vocabulary())[0], np.array(expected, np.float32)
    )
