"""Small immutable records shared by the scientific functions."""

import hashlib
import json
from dataclasses import dataclass
from enum import IntEnum
from typing import Any


class Decision(IntEnum):
    UNRESOLVED = -2
    UNKNOWN = -1
    COMPATIBLE = 0
    INCOMPATIBLE = 1


@dataclass(frozen=True)
class Object:
    object_id: str
    class_id: int
    bbox: tuple[float, float, float, float]


@dataclass(frozen=True)
class Scene:
    image_id: str
    image_path: str
    width: int
    height: int
    objects: tuple[Object, ...]


@dataclass(frozen=True, order=True)
class Relation:
    subject_id: str
    predicate_id: int
    object_id: str


@dataclass(frozen=True)
class Vocabulary:
    object_classes: tuple[dict[str, Any], ...]
    predicates: tuple[dict[str, Any], ...]

    @property
    def digest(self) -> str:
        """Identity of the ordered labels/mappings, independent of file location."""
        value = {"object_classes": self.object_classes, "predicates": self.predicates}
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
