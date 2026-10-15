"""Per-image OWL-RL augmentation and immutable, independently typed decisions."""

import importlib.metadata
import json
import os
import time
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from owlrl import OWLRL_Semantics
from rdflib import OWL, RDF, RDFS, Graph, Namespace, URIRef
from rdflib.compare import to_canonical_graph
from rdflib.term import Node

from .artifacts import (
    atomic_write,
    canonical_digest,
    file_digest,
    read_json,
    write_json,
    write_jsonl,
)
from .data import build_pairs
from .types import Decision, Relation, Scene, Vocabulary


@dataclass(frozen=True)
class CompatibilityTable:
    """Immutable [subject class, predicate, object class] int8 decisions or C1 bool mask."""

    decisions: np.ndarray
    vocabulary_digest: str
    metadata: dict

    def __post_init__(self):
        array = np.array(self.decisions, copy=True)
        if array.ndim != 3 or array.shape[0] != array.shape[2]:
            raise ValueError("Knowledge axes must be [C,P,C]")
        if array.dtype != bool and (
            array.dtype != np.int8 or not np.isin(array, [-2, -1, 0, 1]).all()
        ):
            raise ValueError("Knowledge requires int8 decision codes or Boolean C1 mask")
        # A bytes-backed array cannot be made writable by the caller.
        immutable = np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
        object.__setattr__(self, "decisions", immutable)

    @property
    def digest(self) -> str:
        return canonical_digest(
            {
                "vocabulary_digest": self.vocabulary_digest,
                "metadata": {k: v for k, v in self.metadata.items() if k != "decision_evidence"},
                "dtype": str(self.decisions.dtype),
                "decisions": self.decisions.tolist(),
            }
        )

    def lookup(self, scene: Scene) -> np.ndarray:
        classes = {obj.object_id: obj.class_id for obj in scene.objects}
        pairs = build_pairs(scene)
        if any(c < 0 or c >= self.decisions.shape[0] for c in classes.values()):
            raise ValueError(f"{scene.image_id}: class outside knowledge vocabulary")
        values = (
            np.stack([self.decisions[classes[s], :, classes[o]] for s, o in pairs])
            if pairs
            else np.empty((0, self.decisions.shape[1]), dtype=self.decisions.dtype)
        )
        if (values == Decision.UNRESOLVED).any():
            raise ValueError(f"{scene.image_id}: pair outside prepared reasoning universe")
        return values


def save_knowledge(path: str | Path, table: CompatibilityTable, vocabulary: Vocabulary) -> Path:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Accepted knowledge artifact already exists: {path}")
    if table.vocabulary_digest != vocabulary.digest:
        raise ValueError("Knowledge vocabulary mismatch")
    filename = "simple-mask.npz" if table.decisions.dtype == bool else "decisions.npz"
    atomic_write(
        path.parent / filename,
        lambda stream: np.savez_compressed(stream, decisions=table.decisions),
    )
    write_json(
        path,
        {
            "schema_version": "kgr4vrd.knowledge/1",
            "status": "completed",
            "vocabulary_digest": vocabulary.digest,
            "knowledge_digest": table.digest,
            "metadata": table.metadata,
            "axes": ["subject_class", "predicate", "object_class"],
            "array": {"path": filename, "sha256": file_digest(path.parent / filename)},
        },
    )
    return path


def _load_table(path: str | Path, vocabulary: Vocabulary) -> CompatibilityTable:
    path = Path(path)
    manifest = read_json(path)
    if (
        manifest.get("schema_version") != "kgr4vrd.knowledge/1"
        or manifest.get("status") != "completed"
    ):
        raise ValueError("Knowledge manifest is unsupported or incomplete")
    if manifest["vocabulary_digest"] != vocabulary.digest:
        raise ValueError("Knowledge vocabulary identity mismatch")
    array_path = path.parent / manifest["array"]["path"]
    if file_digest(array_path) != manifest["array"]["sha256"]:
        raise ValueError("Knowledge array checksum mismatch")
    with np.load(array_path, allow_pickle=False) as data:
        table = CompatibilityTable(data["decisions"], vocabulary.digest, manifest["metadata"])
    if table.decisions.shape != (
        len(vocabulary.object_classes),
        len(vocabulary.predicates),
        len(vocabulary.object_classes),
    ):
        raise ValueError("Knowledge axes differ from vocabulary")
    if table.digest != manifest["knowledge_digest"]:
        raise ValueError("Knowledge scientific identity mismatch")
    return table


def load_knowledge(path: str | Path, vocabulary: Vocabulary) -> CompatibilityTable:
    table = _load_table(path, vocabulary)
    if table.decisions.dtype != np.int8:
        raise ValueError("A simple Boolean filter cannot replace full compatibility decisions")
    return table


def load_simple_filter(path: str | Path, vocabulary: Vocabulary) -> CompatibilityTable:
    table = _load_table(path, vocabulary)
    if table.decisions.dtype != bool or table.metadata.get("policy") != "simple_asserted_mask_v1":
        raise ValueError("Expected the asserted-only Boolean control")
    return table


def graph_digest(graph: Graph) -> str:
    return canonical_digest(
        sorted(" ".join(term.n3() for term in triple) for triple in to_canonical_graph(graph))
    )


def load_ontology(path: str | Path) -> Graph:
    """Load a local ontology with an already resolved, pinned import closure."""
    graph = Graph().parse(Path(path))
    if list(graph.triples((None, OWL.imports, None))):
        raise ValueError(
            "Resolve and flatten owl:imports into a pinned local ontology before preparation"
        )
    return graph


def construct_profile(
    ontology: Graph, task_properties: Iterable[str], profile: str = "S_TEU"
) -> tuple[Graph, dict]:
    """Remove disabled asserted families before closure; preserve class schema."""
    switches = {
        "S_TEU": {"symmetry", "transitivity", "equivalence", "subproperty"},
        "all": {"symmetry", "inverse", "transitivity", "equivalence", "subproperty"},
        "none": set(),
    }
    if profile not in switches:
        raise ValueError(f"Unsupported augmentation profile: {profile}")
    props = {URIRef(p) for p in task_properties}
    declarations: dict[Node, str] = {
        OWL.SymmetricProperty: "symmetry",
        OWL.TransitiveProperty: "transitivity",
    }
    relations: dict[Node, str] = {
        OWL.inverseOf: "inverse",
        OWL.equivalentProperty: "equivalence",
        RDFS.subPropertyOf: "subproperty",
    }
    result, removed = Graph(), []
    for subject, predicate, obj in ontology:
        family = declarations.get(obj) if predicate == RDF.type else relations.get(predicate)
        applies = subject in props if predicate == RDF.type else subject in props and obj in props
        if applies and family and family not in switches[profile]:
            removed.append([str(subject), str(predicate), str(obj)])
        else:
            result.add((subject, predicate, obj))
    metadata = {
        "profile": profile,
        "ontology_digest": graph_digest(ontology),
        "removed_axioms": sorted(removed),
        "retained_digest": graph_digest(result),
        "backend": "RDFLib/OWLRL",
        "rdflib_version": importlib.metadata.version("rdflib"),
        "owlrl_version": importlib.metadata.version("owlrl"),
    }
    metadata["profile_digest"] = canonical_digest(metadata)
    return result, metadata


def validate_profile_metadata(record: dict, expected: str = "S_TEU") -> None:
    """Verify the declared pre-closure treatment before admitting inferred targets."""
    keys = (
        "profile",
        "ontology_digest",
        "removed_axioms",
        "retained_digest",
        "backend",
        "rdflib_version",
        "owlrl_version",
    )
    if record.get("profile") != expected or any(key not in record for key in keys):
        raise ValueError(
            "Augmentation profile metadata is missing or differs from the registered profile"
        )
    if canonical_digest({key: record[key] for key in keys}) != record.get("profile_digest"):
        raise ValueError("Augmentation profile digest mismatch")


def _closure(graph: Graph, timeout_seconds: float, max_triples: int) -> None:
    """Apply OWLRL with checked work limits; partial graphs are never returned."""
    if timeout_seconds <= 0 or max_triples < 1:
        raise ValueError("Closure limits must be positive")
    deadline = time.monotonic() + timeout_seconds

    class BoundedClosure(OWLRL_Semantics):
        def rules(self, triple, cycle_num):
            if (
                time.monotonic() > deadline
                or len(self.graph) + len(self.added_triples) > max_triples
            ):
                raise RuntimeError("OWL-RL closure exceeded its declared resource limit")
            return super().rules(triple, cycle_num)

    reasoner = BoundedClosure(graph, axioms=False, daxioms=False)
    reasoner.closure()
    if reasoner.error_messages:
        raise ValueError("Inconsistent OWL-RL graph: " + "; ".join(reasoner.error_messages))
    if time.monotonic() > deadline or len(graph) > max_triples:
        raise RuntimeError("OWL-RL closure exceeded its declared resource limit")


def augment_scene(
    scene: Scene,
    relations: Iterable[Relation],
    vocabulary: Vocabulary,
    ontology: Graph,
    *,
    profile: str = "S_TEU",
    timeout_seconds: float = 30,
    max_triples: int = 1000000,
) -> tuple[tuple[Relation, ...], dict]:
    """Materialize one training image; no mutable graph is shared between images."""
    properties = {entry["id"]: entry.get("ontology_iri") for entry in vocabulary.predicates}
    graph, metadata = construct_profile(ontology, (p for p in properties.values() if p), profile)
    raw = set(relations)
    pairs = set(build_pairs(scene))
    if any(
        (r.subject_id, r.object_id) not in pairs or r.predicate_id not in properties for r in raw
    ):
        raise ValueError(f"{scene.image_id}: relation outside declared pairs/vocabulary")
    ns = Namespace("urn:kgr4vrd:image:" + canonical_digest(scene.image_id) + ":")
    objects = {obj.object_id: URIRef(ns[canonical_digest(obj.object_id)]) for obj in scene.objects}
    for scene_object in scene.objects:
        iri = vocabulary.object_classes[scene_object.class_id].get("ontology_iri")
        if iri:
            graph.add((objects[scene_object.object_id], RDF.type, URIRef(iri)))
    for relation in raw:
        property_iri = properties[relation.predicate_id]
        if property_iri:
            graph.add(
                (objects[relation.subject_id], URIRef(property_iri), objects[relation.object_id])
            )
    _closure(graph, timeout_seconds, max_triples)
    reverse_objects: dict[Node, str] = {iri: key for key, iri in objects.items()}
    reverse_properties: dict[Node, int] = {
        URIRef(iri): key for key, iri in properties.items() if iri
    }
    inferred, excluded_self, excluded_other = set(), 0, 0
    for subject, predicate, obj in graph:
        if predicate not in reverse_properties:
            continue
        if subject not in reverse_objects or obj not in reverse_objects:
            excluded_other += 1
        elif subject == obj:
            excluded_self += 1
        else:
            inferred.add(
                Relation(
                    reverse_objects[subject], reverse_properties[predicate], reverse_objects[obj]
                )
            )
    added = inferred - raw
    result = tuple(sorted(raw | inferred))
    metadata.update(
        {
            "image_id": scene.image_id,
            "asserted_count": len(raw),
            "added_count": len(added),
            "added_relations": [
                {
                    "subject_id": r.subject_id,
                    "predicate_id": r.predicate_id,
                    "object_id": r.object_id,
                    "origin": "inferred",
                    "profile_digest": metadata["profile_digest"],
                }
                for r in sorted(added)
            ],
            "newly_supervised_pairs": len(
                {(r.subject_id, r.object_id) for r in added}
                - {(r.subject_id, r.object_id) for r in raw}
            ),
            "excluded_self": excluded_self,
            "excluded_other_endpoints": excluded_other,
            "closure_triples": len(graph),
        }
    )
    return result, metadata


def simple_asserted_mask(ontology: Graph, vocabulary: Vocabulary) -> CompatibilityTable:
    """C1 uses direct named domain/range/disjointness facts, with no closure."""
    disjoint, unsupported = set(), []
    for left, right in ontology.subject_objects(OWL.disjointWith):
        if isinstance(left, URIRef) and isinstance(right, URIRef):
            disjoint.add(frozenset((left, right)))
        else:
            unsupported.append("anonymous disjointWith")
    for group in ontology.subjects(RDF.type, OWL.AllDisjointClasses):
        head = ontology.value(group, OWL.members)
        if head is None:
            raise ValueError("AllDisjointClasses requires an RDF member list")
        members = list(ontology.items(head))
        named = [member for member in members if isinstance(member, URIRef)]
        disjoint.update(frozenset(pair) for pair in combinations(named, 2))
        unsupported.extend(
            "anonymous AllDisjointClasses member" for m in members if not isinstance(m, URIRef)
        )
    constraints = {}
    for predicate in vocabulary.predicates:
        iri = predicate.get("ontology_iri")
        domain: list[URIRef] = []
        ranges: list[URIRef] = []
        for axis, target in ((RDFS.domain, domain), (RDFS.range, ranges)):
            for obj in ontology.objects(URIRef(iri), axis) if iri else []:
                if isinstance(obj, URIRef):
                    target.append(obj)
                else:
                    unsupported.append(f"anonymous {axis}")
        constraints[predicate["id"]] = (domain, ranges)
    shape = (
        len(vocabulary.object_classes),
        len(vocabulary.predicates),
        len(vocabulary.object_classes),
    )
    mask = np.zeros(shape, dtype=bool)
    for s, p, o in product(*(range(n) for n in shape)):
        subject_iri = vocabulary.object_classes[s].get("ontology_iri")
        object_iri = vocabulary.object_classes[o].get("ontology_iri")
        if not subject_iri or not object_iri or not vocabulary.predicates[p].get("ontology_iri"):
            continue
        domains, ranges = constraints[p]
        for types in ({URIRef(subject_iri), *domains}, {URIRef(object_iri), *ranges}):
            if any(pair <= types for pair in disjoint):
                mask[s, p, o] = True
    facts = {
        str(p): {"domains": sorted(map(str, d)), "ranges": sorted(map(str, r))}
        for p, (d, r) in constraints.items()
    }
    return CompatibilityTable(
        mask,
        vocabulary.digest,
        {
            "policy": "simple_asserted_mask_v1",
            "ontology_digest": graph_digest(ontology),
            "constraints": facts,
            "disjoint_pairs": sorted(sorted(map(str, pair)) for pair in disjoint),
            "unsupported_expressions": sorted(unsupported),
            "blocked_count": int(mask.sum()),
        },
    )


def prepare_compatibility(
    vocabulary: Vocabulary,
    backend: Callable,
    metadata: dict,
    *,
    class_pairs: Iterable[tuple[int, int]] | None = None,
    progress_path: Path | None = None,
) -> CompatibilityTable:
    """Compile independent typed queries; resume only exactly matching successful decisions."""
    count, predicates = len(vocabulary.object_classes), len(vocabulary.predicates)
    pairs = sorted(set(class_pairs if class_pairs is not None else product(range(count), repeat=2)))
    identity = {
        **metadata,
        "vocabulary_digest": vocabulary.digest,
        "class_pairs": pairs,
        "query_scope": "independent_typed_relation",
    }
    identity_digest = canonical_digest(identity)
    completed = {}
    universe = {(s, p, o) for s, o in pairs for p in range(predicates)}
    if any(not 0 <= c < count for pair in pairs for c in pair):
        raise ValueError("Query universe contains class outside vocabulary")
    if progress_path and progress_path.exists():
        content = progress_path.read_bytes()
        if content and not content.endswith(b"\n"):
            content = content[: content.rfind(b"\n") + 1]
            atomic_write(progress_path, lambda stream: stream.write(content))
        for record in (json.loads(line) for line in content.splitlines()):
            if record["identity_digest"] != identity_digest:
                raise ValueError("Interrupted reasoning build input identity changed")
            key = tuple(record["key"])
            if (
                key not in universe
                or record.get("decision") not in (-1, 0, 1)
                or not record.get("reason")
            ):
                raise ValueError("Invalid interrupted reasoning decision")
            if key in completed:
                raise ValueError("Duplicate cached reasoning query")
            completed[key] = record
    values = np.full((count, predicates, count), Decision.UNRESOLVED, dtype=np.int8)
    for s, o in pairs:
        for p in range(predicates):
            key = (s, p, o)
            if key not in completed:
                iris = [
                    vocabulary.object_classes[s].get("ontology_iri"),
                    vocabulary.predicates[p].get("ontology_iri"),
                    vocabulary.object_classes[o].get("ontology_iri"),
                ]
                if not all(iris):
                    decision, evidence = Decision.UNKNOWN, {"reason": "unmapped_label"}
                else:
                    decision, evidence = backend(*iris)
                if decision not in (Decision.UNKNOWN, Decision.COMPATIBLE, Decision.INCOMPATIBLE):
                    raise ValueError("Reasoner returned an unresolved or invalid decision")
                if not evidence.get("reason"):
                    raise ValueError("Reasoning decisions require an explicit reason")
                completed[key] = {
                    "identity_digest": identity_digest,
                    "key": list(key),
                    "decision": int(decision),
                    **evidence,
                }
                if progress_path:
                    progress_path.parent.mkdir(parents=True, exist_ok=True)
                    with progress_path.open("ab") as stream:
                        stream.write(
                            (
                                json.dumps(completed[key], sort_keys=True, allow_nan=False) + "\n"
                            ).encode()
                        )
                        stream.flush()
                        if len(completed) % 64 == 0:
                            os.fsync(stream.fileno())
            values[key] = completed[key]["decision"]
    decision_evidence = [completed[key] for key in sorted(completed)]
    return CompatibilityTable(
        values,
        vocabulary.digest,
        {
            **identity,
            "query_identity": identity_digest,
            "decision_evidence": decision_evidence,
            "unknown_count": int((values == Decision.UNKNOWN).sum()),
        },
    )


def check_conformance(
    table: CompatibilityTable, vocabulary: Vocabulary, backend: Callable, *, sample_size: int = 200
) -> dict:
    """Deterministic stratified fresh reference queries, including every available stratum."""
    strata: dict[tuple, list] = {}
    for s, p, o in np.argwhere(
        np.isin(table.decisions, [Decision.COMPATIBLE, Decision.INCOMPATIBLE])
    ):
        strata.setdefault((int(p), int(table.decisions[s, p, o])), []).append(
            (int(s), int(p), int(o))
        )
    keys: list[tuple[int, int, int]] = []
    while any(strata.values()) and len(keys) < sample_size:
        for stratum in sorted(strata):
            if strata[stratum]:
                keys.append(strata[stratum].pop(0))
    records = []
    for s, p, o in keys:
        result, _ = backend(
            vocabulary.object_classes[s]["ontology_iri"],
            vocabulary.predicates[p]["ontology_iri"],
            vocabulary.object_classes[o]["ontology_iri"],
        )
        if result != table.decisions[s, p, o]:
            raise ValueError(f"Fresh checker disagrees with table at {(s, p, o)}")
        records.append({"key": [s, p, o], "decision": int(result)})
    return {
        "sample": records,
        "sample_count": len(records),
        "required_sample_count": sample_size,
        "status": "passed" if len(records) >= sample_size else "insufficient_available_queries",
        "scope": "sampled independent typed relations; no general OWL completeness claim",
    }


def ontology_inventory(ontology: Graph) -> dict:
    """Conservative supported named-class/object-property profile; reject other semantics."""
    supported_predicates = {
        RDF.type,
        RDFS.subClassOf,
        RDFS.subPropertyOf,
        RDFS.domain,
        RDFS.range,
        OWL.equivalentClass,
        OWL.equivalentProperty,
        OWL.inverseOf,
        OWL.disjointWith,
        OWL.propertyDisjointWith,
        OWL.members,
        RDF.first,
        RDF.rest,
    }
    annotations = {
        RDFS.label,
        RDFS.comment,
        RDFS.seeAlso,
        RDFS.isDefinedBy,
        OWL.versionInfo,
        OWL.versionIRI,
        OWL.priorVersion,
        OWL.backwardCompatibleWith,
        OWL.incompatibleWith,
    }
    supported_types = {
        OWL.Class,
        RDFS.Class,
        OWL.ObjectProperty,
        RDF.Property,
        OWL.Ontology,
        OWL.SymmetricProperty,
        OWL.TransitiveProperty,
        OWL.AsymmetricProperty,
        OWL.IrreflexiveProperty,
        OWL.FunctionalProperty,
        OWL.InverseFunctionalProperty,
        OWL.AllDisjointClasses,
    }
    unsupported = []
    for subject, predicate, obj in ontology:
        if predicate in annotations:
            continue
        if predicate not in supported_predicates or (
            predicate == RDF.type and obj not in supported_types
        ):
            unsupported.append([str(subject), str(predicate), str(obj)])
        elif predicate in {RDFS.subClassOf, RDFS.domain, RDFS.range, OWL.equivalentClass} and not (
            isinstance(subject, URIRef) and isinstance(obj, URIRef)
        ):
            unsupported.append([str(subject), str(predicate), str(obj)])
    return {
        "profile": "named_class_object_property_owlrl_v1",
        "triple_count": len(ontology),
        "supported_predicates": sorted(map(str, supported_predicates)),
        "unsupported_constructs": sorted(unsupported),
    }


def _mapped_vocabulary(path: Path, vocabulary: Vocabulary) -> tuple[Vocabulary, str]:
    document = read_json(path)
    records = document.get("mappings") if isinstance(document, dict) else document
    if not isinstance(records, list):
        raise ValueError("Mapping must be a list or an object with a mappings list")
    entries = {
        "object_class": [dict(v) for v in vocabulary.object_classes],
        "predicate": [dict(v) for v in vocabulary.predicates],
    }
    seen = set()
    for record in records:
        required = {"label_kind", "label_id", "iri", "mapping_relation", "rationale", "provenance"}
        if not required <= record.keys():
            raise ValueError(
                "Mapping records require kind, ID, IRI, relation, rationale and provenance"
            )
        if any(
            not isinstance(record[key], str) or not record[key].strip()
            for key in ("rationale", "provenance")
        ):
            raise ValueError("Mapping rationale and provenance must be nonempty strings")
        kind, label_id = record["label_kind"], record["label_id"]
        if (
            kind not in entries
            or type(label_id) is not int
            or not 0 <= label_id < len(entries[kind])
        ):
            raise ValueError("Mapping label outside canonical vocabulary")
        if (kind, label_id) in seen:
            raise ValueError("Duplicate mapping label")
        seen.add((kind, label_id))
        relation = record["mapping_relation"]
        if relation not in {"exact", "unmapped"}:
            raise ValueError("Broader mappings require a separately reviewed semantics profile")
        iri = record["iri"] if relation == "exact" else None
        if iri is not None and (
            not isinstance(iri, str) or ":" not in iri or any(c in iri for c in '<>" {}\\\n')
        ):
            raise ValueError("Invalid ontology IRI")
        if relation == "exact" and not iri:
            raise ValueError("Exact mapping requires a nonempty ontology IRI")
        entries[kind][label_id]["ontology_iri"] = iri
    if len(seen) != sum(map(len, entries.values())):
        raise ValueError("Mapping must explicitly cover every canonical class and predicate")
    return Vocabulary(
        tuple(entries["object_class"]), tuple(entries["predicate"])
    ), canonical_digest(records)


def _relation_records(relations: dict[str, tuple[Relation, ...]]) -> list[dict]:
    return [
        {
            "image_id": image,
            "relations": [
                {
                    "subject_id": r.subject_id,
                    "predicate_id": r.predicate_id,
                    "object_id": r.object_id,
                }
                for r in values
            ],
        }
        for image, values in sorted(relations.items())
    ]


def prepare_reasoning(config: dict) -> Path:
    """Record failures throughout preparation, including closure/control computation."""
    path = Path(config["paths"]["data_root"]) / config["intervention"]["knowledge_manifest"]
    previously_accepted = path.exists()
    try:
        return _prepare_reasoning(config)
    except Exception as error:
        if not previously_accepted:
            write_json(
                path.parent / "preparation-status.json", {"status": "failed", "reason": str(error)}
            )
        raise


def _prepare_reasoning(config: dict) -> Path:
    """Prepare training augmentation, C1/C2 and reference decisions without held-out targets."""
    from .adapters.graphdb import GraphDB
    from .controls import frequency_weights, shuffle_added_labels
    from .data import load_manifest, load_objects, load_relations, load_vocabulary

    root = Path(config["paths"]["data_root"])
    preparation = config.get("preparation", {})
    required = {
        "augmentation_ontology",
        "compatibility_ontology",
        "mapping",
        "closure_timeout_seconds",
        "closure_max_triples",
    }
    if required - preparation.keys() or not config.get("graphdb"):
        raise ValueError(
            "Reasoning preparation requires ontology/mapping paths, closure limits and GraphDB configuration"
        )
    dataset_path = root / config["dataset"]["manifest"]
    dataset = load_manifest(dataset_path)
    vocabulary = load_vocabulary(dataset_path)
    mapped, mapping_digest = _mapped_vocabulary(root / preparation["mapping"], vocabulary)
    knowledge_path = root / config["intervention"]["knowledge_manifest"]
    augmentation_path = root / config["intervention"]["augmentation_manifest"]
    simple_path = root / config["evaluation"]["simple_filter_manifest"]
    control_path = root / config["intervention"]["control_manifest"]
    if any(
        path.exists() for path in (knowledge_path, augmentation_path, simple_path, control_path)
    ):
        raise FileExistsError(
            "Reasoning output already accepted; choose a new preparation directory"
        )
    checker_path = root / preparation["compatibility_ontology"]
    augmentation_source = root / preparation["augmentation_ontology"]
    ontology, augmentation_ontology = (
        load_ontology(checker_path),
        load_ontology(augmentation_source),
    )
    inventories = {
        "compatibility": ontology_inventory(ontology),
        "augmentation": ontology_inventory(augmentation_ontology),
    }
    write_json(knowledge_path.parent / "ontology-inventory.json", inventories)
    if any(item["unsupported_constructs"] for item in inventories.values()):
        raise ValueError(
            "Unsupported ontology constructs; inspect ontology-inventory.json before defining a new profile"
        )
    train_scenes, raw = load_objects(dataset_path, "train"), load_relations(dataset_path, "train")
    class_pairs: set[tuple[int, int]] = set()
    for split in ("train", "validation", "test"):
        for scene in load_objects(dataset_path, split).values():
            classes = {obj.object_id: obj.class_id for obj in scene.objects}
            class_pairs.update((classes[s], classes[o]) for s, o in build_pairs(scene))
    started = time.monotonic()
    backend = GraphDB(config["graphdb"])
    status_path = knowledge_path.parent / "preparation-status.json"
    write_json(status_path, {"status": "running"})
    try:
        verified = backend.verify(ontology)
        backend_identity = {
            key: verified[key] for key in ("backend", "version", "ruleset", "query_scope")
        }
        metadata = {
            **backend_identity,
            "ontology_digest": file_digest(checker_path),
            "mapping_digest": mapping_digest,
            "inventory": inventories["compatibility"],
        }
        table = prepare_compatibility(
            mapped,
            backend,
            metadata,
            class_pairs=class_pairs,
            progress_path=knowledge_path.parent / "completed-decisions.jsonl",
        )
        conflicts = []
        for image, relations in raw.items():
            scene = train_scenes[image]
            classes = {obj.object_id: obj.class_id for obj in scene.objects}
            for r in relations:
                if (
                    table.decisions[classes[r.subject_id], r.predicate_id, classes[r.object_id]]
                    == Decision.INCOMPATIBLE
                ):
                    conflicts.append(
                        {
                            "image_id": image,
                            "subject_id": r.subject_id,
                            "predicate_id": r.predicate_id,
                            "object_id": r.object_id,
                        }
                    )
        write_json(knowledge_path.parent / "training-conflicts.json", {"conflicts": conflicts})
        if conflicts:
            raise ValueError(
                "Raw training positives contradict checker; adjudicate source/ontology and rebuild"
            )
        conformance = check_conformance(table, mapped, backend)
        write_json(
            knowledge_path.parent / "conformance.json", {**conformance, "backend_fixture": verified}
        )
        table = CompatibilityTable(
            table.decisions, vocabulary.digest, {**table.metadata, "conformance": conformance}
        )
    except Exception as error:
        write_json(status_path, {"status": "failed", "reason": str(error)})
        raise
    finally:
        backend.close()
    expanded, derivations = {}, []
    for image, scene in train_scenes.items():
        expanded[image], metadata = augment_scene(
            scene,
            raw[image],
            mapped,
            augmentation_ontology,
            profile=config["intervention"]["augmentation_profile"],
            timeout_seconds=preparation["closure_timeout_seconds"],
            max_triples=preparation["closure_max_triples"],
        )
        derivations.append(metadata)
    raw_counts = np.zeros(len(vocabulary.predicates), dtype=np.int64)
    added_counts = np.zeros_like(raw_counts)
    for image in raw:
        for relation in raw[image]:
            raw_counts[relation.predicate_id] += 1
        for relation in set(expanded[image]) - set(raw[image]):
            added_counts[relation.predicate_id] += 1
    weights, control = frequency_weights(raw_counts, added_counts)
    common = {
        "status": "completed",
        "vocabulary_digest": vocabulary.digest,
        "training_relations_digest": dataset["partitions"]["train"]["relations"]["sha256"],
        "training_objects_digest": dataset["partitions"]["train"]["objects"]["sha256"],
    }
    control.update(
        {**common, "schema_version": "kgr4vrd.control/1", "selected_condition": "frequency_control"}
    )
    if control["requires_shuffle"]:
        shuffled, shuffle_metadata = shuffle_added_labels(raw, expanded)
        control.update(shuffle_metadata)
        control["selected_condition"] = "shuffled_augmentation_control"
        write_jsonl(control_path.parent / "relations.jsonl", _relation_records(shuffled))
        control["relations"] = {
            "path": "relations.jsonl",
            "sha256": file_digest(control_path.parent / "relations.jsonl"),
        }
    write_jsonl(augmentation_path.parent / "relations.jsonl", _relation_records(expanded))
    write_jsonl(augmentation_path.parent / "derivations.jsonl", derivations)
    _, profile = construct_profile(
        augmentation_ontology,
        (entry["ontology_iri"] for entry in mapped.predicates if entry.get("ontology_iri")),
    )
    augmentation = {
        **common,
        "schema_version": "kgr4vrd.augmentation/1",
        **profile,
        "mapping_digest": mapping_digest,
        "source_sha256": file_digest(augmentation_source),
        "relations": {
            "path": "relations.jsonl",
            "sha256": file_digest(augmentation_path.parent / "relations.jsonl"),
        },
        "derivations": {
            "path": "derivations.jsonl",
            "sha256": file_digest(augmentation_path.parent / "derivations.jsonl"),
        },
        "raw_counts": raw_counts.tolist(),
        "added_counts": added_counts.tolist(),
        "closure_limits": {
            "timeout_seconds": preparation["closure_timeout_seconds"],
            "max_triples": preparation["closure_max_triples"],
        },
    }
    simple = simple_asserted_mask(ontology, mapped)
    simple = CompatibilityTable(
        simple.decisions, vocabulary.digest, {**simple.metadata, "mapping_digest": mapping_digest}
    )
    save_knowledge(knowledge_path, table, vocabulary)
    save_knowledge(simple_path, simple, vocabulary)
    write_json(augmentation_path, augmentation)
    write_json(control_path, control)
    write_json(
        status_path,
        {
            "status": "completed",
            "elapsed_seconds": time.monotonic() - started,
            "query_count": len(table.metadata["decision_evidence"]),
            "conformance_status": conformance["status"],
        },
    )
    return knowledge_path
