"""Preparation executes real OWLRL locally; GraphDB acceptance stays an opt-in check."""

import json
import os
from pathlib import Path

import pytest
from owlrl import OWLRL_Semantics
from rdflib import OWL, RDF, RDFS, Graph, Namespace, URIRef

from kgr4vrd import data
from kgr4vrd.adapters import graphdb
from kgr4vrd.artifacts import read_json, write_json
from kgr4vrd.reasoning import (
    check_conformance,
    load_knowledge,
    load_simple_filter,
    prepare_compatibility,
    prepare_reasoning,
)
from kgr4vrd.smoke import prepare_smoke
from kgr4vrd.types import Decision, Vocabulary

EX = Namespace("urn:kgr4vrd:preparation-test:")


class LocalReference:
    """Test-only independent OWLRL typed-query oracle; not a GraphDB substitute."""

    def __init__(self, config):
        self.ontology = Graph()

    def verify(self, ontology):
        self.ontology = ontology
        reasoner = OWLRL_Semantics(ontology, False, False)
        # A copy prevents the source graph from becoming an inferred C1 input.
        copy = Graph()
        for triple in ontology:
            copy.add(triple)
        reasoner = OWLRL_Semantics(copy, False, False)
        reasoner.closure()
        if reasoner.error_messages:
            raise ValueError("inconsistent background")
        return {
            "backend": "test-OWLRL",
            "version": "test",
            "ruleset": "test",
            "query_scope": "independent_typed_relation",
        }

    def __call__(self, subject, predicate, obj):
        graph = Graph()
        for triple in self.ontology:
            graph.add(triple)
        graph.add((EX.a, RDF.type, URIRef(subject)))
        graph.add((EX.b, RDF.type, URIRef(obj)))
        graph.add((EX.a, URIRef(predicate), EX.b))
        reasoner = OWLRL_Semantics(graph, False, False)
        reasoner.closure()
        return (
            Decision.INCOMPATIBLE if reasoner.error_messages else Decision.COMPATIBLE,
            {"reason": "; ".join(reasoner.error_messages) or "supported_consistent"},
        )

    def close(self):
        pass


def preparation_config(tmp_path):
    config = prepare_smoke(tmp_path)
    root = Path(config["paths"]["data_root"])
    vocabulary = data.load_vocabulary(root / config["dataset"]["manifest"])
    mappings = []
    for kind, entries in (
        ("object_class", vocabulary.object_classes),
        ("predicate", vocabulary.predicates),
    ):
        for entry in entries:
            mapped = entry["label"] != "unmapped"
            mappings.append(
                {
                    "label_kind": kind,
                    "label_id": entry["id"],
                    "iri": str(EX[entry["label"]]) if mapped else None,
                    "mapping_relation": "exact" if mapped else "unmapped",
                    "rationale": "synthetic identity",
                    "provenance": "test fixture",
                }
            )
    write_json(root / "mapping.json", {"mappings": mappings})
    ontology = Graph()
    for triple in (
        (EX.Person, RDFS.subClassOf, EX.Animal),
        (EX.Horse, RDFS.subClassOf, EX.Animal),
        (EX.Car, RDFS.subClassOf, EX.Vehicle),
        (EX.Animal, OWL.disjointWith, EX.Vehicle),
        (EX.rides, RDFS.range, EX.Animal),
        (EX.near, RDF.type, OWL.SymmetricProperty),
    ):
        ontology.add(triple)
    ontology.serialize(root / "ontology.ttl", format="turtle")
    config["preparation"] = {
        "mapping": "mapping.json",
        "augmentation_ontology": "ontology.ttl",
        "compatibility_ontology": "ontology.ttl",
        "closure_timeout_seconds": 30,
        "closure_max_triples": 10000,
    }
    config["graphdb"] = {"endpoint": "unused-by-test"}
    config["intervention"].update(
        knowledge_manifest="new-knowledge/full/manifest.json",
        augmentation_manifest="new-knowledge/augmentation/manifest.json",
        control_manifest="new-knowledge/control/manifest.json",
    )
    config["evaluation"]["simple_filter_manifest"] = "new-knowledge/simple/manifest.json"
    return config, vocabulary


def test_preparation_reads_only_train_targets_and_preserves_mapping_identity(tmp_path, monkeypatch):
    config, vocabulary = preparation_config(tmp_path)
    original = data.load_relations
    seen = []

    def training_only(path, split):
        seen.append(split)
        assert split == "train", "held-out targets reached reasoning preparation"
        return original(path, split)

    monkeypatch.setattr(data, "load_relations", training_only)
    monkeypatch.setattr(graphdb, "GraphDB", LocalReference)
    path = prepare_reasoning(config)
    assert seen == ["train"]
    table = load_knowledge(path, vocabulary)
    assert table.vocabulary_digest == vocabulary.digest
    assert table.decisions[0, 1, 2] == Decision.INCOMPATIBLE
    assert table.decisions[0, 1, 1] == Decision.COMPATIBLE
    assert table.decisions[0, 3, 1] == Decision.UNKNOWN
    root = Path(config["paths"]["data_root"])
    simple = load_simple_filter(root / config["evaluation"]["simple_filter_manifest"], vocabulary)
    assert not simple.decisions[0, 1, 2], "C1 must not infer subclass disjointness"
    augmentation = read_json(root / config["intervention"]["augmentation_manifest"])
    assert augmentation["vocabulary_digest"] == vocabulary.digest
    assert augmentation["added_counts"] == [2, 0, 0, 0]
    control = read_json(root / config["intervention"]["control_manifest"])
    assert control["selected_condition"] == "frequency_control"
    assert control["weights"] == [2, 1, 1, 1]
    with pytest.raises(FileExistsError):
        prepare_reasoning(config)


def test_conflicting_raw_positive_never_publishes_accepted_artifact(tmp_path, monkeypatch):
    config, _ = preparation_config(tmp_path)

    class ConflictReference(LocalReference):
        def __call__(self, *args):
            return Decision.INCOMPATIBLE, {"reason": "controlled synthetic conflict"}

    monkeypatch.setattr(graphdb, "GraphDB", ConflictReference)
    root = Path(config["paths"]["data_root"])
    with pytest.raises(ValueError, match="Raw training positives"):
        prepare_reasoning(config)
    path = root / config["intervention"]["knowledge_manifest"]
    assert not path.exists()
    assert read_json(path.parent / "training-conflicts.json")["conflicts"]
    assert read_json(path.parent / "preparation-status.json")["status"] == "failed"


@pytest.mark.integration
def test_real_graphdb_fixtures_and_200_fresh_conformance_queries():
    location = os.environ.get("KGR4VRD_GRAPHDB_CONFIG")
    if not location:
        pytest.skip(
            "Requires KGR4VRD_GRAPHDB_CONFIG JSON for an exclusive empty GraphDB repository"
        )
    config = json.loads(Path(location).read_text())
    vocabulary = Vocabulary(
        tuple(
            {"id": i, "label": f"C{i}", "source_id": str(i), "ontology_iri": str(EX[f"C{i}"])}
            for i in range(9)
        ),
        tuple(
            {"id": i, "label": f"p{i}", "source_id": str(i), "ontology_iri": str(EX[f"p{i}"])}
            for i in range(3)
        ),
    )
    ontology = Graph()
    ontology.add((EX.C0, OWL.disjointWith, EX.C1))
    ontology.add((EX.p0, RDFS.range, EX.C0))
    ontology.add((EX.C2, RDFS.subClassOf, EX.C1))
    client = graphdb.GraphDB(config)
    try:
        identity = client.verify(ontology)
        table = prepare_compatibility(
            vocabulary, client, {k: v for k, v in identity.items() if k != "consistency_fixture"}
        )
        assert table.decisions[0, 0, 0] == Decision.COMPATIBLE
        assert table.decisions[0, 0, 2] == Decision.INCOMPATIBLE
        assert (
            table.decisions[0, 0, 3] == Decision.COMPATIBLE
        )  # Open-world mismatch alone is not contradiction.
        result = check_conformance(table, vocabulary, client)
        assert result["status"] == "passed" and result["sample_count"] >= 200
    finally:
        client.close()


def test_closure_failure_records_failed_stage_without_accepted_manifest(tmp_path, monkeypatch):
    config, _ = preparation_config(tmp_path)
    monkeypatch.setattr(graphdb, "GraphDB", LocalReference)

    def exhausted(*args, **kwargs):
        raise RuntimeError("controlled closure resource limit")

    monkeypatch.setattr("kgr4vrd.reasoning.augment_scene", exhausted)
    with pytest.raises(RuntimeError, match="resource limit"):
        prepare_reasoning(config)
    path = Path(config["paths"]["data_root"]) / config["intervention"]["knowledge_manifest"]
    assert not path.exists()
    assert read_json(path.parent / "preparation-status.json")["status"] == "failed"
