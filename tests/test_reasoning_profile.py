"""Actual library rules and GraphDB guard query, without claiming real service conformance."""

from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pytest
from owlrl import OWLRL_Semantics
from rdflib import OWL, RDFS, BNode, Graph, Namespace
from rdflib.collection import Collection

from kgr4vrd.adapters.graphdb import GraphDB, Response
from kgr4vrd.reasoning import (
    CompatibilityTable,
    check_conformance,
    graph_digest,
    load_ontology,
    ontology_inventory,
    simple_asserted_mask,
)
from kgr4vrd.types import Decision, Vocabulary

EX = Namespace("urn:kgr4vrd:boolean:")
FIXTURE = Path(__file__).parent / "fixtures/boolean-vrd-style.owl"


def local_client(monkeypatch, schema=None):
    client = GraphDB({"endpoint": "http://unused", "repository": "unused"})
    client.verified = True
    state = {"queries": [], "cleanup": []}

    def insert(graph, context):
        closure = Graph()
        for source in (schema or Graph(), graph):
            for triple in source:
                closure.add(triple)
        engine = OWLRL_Semantics(closure, False, False)
        engine.closure()
        state["closure"] = closure
        if engine.error_messages:
            return Response(
                500,
                (
                    "ConsistencyException: disjoint consistency rule: "
                    + ";".join(engine.error_messages)
                ).encode(),
            )
        return Response(204)

    def query(url):
        parameters = parse_qs(urlsplit(url).query)
        assert parameters["infer"] == ["true"]
        state["queries"].append(parameters["query"][0])
        return {"boolean": bool(state["closure"].query(parameters["query"][0]).askAnswer)}

    monkeypatch.setattr(client, "_insert", insert)
    monkeypatch.setattr(client, "_json", query)
    monkeypatch.setattr(client, "cleanup", state["cleanup"].append)
    return client, state


def test_explicit_turtle_owl_and_structural_inventory(tmp_path):
    graph = load_ontology(FIXTURE, format="turtle")
    inventory = ontology_inventory(graph)
    assert inventory["unsupported_constructs"] == []
    assert len(inventory["class_expressions"]) == 5
    assert inventory["datatype_schema_properties"] == [str(EX.coordinate)]
    renamed = tmp_path / "same.nt"
    renamed.write_bytes(FIXTURE.read_bytes())
    assert graph_digest(load_ontology(renamed, format="turtle")) == graph_digest(graph)
    with pytest.raises(Exception):
        load_ontology(FIXTURE, format="xml")
    with pytest.raises(ValueError, match="format"):
        load_ontology(FIXTURE, format="guess")


@pytest.mark.parametrize(
    "subject,predicate,obj,expected",
    [
        (EX.A, EX.rangeUnion, EX.B, Decision.COMPATIBLE),
        (EX.A, EX.rangeUnion, EX.Unknown, Decision.UNKNOWN),
        (EX.A, EX.rangeUnion, EX.Neither, Decision.UNKNOWN),
        (EX.Both, EX.domainForbidden, EX.A, Decision.INCOMPATIBLE),
        (EX.NotA, EX.domainIntersection, EX.A, Decision.INCOMPATIBLE),
        (EX.Unknown, EX.directUnion, EX.A, Decision.UNKNOWN),
        (EX.A, EX.rangeOuter, EX.Inner, Decision.UNKNOWN),
        (EX.A, EX.rangeOuter, EX.C, Decision.COMPATIBLE),
    ],
)
def test_witnessed_partial_query_semantics(monkeypatch, subject, predicate, obj, expected):
    client, state = local_client(monkeypatch, load_ontology(FIXTURE))
    decision, evidence = client(str(subject), str(predicate), str(obj))
    assert decision == expected
    if decision == Decision.UNKNOWN:
        assert evidence["reason"] == "union_without_member_witness"
    if decision != Decision.INCOMPATIBLE:
        query = state["queries"][-1]
        assert query.count("urn:kgr4vrd:individual:") == 2
    assert len(state["cleanup"]) == 1 and not client.contexts


def test_live_startup_boolean_cases_have_library_witnesses(monkeypatch):
    client, _ = local_client(monkeypatch)
    evidence = client._verify_boolean_profile()
    assert [row["decision"] for row in evidence] == [0, -1, -1, 1, 1]


def test_c1_keeps_anonymous_union_while_full_guard_returns_unknown(monkeypatch):
    vocab = Vocabulary(
        ({"id": 0, "ontology_iri": str(EX.Unknown)},),
        ({"id": 0, "ontology_iri": str(EX.directUnion)},),
    )
    graph = load_ontology(FIXTURE)
    control = simple_asserted_mask(graph, vocab)
    assert not control.decisions[0, 0, 0]
    assert control.metadata["unsupported_expressions"]
    client, _ = local_client(monkeypatch, graph)
    assert client(str(EX.Unknown), str(EX.directUnion), str(EX.Unknown))[0] == Decision.UNKNOWN


@pytest.mark.parametrize(
    "axioms",
    [
        "ex:note a owl:AnnotationProperty; rdfs:subPropertyOf ex:p. ex:a ex:note ex:b.",
        "ex:note a owl:AnnotationProperty; rdfs:domain ex:A. ex:a ex:note 'text'.",
        "rdfs:label rdfs:subPropertyOf ex:p. ex:a rdfs:label ex:b.",
        "ex:U owl:unionOf (ex:A [owl:intersectionOf (ex:B ex:C)]).",
        "ex:U owl:unionOf ().",
        "owl:Thing rdfs:subClassOf ex:Impossible. ex:Impossible rdfs:subClassOf owl:Nothing.",
        "ex:a a ex:A.",
        "owl:Class owl:unionOf (owl:Nothing ex:B). ex:B rdfs:subClassOf owl:Nothing. ex:C a owl:Class.",
        "rdf:type rdfs:range ex:U. ex:U owl:unionOf (owl:Nothing ex:B). ex:B rdfs:subClassOf owl:Nothing. ex:C a owl:Class.",
        "owl:Class rdfs:subClassOf ex:U. ex:U owl:unionOf (owl:Nothing ex:B). ex:B rdfs:subClassOf owl:Nothing.",
        "ex:p a owl:DatatypeProperty. ex:a ex:p 2.",
        "ex:A owl:equivalentClass [a owl:Restriction; owl:onProperty ex:p; owl:someValuesFrom ex:B].",
    ],
)
def test_unsupported_or_non_inert_graphs_are_rejected(axioms):
    graph = Graph().parse(
        data="""@prefix ex: <urn:unsupported:> .
        @prefix owl: <http://www.w3.org/2002/07/owl#> .
        @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
        @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> ."""
        + axioms,
        format="turtle",
    )
    assert ontology_inventory(graph)["unsupported_constructs"]


def test_mapped_unknown_queries_rechecked_separately_from_200_decisions():
    vocab = Vocabulary(
        ({"id": 0, "ontology_iri": str(EX.A)},),
        ({"id": 0, "ontology_iri": str(EX.rangeUnion)}, {"id": 1}),
    )
    table = CompatibilityTable(np.full((1, 2, 1), -1, dtype=np.int8), vocab.digest, {})
    calls = []

    def reference(*query):
        calls.append(query)
        return Decision.UNKNOWN, {"reason": "union_without_member_witness"}

    result = check_conformance(table, vocab, reference)
    assert result["sample_count"] == 0 and result["mapped_unknown_sample_count"] == 1
    assert len(calls) == 1 and result["status"] == "insufficient_available_queries"


def test_unknown_conformance_never_queries_unprepared_mapped_classes():
    vocab = Vocabulary(
        tuple({"id": i, "ontology_iri": str(EX[f"C{i}"])} for i in range(2)),
        ({"id": 0, "ontology_iri": str(EX.rangeUnion)},),
    )
    decisions = np.full((2, 1, 2), Decision.UNRESOLVED, dtype=np.int8)
    decisions[0, 0, 0] = Decision.UNKNOWN
    table = CompatibilityTable(decisions, vocab.digest, {})
    calls = []

    def backend(subject, predicate, obj):
        calls.append((subject, predicate, obj))
        assert subject == obj == str(EX.C0)
        return Decision.UNKNOWN, {"reason": "union_without_member_witness"}

    result = check_conformance(table, vocab, backend)
    assert result["mapped_unknown_sample_count"] == 1 and len(calls) == 1


def test_real_boolean_training_conflict_stops_preparation(tmp_path, monkeypatch):
    from test_reasoning_preparation import EX as PREP
    from test_reasoning_preparation import LocalReference, preparation_config

    from kgr4vrd.adapters import graphdb
    from kgr4vrd.artifacts import read_json
    from kgr4vrd.reasoning import prepare_reasoning

    config, _ = preparation_config(tmp_path)
    root = Path(config["paths"]["data_root"])
    graph = load_ontology(root / "ontology.ttl")
    expression, head = BNode(), BNode()
    Collection(graph, head, [PREP.Animal, PREP.Vehicle])
    graph.add((expression, OWL.intersectionOf, head))
    graph.add((PREP.rides, RDFS.domain, expression))
    graph.serialize(root / "ontology.ttl", format="turtle")
    monkeypatch.setattr(graphdb, "GraphDB", LocalReference)
    with pytest.raises(ValueError, match="Raw training positives"):
        prepare_reasoning(config)
    output = root / config["intervention"]["knowledge_manifest"]
    assert not output.exists()
    assert read_json(output.parent / "training-conflicts.json")["conflicts"]
