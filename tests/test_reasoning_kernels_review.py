"""Independent, hand-specified scientific fixtures; no live reasoner or network."""

from collections import Counter
from dataclasses import replace

import numpy as np
import pytest
from owlrl import OWLRL_Semantics
from rdflib import OWL, RDF, RDFS, Graph, Namespace, URIRef

from kgr4vrd.artifacts import read_json, read_jsonl, write_json, write_jsonl
from kgr4vrd.controls import frequency_weights, shuffle_added_labels
from kgr4vrd.reasoning import (
    CompatibilityTable,
    augment_scene,
    construct_profile,
    graph_digest,
    load_knowledge,
    load_simple_filter,
    prepare_compatibility,
    save_knowledge,
    simple_asserted_mask,
)
from kgr4vrd.types import Decision, Object, Relation, Scene, Vocabulary

EX = Namespace("urn:kgr4vrd:independent-review:")


def vocabulary(classes, predicates):
    def entries(names):
        return tuple(
            {
                "id": index,
                "label": name,
                "source_id": str(index),
                **({"ontology_iri": str(EX[name])} if name != "unmapped" else {}),
            }
            for index, name in enumerate(names)
        )

    return Vocabulary(entries(classes), entries(predicates))


def scene(image_id="one", class_ids=(0, 0, 0)):
    return Scene(
        image_id,
        "synthetic-unused.png",
        20,
        20,
        tuple(
            Object(chr(97 + index), class_id, (0.0, 0.0, 1.0, 1.0))
            for index, class_id in enumerate(class_ids)
        ),
    )


def augmentation_fixture():
    vocab = vocabulary(
        ["Person"],
        [
            "symmetric",
            "transitive",
            "equivalent_a",
            "equivalent_b",
            "child_property",
            "parent_property",
            "inverse_a",
            "inverse_b",
        ],
    )
    ontology = Graph()
    for triple in (
        (EX.symmetric, RDF.type, OWL.SymmetricProperty),
        (EX.transitive, RDF.type, OWL.TransitiveProperty),
        (EX.equivalent_a, OWL.equivalentProperty, EX.equivalent_b),
        (EX.child_property, RDFS.subPropertyOf, EX.parent_property),
        (EX.inverse_a, OWL.inverseOf, EX.inverse_b),
        (EX.Person, RDFS.subClassOf, EX.Animal),
    ):
        ontology.add(triple)
    raw = (
        Relation("a", 0, "b"),
        Relation("a", 1, "b"),
        Relation("b", 1, "c"),
        Relation("a", 2, "b"),
        Relation("a", 4, "b"),
        Relation("a", 6, "b"),
    )
    return vocab, ontology, raw


def test_steu_materialisation_is_exact_deduplicated_and_isolated_per_image():
    vocab, ontology, raw = augmentation_fixture()
    original_digest = graph_digest(ontology)
    expected_added = {
        Relation("b", 0, "a"),
        Relation("a", 1, "c"),
        Relation("a", 3, "b"),
        Relation("a", 5, "b"),
    }
    expanded, metadata = augment_scene(scene(), raw + raw, vocab, ontology)
    assert set(expanded) == set(raw) | expected_added
    assert len(expanded) == 10
    assert metadata["asserted_count"] == 6
    assert metadata["added_count"] == 4
    assert metadata["newly_supervised_pairs"] == 2
    assert {r["origin"] for r in metadata["added_relations"]} == {"inferred"}
    assert Relation("b", 7, "a") not in expanded
    assert metadata["removed_axioms"] == [
        [str(EX.inverse_a), str(OWL.inverseOf), str(EX.inverse_b)]
    ]
    again, _ = augment_scene(scene(), reversed(raw), vocab, ontology)
    assert again == expanded
    other, other_metadata = augment_scene(scene("two"), (), vocab, ontology)
    assert other == () and other_metadata["added_count"] == 0
    assert graph_digest(ontology) == original_digest


def test_profiles_remove_asserted_families_before_closure_and_resource_limits_fail():
    vocab, ontology, raw = augmentation_fixture()
    properties = [entry["ontology_iri"] for entry in vocab.predicates]
    stripped, record = construct_profile(ontology, properties, "none")
    assert set(stripped) == {(EX.Person, RDFS.subClassOf, EX.Animal)}
    assert len(record["removed_axioms"]) == 5
    no_rules, _ = augment_scene(scene(), raw, vocab, ontology, profile="none")
    all_rules, _ = augment_scene(scene(), raw, vocab, ontology, profile="all")
    assert set(no_rules) == set(raw)
    assert Relation("b", 7, "a") in all_rules
    cyclic, metadata = augment_scene(
        scene(), (Relation("a", 1, "b"), Relation("b", 1, "a")), vocab, ontology
    )
    assert set(cyclic) == {Relation("a", 1, "b"), Relation("b", 1, "a")}
    assert metadata["excluded_self"] == 2
    with pytest.raises(RuntimeError, match="resource limit"):
        augment_scene(scene(), raw, vocab, ontology, max_triples=1)


def typed_fixture():
    vocab = vocabulary(
        ["Person", "Horse", "Car", "Animal", "Vehicle", "Rock"], ["relation", "unmapped"]
    )
    ontology = Graph()
    for triple in (
        (EX.Person, RDFS.subClassOf, EX.Animal),
        (EX.Horse, RDFS.subClassOf, EX.Animal),
        (EX.Car, RDFS.subClassOf, EX.Vehicle),
        (EX.Animal, OWL.disjointWith, EX.Vehicle),
        (EX.relation, RDFS.range, EX.Animal),
    ):
        ontology.add(triple)

    calls = []

    def local_fixture_backend(subject_class, predicate, object_class):
        # An independent fresh OWLRL graph is a fixture oracle, not a claim of
        # equivalence to the untested real GraphDB implementation.
        calls.append((subject_class, predicate, object_class))
        query = Graph()
        for triple in ontology:
            query.add(triple)
        query.add((EX.subject, RDF.type, URIRef(subject_class)))
        query.add((EX.object, RDF.type, URIRef(object_class)))
        query.add((EX.subject, URIRef(predicate), EX.object))
        reasoner = OWLRL_Semantics(query, axioms=False, daxioms=False)
        reasoner.closure()
        if reasoner.error_messages:
            return Decision.INCOMPATIBLE, {"reason": "; ".join(reasoner.error_messages)}
        return Decision.COMPATIBLE, {"reason": "consistent in the independent fixture profile"}

    return vocab, ontology, local_fixture_backend, calls


def test_asserted_control_differs_from_inferred_contradiction_and_open_world_absence():
    vocab, ontology, backend, calls = typed_fixture()
    pairs = [(0, 1), (1, 0), (0, 2), (2, 0), (0, 4), (4, 0), (0, 5), (5, 0)]
    full = prepare_compatibility(
        vocab, backend, {"profile": "independent fixture"}, class_pairs=pairs
    )
    simple = simple_asserted_mask(ontology, vocab)
    assert full.decisions[0, 0, 1] == Decision.COMPATIBLE
    assert full.decisions[0, 0, 2] == Decision.INCOMPATIBLE
    assert not simple.decisions[0, 0, 2]  # Car -> Vehicle requires subclass inference.
    assert simple.decisions[0, 0, 4]  # Vehicle/Animal is explicitly disjoint.
    assert full.decisions[0, 0, 4] == Decision.INCOMPATIBLE
    assert full.decisions[0, 0, 5] == Decision.COMPATIBLE  # Rock not known disjoint.
    assert not simple.decisions[0, 0, 5]
    assert np.all(full.decisions[0, 1, [1, 2, 4, 5]] == Decision.UNKNOWN)
    assert len(calls) == len(pairs)  # Unmapped predicates never invoke a backend.
    one = full.lookup(scene("first", (0, 2)))
    two = full.lookup(scene("later", (0, 2)))
    np.testing.assert_array_equal(one, two)
    assert one[0, 0] == Decision.INCOMPATIBLE
    with pytest.raises(ValueError, match="outside prepared reasoning universe"):
        full.lookup(scene("unprepared", (1, 2)))
    ontology.add((EX.background_individual, RDF.type, EX.Animal))
    ontology.add((EX.background_individual, RDF.type, EX.Vehicle))
    with pytest.raises(ValueError, match="Inconsistent OWL-RL graph"):
        augment_scene(scene(), (), vocab, ontology)


def test_frozen_table_content_identity_serialization_and_progress_cache(tmp_path):
    vocab, _, backend, calls = typed_fixture()
    progress = tmp_path / "progress.jsonl"
    table = prepare_compatibility(
        vocab,
        backend,
        {"profile_digest": "fixed"},
        class_pairs=[(0, 1), (1, 0)],
        progress_path=progress,
    )
    first_calls = len(calls)
    resumed = prepare_compatibility(
        vocab,
        backend,
        {"profile_digest": "fixed"},
        class_pairs=[(0, 1), (1, 0)],
        progress_path=progress,
    )
    assert len(calls) == first_calls
    assert resumed.digest == table.digest
    with pytest.raises(ValueError, match="identity changed"):
        prepare_compatibility(
            vocab,
            backend,
            {"profile_digest": "changed"},
            class_pairs=[(0, 1), (1, 0)],
            progress_path=progress,
        )
    with pytest.raises(ValueError):
        table.decisions[0, 0, 1] = 1
    with pytest.raises(ValueError):
        table.decisions.setflags(write=True)
    source = np.zeros((6, 2, 6), dtype=np.int8)
    copied = CompatibilityTable(source, vocab.digest, {"profile": "copy"})
    source[0, 0, 0] = 1
    assert copied.decisions[0, 0, 0] == 0
    path = save_knowledge(tmp_path / "table/manifest.json", table, vocab)
    assert load_knowledge(path, vocab).digest == table.digest
    altered = replace(vocab, predicates=tuple(reversed(vocab.predicates)))
    with pytest.raises(ValueError, match="vocabulary identity"):
        load_knowledge(path, altered)
    metadata = read_json(path)
    metadata["metadata"]["profile_digest"] = "tampered"
    write_json(path, metadata)
    with pytest.raises(ValueError, match="scientific identity"):
        load_knowledge(path, vocab)


def test_simple_boolean_artifact_cannot_replace_full_compatibility(tmp_path):
    vocab, ontology, _, _ = typed_fixture()
    simple = simple_asserted_mask(ontology, vocab)
    path = save_knowledge(tmp_path / "simple/manifest.json", simple, vocab)
    assert load_simple_filter(path, vocab).digest == simple.digest
    with pytest.raises(ValueError, match="cannot replace full compatibility"):
        load_knowledge(path, vocab)


def test_reasoning_progress_validates_records_and_recovers_only_truncated_tail(tmp_path):
    vocab, _, backend, calls = typed_fixture()
    path = tmp_path / "progress.jsonl"
    table = prepare_compatibility(
        vocab,
        backend,
        {"profile_digest": "fixed"},
        class_pairs=[(0, 1), (1, 0)],
        progress_path=path,
    )
    original = read_jsonl(path)
    for field, value in (("key", [0, 0, 0]), ("decision", -2), ("reason", "")):
        broken = [dict(row) for row in original]
        broken[0][field] = value
        write_jsonl(path, broken)
        with pytest.raises(ValueError, match="Invalid interrupted reasoning decision"):
            prepare_compatibility(
                vocab,
                backend,
                {"profile_digest": "fixed"},
                class_pairs=[(0, 1), (1, 0)],
                progress_path=path,
            )
    write_jsonl(path, original[:2])
    with path.open("ab") as stream:
        stream.write(b'{"identity_digest":"interrupted-write')
    before = len(calls)
    resumed = prepare_compatibility(
        vocab,
        backend,
        {"profile_digest": "fixed"},
        class_pairs=[(0, 1), (1, 0)],
        progress_path=path,
    )
    assert len(calls) == before + 1  # Only the uncached mapped query is repeated.
    assert resumed.digest == table.digest
    assert len(read_jsonl(path)) == 4 and path.read_bytes().endswith(b"\n")


def test_frequency_weights_match_only_supported_positive_mass_and_threshold():
    weights, record = frequency_weights([2, 4, 0, 0], [4, 2, 1, 0])
    np.testing.assert_array_equal(weights, [3.0, 1.5, 1.0, 1.0])
    np.testing.assert_array_equal(weights[:2] * [2, 4], [6, 6])
    assert record["unmatched_count"] == 1
    assert record["unmatched_fraction"] == pytest.approx(1 / 7)
    assert record["requires_shuffle"]
    _, boundary = frequency_weights([1, 0], [19, 1])
    assert boundary["unmatched_fraction"] == 0.05 and not boundary["requires_shuffle"]
    _, degenerate = frequency_weights([1, 0], [0, 0])
    assert degenerate["degenerate"] and not degenerate["requires_shuffle"]
    with pytest.raises(ValueError, match="nonnegative integers"):
        frequency_weights([1.5], [1])


def test_shuffle_preserves_pair_predicate_counts_and_raw_labels_without_collision():
    raw = {"one": (Relation("a", 9, "b"),), "two": (Relation("c", 9, "d"),)}
    added = {
        "one": tuple(Relation(f"s{i}", i % 3, f"o{i}") for i in range(15)),
        "two": tuple(Relation(f"s{i}", (i + 1) % 3, f"o{i}") for i in range(15)),
    }
    expanded = {image: raw[image] + relations for image, relations in added.items()}
    shuffled, record = shuffle_added_labels(raw, expanded)
    repeated, repeated_record = shuffle_added_labels(
        dict(reversed(list(raw.items()))), dict(reversed(list(expanded.items())))
    )
    assert shuffled == repeated and record == repeated_record
    original_cells = {
        (image, r.subject_id, r.object_id, r.predicate_id)
        for image, relations in added.items()
        for r in relations
    }
    final_cells = {
        (image, r.subject_id, r.object_id, r.predicate_id)
        for image, relations in shuffled.items()
        for r in set(relations) - set(raw[image])
    }
    assert Counter(c[-1] for c in final_cells) == Counter(c[-1] for c in original_cells)
    assert Counter(c[:-1] for c in final_cells) == Counter(c[:-1] for c in original_cells)
    assert len(final_cells) == len(original_cells) == record["added_count"]
    assert record["attempts"] == 3000
    assert record["changed_cell_fraction"] == len(final_cells - original_cells) / 30
    for image in raw:
        assert set(raw[image]) <= set(shuffled[image])
        assert len(shuffled[image]) == len(set(shuffled[image]))


def test_shuffle_reports_infeasibility_and_degenerate_input():
    raw = {"one": (Relation("a", 1, "b"), Relation("c", 0, "d"))}
    # The only proposed exchange would collide with raw annotations.
    expanded = {"one": raw["one"] + (Relation("a", 0, "b"), Relation("c", 1, "d"))}
    shuffled, record = shuffle_added_labels(raw, expanded)
    assert set(shuffled["one"]) == set(expanded["one"])
    assert record["attempts"] == 200 and record["accepted_swaps"] == 0
    assert record["changed_cell_fraction"] == 0 and not record["adequate"]
    unchanged, empty = shuffle_added_labels(raw, raw)
    assert set(unchanged["one"]) == set(raw["one"])
    assert empty["degenerate"] and empty["attempts"] == 100 and not empty["adequate"]
