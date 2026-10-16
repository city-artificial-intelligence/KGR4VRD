"""Inventory the acquired public inputs, prepare canonical data and write the finite plan."""

import argparse
import hashlib
import os
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

from rdflib import OWL, RDF, Graph, URIRef
from rdflib.compare import graph_diff, to_isomorphic

from kgr4vrd.adapters.nesy4vrd import prepare_dataset
from kgr4vrd.artifacts import file_digest, read_json, utc_now, write_json
from kgr4vrd.campaign import plan
from kgr4vrd.config import load_config
from kgr4vrd.reasoning import ontology_inventory


def only(root, filename):
    paths = list(root.rglob(filename))
    if len(paths) != 1:
        raise ValueError(f"Expected one {filename}; found {paths}")
    return paths[0]


def extract_archive(archive, destination):
    with zipfile.ZipFile(archive) as stream:
        for entry in stream.infolist():
            target = destination / entry.filename
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError("Archive path escapes extraction root")
            if "__MACOSX" in target.parts or target.name == ".DS_Store":
                continue
            stream.extract(entry, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    args = parser.parse_args()
    repo, campaign = args.repository.resolve(), args.campaign.resolve()
    assert os.environ.get("SLURM_STEP_ID", "").isdigit()
    data = repo / "data"
    archive = data / "raw/NeSy4VRD-1.1.zip"
    with archive.open("rb") as stream:
        md5 = hashlib.file_digest(stream, "md5").hexdigest()
    if md5 != "4c83779814209077e25a7a707f441eed":
        raise ValueError("NeSy4VRD 1.1 archive does not match the publisher checksum")
    extracted = data / "raw/NeSy4VRD-1.1"
    if not extracted.exists():
        staging = Path(tempfile.mkdtemp(prefix="nesy4vrd-extract-", dir=data / "raw"))
        extract_archive(archive, staging)
        staging.rename(extracted)
    images = extracted / "images"
    if not images.exists():
        staging = Path(tempfile.mkdtemp(prefix="images-extract-", dir=extracted))
        for filename in ("vrd_train_images.zip", "vrd_test_images.zip"):
            extract_archive(only(extracted, filename), staging)
        staging.rename(images)
    recovered_split = campaign / "recovered-split.json"
    split_sources = {
        "train": only(extracted, "nesy4vrd_image_names_train_training.json"),
        "validation": only(extracted, "nesy4vrd_image_names_train_validation.json"),
    }
    write_json(recovered_split, {key: read_json(path) for key, path in split_sources.items()})
    sources = {
        "object_vocabulary": only(extracted, "nesy4vrd_objects.json"),
        "predicate_vocabulary": only(extracted, "nesy4vrd_predicates.json"),
        "train_annotations": only(extracted, "nesy4vrd_annotations_train.json"),
        "test_annotations": only(extracted, "nesy4vrd_annotations_test.json"),
        "images": images,
        "historical_split": recovered_split,
    }
    author = data / "raw/NeSy4VRD-source"
    author_commit = subprocess.check_output(
        ["git", "-C", str(author), "rev-parse", "HEAD"], text=True
    ).strip()
    # The exact historical variants are unavailable; use an explicitly named public profile.
    ontology_source = only(extracted, "vrd_world_v1_2.owl")
    assert file_digest(ontology_source) == file_digest(author / "ontology/vrd_world_v1_2.owl")
    graph = Graph().parse(ontology_source, format="turtle")
    inventory = ontology_inventory(graph)
    write_json(campaign / "ontology-structural-inventory.json", inventory)
    if inventory["unsupported_constructs"]:
        raise ValueError("Public ontology has unsupported constructs; do not weaken it silently")
    profile = data / "raw/public-vrd-world-1.2-20261016"
    profile.mkdir(exist_ok=True)
    for name in ("asserted-augmentation.ttl", "asserted-checker.ttl"):
        if (profile / name).exists():
            assert file_digest(profile / name) == file_digest(ontology_source)
        else:
            shutil.copyfile(ontology_source, profile / name)
    mappings = []
    prefix = "http://www.semanticweb.org/nesy4vrd/ontologies/vrd_world#"
    for kind, filename, declaration in (
        ("object_class", "object_vocabulary", OWL.Class),
        ("predicate", "predicate_vocabulary", OWL.ObjectProperty),
    ):
        for index, label in enumerate(read_json(sources[filename])):
            words = label.lower().split()
            local = "".join(word.capitalize() for word in words)
            if kind == "predicate":
                local = words[0] + "".join(word.capitalize() for word in words[1:])
            iri = prefix + local
            if kind == "object_class" and label in {"road", "coat", "plane"}:
                mappings.append(
                    dict(
                        label_kind=kind,
                        label_id=index,
                        iri=None,
                        mapping_relation="unmapped",
                        rationale="Publisher retains this deprecated vocabulary ID but states that no annotations use it; preserve the column and explicit unknown status.",
                        provenance="NeSy4VRD 1.1 nesy4vrd_annotations/readme.txt section 3 (archive checksum recorded in acquisition.json)",
                    )
                )
                continue
            if (URIRef(iri), RDF.type, declaration) not in graph:
                raise ValueError(f"Author mapping lacks the expected declaration: {label}: {iri}")
            mappings.append(
                dict(
                    label_kind=kind,
                    label_id=index,
                    iri=iri,
                    mapping_relation="exact",
                    rationale="Author-specified case conversion; declaration verified in the pinned ontology.",
                    provenance=f"https://github.com/djherron/NeSy4VRD/blob/{author_commit}/sample_code/nesy4vrd_utils4.py",
                )
            )
    write_json(profile / "mapping.json", dict(mappings=mappings))
    original = Graph().parse(author / "ontology/vrd_world_v1.owl", format="turtle")
    shared, removed, added = graph_diff(to_isomorphic(original), to_isomorphic(graph))
    removed.serialize(campaign / "public-v1-to-v1.2-removed.nt", format="nt")
    added.serialize(campaign / "public-v1-to-v1.2-added.nt", format="nt")
    weights = data / "weights/resnet50-11ad3fa6.pth"
    write_json(
        data / "weights/resnet50-v2.json",
        dict(
            schema_version="kgr4vrd.encoder/1",
            encoder="resnet50",
            weights="IMAGENET1K_V2",
            torchvision_version="0.17.2",
            weights_file=dict(path=weights.name, sha256=file_digest(weights)),
            transform=dict(
                resize_size=232,
                crop_size=224,
                interpolation="bilinear",
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                antialias=True,
            ),
        ),
    )
    source = Path(__file__).resolve().parents[1]
    config = load_config(
        source / "configs/nesy4vrd-cuda.toml", data_root=data, artifact_root=campaign / "research"
    )
    tag = "cluster-20261016-01"
    config["dataset"].update(
        manifest=f"prepared/nesy4vrd/{tag}/dataset.json",
        split=f"prepared/nesy4vrd/{tag}/splits.json",
    )
    config["intervention"].update(
        augmentation_manifest=f"knowledge/{tag}/augmentation/manifest.json",
        knowledge_manifest=f"knowledge/{tag}/compatibility/manifest.json",
        control_manifest=f"knowledge/{tag}/control/manifest.json",
    )
    config["evaluation"]["simple_filter_manifest"] = f"knowledge/{tag}/simple-filter/manifest.json"
    config["preparation"] = dict(
        source_version="1.1",
        output_directory=f"prepared/nesy4vrd/{tag}",
        sources={key: os.path.relpath(path, data) for key, path in sources.items()},
        augmentation_ontology=str((profile / "asserted-augmentation.ttl").relative_to(data)),
        compatibility_ontology=str((profile / "asserted-checker.ttl").relative_to(data)),
        augmentation_format="turtle",
        compatibility_format="turtle",
        mapping=str((profile / "mapping.json").relative_to(data)),
        closure_timeout_seconds=30,
        closure_max_triples=1000000,
        license_provenance=dict(
            reference="https://zenodo.org/records/17076303; retained source package and ontology declarations",
            redistribution_status="Local research use only; no redistribution performed or image rights inferred from the code license.",
        ),
    )
    config["graphdb"] = dict(
        endpoint="http://127.0.0.1:17200",
        repository="kgr4vrd-20261016",
        ruleset="owl2-rl",
        expected_version="10.8.14",
    )
    config["visual"] = dict(
        weights_manifest="weights/resnet50-v2.json",
        features_manifest=f"features/{tag}/manifest.json",
        crop_batch_size=16,
    )
    config_path = campaign / "config.json"
    write_json(config_path, config)
    config = load_config(config_path)
    write_json(
        campaign / "acquisition.json",
        dict(
            retrieved_at=utc_now(),
            source_url="https://zenodo.org/records/17076303",
            archive_md5=md5,
            archive_sha256=file_digest(archive),
            author_source_commit=author_commit,
            public_profile="public_vrd_world_1_2_shared_asserted_S_TEU",
            ontology_sha256=file_digest(ontology_source),
            mapping_sha256=file_digest(profile / "mapping.json"),
            public_v1_difference=dict(common=len(shared), removed=len(removed), added=len(added)),
            historical_variants="v1_1p augmentation and v1_2_disjoint checker not recovered; no historical replication claimed",
            asserted_inputs="Separate byte-identical v1.2 augmentation/checker snapshots; S_TEU removes asserted inverse axioms only for augmentation",
            sources={
                key: dict(path=str(path), sha256=file_digest(path))
                for key, path in sources.items()
                if path.is_file()
            },
            weights_sha256=file_digest(weights),
        ),
    )
    dataset = prepare_dataset(config)
    from kgr4vrd.data import load_objects

    deprecated = {row["label_id"] for row in mappings if row["mapping_relation"] == "unmapped"}
    observed = {
        obj.class_id
        for split in ("train", "validation", "test")
        for scene in load_objects(dataset, split).values()
        for obj in scene.objects
    }
    if deprecated & observed:
        raise ValueError(
            "Publisher-deprecated class IDs occur in supplied objects; mapping adjudication required"
        )
    write_json(
        campaign / "vocabulary-coverage.json",
        dict(
            vocabulary_classes=112,
            observed_classes=len(observed),
            deprecated_unused_ids=sorted(deprecated),
            interpretation="112 stable vocabulary slots, 109 active classes as documented by publisher; no class reindexing",
        ),
    )
    campaign_input = campaign / "campaign.toml"
    campaign_input.write_text(
        'schema_version = "kgr4vrd.campaign-config/1"\nbase_config = "config.json"\n'
    )
    plan_path = plan(campaign_input, campaign / "plan")
    print(dict(dataset=str(dataset), plan=str(plan_path)), flush=True)


if __name__ == "__main__":
    main()
