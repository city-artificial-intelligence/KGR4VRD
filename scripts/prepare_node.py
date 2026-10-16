"""Dedicated GraphDB lifetime and real integration checks for this serial campaign."""

import argparse
import copy
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from kgr4vrd.artifacts import canonical_digest, file_digest, read_json, write_json, write_jsonl
from kgr4vrd.config import data_path, load_config
from kgr4vrd.data import load_objects


def encoder_fixture(config, directory):
    """Use three training images and supplied objects; copy no relation targets."""
    source = data_path(config, config["dataset"]["manifest"])
    original = read_json(source)
    scenes = load_objects(source, "train")
    chosen = sorted(key for key, scene in scenes.items() if len(scene.objects) >= 2)[:3]
    assert len(chosen) == 3
    directory.mkdir(parents=True, exist_ok=False)
    manifest = copy.deepcopy(original)
    shutil.copyfile(source.parent / original["vocabulary"]["path"], directory / "vocabulary.json")
    manifest["vocabulary"] = dict(
        path="vocabulary.json", sha256=file_digest(directory / "vocabulary.json")
    )
    splits = dict(
        schema_version="kgr4vrd.splits/1",
        provenance="encoder acceptance only; three training images; no relation targets",
    )
    manifest["partitions"] = {}
    for split, key in zip(("train", "validation", "test"), chosen):
        splits[split] = [key]
        write_jsonl(directory / "objects" / (split + ".jsonl"), [asdict(scenes[key])])
        write_jsonl(
            directory / "relations" / (split + ".jsonl"), [dict(image_id=key, relations=[])]
        )
        manifest["partitions"][split] = {
            kind: dict(
                path=f"{kind}/{split}.jsonl",
                sha256=file_digest(directory / kind / (split + ".jsonl")),
            )
            for kind in ("objects", "relations")
        }
        count = len(scenes[key].objects)
        manifest["partitions"][split].update(
            image_count=1, object_count=count, pair_count=count * (count - 1), relation_count=0
        )
    write_json(directory / "splits.json", splits)
    manifest["splits"] = dict(path="splits.json", sha256=file_digest(directory / "splits.json"))
    manifest["split_digest"] = canonical_digest(splits)
    manifest["image_root"] = os.path.relpath(source.parent / original["image_root"], directory)
    manifest["image_manifest"] = [r for r in original["image_manifest"] if r["image_id"] in chosen]
    manifest["dataset_id"] = "nesy4vrd-encoder-acceptance-only"
    write_json(directory / "dataset.json", manifest)
    fixture = copy.deepcopy(config)
    fixture["dataset"]["manifest"] = str(directory / "dataset.json")
    fixture["dataset"]["split"] = str(directory / "splits.json")
    write_json(directory / "config.json", fixture)
    return directory / "config.json"


def repository_config(repository):
    return f'''@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix rep: <http://www.openrdf.org/config/repository#> .
@prefix sr: <http://www.openrdf.org/config/repository/sail#> .
@prefix sail: <http://www.openrdf.org/config/sail#> .
@prefix graphdb: <http://www.ontotext.com/config/graphdb#> .
[] a rep:Repository; rep:repositoryID "{repository}";
 rdfs:label "Exclusive KGR4VRD preparation";
 rep:repositoryImpl [rep:repositoryType "graphdb:SailRepository";
 sr:sailImpl [sail:sailType "graphdb:Sail";
 graphdb:ruleset "owl2-rl"; graphdb:check-for-inconsistencies "true";
 graphdb:disable-sameAs "false"; graphdb:enable-context-index "true";
 graphdb:repository-type "file-repository"; graphdb:storage-folder "storage";
 graphdb:entity-index-size "1000000"; graphdb:query-timeout "30";
 graphdb:throw-QueryEvaluationException-on-timeout "true"]].
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("acceptance", "reasoning"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--graphdb", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    assert os.environ.get("SLURM_STEP_ID", "").isdigit()
    endpoint = config["graphdb"]["endpoint"]
    # Each attempt gets its own home; never clear or take over an existing service.
    home = work / ("graphdb-" + os.environ["SLURM_JOB_ID"] + "-" + os.environ["SLURM_STEP_ID"])
    home.mkdir(exist_ok=False)
    port = endpoint.rsplit(":", 1)[1]
    assert endpoint == "http://127.0.0.1:" + port
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", int(port)))
    environment = dict(
        os.environ, GDB_MIN_MEM="1g", GDB_MAX_MEM="8g", GDB_JAVA_OPTS="-XX:ActiveProcessorCount=6"
    )
    with (work / "graphdb.log").open("a") as log:
        service = subprocess.Popen(
            [
                str(args.graphdb.resolve()),
                "-Dgraphdb.home=" + str(home),
                "-Dgraphdb.connector.port=" + port,
                "-Dgraphdb.connector.address=127.0.0.1",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        try:
            for _ in range(180):
                if service.poll() is not None:
                    raise RuntimeError("GraphDB exited; see " + str(work / "graphdb.log"))
                try:
                    with urlopen(endpoint + "/rest/info/version", timeout=2) as response:
                        version = json.load(response)
                    break
                except (URLError, TimeoutError):
                    time.sleep(1)
            else:
                raise RuntimeError("GraphDB startup exceeded 180 seconds")
            assert version["productVersion"] == config["graphdb"]["expected_version"], version
            write_json(work / "graphdb-version.json", version)
            repository = config["graphdb"]["repository"]
            template = work / "repository.ttl"
            template.write_text(repository_config(repository))
            subprocess.run(
                [
                    "curl",
                    "-fSs",
                    "-X",
                    "POST",
                    "-F",
                    "config=@" + str(template),
                    endpoint + "/rest/repositories",
                ],
                check=True,
            )
            if args.stage == "acceptance":
                backend = work / "backend.json"
                write_json(backend, config["graphdb"])
                fixture = encoder_fixture(config, work / "encoder-fixture")
                test_env = dict(
                    os.environ,
                    KGR4VRD_CUDA_TEST="1",
                    KGR4VRD_GRAPHDB_CONFIG=str(backend),
                    KGR4VRD_ENCODER_CONFIG=str(fixture),
                )
                xml = work / "integration.xml"
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pytest",
                        "-q",
                        "-m",
                        "integration",
                        "--junitxml=" + str(xml),
                    ],
                    env=test_env,
                    check=True,
                )
                root = ET.parse(xml).getroot()
                suites = list(root.iter("testsuite"))
                assert sum(int(s.get("tests", "0")) for s in suites) > 0
                assert not any(
                    int(s.get(field, "0"))
                    for s in suites
                    for field in ("skipped", "errors", "failures")
                )
                write_json(
                    work / "accepted.json",
                    dict(
                        status="passed",
                        junit_sha256=file_digest(xml),
                        config_sha256=file_digest(args.config),
                    ),
                )
            else:
                from kgr4vrd.reasoning import prepare_reasoning

                prepare_reasoning(config)
        finally:
            service.terminate()
            try:
                service.wait(timeout=30)
            except subprocess.TimeoutExpired:
                service.kill()
                service.wait()


if __name__ == "__main__":
    main()
