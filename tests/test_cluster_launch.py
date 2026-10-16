"""The detached worker must retain failures and refuse duplicate attempts."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


def test_encoder_fixture_reads_only_training_objects(tmp_path, monkeypatch):
    from test_data import source_fixture

    from kgr4vrd.adapters.nesy4vrd import prepare_dataset
    from kgr4vrd.artifacts import read_json
    from kgr4vrd.data import load_objects, load_relations

    config = source_fixture(tmp_path / "original")
    original_path = prepare_dataset(config)
    config["dataset"] = {"manifest": str(original_path)}
    script = Path(__file__).resolve().parents[1] / "scripts/prepare_node.py"
    specification = importlib.util.spec_from_file_location("prepare_node", script)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    original_open = Path.open

    def reject_targets(path, *args, **kwargs):
        if path.is_relative_to(original_path.parent / "relations"):
            raise AssertionError("Original relation targets must not be opened")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", reject_targets)
    destination = module.encoder_fixture(config, tmp_path / "fixture")
    fixture = read_json(destination)
    dataset = Path(fixture["dataset"]["manifest"])
    training_ids = set(load_objects(original_path, "train"))
    for split in ("train", "validation", "test"):
        objects = load_objects(dataset, split)
        assert len(objects) == 1 and set(objects) <= training_ids
        assert all(not value for value in load_relations(dataset, split).values())


def test_worker_receipts_failure_pause_and_duplicate_ownership(tmp_path):
    worker = Path(__file__).resolve().parents[1] / "scripts/slurm_worker.py"
    environment = dict(os.environ, SLURM_JOB_ID="test", SLURM_STEP_ID="1")
    for name, command, paused, expected in (
        ("complete", [sys.executable, "-c", "pass"], False, 0),
        ("failure", [sys.executable, "-c", "raise SystemExit(7)"], False, 7),
        ("paused", [sys.executable, "-c", "raise AssertionError('must not run')"], True, 1),
    ):
        directory = tmp_path / name
        directory.mkdir()
        pause = directory / "PAUSE"
        if paused:
            pause.touch()
        spec = dict(
            directory=str(directory),
            allocation="test",
            nonce="test_nonce_0001",
            resources=dict(cpus=1, gpus=0, memory_mb=128),
            cwd=str(tmp_path),
            commands=[command],
            pause_paths=[str(pause)],
        )
        path = directory / "spec.json"
        path.write_text(json.dumps(spec))
        result = subprocess.run(
            [sys.executable, str(worker), str(path)],
            env=environment,
            capture_output=True,
            text=True,
        )
        assert result.returncode == expected, result.stderr
        receipt = json.loads((directory / "completion.json").read_text())
        assert receipt["status"] == ("complete" if expected == 0 else "failed")
        assert receipt["step_id"] == "test.1"
        assert receipt["dispatch_nonce"] == spec["nonce"]
        assert int((directory / "exit-code").read_text()) == expected
        previous = (directory / "completion.json").read_bytes()
        duplicate = subprocess.run(
            [sys.executable, str(worker), str(path)],
            env=environment,
            capture_output=True,
            text=True,
        )
        assert duplicate.returncode != 0
        assert (directory / "completion.json").read_bytes() == previous
