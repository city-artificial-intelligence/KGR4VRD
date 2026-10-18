"""Attaching a launch must preserve live registry rows and existing dependencies."""

import copy
import json
import sys
import types

import pytest
import queue_step


def test_attach_existing_obligation_is_atomic_and_preserves_history(tmp_path, monkeypatch):
    dispatch = types.ModuleType("exact.experiments.dispatch")
    dispatch._validate = lambda batch, allocation: batch["launch"]
    monkeypatch.setitem(sys.modules, "exact.experiments.dispatch", dispatch)
    resources = {"cpus": 6, "gpus": 1, "memory_mb": 49152}
    pending = {
        "id": "pilot",
        "depends_on": ["reasoning"],
        "resources": resources,
        "scope": "preserve",
        "history": [{"original": True}],
    }
    registry = {
        "runs": [{"id": "reasoning", "step_id": "14482.26"}],
        "pending_batches": [pending, {"id": "development", "depends_on": ["pilot"]}],
        "remaining_work_status": "pending",
        "concurrent_field": "retained",
    }
    supervisor = tmp_path / "supervisor"
    supervisor.mkdir()
    path = supervisor / "registry.json"
    path.write_text(json.dumps(registry))
    spec = {
        "id": "pilot",
        "depends_on": ["reasoning"],
        "resources": resources,
        "directory": str(tmp_path / "attempt"),
        "python": sys.executable,
        "allocation": "14482",
        "cwd": str(tmp_path),
        "nonce": "test_nonce_000001",
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    result = queue_step.publish(spec_path, supervisor, tmp_path, prepare_existing=True)
    after = json.loads(path.read_text())
    assert after["runs"] == registry["runs"]
    assert after["pending_batches"][1] == registry["pending_batches"][1]
    assert after["concurrent_field"] == "retained"
    assert after["pending_batches"][0] == {**pending, **result}
    assert after["pending_batches"][0]["launch"]["tmux_socket"] == str(supervisor / "tmux.sock")
    before = path.read_bytes()
    # A fresh attempt cannot overwrite a ready/possibly dispatched descriptor.
    spec["directory"] = str(tmp_path / "duplicate")
    spec_path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="already registered"):
        queue_step.publish(spec_path, supervisor, tmp_path, prepare_existing=True)
    assert path.read_bytes() == before
    # A changed dependency or resource demand cannot replace an obligation either.
    changed = copy.deepcopy(registry)
    changed["pending_batches"][0]["depends_on"] = ["different"]
    path.write_text(json.dumps(changed))
    spec["directory"] = str(tmp_path / "mismatch")
    spec_path.write_text(json.dumps(spec))
    before = path.read_bytes()
    with pytest.raises(ValueError, match="already registered"):
        queue_step.publish(spec_path, supervisor, tmp_path, prepare_existing=True)
    assert path.read_bytes() == before
