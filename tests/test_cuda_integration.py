"""Opt-in node acceptance: registered FP32 heads and exact CUDA resume state.

Run with KGR4VRD_CUDA_TEST=1. Tests retain production process numerical defaults
and set only the node handoff's cuBLAS workspace before CUDA initialization.
Runtime flags are recorded in JUnit properties and checked for unexpected changes;
thread/RNG state is restored. If CUDA was initialized earlier, start the process
with CUBLAS_WORKSPACE_CONFIG=:4096:8 (or :16:8) as well.
"""

import copy
import json
import os
import random

import numpy as np
import pytest
import torch

from kgr4vrd.artifacts import read_json, source_snapshot, verify_run
from kgr4vrd.losses import compute_loss
from kgr4vrd.models import build_model
from kgr4vrd.smoke import prepare_smoke
from kgr4vrd.training import train

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("KGR4VRD_CUDA_TEST") != "1",
        reason="KGR4VRD_CUDA_TEST=1 and an actual CUDA device are required",
    ),
]


def runtime_flags():
    return {
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "matmul_precision": torch.get_float32_matmul_precision(),
    }


@pytest.fixture(scope="module")
def cuda_device(record_testsuite_property):
    previous_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    valid_workspaces = {":4096:8", ":16:8"}
    if torch.cuda.is_initialized() and previous_workspace not in valid_workspaces:
        pytest.fail("Start this process with CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA use")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = (
        previous_workspace if previous_workspace in valid_workspaces else ":4096:8"
    )
    flags = runtime_flags()
    record_testsuite_property("cuda_runtime_flags", json.dumps(flags, sort_keys=True))
    record_testsuite_property("CUBLAS_WORKSPACE_CONFIG", os.environ["CUBLAS_WORKSPACE_CONFIG"])
    previous = {
        "threads": torch.get_num_threads(),
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
    }
    cuda_rng = None
    try:
        if not torch.cuda.is_available():
            pytest.fail(
                "CUDA acceptance was requested, but this PyTorch runtime has no CUDA device"
            )
        cuda_rng = torch.cuda.get_rng_state_all()
        torch.set_num_threads(1)
        device = torch.device("cuda", torch.cuda.current_device())
        assert torch.empty(1, dtype=torch.float32, device=device).device == device
        yield device
        torch.cuda.synchronize(device)
        assert runtime_flags() == flags, "Production execution changed numerical runtime flags"
    finally:
        torch.use_deterministic_algorithms(flags["deterministic"], warn_only=flags["warn_only"])
        torch.backends.cudnn.benchmark = flags["benchmark"]
        torch.backends.cudnn.deterministic = flags["cudnn_deterministic"]
        torch.backends.cudnn.allow_tf32 = flags["cudnn_tf32"]
        torch.backends.cuda.matmul.allow_tf32 = flags["matmul_tf32"]
        torch.set_float32_matmul_precision(flags["matmul_precision"])
        torch.set_num_threads(previous["threads"])
        random.setstate(previous["python_rng"])
        np.random.set_state(previous["numpy_rng"])
        torch.set_rng_state(previous["torch_rng"])
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        if previous_workspace is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = previous_workspace


@pytest.mark.parametrize("model_name", ["geometry_mlp", "visual_mlp_v1"])
def test_registered_heads_forward_backward_on_cuda(cuda_device, model_name):
    # Twelve ordered pairs represent one four-object image, preserving image batching.
    dimension = 2 * 3 + 15 + (6144 if model_name == "visual_mlp_v1" else 0)
    model = build_model(model_name, dimension, 4).to(device=cuda_device, dtype=torch.float32)
    features = torch.randn(12, dimension, dtype=torch.float32, device=cuda_device)
    logits = model(features)
    assert logits.shape == (12, 5) and logits.dtype == torch.float32
    assert logits.device == cuda_device
    targets = torch.zeros_like(logits)
    targets[0, 0] = 1
    loss = compute_loss(
        logits,
        targets,
        torch.ones(12, dtype=torch.bool, device=cuda_device),
        torch.zeros(12, 4, dtype=torch.bool, device=cuda_device),
    )
    assert loss is not None
    loss[0].backward()
    for parameter in model.parameters():
        assert parameter.device == cuda_device and parameter.dtype == torch.float32
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    torch.cuda.synchronize(cuda_device)


def assert_state_equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_state_equal(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("interruption_epoch", [2, 3])
def test_combined_cuda_resume_across_semantic_gate(
    tmp_path, monkeypatch, cuda_device, interruption_epoch
):
    snapshot = source_snapshot()
    monkeypatch.setattr("kgr4vrd.training.source_snapshot", lambda: snapshot)
    config = prepare_smoke(tmp_path / "complete")
    config["intervention"]["condition"] = "combined"
    # This fixture seed gives positive incompatible logits, so the gate is exercised.
    config["training"].update(device="cuda", seed=23)
    assert config["training"]["precision"] == "float32"
    assert config["training"]["batch_images"] == 1
    complete = train(config)
    resumed_config = copy.deepcopy(config)
    resumed_config["paths"]["artifact_root"] = str(tmp_path / "resumed")
    interrupted = train(resumed_config, stop_after_epochs=interruption_epoch)
    assert read_json(interrupted / "run.json")["status"] == "interrupted"
    boundary = torch.load(
        interrupted / "checkpoints/latest.pt", map_location="cpu", weights_only=False
    )
    assert boundary["completed_epoch"] == interruption_epoch
    active_cells = sum(int(mask.sum()) for mask in boundary["active"].values())
    assert (active_cells > 0) == (interruption_epoch >= 3)
    resumed = train(resumed_config, resume=True)
    assert complete.name == resumed.name
    left, right = [
        torch.load(path / "checkpoints/latest.pt", map_location="cpu", weights_only=False)
        for path in (complete, resumed)
    ]
    for key in (
        "model",
        "optimiser",
        "active",
        "best_model",
        "best_epoch",
        "best_metric",
        "bad_epochs",
        "updates",
        "rng",
        "completed_epoch",
    ):
        assert_state_equal(left[key], right[key])
    assert left["completed_epoch"] == 4
    assert [row["active_cells"] for row in left["history"][:2]] == [0, 0]
    assert left["history"][2]["active_cells"] > 0 and left["history"][2]["semantic_loss"] > 0
    for a, b in zip(left["history"], right["history"]):
        assert_state_equal(
            {key: value for key, value in a.items() if not key.endswith("_seconds")},
            {key: value for key, value in b.items() if not key.endswith("_seconds")},
        )
    for path in (complete, resumed):
        assert verify_run(path)["status"] == "verified"
        assert read_json(path / "run.json")["effective_device"] == cuda_device.type
    a, b = [read_json(path / "selected.json") for path in (complete, resumed)]
    for key in ("epoch", "metric", "value", "policy_digest"):
        assert a[key] == b[key]
