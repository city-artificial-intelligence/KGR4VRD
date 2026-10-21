"""Reporting must recompute saved scores and reject incomplete accounting."""

import copy
import csv

import pytest
import report_stage
import torch

from kgr4vrd.artifacts import file_digest, read_json, write_json
from kgr4vrd.prediction import evaluate_predictions, predict
from kgr4vrd.reporting import report_evaluations
from kgr4vrd.smoke import prepare_smoke
from kgr4vrd.training import train


def test_real_saved_score_regeneration_and_wrong_saved_metrics(tmp_path):
    torch.set_num_threads(1)
    config = prepare_smoke(tmp_path / "fixture")
    config["training"]["max_epochs"] = 1
    run = train(config)
    scores = predict(run / "selected.json", split="validation")
    evaluation = evaluate_predictions(scores)
    original = evaluation.read_bytes()
    result = report_stage.regenerate_view(scores, None, evaluation, tmp_path / "regenerated")
    assert result["status"] == "verified"
    assert evaluation.read_bytes() == original
    assert len(result["outputs"]) == 4
    # A self-consistent but false accepted metric must fail fresh computation.
    metrics = evaluation.parent / "metrics.json"
    record = read_json(metrics)
    record["metrics"]["recall_image_mean_at_50"] = 0.123456789
    write_json(metrics, record)
    manifest = read_json(evaluation)
    manifest["outputs"]["metrics.json"] = file_digest(metrics)
    write_json(evaluation, manifest)
    with pytest.raises(ValueError, match="regeneration differs"):
        report_stage.regenerate_view(scores, None, evaluation, tmp_path / "wrong")


def test_report_regeneration_compares_actual_tables_and_pngs(tmp_path):
    from kgr4vrd.smoke import smoke

    manifest = smoke(tmp_path / "smoke")
    first = manifest.parent
    with (first / "evidence.csv").open() as stream:
        evaluations = [row["path"] for row in csv.DictReader(stream)]
    second = tmp_path / "second"
    report_evaluations(evaluations, second)
    assert report_stage.compare_reports(first, second)
    # Optimized reporters may add operational receipts; their timing can differ.
    for index, directory in enumerate((first, second)):
        write_json(
            directory / "figure-progress.json",
            dict(
                stage="report figures",
                status="completed",
                completed=1,
                total=1,
                elapsed_seconds=index + 1,
            ),
        )
    assert report_stage.compare_reports(first, second)
    progress = second / "figure-progress.json"
    saved_progress = progress.read_bytes()
    record = read_json(progress)
    record["status"] = "running"
    write_json(progress, record)
    with pytest.raises(ValueError, match="Incomplete report progress"):
        report_stage.compare_reports(first, second)
    progress.write_bytes(saved_progress)
    png = next(second.glob("*.png"))
    png.write_bytes(b"altered rendered figure")
    with pytest.raises(ValueError, match="regeneration differs"):
        report_stage.compare_reports(first, second)


def test_inventory_rejects_missing_duplicate_failed_and_extra_rows():
    development = [dict(key=f"d{i}") for i in range(8)]
    final = [dict(key=f"f{i}") for i in range(50)]
    views = [
        dict(key=f"e{i}", filter_mode="simple_asserted" if i < 10 else "none", condition="baseline")
        for i in range(110)
    ]
    plan = dict(development_runs=development, runs=final, evaluations=views)
    inventory = dict(
        runs=[dict(r, status="completed") for r in development + final],
        evaluations=[dict(r, status="completed") for r in views],
    )
    report_stage.exact_inventory(plan, inventory)
    for mutation in ("missing", "duplicate", "failed", "unplanned"):
        changed = copy.deepcopy(inventory)
        if mutation == "missing":
            changed["runs"].pop()
        elif mutation == "duplicate":
            changed["runs"][-1] = changed["runs"][0]
        elif mutation == "failed":
            changed["evaluations"][0]["status"] = "failed"
        else:
            changed["evaluations"][0]["key"] = "unplanned"
        with pytest.raises(ValueError):
            report_stage.exact_inventory(plan, changed)


def test_completion_preserves_registry_and_rejects_concurrent_work(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "14482")
    monkeypatch.setenv("SLURM_STEP_ID", "999")
    receipt = tmp_path / "completion.json"
    write_json(receipt, dict(status="complete", exit_code=0))
    registry = dict(
        runs=[
            dict(id="before", enabled=True, completion_path=str(receipt)),
            dict(id="report", step_id="14482.999"),
        ],
        pending_batches=[dict(id="concurrent")],
        campaign_context=dict(preserve="evidence"),
        remaining_work_status="pending",
        history=["retained"],
    )
    path = tmp_path / "registry.json"
    write_json(path, registry)
    acceptance = tmp_path / "accepted.json"
    write_json(
        acceptance,
        dict(
            status="complete",
            counts=dict(development=8, final=50, raw_scores=50, evaluations=110),
            report_directory="report",
        ),
    )
    before = path.read_bytes()
    with pytest.raises(ValueError, match="Concurrent"):
        report_stage.complete_registry(tmp_path, "report", acceptance, [])
    assert path.read_bytes() == before
    registry["pending_batches"] = []
    write_json(path, registry)
    marker = tmp_path / "PAUSE"
    marker.touch()
    with pytest.raises(RuntimeError, match="STOP/PAUSE"):
        report_stage.complete_registry(tmp_path, "report", acceptance, [marker])
    marker.unlink()
    report_stage.complete_registry(tmp_path, "report", acceptance, [marker])
    after = read_json(path)
    assert after["remaining_work_status"] == "complete"
    assert after["runs"] == registry["runs"] and after["history"] == registry["history"]
    assert after["campaign_context"]["preserve"] == "evidence"
    assert after["campaign_context"]["stronger_predictor_claim_supported"] is False
