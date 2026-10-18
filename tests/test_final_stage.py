"""Final orchestration must preserve complete-group gating and failure evidence."""

import subprocess
from pathlib import Path

import final_stage
import pytest

from kgr4vrd.artifacts import read_json, write_json


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "14482")
    monkeypatch.setenv("SLURM_STEP_ID", "123")
    path = tmp_path / "campaign.json"
    write_json(path, {})
    write_json(
        tmp_path / "status.json",
        {
            "runs": [{"key": "a", "status": "not_started"}, {"key": "b", "status": "not_started"}],
            "evaluations": [
                {"key": "a:none", "status": "not_started"},
                {"key": "a:ontology", "status": "not_started"},
            ],
            "unrelated": {"preserve": True},
        },
    )
    return path


def test_training_failure_and_retry_retain_history(campaign, monkeypatch):
    calls = []

    def failed(command, check):
        calls.append(command)
        raise subprocess.CalledProcessError(4, command)

    monkeypatch.setattr(final_stage.subprocess, "run", failed)
    with pytest.raises(subprocess.CalledProcessError):
        final_stage.train_row(campaign, "a", campaign.parent)
    first = read_json(campaign.parent / "status.json")
    assert first["runs"][0]["status"] == "failed"
    assert first["runs"][1]["status"] == "not_started"
    assert first["unrelated"] == {"preserve": True}
    assert "--resume" not in calls[0]

    def completed(command, check):
        calls.append(command)
        final_stage.update_inventory(
            campaign, "runs", "a", {"selection": "selected.json", "status": "completed"}
        )

    monkeypatch.setattr(final_stage.subprocess, "run", completed)
    final_stage.train_row(campaign, "a", campaign.parent, resume=True)
    row = final_stage.inventory_row(campaign, "runs", "a")
    assert calls[1][-1] == "--resume"
    assert [event["status"] for event in row["execution_history"]] == ["failed", "completed"]
    assert all(event["elapsed_seconds"] >= 0 for event in row["execution_history"])
    assert all(event["step_id"] == "14482.123" for event in row["execution_history"])


def test_incomplete_group_never_predicts(campaign, monkeypatch):
    from kgr4vrd import prediction

    final_stage.update_inventory(
        campaign, "runs", "a", {"status": "completed", "selection": "selected.json"}
    )
    monkeypatch.setattr(final_stage, "load_campaign", lambda *a, **kw: {})

    def reject(*args):
        raise ValueError("incomplete group")

    monkeypatch.setattr(final_stage, "assert_test_selection", reject)
    monkeypatch.setattr(
        prediction, "predict", lambda *a, **kw: pytest.fail("test inference before group freeze")
    )
    with pytest.raises(ValueError, match="incomplete group"):
        final_stage.evaluate_row(campaign, "a")


def test_one_score_artifact_for_registered_views_and_visible_failure(campaign, monkeypatch):
    from kgr4vrd import prediction

    final_stage.update_inventory(
        campaign, "runs", "a", {"status": "completed", "selection": "selected.json"}
    )
    views = [
        dict(
            key=f"a:{mode}",
            run_key="a",
            filter_mode=mode,
            model="geometry_mlp",
            condition="combined",
            seed=11,
        )
        for mode in ("none", "ontology")
    ]
    monkeypatch.setattr(
        final_stage, "load_campaign", lambda *a, **kw: {"plan_digest": "plan", "evaluations": views}
    )
    monkeypatch.setattr(final_stage, "assert_test_selection", lambda *a: {})
    predictions = campaign.parent / "scores.json"
    calls = []
    monkeypatch.setattr(
        prediction, "predict", lambda *a, **kw: calls.append("predict") or predictions
    )

    def evaluate(path, *, campaign, filter_mode):
        assert path == predictions
        calls.append(filter_mode)
        if filter_mode == "ontology":
            raise ValueError("injected corrupt score")
        result = Path(campaign).parent / "evaluation.json"
        write_json(result, {**views[0], "status": "completed", "plan_digest": "plan"})
        return result

    monkeypatch.setattr(prediction, "evaluate_predictions", evaluate)
    with pytest.raises(ValueError, match="corrupt score"):
        final_stage.evaluate_row(campaign, "a")
    assert calls == ["predict", "none", "ontology"]
    inventory = read_json(campaign.parent / "status.json")
    assert [row["status"] for row in inventory["evaluations"]] == ["completed", "failed"]
    assert inventory["runs"][0]["predictions"] == str(predictions)
    assert inventory["evaluations"][0]["path"].endswith("evaluation.json")


def test_pause_and_duplicate_inventory_are_not_silently_accepted(campaign):
    marker = campaign.parent / "PAUSE"
    marker.touch()
    with pytest.raises(RuntimeError, match="STOP/PAUSE"):
        final_stage.check_pause([marker])
    inventory = read_json(campaign.parent / "status.json")
    inventory["runs"].append(dict(inventory["runs"][0]))
    write_json(campaign.parent / "status.json", inventory)
    before = (campaign.parent / "status.json").read_bytes()
    with pytest.raises(ValueError, match="one registered"):
        final_stage.update_inventory(campaign, "runs", "a", {"status": "completed"})
    assert (campaign.parent / "status.json").read_bytes() == before
