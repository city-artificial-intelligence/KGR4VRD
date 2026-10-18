"""Record finite final-stage execution around the frozen package's existing APIs."""

import argparse
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

from kgr4vrd.artifacts import environment, file_digest, read_json, source_snapshot, write_json
from kgr4vrd.campaign import assert_test_selection, load_campaign


def check_pause(paths):
    present = [str(path) for path in paths if Path(path).exists()]
    if present:
        raise RuntimeError(f"Campaign STOP/PAUSE prevents execution: {present}")


def update_inventory(campaign, section, key, changes, event=None):
    path = Path(campaign).parent / "status.json"
    with path.with_suffix(".json.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        inventory = read_json(path)
        matches = [row for row in inventory[section] if row["key"] == key]
        if len(matches) != 1:
            raise ValueError(f"Expected one registered inventory row: {key}")
        matches[0].update(changes)
        if event is not None:
            matches[0].setdefault("execution_history", []).append(event)
        write_json(path, inventory)


def inventory_row(campaign, section, key):
    rows = read_json(Path(campaign).parent / "status.json")[section]
    matches = [row for row in rows if row["key"] == key]
    if len(matches) != 1:
        raise ValueError(f"Expected one registered inventory row: {key}")
    return matches[0]


def execute(campaign, section, key, operation):
    started = time.time()
    step = f"{os.environ['SLURM_JOB_ID']}.{os.environ['SLURM_STEP_ID']}"
    event = {"step_id": step, "started_epoch": started}
    try:
        result = operation()
    except BaseException as error:
        event.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    else:
        event["status"] = "completed"
        return result
    finally:
        event.update(ended_epoch=time.time(), elapsed_seconds=time.time() - started)
        update_inventory(campaign, section, key, {"status": event["status"]}, event)


def train_row(campaign, key, source, *, resume=False):
    command = [
        sys.executable,
        str(Path(source) / "scripts/run_planned.py"),
        "--campaign",
        str(campaign),
        "--key",
        key,
    ]
    if resume:
        command.append("--resume")

    def train_and_check():
        subprocess.run(command, check=True)
        row = inventory_row(campaign, "runs", key)
        if row["status"] != "completed" or not row.get("selection"):
            raise RuntimeError(f"Training returned without an inventory selection: {key}")

    execute(campaign, "runs", key, train_and_check)


def evaluate_row(campaign, key, pause_paths=()):
    from kgr4vrd.prediction import evaluate_predictions, predict

    plan = load_campaign(campaign, frozen=True)
    row = inventory_row(campaign, "runs", key)
    if row["status"] != "completed":
        raise ValueError(f"Cannot evaluate incomplete training: {key}")
    selection = Path(campaign).parent / row["selection"]
    # This validates the complete group before the first model forward pass.
    assert_test_selection(campaign, selection)
    check_pause(pause_paths)
    predictions = predict(selection, split="test", campaign=campaign)
    update_inventory(campaign, "runs", key, {"predictions": str(predictions)})
    for view in [item for item in plan["evaluations"] if item["run_key"] == key]:
        check_pause(pause_paths)

        def evaluate_view():
            path = evaluate_predictions(
                predictions, campaign=campaign, filter_mode=view["filter_mode"]
            )
            record = read_json(path)
            expected = {name: view[name] for name in ("model", "condition", "seed", "filter_mode")}
            if any(record[name] != value for name, value in expected.items()):
                raise ValueError(f"Evaluation identity mismatch: {view['key']}")
            if record["plan_digest"] != plan["plan_digest"] or record["status"] != "completed":
                raise ValueError("Evaluation is not complete under this frozen plan")
            update_inventory(campaign, "evaluations", view["key"], {"path": str(path)})
            return path

        execute(campaign, "evaluations", view["key"], evaluate_view)


def verify_group(campaign, group, output):
    from kgr4vrd.artifacts import verify_run
    from kgr4vrd.reporting import load_evaluation

    plan = load_campaign(campaign, frozen=True)
    wanted = next(row for row in plan["groups"] if row["name"] == group)
    bindings = read_json(Path(campaign).parent / "selections" / f"{group}.json")
    if bindings["plan_digest"] != plan["plan_digest"]:
        raise ValueError("Group binding uses another plan")
    if {r["key"] for r in bindings["selections"]} != set(wanted["run_keys"]):
        raise ValueError("Group selection is incomplete")
    selections, predictions, evaluations = {}, {}, {}
    for key in wanted["run_keys"]:
        row = inventory_row(campaign, "runs", key)
        if row["status"] != "completed":
            raise ValueError(f"Incomplete final run: {key}")
        selection = Path(campaign).parent / row["selection"]
        verify_run(selection.parent)
        binding = assert_test_selection(campaign, selection)
        selections[key] = binding
        prediction = Path(campaign).parent / row["predictions"]
        record = read_json(prediction)
        if record["status"] != "completed" or record["run_id"] != binding["run_id"]:
            raise ValueError(f"Invalid raw scores: {key}")
        predictions[key] = str(prediction.resolve())
    for view in [r for r in plan["evaluations"] if r["group"] == group]:
        row = inventory_row(campaign, "evaluations", view["key"])
        if row["status"] != "completed":
            raise ValueError(f"Incomplete evaluation: {view['key']}")
        path = Path(campaign).parent / row["path"]
        record = load_evaluation(path)["manifest"]
        binding = selections[view["run_key"]]
        expected = {name: view[name] for name in ("model", "condition", "seed", "filter_mode")}
        expected.update({name: binding[name] for name in ("run_id", "checkpoint_digest")})
        expected.update(plan_digest=plan["plan_digest"], split="test", status="completed")
        if any(record[name] != value for name, value in expected.items()):
            raise ValueError(f"Evaluation binding mismatch: {view['key']}")
        if (path.parent / record["predictions"]).resolve() != Path(predictions[view["run_key"]]):
            raise ValueError("Views do not reuse the registered raw score artifact")
        evaluations[view["key"]] = {"path": str(path), "sha256": file_digest(path)}
    if len(set(predictions.values())) != len(wanted["run_keys"]):
        raise ValueError("Distinct final runs must retain distinct raw score artifacts")
    result = dict(
        status="complete",
        group=group,
        plan_digest=plan["plan_digest"],
        runs=len(selections),
        raw_scores=len(predictions),
        views=len(evaluations),
        selections=selections,
        predictions=predictions,
        evaluations=evaluations,
        step_id=f"{os.environ['SLURM_JOB_ID']}.{os.environ['SLURM_STEP_ID']}",
    )
    write_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "evaluate", "verify-group"))
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--allocation", required=True)
    parser.add_argument("--key")
    parser.add_argument("--group")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pause-path", type=Path, action="append", default=[])
    args = parser.parse_args()
    if (
        os.environ.get("SLURM_JOB_ID") != args.allocation
        or not os.environ.get("SLURM_STEP_ID", "").isdigit()
    ):
        raise RuntimeError("Final stages require the retained allocation's numeric Slurm step")
    check_pause(args.pause_path)
    plan = load_campaign(args.campaign, frozen=True)
    import kgr4vrd

    if not Path(kgr4vrd.__file__).resolve().is_relative_to(args.source.resolve()):
        raise RuntimeError("Wrong frozen package source")
    execution = {"source_digest": source_snapshot()["digest"], "environment": environment()}
    if execution != plan["development"]["execution_identity"]:
        raise ValueError("Final execution differs from accepted development")
    if args.stage == "verify-group":
        if not args.group or not args.output:
            parser.error("verify-group requires --group and --output")
        verify_group(args.campaign, args.group, args.output)
    else:
        matches = [r for r in plan["runs"] if r["key"] == args.key]
        if len(matches) != 1:
            raise ValueError("Expected exactly one registered final run")
        if file_digest(args.campaign.parent / matches[0]["config"]) != matches[0]["config_digest"]:
            raise ValueError("Final configuration changed after freeze")
        if args.stage == "train":
            train_row(args.campaign, args.key, args.source, resume=args.resume)
        else:
            evaluate_row(args.campaign, args.key, args.pause_path)


if __name__ == "__main__":
    main()
