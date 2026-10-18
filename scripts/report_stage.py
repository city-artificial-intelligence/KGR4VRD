"""Verify the bounded campaign and regenerate evidence using the frozen package."""

import argparse
import csv
import fcntl
import os
import time
from pathlib import Path

from final_stage import check_pause, verify_group

from kgr4vrd.artifacts import (
    environment,
    file_digest,
    read_json,
    read_jsonl,
    source_snapshot,
    verify_run,
    write_json,
    write_jsonl,
)
from kgr4vrd.campaign import load_campaign


def exact_inventory(plan, inventory):
    expected_runs = plan["development_runs"] + plan["runs"]
    for expected, actual, count in (
        (expected_runs, inventory["runs"], 58),
        (plan["evaluations"], inventory["evaluations"], 110),
    ):
        keys = [row["key"] for row in actual]
        if len(expected) != count or len(keys) != count or len(set(keys)) != count:
            raise ValueError("Incomplete or duplicate registered inventory")
        if set(keys) != {row["key"] for row in expected}:
            raise ValueError("Inventory differs from the registered plan")
        if any(row["status"] != "completed" for row in actual):
            raise ValueError("Failed or missing rows remain; preserve them and do not complete")
    if len(plan["development_runs"]) != 8 or len(plan["runs"]) != 50:
        raise ValueError("Expected exactly eight development and fifty final rows")
    simple = [row for row in plan["evaluations"] if row["filter_mode"] == "simple_asserted"]
    if len(simple) != 10 or any(row["condition"] != "baseline" for row in simple):
        raise ValueError("Simple asserted filtering must cover only ten baselines")


def regenerate_view(predictions, campaign, accepted, output):
    """Validate accepted identities, then recompute without accepted image journals."""
    from kgr4vrd import prediction
    from kgr4vrd.config import data_path, load_config
    from kgr4vrd.data import load_objects, load_relations, load_vocabulary
    from kgr4vrd.reasoning import load_knowledge, load_simple_filter

    manifest = read_json(accepted)
    before = file_digest(accepted)
    actual = prediction.evaluate_predictions(
        predictions, campaign=campaign, filter_mode=manifest["filter_mode"]
    )
    if Path(actual).resolve() != Path(accepted).resolve():
        raise ValueError("Expected the registered evaluation identity")
    saved = read_json(predictions)
    selection_path = (Path(predictions).parent / saved["selection"]).resolve()
    selection = read_json(selection_path)
    config = load_config(selection_path.parent / selection["config"])
    dataset = data_path(config, config["dataset"]["manifest"])
    vocabulary = load_vocabulary(dataset)
    scenes = load_objects(dataset, saved["split"])
    targets = load_relations(dataset, saved["split"])
    table = load_knowledge(
        data_path(config, config["intervention"]["knowledge_manifest"]), vocabulary
    )
    decisions = {key: table.lookup(scene) for key, scene in scenes.items()}
    simple = None
    if manifest["filter_mode"] == "simple_asserted":
        simple_table = load_simple_filter(
            data_path(config, config["evaluation"]["simple_filter_manifest"]), vocabulary
        )
        simple = {key: simple_table.lookup(scene) for key, scene in scenes.items()}
    # No checkpoint_directory: every image is recomputed from immutable score shards.
    result = prediction.evaluate(
        tuple(scenes),
        targets,
        prediction.ScoreShards(predictions, scenes),
        scenes,
        len(vocabulary.predicates),
        decisions=decisions,
        filter_mode=manifest["filter_mode"],
        simple_masks=simple,
    )
    if result["policy"] != manifest["policy"]:
        raise ValueError("Regenerated metric policy differs")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("per_image", "per_predicate"):
        prediction.write_csv(
            output / f"{name}.csv",
            [
                {**row, "run_id": manifest["run_id"], "evaluation_id": manifest["evaluation_id"]}
                for row in result[name]
            ],
        )
    ranked = result["ranked_predictions"]
    if isinstance(ranked, dict):
        ranked = [{"image_id": key, "predictions": values} for key, values in ranked.items()]
    write_jsonl(output / "ranked_predictions.jsonl", ranked)
    write_json(
        output / "metrics.json",
        {key: result[key] for key in ("metrics", "metric_details", "diagnostics")},
    )
    outputs = {name: file_digest(output / name) for name in manifest["outputs"]}
    if outputs != manifest["outputs"]:
        raise ValueError(f"Saved-score regeneration differs: {accepted}")
    if file_digest(accepted) != before:
        raise ValueError("Accepted evaluation changed during verification")
    receipt = dict(status="verified", accepted=str(accepted), sha256=before, outputs=outputs)
    write_json(output / "verified.json", receipt)
    return receipt


def compare_reports(first, second):
    left = {str(p.relative_to(first)): p for p in first.rglob("*") if p.is_file()}
    right = {str(p.relative_to(second)): p for p in second.rglob("*") if p.is_file()}
    if set(left) != set(right) or not left:
        raise ValueError("Report regeneration has a different file inventory")
    checks = {}
    for name, path in left.items():
        # Vector exports have creation timestamps and generated SVG identifiers.
        # Their numerical inputs and rendered PNGs must reproduce byte-for-byte.
        equal = file_digest(path) == file_digest(right[name])
        vector = path.suffix in {".pdf", ".svg"}
        if vector and str(Path(name).with_suffix(".png")) not in left:
            raise ValueError("Vector report figure lacks a comparable rendered PNG")
        operational = name in {"progress.json", "load-progress.json", "figure-progress.json"}
        if operational:
            a, b = read_json(path), read_json(right[name])
            fields = ("stage", "status", "completed", "total")
            if any(a[k] != b[k] for k in fields) or a["status"] != "completed":
                raise ValueError(f"Incomplete report progress receipt: {name}")
        if not equal and not vector and not operational:
            raise ValueError(f"Report regeneration differs: {name}")
        if path.stat().st_size == 0 or right[name].stat().st_size == 0:
            raise ValueError(f"Empty report artifact: {name}")
        checks[name] = dict(
            sha256=file_digest(path),
            regenerated_sha256=file_digest(right[name]),
            bytes_equal=equal,
            comparison=(
                "completed stage/count equality; timing retained"
                if operational
                else "rendered PNG and numerical input equality"
                if vector
                else "exact bytes"
            ),
        )
    return checks


def resource_record(campaign, inventory, base):
    rows = []
    for row in inventory["runs"]:
        directory = (campaign.parent / row["selection"]).parent
        run = read_json(directory / "run.json")
        epochs = read_jsonl(directory / "epochs.jsonl")
        rows.append(
            dict(
                key=row["key"],
                run_id=run["run_id"],
                started_at=run["started_at"],
                ended_at=run["ended_at"],
                completed_epochs=run["completed_epochs"],
                completed_updates=run["completed_updates"],
                peak_process_memory_bytes=run["peak_process_memory_bytes"],
                memory_scope=run["memory_scope"],
                epoch_measurements=epochs,
                artifact_bytes=sum(p.stat().st_size for p in directory.rglob("*") if p.is_file()),
            )
        )
    attempts = []
    for path in sorted((base / "attempts").glob("*/completion.json")):
        attempts.append(dict(path=str(path), sha256=file_digest(path), **read_json(path)))
    return dict(
        runs=rows,
        attempts_including_failures=attempts,
        cumulative_attempt_seconds=sum(r["elapsed_seconds"] for r in attempts),
        timing_note="Attempt totals retain retries and overlap instrumentation; not allocation wall time.",
        pilot=read_json(base / "pilot-02/frozen.json"),
        preparation=read_json(base / "pilot-02/inventory.json"),
        pilot_cases=[
            read_json(r["path"]) for r in read_json(base / "pilot-02/frozen.json")["cases"]
        ],
    )


def complete_registry(supervisor, run_id, acceptance, pause_paths):
    check_pause(pause_paths)
    path = supervisor / "registry.json"
    with (supervisor / "registry.json.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        registry = read_json(path)
        check_pause(list(pause_paths) + registry.get("pause_paths", []))
        if registry["pending_batches"]:
            raise ValueError("Concurrent pending work prevents completion")
        own = [r for r in registry["runs"] if r["id"] == run_id]
        step = f"{os.environ['SLURM_JOB_ID']}.{os.environ['SLURM_STEP_ID']}"
        if len(own) != 1 or own[0]["step_id"] != step:
            raise ValueError("Report worker is not registered to this numeric step")
        for row in registry["runs"]:
            if row["id"] == run_id or not row.get("enabled", True):
                continue
            receipt = read_json(row["completion_path"])
            if receipt["exit_code"] != 0 or receipt["status"] != "complete":
                raise ValueError(f"Enabled predecessor incomplete: {row['id']}")
        evidence = read_json(acceptance)
        if evidence["status"] != "complete" or evidence["counts"] != {
            "development": 8,
            "final": 50,
            "raw_scores": 50,
            "evaluations": 110,
        }:
            raise ValueError("Report accounting is incomplete")
        registry["campaign_context"].update(
            report_acceptance=str(acceptance),
            report_directory=evidence["report_directory"],
            report_counts=evidence["counts"],
            stronger_predictor_claim_supported=False,
            visual_qualification_passed=False,
        )
        registry["remaining_work_status"] = "complete"
        write_json(path, registry)


def run(args):
    from kgr4vrd.reporting import summarise

    started = time.monotonic()
    campaign, output, base = args.campaign.resolve(), args.output.resolve(), args.base.resolve()
    plan = load_campaign(campaign, frozen=True)
    inventory_path = campaign.parent / "status.json"
    inventory_digest = file_digest(inventory_path)
    inventory = read_json(inventory_path)
    exact_inventory(plan, inventory)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "inventory-before.json", inventory)
    verified_runs = {}
    planned_runs = {r["key"]: r for r in plan["development_runs"] + plan["runs"]}
    for row in inventory["runs"]:
        check_pause(args.pause_path)
        selection = campaign.parent / row["selection"]
        verified_runs[row["key"]] = verify_run(selection.parent)
        planned = planned_runs[row["key"]]
        if file_digest(campaign.parent / planned["config"]) != file_digest(
            selection.parent / "resolved-config.json"
        ):
            raise ValueError(f"Registered configuration differs: {row['key']}")
        write_json(output / "verified-runs.json", verified_runs)
    for group in plan["groups"]:
        check_pause(args.pause_path)
        verify_group(campaign, group["name"], output / f"{group['name']}-verified.json")
    development = {r["key"]: r for r in inventory["runs"]}
    frozen_development_paths = {
        Path(r["selection"]).resolve() for r in plan["development"]["records"]
    }
    actual_development_paths = {
        (campaign.parent / development[r["key"]]["selection"]).resolve()
        for r in plan["development_runs"]
    }
    if actual_development_paths != frozen_development_paths or len(actual_development_paths) != 8:
        raise ValueError("Development inventory differs from the frozen selections")
    for row in plan["development"]["records"]:
        path = Path(row["selection"])
        if file_digest(path) != row["selection_digest"]:
            raise ValueError("Development selection differs from freeze")
    predictions = {
        r["key"]: (campaign.parent / development[r["key"]]["predictions"]).resolve()
        for r in plan["runs"]
    }
    if len(set(predictions.values())) != 50:
        raise ValueError("Expected fifty distinct raw-score artifacts")
    views = {r["key"]: r for r in inventory["evaluations"]}
    regenerated = {}
    for row in plan["evaluations"]:
        check_pause(args.pause_path)
        accepted = (campaign.parent / views[row["key"]]["path"]).resolve()
        regenerated[row["key"]] = regenerate_view(
            predictions[row["run_key"]],
            campaign,
            accepted,
            output / "regenerated-evaluations" / read_json(accepted)["evaluation_id"],
        )
        write_json(
            output / "regeneration-progress.json",
            dict(
                completed=len(regenerated),
                total=110,
                evaluations=regenerated,
            ),
        )
        print(f"Verified saved-score view {len(regenerated)}/110: {row['key']}", flush=True)
    first, second = output / "report", output / "report-regenerated"
    for target in (first, second):
        check_pause(args.pause_path)
        summarise(campaign, target)
        report = read_json(target / "report.json")
        if report["status"] != "completed" or report["completed_evaluations"] != 110:
            raise ValueError("Generated report is incomplete")
    comparison = compare_reports(first, second)
    write_json(output / "report-regeneration.json", comparison)
    write_json(output / "resources.json", resource_record(campaign, inventory, base))
    qualifications = plan["development"]["visual_qualification"]
    references = [
        "acquisition.json",
        "recovered-split.json",
        "ontology-structural-inventory.json",
        "vocabulary-coverage.json",
        "optimization-03/accepted.json",
        "pilot-02/frozen.json",
        "pilot-02/inventory.json",
        "freeze-01/accepted.json",
        "final-01/accepted.json",
        "freeze-01/reference-review/REVIEW.md",
        "freeze-01/reference-review/environment-probe.json",
    ]
    write_json(
        output / "scope.json",
        dict(
            scientific_profile=plan["metadata"]["scientific_profile"],
            limitations=plan["metadata"]["scope_limitations"],
            visual_qualification=qualifications,
            control=plan["control"],
            input_evidence=plan["input_evidence"],
            references={
                n: dict(path=str(base / n), sha256=file_digest(base / n)) for n in references
            },
        ),
    )
    with (first / "paired_effects.csv").open() as stream:
        effects = [r for r in csv.DictReader(stream) if r["metric"] == "recall_image_mean_at_50"]
    lines = [
        "# Fresh corrected public ontology study",
        "",
        "All 8 development, 50 final and 110 evaluation inventory rows were verified. "
        "The 110 views reproduce exactly from 50 saved raw-score artifacts, without training "
        "or a live GraphDB service. Report tables and rendered PNGs reproduce exactly; vector "
        "export metadata may differ. See report-regeneration.json.",
        "",
        "The visual model failed validation qualification: image-mean recall@50 difference "
        f"{100 * qualifications['image_mean_difference']:.4f} percentage points; predicate-mean "
        f"difference {100 * qualifications['predicate_mean_difference']:.4f} points. "
        "W3/G4 remains incomplete as a stronger-predictor study. The bounded reference review "
        "and measured environment probe are retained in scope.json. Closing that claim needs "
        "a pinned compatible reference implementation with the NeSy4VRD multi-label, intervention "
        "and raw-score adapter, resource validation and validation qualification before test use. "
        "A later scientific design change is exploratory; no new model is authorized here.",
        "",
        "This is not historical replication. Exact historical checkpoints and ontology variants "
        "were not recovered. The acquired public profile uses explicitly partial Boolean/RL "
        "semantics; zero observed unknown coverage does not establish general OWL completeness. "
        "Source, mapping, original terms and split provenance remain in the referenced input records.",
        "",
        "Single retrospectively studied NeSy4VRD benchmark; no independent confirmation. "
        "Annotation AP is not audited visual precision. Filtered zero known violations is a "
        "filtering property, not learned correctness. Missing labels, fixed lambda and "
        "frequency-control limitations remain. Negative and uncertain effects are retained below.",
        "",
        "Measured preparation, cached reuse, per-epoch training/validation, peak host memory, "
        "pilot CUDA peaks, storage and failed-attempt cumulative time are in resources.json. "
        "Pilot checkpoints are excluded from development/final results. Inference times remain "
        "in the 50 prediction manifests. The original pilot projections remain frozen.",
        "",
        "| Contrast | Recall@50 change (pp) | 95% interval (pp) | Interpretation |",
        "| --- | ---: | --- | --- |",
    ]
    for row in effects:
        value = float(row["effect"])
        lo, hi = row["lower"], row["upper"]
        interval = (
            "undefined" if not lo or not hi else f"[{100 * float(lo):.3f}, {100 * float(hi):.3f}]"
        )
        label = (
            "tradeoff: decrease exceeds 1 pp"
            if value < -0.01
            else "negative estimate"
            if value < 0
            else "estimate; assess interval"
        )
        lines.append(f"| {row['contrast']} | {100 * value:.3f} | {interval} | {label} |")
    lines += [
        "",
        "Reproduce with the frozen report spec commands and source-08 environment. "
        "Use a fresh report output directory; accepted runs, selections, scores and evaluation "
        "artifacts remain immutable. The paper was not changed.",
    ]
    (output / "CAMPAIGN-REPORT.md").write_text("\n".join(lines) + "\n")
    if file_digest(inventory_path) != inventory_digest:
        raise ValueError("Inventory changed during report verification")
    exact_inventory(plan, read_json(inventory_path))
    acceptance = dict(
        status="complete",
        counts=dict(development=8, final=50, raw_scores=50, evaluations=110),
        plan_digest=plan["plan_digest"],
        inventory_sha256=inventory_digest,
        source_digest=source_snapshot()["digest"],
        report_directory=str(first),
        report_review=str(output / "CAMPAIGN-REPORT.md"),
        stronger_predictor_claim_supported=False,
        visual_qualification_passed=False,
        elapsed_seconds=time.monotonic() - started,
        step_id=f"{os.environ['SLURM_JOB_ID']}.{os.environ['SLURM_STEP_ID']}",
        artifacts={
            n: file_digest(output / n)
            for n in [
                "verified-runs.json",
                "regeneration-progress.json",
                "report-regeneration.json",
                "resources.json",
                "scope.json",
                "CAMPAIGN-REPORT.md",
            ]
        },
    )
    write_json(output / "accepted.json", acceptance)
    complete_registry(args.supervisor, args.run_id, output / "accepted.json", args.pause_path)
    with (base / "handoff.json.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        handoff = read_json(base / "handoff.json")
        handoff.update(
            status="registered_campaign_complete",
            remaining_work_status="complete",
            report=acceptance,
            active_worker=None,
            unprepared_obligations=[],
            next_action="Registered 8/50/110 execution and regeneration complete. "
            "Review report-01/CAMPAIGN-REPORT.md; stronger-predictor claim remains unsupported.",
        )
        write_json(base / "handoff.json", handoff)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign", "source", "base", "output", "supervisor"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--allocation", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--pause-path", type=Path, action="append", default=[])
    args = parser.parse_args()
    if (
        os.environ.get("SLURM_JOB_ID") != args.allocation
        or not os.environ.get("SLURM_STEP_ID", "").isdigit()
    ):
        raise RuntimeError("Report execution requires the retained allocation's numeric step")
    check_pause(args.pause_path)
    import kgr4vrd

    if not Path(kgr4vrd.__file__).resolve().is_relative_to(args.source.resolve()):
        raise RuntimeError("Wrong frozen scientific source")
    plan = load_campaign(args.campaign, frozen=True)
    identity = dict(source_digest=source_snapshot()["digest"], environment=environment())
    if identity != plan["development"]["execution_identity"]:
        raise ValueError("Report source/environment differs from accepted development")
    run(args)


if __name__ == "__main__":
    main()
