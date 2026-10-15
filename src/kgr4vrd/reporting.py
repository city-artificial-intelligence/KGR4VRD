"""Regenerate evidence-bounded tables and figures from saved evaluations only."""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import file_digest, read_json, write_json
from .campaign import load_campaign


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for name, value in row.items():
            if name not in {"image_id", "run_id", "evaluation_id"}:
                row[name] = None if value == "" else float(value)
    return rows


def load_evaluation(path: str | Path) -> dict[str, Any]:
    """Read one explicit completed artifact; never discover a preferred attempt."""
    path = Path(path).resolve()
    manifest = read_json(path)
    if manifest.get("status") != "completed":
        raise ValueError(f"Evaluation is incomplete: {path}")
    for name, digest in manifest.get("outputs", {}).items():
        if file_digest(path.parent / name) != digest:
            raise ValueError(f"Evaluation output checksum mismatch: {name}")
    result = read_json(path.parent / manifest.get("metrics", "metrics.json"))
    metrics = result["metrics"] if "metrics" in result else result
    images = _read_csv(path.parent / "per_image.csv")
    if len({r["image_id"] for r in images}) != len(images):
        raise ValueError("Duplicate per-image sufficient statistics")
    return {
        "manifest": manifest,
        "metrics": metrics,
        "per_image": images,
        "path": str(path),
        "digest": file_digest(path),
    }


def _metric_arrays(
    records: list[dict[str, Any]], metric: str
) -> tuple[np.ndarray, np.ndarray | None]:
    if metric.startswith("recall_image_mean_at_"):
        k = metric.rsplit("_", 1)[1]
        return np.array([r[f"hits_at_{k}"] / r["ground_truth_count"] for r in records]), np.ones(
            len(records)
        )
    if metric.startswith("annotation_ap_image_mean_at_"):
        k = metric.rsplit("_", 1)[1]
        return np.array([r[f"ap_sum_at_{k}"] / r["ground_truth_count"] for r in records]), np.ones(
            len(records)
        )
    if metric.startswith("recall_global_at_"):
        k = metric.rsplit("_", 1)[1]
        return np.array([r[f"hits_at_{k}"] for r in records]), np.array(
            [r["ground_truth_count"] for r in records]
        )
    names = {
        "incompatible_rate_all": ("incompatible_count", "prediction_count"),
        "incompatible_rate_decided": ("incompatible_count", "decided"),
        "reasoning_coverage": ("decided", "prediction_count"),
        "predictions_per_image": ("prediction_count", "images"),
    }
    if metric in names:
        numerator, denominator = names[metric]

        def values(name: str) -> np.ndarray:
            if name == "images":
                return np.ones(len(records))
            if name == "decided":
                return np.array([r["compatible_count"] + r["incompatible_count"] for r in records])
            return np.array([r[name] for r in records])

        return values(numerator), values(denominator)
    if metric in {"prediction_count", "incompatible_count", "compatible_count", "unknown_count"}:
        return np.array([r[metric] for r in records]), None
    raise ValueError(f"No registered bootstrap definition for metric: {metric}")


def paired_bootstrap(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    metric: str,
    *,
    draws: int = 10000,
    seed: int = 2028,
) -> dict[str, Any]:
    """Paired crossed bootstrap: one shared image sample for every sampled seed.

    Predicate-macro recall instead uses seed-only resampling with fixed support.
    Undefined draws remain undefined; an interval needs at least 95% valid draws.
    """
    if not left or len(left) != len(right) or draws <= 0:
        raise ValueError("Bootstrap needs paired runs and a positive draw count")
    count = len(left)
    for a, b in zip(left, right, strict=True):
        for key in ("seed", "split", "target_digest", "vocabulary_digest"):
            if a["manifest"].get(key) is None or a["manifest"].get(key) != b["manifest"].get(key):
                raise ValueError(f"Unpaired comparison identity: {key}")
        for key in ("ranking", "matching", "threshold", "top_ks"):
            if a["manifest"].get("policy", {}).get(key) != b["manifest"].get("policy", {}).get(key):
                raise ValueError(f"Unpaired scoring policy: {key}")
    differences = [
        None
        if a["metrics"].get(metric) is None or b["metrics"].get(metric) is None
        else a["metrics"][metric] - b["metrics"][metric]
        for a, b in zip(left, right, strict=True)
    ]
    rng = np.random.default_rng(seed)
    values = np.full(draws, np.nan)
    seed_only = metric.startswith("recall_predicate_mean_at_")
    if seed_only:
        source = np.array([np.nan if v is None else v for v in differences])
        values = source[rng.integers(count, size=(draws, count))].mean(axis=1)
    else:
        maps = [{r["image_id"]: r for r in item["per_image"]} for item in left + right]
        image_ids = sorted(maps[0])
        if any(set(records) != set(image_ids) for records in maps):
            raise ValueError("Paired bootstrap requires the same complete image set")
        if any(
            records[i]["ground_truth_count"] != maps[0][i]["ground_truth_count"]
            for records in maps
            for i in image_ids
        ):
            raise ValueError("Paired bootstrap requires identical raw target supports")
        if metric.startswith(("recall_", "annotation_ap_")):
            image_ids = [i for i in image_ids if maps[0][i]["ground_truth_count"]]
        if image_ids:
            arrays = [_metric_arrays([records[i] for i in image_ids], metric) for records in maps]
            numerators = np.stack([a[0] for a in arrays])
            denominators = (
                None
                if arrays[0][1] is None
                else np.stack([a[1] for a in arrays if a[1] is not None])
            )
            for start in range(0, draws, 128):
                size = min(128, draws - start)
                sampled_seeds = rng.integers(count, size=(size, count))
                sampled_images = rng.integers(len(image_ids), size=(size, len(image_ids)))
                totals = numerators[:, sampled_images].sum(axis=2).T
                if denominators is not None:
                    den = denominators[:, sampled_images].sum(axis=2).T
                    totals = np.divide(
                        totals, den, out=np.full_like(totals, np.nan, dtype=float), where=den != 0
                    )
                paired = totals[:, :count] - totals[:, count:]
                values[start : start + size] = np.take_along_axis(
                    paired, sampled_seeds, axis=1
                ).mean(axis=1)
    valid = values[np.isfinite(values)]
    enough = len(valid) >= 0.95 * draws
    interval = np.quantile(valid, [0.025, 0.975]).tolist() if enough else [None, None]
    defined_differences = [v for v in differences if v is not None]
    return {
        "effect": float(np.mean(defined_differences))
        if len(defined_differences) == count
        else None,
        "per_seed_differences": differences,
        "lower": interval[0],
        "upper": interval[1],
        "defined_draws": len(valid),
        "undefined_draws": draws - len(valid),
        "draws": draws,
        "interval_method": "paired_seed_only" if seed_only else "paired_crossed_seed_image",
        "null_reason": None if enough else "fewer than 95% of bootstrap draws are defined",
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row)) or ["status"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _summary(
    expected: list[dict[str, Any]], available: dict[str, dict[str, Any]], statuses: dict[str, str]
) -> list[dict[str, Any]]:
    rows, groups = [], defaultdict(list)
    metric_names = sorted(
        {
            key
            for value in available.values()
            for key, metric in value["metrics"].items()
            if metric is None or isinstance(metric, (float, int))
        }
    )
    if not metric_names:
        metric_names = ["recall_image_mean_at_50", "incompatible_rate_all", "reasoning_coverage"]
    for item in expected:
        result = available.get(item["key"])
        common = {
            "model": item["model"],
            "condition": item["condition"],
            "filter_mode": item["filter_mode"],
            "dataset": item.get("dataset", "unspecified")
            if result is None
            else result["manifest"].get("dataset", "unspecified"),
        }
        groups[(item["model"], item["condition"], item["filter_mode"])].append(item)
        for metric in metric_names:
            rows.append(
                {
                    **common,
                    "seed": item["seed"],
                    "statistic": "seed",
                    "metric": metric,
                    "value": None if result is None else result["metrics"].get(metric),
                    "status": statuses.get(item["key"], "missing"),
                    "support": None
                    if result is None
                    else sum(r["ground_truth_count"] for r in result["per_image"]),
                    "evaluation_ids": "" if result is None else result["manifest"]["evaluation_id"],
                }
            )
    for (model, condition, mode), items in groups.items():
        complete = all(item["key"] in available for item in items)
        for metric in metric_names:
            numbers = (
                [available[i["key"]]["metrics"].get(metric) for i in items] if complete else []
            )
            defined = complete and all(v is not None for v in numbers)
            for statistic in ("mean", "sample_sd"):
                value = None
                if defined and (statistic == "mean" or len(numbers) > 1):
                    value = (
                        float(np.mean(numbers))
                        if statistic == "mean"
                        else float(np.std(numbers, ddof=1))
                    )
                rows.append(
                    {
                        "model": model,
                        "condition": condition,
                        "filter_mode": mode,
                        "dataset": items[0].get("dataset", "unspecified"),
                        "seed": "all",
                        "statistic": statistic,
                        "metric": metric,
                        "value": value,
                        "status": "completed" if complete else "incomplete",
                        "support": len(numbers),
                        "evaluation_ids": ";".join(
                            available[i["key"]]["manifest"]["evaluation_id"]
                            for i in items
                            if i["key"] in available
                        ),
                    }
                )
    return rows


def _effects(
    campaign: dict[str, Any], available: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    lookup = {
        (r["model"], r["condition"], r["seed"], r["filter_mode"]): available.get(r["key"])
        for r in campaign["evaluations"]
    }
    rows = []
    metrics = [
        f"{metric}_at_{k}"
        for metric in (
            "recall_image_mean",
            "recall_global",
            "recall_predicate_mean",
            "annotation_ap_image_mean",
        )
        for k in campaign["metric_policy"]["top_ks"]
    ] + [
        "incompatible_count",
        "incompatible_rate_all",
        "incompatible_rate_decided",
        "reasoning_coverage",
        "prediction_count",
    ]
    for contrast in campaign.get("contrasts", []):
        left = [
            lookup.get(
                (contrast["model"], contrast["left_condition"], seed, contrast["left_filter"])
            )
            for seed in campaign["final_seeds"]
        ]
        right = [
            lookup.get(
                (
                    contrast.get("right_model", contrast["model"]),
                    contrast["right_condition"],
                    seed,
                    contrast["right_filter"],
                )
            )
            for seed in campaign["final_seeds"]
        ]
        complete = all(value is not None for value in left + right)
        for metric in metrics:
            row = {
                "contrast": contrast["name"],
                "family": contrast["family"],
                "metric": metric,
                "status": "completed" if complete else "incomplete",
            }
            if complete:
                result = paired_bootstrap(
                    [r for r in left if r is not None],
                    [r for r in right if r is not None],
                    metric,
                    draws=campaign["metric_policy"]["bootstrap_draws"],
                    seed=campaign["metric_policy"]["uncertainty_seed"],
                )
                row.update({k: v for k, v in result.items() if k != "per_seed_differences"})
                row["per_seed_differences"] = ";".join(
                    str(v) for v in result["per_seed_differences"]
                )
            rows.append(row)
    return rows


def _figures(output: Path, summary: list[dict[str, Any]], effects: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "baseline": "#333333",
        "augment": "#0072B2",
        "semantic_loss": "#D55E00",
        "combined": "#009E73",
        "frequency_control": "#CC79A7",
        "shuffled_augmentation_control": "#CC79A7",
    }
    figure_data = output / "figure_data"
    selected = [
        r for r in summary if r["metric"] == "recall_image_mean_at_50" and r["statistic"] == "seed"
    ]
    _write_csv(figure_data / "absolute-recall.csv", selected)
    models = list(dict.fromkeys(r["model"] for r in selected)) or ["No results"]
    fig, axes = plt.subplots(len(models), 1, figsize=(9, 5 * len(models)), squeeze=False)
    for model, axis in zip(models, axes[:, 0], strict=True):
        subset = [r for r in selected if r["model"] == model]
        labels = list(dict.fromkeys(f"{r['condition']} / {r['filter_mode']}" for r in subset))
        seeds = list(dict.fromkeys(r["seed"] for r in subset))
        for row in subset:
            label = f"{row['condition']} / {row['filter_mode']}"
            y = labels.index(label) + 0.05 * (seeds.index(row["seed"]) - (len(seeds) - 1) / 2)
            if row["value"] is not None:
                axis.scatter(100 * row["value"], y, color=colors[row["condition"]], s=20)
            else:
                axis.text(0, labels.index(label), "missing", va="center", fontsize=9)
        axis.set(
            yticks=range(len(labels)),
            yticklabels=labels,
            xlabel="Image-mean recall@50 (%)",
            title=f"{model}: absolute seed results; no confidence threshold",
        )
        axis.invert_yaxis()
        axis.tick_params(labelsize=9)
    _save_figure(fig, output / "absolute-recall")
    effect_data = [r for r in effects if r["metric"] == "recall_image_mean_at_50"]
    _write_csv(figure_data / "paired-effects.csv", effect_data)
    fig, axis = plt.subplots(figsize=(12, max(3, 0.4 * len(effect_data))))
    for index, row in enumerate(effect_data):
        if row.get("effect") is not None:
            axis.scatter(100 * row["effect"], index, color="#0072B2", s=20)
            if row.get("lower") is not None:
                axis.plot([100 * row["lower"], 100 * row["upper"]], [index, index], color="#0072B2")
        else:
            axis.text(0, index, "missing", va="center", fontsize=9)
    axis.axvline(0, color="grey", linewidth=0.6)
    axis.set(
        yticks=range(len(effect_data)),
        yticklabels=[r["contrast"] for r in effect_data],
        xlabel="Recall@50 difference (percentage points)",
        title="Planned effects: 95% paired crossed seed/image percentile intervals",
    )
    axis.invert_yaxis()
    axis.tick_params(labelsize=9)
    _save_figure(fig, output / "paired-effects")
    _tradeoff_figure(output, summary, colors, plt)
    fig, axis = plt.subplots(figsize=(8, 2.5))
    axis.axis("off")
    for position, label in (
        (0.1, "Raw training\nannotations"),
        (0.36, "Targets + optional\noffline augmentation"),
        (0.63, "Shared trainer\n+ optional semantic loss"),
        (0.9, "Selected checkpoint\nraw logits → optional filter"),
    ):
        axis.text(
            position,
            0.6,
            label,
            ha="center",
            va="center",
            fontsize=8,
            bbox={"boxstyle": "round", "facecolor": "#eef3f7"},
            transform=axis.transAxes,
        )
    for position in (0.21, 0.48, 0.75):
        axis.annotate(
            "",
            (position + 0.06, 0.6),
            (position, 0.6),
            xycoords="axes fraction",
            arrowprops={"arrowstyle": "->"},
        )
    axis.text(
        0.5,
        0.15,
        "Frozen ontology artifacts feed augmentation, loss and filtering. Raw validation selects; raw test labels score.",
        ha="center",
        fontsize=8,
        transform=axis.transAxes,
    )
    _save_figure(fig, output / "method")


def _save_figure(fig: Any, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig.tight_layout()
    for extension in ("svg", "pdf", "png"):
        fig.savefig(path.with_suffix(f".{extension}"), dpi=300)
    plt.close(fig)


def _tradeoff_figure(
    output: Path, summary: list[dict[str, Any]], colors: dict[str, str], plt: Any
) -> None:
    index = {
        (r["model"], r["condition"], r["filter_mode"], r["seed"], r["metric"]): r
        for r in summary
        if r["statistic"] == "seed"
    }
    points = []
    for key, row in index.items():
        if key[-1] != "recall_image_mean_at_50":
            continue
        violation = index.get((*key[:-1], "incompatible_rate_all"), {})
        count = index.get((*key[:-1], "incompatible_count"), {})
        points.append(
            {
                **row,
                "incompatible_rate_all": violation.get("value"),
                "incompatible_count": count.get("value"),
            }
        )
    _write_csv(output / "figure_data" / "recall-violations.csv", points)
    models = list(dict.fromkeys(p["model"] for p in points)) or ["No results"]
    modes = list(dict.fromkeys(p["filter_mode"] for p in points)) or ["none"]
    fig, axes = plt.subplots(
        len(models), len(modes), figsize=(5 * len(modes), 4 * len(models)), squeeze=False
    )
    for model_index, model in enumerate(models):
        for mode_index, mode in enumerate(modes):
            axis = axes[model_index, mode_index]
            annotated = set()
            for point in points:
                if point["model"] != model or point["filter_mode"] != mode:
                    continue
                if point["value"] is not None and point["incompatible_rate_all"] is not None:
                    x, y = 100 * point["incompatible_rate_all"], 100 * point["value"]
                    label = point["condition"]
                    axis.scatter(x, y, color=colors[label], s=20, label=label)
                    point_key = (x, y, point["incompatible_count"])
                    if point_key not in annotated:
                        axis.annotate(
                            str(int(point["incompatible_count"])),
                            (x, y),
                            xytext=(4, 4),
                            textcoords="offset points",
                            fontsize=8,
                        )
                        annotated.add(point_key)
            handles, labels = axis.get_legend_handles_labels()
            unique = dict(zip(labels, handles, strict=True))
            if unique:
                axis.legend(unique.values(), unique.keys(), fontsize=8)
            axis.set(
                xlabel="Known incompatibility in top 100 (%)",
                ylabel="Image-mean recall@50 (%)",
                title=f"{model} / {mode}\nLabels: absolute violation counts",
            )
            axis.tick_params(labelsize=9)
    _save_figure(fig, output / "recall-violations")


def _write_report(
    output: Path,
    expected: list[dict[str, Any]],
    available: dict[str, dict[str, Any]],
    statuses: dict[str, str],
    label: str,
    campaign: dict[str, Any] | None = None,
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    summary = _summary(expected, available, statuses)
    effects = _effects(campaign, available) if campaign is not None else []
    _write_csv(output / "summary.csv", summary)
    _write_csv(output / "paired_effects.csv", effects)
    _write_csv(
        output / "controls.csv",
        [
            row
            for row in summary
            if row["condition"]
            in {"baseline", "augment", "frequency_control", "shuffled_augmentation_control"}
        ],
    )
    _write_csv(
        output / "evidence.csv",
        [
            {
                "evaluation_id": v["manifest"]["evaluation_id"],
                "run_id": v["manifest"]["run_id"],
                "path": v["path"],
                "claim_scope": label,
            }
            for v in available.values()
        ],
    )
    _figures(output, summary, effects)
    missing = len(expected) - len(available)
    tradeoffs = [
        r
        for r in effects
        if r["metric"] == "recall_image_mean_at_50"
        and r.get("effect") is not None
        and r["effect"] < -0.01
    ]
    narrative = (
        f"# {label}\n\n{len(available)} of {len(expected)} expected evaluation cells are complete; {missing} are missing or failed. "
        "Incomplete groups have no aggregate mean. All seed rows remain visible.\n\n"
        "Recall and truncated annotation-based AP use raw annotations and exact object identities. "
        "AP is not audited visual precision. Zero remaining known violations after ontology filtering is a property of that filter, "
        "not evidence of learned visual correctness. Unknown decisions remain explicit.\n\n"
        "Scope is the sampled benchmark, supplied objects, tested predictors, ontology profile and fixed lambda. "
        "NeSy4VRD has historical test exposure and a closely related ontology; fresh seeds do not supply independent-dataset confirmation. "
        "Incomplete annotations make unlabelled predicates within annotated pairs assumed negatives. "
        "Frequency weighting matches supported positive-loss mass, not all structural effects of augmentation. "
        "The simple filter uses only asserted constraints. Runtime claims need measured preparation and inference costs as well as training.\n\n"
        "Intervals use 10,000 paired crossed seed/image draws; predicate-mean recall uses seed-only draws and describes run variation only. "
        "Undefined draws are disclosed and intervals require at least 95% defined draws.\n"
    )
    if campaign is None:
        narrative += "\nThis is a synthetic integration report; it does not complete G3/G4/G5 or establish benchmark effects. No campaign uncertainty is claimed.\n"
    for row in tradeoffs:
        narrative += f"\nTradeoff: {row['contrast']} lowers recall@50 by {-100 * row['effect']:.3f} percentage points, exceeding the registered one-point threshold.\n"
    (output / "report.md").write_text(narrative, encoding="utf-8")
    write_json(
        output / "report.json",
        {
            "schema_version": "kgr4vrd.report/1",
            "label": label,
            "status": "completed" if not missing else "incomplete",
            "expected_evaluations": len(expected),
            "completed_evaluations": len(available),
            "inputs": {key: value["digest"] for key, value in available.items()},
            "plan_digest": None if campaign is None else campaign.get("plan_digest"),
        },
    )
    return output / "report.json"


def report_evaluations(
    evaluation_paths: Sequence[str | Path],
    output: str | Path,
    *,
    label: str = "Synthetic integration evidence",
) -> Path:
    """Report explicit smoke artifacts without implying a frozen research campaign."""
    available, expected = {}, []
    for path in evaluation_paths:
        value = load_evaluation(path)
        row = value["manifest"]
        key = row["evaluation_id"]
        if key in available:
            raise ValueError("Duplicate evaluation artifact")
        available[key] = value
        expected.append(
            {
                "key": key,
                "dataset": row.get("dataset", "unspecified"),
                **{name: row[name] for name in ("model", "condition", "seed", "filter_mode")},
            }
        )
    return _write_report(
        Path(output), expected, available, dict.fromkeys(available, "completed"), label
    )


def summarise(campaign_path: str | Path, output: str | Path) -> Path:
    """Rebuild the complete registered inventory, preserving failed/missing rows."""
    path = Path(campaign_path).resolve()
    campaign = load_campaign(path)
    status = read_json(path.parent / "status.json")
    rows = status.get("evaluations", [])
    if len({row["key"] for row in rows}) != len(rows):
        raise ValueError(
            "Duplicate evaluation attempts: select the registered representative explicitly"
        )
    expected = {r["key"]: r for r in campaign["evaluations"]}
    if any(row["key"] not in expected for row in rows):
        raise ValueError("Evaluation inventory contains unplanned rows")
    available, statuses = {}, {r["key"]: r["status"] for r in rows}
    for row in rows:
        if row["status"] == "completed":
            result = load_evaluation(path.parent / row["path"])
            for key in ("model", "condition", "seed", "filter_mode"):
                if result["manifest"][key] != expected[row["key"]][key]:
                    raise ValueError(f"Evaluation identity mismatch: {key}")
            if result["manifest"]["split"] != "test" or campaign["status"] != "frozen":
                raise ValueError("Final campaign reporting requires frozen test evaluations")
            if result["manifest"].get("plan_digest") != campaign["plan_digest"]:
                raise ValueError("Evaluation belongs to another frozen analysis plan")
            selection = read_json(
                path.parent / "selections" / f"{expected[row['key']]['group']}.json"
            )
            matches = [
                r for r in selection["selections"] if r["key"] == expected[row["key"]]["run_key"]
            ]
            if len(matches) != 1 or any(
                matches[0][key] != result["manifest"][key]
                for key in ("run_id", "checkpoint_digest")
            ):
                raise ValueError(
                    "Evaluation does not use the registered group's selected checkpoint"
                )
            available[row["key"]] = result
    return _write_report(
        Path(output),
        list(expected.values()),
        available,
        statuses,
        "Registered NeSy4VRD results",
        campaign,
    )
