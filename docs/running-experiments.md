# Running the corrected package

The only supported entry point is `kgr4vrd` (`python -m kgr4vrd` is equivalent).
The old scripts are historical references. The submitted paper is not modified.
All commands below refer to the repository root unless an absolute path is shown.

## Install and offline example

Use Python 3.11 and the committed `uv.lock`. The tested local environment is Intel
macOS CPU. Linux CPU is the required reference environment in CI; its status is
reported separately in the implementation report. CUDA/MPS are not verified here.

```sh
uv sync --locked --all-extras --group dev --python 3.11
uv run --locked kgr4vrd --help
uv run --locked kgr4vrd smoke --output artifacts/smoke
```

Choose a fresh output directory. The example builds synthetic inputs and runs all
four conditions for four epochs on CPU, including the epoch-3 semantic gate. It
writes checkpoints, complete raw score shards, eight validation evaluation views,
CSV tables and SVG/PDF/PNG figures. It needs no network, GraphDB, credentials,
images or pretrained weights after installation. These are synthetic integration
results, not NeSy4VRD experiments. Synthetic evaluation uses validation labels so
it cannot accidentally satisfy the real campaign's test-freeze gate.

The package itself does not require `uv` at runtime. To test a wheel separately:

```sh
uv build
uv venv /tmp/kgr4vrd-wheel --python 3.11
uv export --locked --all-extras --no-dev --no-emit-project > /tmp/kgr4vrd-requirements.txt
uv pip install --python /tmp/kgr4vrd-wheel/bin/python -r /tmp/kgr4vrd-requirements.txt
uv pip install --python /tmp/kgr4vrd-wheel/bin/python --no-deps dist/*.whl
cd /tmp
/tmp/kgr4vrd-wheel/bin/kgr4vrd smoke --output /tmp/kgr4vrd-example
```

## Prepare and plan a real campaign

Acquire and inventory the inputs described in [data preparation](data-preparation.md).
The checked-in NeSy configuration intentionally contains no invented raw filenames.
Paths in `[paths]` resolve relative to the configuration file. Other input paths
resolve against `data_root`, component paths against their manifest's directory.
`--data-root` and `--artifact-root` rebind execution roots explicitly.

```sh
kgr4vrd inspect --config configs/nesy4vrd.toml
kgr4vrd prepare --config configs/nesy4vrd.toml --stage data
kgr4vrd prepare --config configs/nesy4vrd.toml --stage reasoning
kgr4vrd prepare --config configs/nesy4vrd.toml --stage features
kgr4vrd plan --campaign configs/campaign.toml --output artifacts/plan
```

The finite plan contains 40 core final jobs, 10 control jobs, eight development
jobs and 110 primary evaluation variants. It does not launch them. First run the
eight development configurations, record the measured preparation/one-epoch pilot,
and supply the development-selection and control-preparation records. Development
uses only train and raw validation data. The default learning-rate comparison is
`[1e-4, 3e-4]` with seeds `[101,103]`; each predictor's recipe applies to every
intervention. A failure of the registered visual qualification remains visible.

`campaign.json` is a draft until all decisions and input digests are verified by
`freeze`. `status.json` is an explicit mutable inventory, not a source of scientific
choices. For each completed run, set its existing row's `status` to `completed`
and `selection` to that run's `selected.json`; retain failed/not-started rows. Do
not add another seed because a result is poor, choose a favourable attempt, or
replace the inventory with only successful runs.

For example, a completed inventory row is
`{"key":"geometry_mlp:development:101:0.0001","status":"completed","selection":"/absolute/artifacts/runs/RUN_ID/selected.json"}`.
Use the exact existing `key` and actual selected path. The development evidence is
a list of eight records with `model`, `seed`, `learning_rate`, and `selection`; the
selector reads both recall measures from the saved selected validation epoch.
The control decision contains `selected_condition`, `unmatched_fraction`, and the
prepared control manifest's `artifact_digest`, plus `changed_cell_fraction` for
the triggered shuffle fallback. These are measured evidence, not manual scores.

After recording all eight development selections in `status.json`, this snippet
fills those records and the required input identities from the prepared files.
Run it from the repository root while the plan is still a draft:

```python
from pathlib import Path
from kgr4vrd import features
from kgr4vrd.artifacts import canonical_digest, file_digest, read_json, write_json
from kgr4vrd.config import data_path
from kgr4vrd.reasoning import validate_profile_metadata

root = Path("artifacts/plan").resolve()
plan = read_json(root / "campaign.json")
assert plan["status"] == "draft"
configs = {
    model: read_json(root / next(r["config"] for r in plan["runs"] if r["model"] == model))
    for model in ("geometry_mlp", "visual_mlp_v1")
}
geometry, visual = configs["geometry_mlp"], configs["visual_mlp_v1"]
dataset_path = data_path(geometry, geometry["dataset"]["manifest"])
dataset = read_json(dataset_path)
feature_cache = read_json(data_path(visual, visual["visual"]["features_manifest"]))
weights_digest = file_digest(data_path(visual, visual["visual"]["weights_manifest"]))
assert feature_cache["weights_manifest"]["sha256"] == weights_digest
assert feature_cache["identity"]["image_manifest_digest"] == canonical_digest(dataset["image_manifest"])
augmentation = read_json(data_path(geometry, geometry["intervention"]["augmentation_manifest"]))
validate_profile_metadata(augmentation, "S_TEU")
assert augmentation["training_objects_digest"] == dataset["partitions"]["train"]["objects"]["sha256"]
plan["inputs"] = {
    "dataset": file_digest(dataset_path),
    "split": dataset["split_digest"],
    "vocabulary": dataset["vocabulary_digest"],
    "augmentation": file_digest(data_path(geometry, geometry["intervention"]["augmentation_manifest"])),
    "compatibility": file_digest(data_path(geometry, geometry["intervention"]["knowledge_manifest"])),
    "mapping": file_digest(data_path(geometry, geometry["preparation"]["mapping"])),
    "geometry_features": canonical_digest({
        "profile": geometry["model"]["feature_profile"], "source": file_digest(features.__file__)
    }),
    "visual_features": file_digest(data_path(visual, visual["visual"]["features_manifest"])),
    "visual_weights": weights_digest,
}
inventory = {row["key"]: row for row in read_json(root / "status.json")["runs"]}
assert all(inventory[row["key"]]["status"] == "completed" for row in plan["development_runs"])
plan["development"]["records"] = [
    {**{key: row[key] for key in ("model", "seed", "learning_rate")},
     "selection": inventory[row["key"]]["selection"]}
    for row in plan["development_runs"]
]
control_path = data_path(geometry, geometry["intervention"]["control_manifest"])
control = read_json(control_path)
assert control["training_objects_digest"] == dataset["partitions"]["train"]["objects"]["sha256"]
plan["control"] = {"record": {
    **{key: control[key] for key in
       ("selected_condition", "unmatched_fraction", "changed_cell_fraction") if key in control},
    "artifact_digest": file_digest(control_path),
}}
write_json(root / "campaign.json", plan)
```

Set `pilot.runtime_hours_range` and `pilot.storage_gb_range` to two-element
`[lower, upper]` arrays based on the measured preparation and one-epoch pilot;
retain the measurement record under `metadata`. No placeholder estimate satisfies
the real-data readiness gate. `freeze` independently verifies component checksums,
common training/validation inputs, source/environment identities, C2 coverage,
the augmentation profile, the cache's weight and source-image inventory, and the
visual qualification rule. If qualification fails and the single bounded
reference integration is unavailable, record
`development.fallback_decision={"status":"unavailable","reason":"actual dependency or integration limitation"}`.
That records the unresolved dependency and leaves the stronger-predictor claim
unsupported; it does not turn the failed qualification into a pass.

```sh
kgr4vrd train --config artifacts/plan/configs/CONFIG_DIGEST.json
kgr4vrd train --config artifacts/plan/configs/CONFIG_DIGEST.json --resume
kgr4vrd freeze --campaign artifacts/plan/campaign.json
```

Freezing resolves the selected recipes into the final configurations. Train the
final groups using those configurations, record their selected checkpoints in the
inventory, then bind each complete group before its first test prediction:

```sh
kgr4vrd freeze --campaign artifacts/plan/campaign.json --group geometry-core
kgr4vrd predict --selection artifacts/runs/RUN_ID/selected.json --split test --campaign artifacts/plan/campaign.json
kgr4vrd evaluate --predictions artifacts/predictions/PREDICTION_ID/manifest.json --campaign artifacts/plan/campaign.json --filter none
kgr4vrd evaluate --predictions artifacts/predictions/PREDICTION_ID/manifest.json --campaign artifacts/plan/campaign.json --filter ontology
```

Use `--filter simple_asserted` only for the ten baseline runs. Record evaluation
paths/statuses in the corresponding existing `status.json` rows. All views reuse
one validation-selected checkpoint and one raw score artifact. Later groups bind
their own selections to the same frozen plan. A changed scientific choice after
test use is an exploratory revision, not a replacement frozen plan.

## Verification and regeneration

```sh
kgr4vrd verify --run artifacts/runs/RUN_ID
kgr4vrd report --campaign artifacts/plan/campaign.json --output artifacts/report
uv run --locked pytest -q
uv run --locked ruff check src tests
uv run --locked ruff format --check src tests
uv run --locked mypy src/kgr4vrd
```

Re-evaluation needs raw scores, targets and immutable knowledge artifacts, but no
model forward pass or GraphDB. Reporting needs the campaign and saved evaluations;
it does not require a GPU, reasoner or original console logs. Missing seed cells
remain visible and prevent complete-group means. Figures are generated from CSV
figure data, with the same plotting functions exporting all three formats.

Runs are content identified. Resume restores the last completed epoch, optimizer,
RNG states and per-image semantic masks; it rejects changed source, input or
environment identities. A completed run is not overwritten. Source snapshots
contain the actual implementation files, including any dirty changes, and exclude
research files, data and credentials. Before a real campaign, use a clean commit.

To relocate data, copy the whole prepared tree and rebind `--data-root`; component
hashes and scientific identity remain unchanged. Keep images at the relative
`image_root` recorded by the dataset manifest, or move their containing tree too.
Run and prediction references inside their artifact tree are relative. Preserve
the explicit campaign inventory or update its location references before freezing.

CLI exit codes are 0 success, 2 invalid input/configuration, 3 missing prerequisite,
and 4 execution/service failure. Inspecting a missing preparation is read-only and
prints the missing input and next stage.
