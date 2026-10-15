# Compute-node handoff

Updated: 16 October 2026. Start with [the specification index and all five linked
documents](specs/README.md), then [the implementation review](implementation-review-2026-10-15.md).
The specifications remain the scientific contract. This handoff adds launch fixes,
resource checks and execution order; it does not add experiments or replace the
[data preparation](data-preparation.md) and [command guide](running-experiments.md).

**Current launch status:** the R1–R4 software corrections are available on
`codex/dev-node-experiments`. Real GraphDB, CUDA and pinned-encoder acceptance,
acquired input provenance and the resource pilot still gate the full campaign.
R1 uses explicitly partial Boolean/RL semantics: unsupported constructs fail and
unwitnessed union membership stays unknown. Record its coverage and scientific
consequence on the acquired ontologies before accepting the profile.

Fetch and check out `origin/codex/dev-node-experiments` on the node; record the exact
commit, then follow [the locked CUDA installation](cuda-environment.md). Preserve
the architecture and the specification's bounded model-selection scope.

## Node allocation and pilot

One RTX 2080 with 8 GB VRAM, six CPU cores and 64 GB host RAM is a reasonable
initial allocation for sequential execution. This is an estimate from the code,
not a measured GPU capacity result. Confirm the actual GPU model and available VRAM.

| Resource | Starting arrangement |
| --- | --- |
| GPU | One process at a time. Frozen encoder extraction first; cached-feature head training afterwards. Start extraction with 16 crops per batch and measure. |
| Training | Preserve FP32, one image per batch and `num_workers=0`. Do not introduce mixed precision or change the training batch to make a pilot fit. |
| CPU/RAM | Run preparation and training as separate stages. An initial 8 GiB GraphDB heap leaves room for Python and the OS; increase only from measured need. Limit competing thread pools to the allocated cores. |
| Storage | Provision roughly 100–200 GB of free local SSD as an initial reserve, then replace this estimate with measured cache/checkpoint/prediction sizes. This is not a measured minimum. |

With the historical 109-class/71-predicate vocabulary, the implemented heads have
669,947 and 3,415,368 trainable parameters. The encoder is frozen, and visual inputs
are materialized one image at a time. Capacity therefore depends more on candidate
pairs and cached artifacts than on head parameters alone. For ordered distinct
pairs, an image with `n` objects has `n*(n-1)` candidates. Useful lower-bound storage
calculations, excluding container/allocator overhead and compression, are:

- Current device masks: `2 * N_train_pairs * P` bytes across all training images,
  even in conditions that do not use the semantic penalty.
- Visual input tensor: `4 * N_image_pairs * (2*C + 15 + 3*2048)` bytes, plus
  activations, gradients and optimizer state.
- Saved FP32 scores: `4 * N_test_pairs * (P+1)` bytes per final model. There are
  50 models, with score reuse across filtering views.
- Feature vectors: `4 * 2048 * N_unique_crops` bytes, plus indices and metadata.

Before budgeting the campaign, record actual pair/crop counts, largest-image peak
VRAM, host peak RSS, free disk, extraction throughput, reasoning throughput,
checkpoint sizes, and a complete training-plus-validation epoch for each predictor.
Exercise the combined condition after the epoch-3 semantic gate as well. Use
isolated pilot artifacts and train/validation targets; shortened pilot runs do not
count as development or final runs. The current training timing record reports
host RSS, so collect CUDA allocated/reserved peaks separately.

At 109 classes and 71 predicates, the compatibility table has at most 843,551 typed
queries; actual preparation uses observed object-class pairs. Measure end-to-end
query throughput, including insertion and cleanup. The GPU does not accelerate
GraphDB. Project preparation separately from training and validation. The planned
58 training runs have a ceiling of 5,800 epochs at the registered maximum, before
early stopping; pilot checks and any explicitly triggered fallback are additional.
Do not promise a wall-clock duration before these measurements exist.

## 1. Finish installation and integration acceptance

Acquire the inputs in the data guide and record a clean source revision and one
of the committed Linux CUDA environments. The current `uv.lock` selects Linux CPU
wheels; `requirements/linux-cu118.txt` and `requirements/linux-cu121.txt` provide
separate hash-locked GPU profiles. Use one validated environment consistently
for development and final runs. The commands below assume its `python` and
`kgr4vrd` executables are active, without an implicit CPU re-sync.

Verify the driver/GPU inventory and an actual CUDA allocation:

```sh
nvidia-smi
python -c 'import torch; assert torch.cuda.is_available(); x=torch.ones(1, device="cuda"); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), x)'
python -m pytest -q
python -m ruff check src tests
python -m ruff format --check src tests
python -m mypy src/kgr4vrd
```

The corrected ontology and device paths have focused local regression checks.
Exercise the supplied real encoder and CUDA checks before development. The CUDA
resume check requires exact equality; the encoder-repeat check uses `atol=1e-6`,
`rtol=1e-5`. Retain numerical settings and measured results with node acceptance.

The external checks use these three environment variables:

- `KGR4VRD_GRAPHDB_CONFIG`: a path to **backend-only JSON**, containing the GraphDB
  fields documented in the data guide. Use an exclusive empty repository with
  consistency checking enabled. The test uses synthetic fixtures and at least 200
  fresh queries; campaign preparation must additionally validate the actual ontology.
- `KGR4VRD_ENCODER_CONFIG`: a path to a **main TOML/JSON configuration**, pointing to
  real pinned weights and a tiny separate canonical image fixture. The test extracts
  every configured image twice; do not point it at the complete research cache.
- `KGR4VRD_CUDA_TEST=1`: enables actual FP32 forward/backward and resume checks for
  both heads and the visual device path. An unavailable requested GPU fails.

Set both paths plus `KGR4VRD_CUDA_TEST=1`, export
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, and run
`python -m pytest -q -m integration --junitxml=artifacts/node-acceptance/integration.xml`;
no integration test may remain skipped when claiming acceptance. The encoder
configuration must select CUDA on this node. Preserve results.
Never clear a shared GraphDB repository to satisfy an empty-repository check.

## 2. Acquire and prepare immutable campaign inputs

Use the [data preparation guide](data-preparation.md) to inventory the NeSy4VRD 1.1
archive, images, labels, mappings, separate asserted augmentation/checker ontologies,
licenses and pinned ResNet-50 V2 weights. Record original SHA-256 hashes and any
documented conversion. Public reference ontologies are not automatically the exact
historical variants. Do not invent missing mappings, versions or recovered results.
As decisions D01/D02 specify, missing historical checkpoints do not block a fresh
corrected study. If exact historical ontologies cannot be recovered, explicitly
name the acquired/new profile, record its axiom differences and hashes, and validate
it before execution; do not present that study as historical replication.

Complete `configs/nesy4vrd-cuda.toml` and its acquired paths before planning. It
already selects `training.device="cuda"`. Set each ontology's explicit format
(`turtle`, `xml` or `nt`); a `.owl` suffix does not determine serialization. Preserve
the fixed split, ontology profiles and all scientific defaults, then run:

```sh
kgr4vrd prepare --config configs/nesy4vrd-cuda.toml --stage data
kgr4vrd inspect --config configs/nesy4vrd-cuda.toml
kgr4vrd prepare --config configs/nesy4vrd-cuda.toml --stage reasoning
kgr4vrd prepare --config configs/nesy4vrd-cuda.toml --stage features
```

Check inventory discrepancies and ontology capability before the expensive stages.
Raw training conflicts, failed conformance, service failures, unsupported semantics
or insufficient control coverage require resolution; they are not empty results.
Do not adapt ontology or mappings using test relation labels. The simple asserted
filter must stay distinct from the full checker.

Current feature extraction verifies completed caches but does not resume partial
ones. Use uninterrupted extraction or, after investigating failure, a fresh output
directory. Do not publish an incomplete manifest. Keep one accepted cache shared
across runs. Measure the resource pilot above before full training.

## 3. Run development and freeze the scientific choices

```sh
kgr4vrd plan --campaign configs/campaign-cuda.toml --output artifacts/plan
```

This writes a plan; it launches no training. Run the eight `development_runs` rows
serially using each row's generated configuration: two predictors, learning rates
`1e-4`/`3e-4`, and seeds `101`/`103`. Each is a baseline run under the registered
100-epoch maximum and patience 25. Configuration paths in plan rows are relative to
`artifacts/plan`: use `kgr4vrd train --config artifacts/plan/configs/CONFIG_DIGEST.json`
with the actual row's filename. Use `--resume` only for a matching interrupted run.

Record each actual `selected.json` and status in the existing `status.json` row.
Follow the command guide's evidence-population snippet to record prepared-input
digests, all eight development selections and the C2 decision. Add measured pilot
runtime/storage ranges and their underlying measurements. The visual predictor
must meet the specified validation qualification; a failed qualification remains
visible and follows only the bounded fallback rule in the specification.

```sh
kgr4vrd freeze --campaign artifacts/plan/campaign.json
```

This resolves the selected learning rate and C2 condition into the final configs.
Read their paths again after freezing. Do not start final jobs from draft configs.
All development choices for both models and C2 must be complete before any test
evaluation. Keep source, environment and input identities stable; a later code or
dependency correction requires resolving the affected development evidence before
test use, rather than editing hashes to admit old results.

## 4. Execute the final matrix serially

| Group | Training runs | Test evaluation views |
| --- | ---: | ---: |
| `geometry-core` | 4 conditions × 5 seeds = 20 | 40 unfiltered/full-filter + 5 simple-filter baseline views |
| `visual-core` | 4 conditions × 5 seeds = 20 | 40 unfiltered/full-filter + 5 simple-filter baseline views |
| `geometry-control` | 5 C2 runs | 10 unfiltered/full-filter views |
| `visual-control` | 5 C2 runs | 10 unfiltered/full-filter views |
| **Total** | **50 final runs, plus 8 development runs** | **110 views; each includes K = 25, 50, 100** |

Core conditions are `baseline`, `augment`, `semantic_loss`, `combined`; final seeds
are `11, 23, 37, 53, 71`. C2 is the frequency control or its registered triggered
shuffle fallback, not both as an extra sweep. Filtering creates no training jobs.

Train one complete group, record every selection/status, then bind it before test
prediction. Geometry core may finish first, after the common development freeze.
For example:

```sh
kgr4vrd freeze --campaign artifacts/plan/campaign.json --group geometry-core
kgr4vrd predict --selection RUN_DIRECTORY/selected.json --split test --campaign artifacts/plan/campaign.json
kgr4vrd evaluate --predictions PREDICTION_DIRECTORY/manifest.json --campaign artifacts/plan/campaign.json --filter none
kgr4vrd evaluate --predictions PREDICTION_DIRECTORY/manifest.json --campaign artifacts/plan/campaign.json --filter ontology
```

Replace placeholders with actual paths returned by training/prediction. Also run
`--filter simple_asserted` for baseline runs only. Repeat with each group's own
freeze and selections. All views and K values reuse the single checkpoint selected
by **unfiltered raw validation image-mean recall@50**, and the same saved raw scores.
Update the existing evaluation inventory rows with their actual paths/statuses.
Keep failed/missing runs visible; never replace a poor seed or select on test scores.

## 5. Verify and deliver

Run `kgr4vrd verify --run RUN_DIRECTORY` for every completed run, then:

```sh
kgr4vrd report --campaign artifacts/plan/campaign.json --output artifacts/report
```

Check all eight development rows, 50 final rows and 110 evaluation rows, rather than
just a successful report command. Preserve the frozen plan/group bindings, source
and environment records, prepared-input manifests, checkpoint selections, raw
scores, metrics, uncertainty estimates, figure data and generated figures. Confirm
evaluation/report regeneration from those artifacts without retraining or GraphDB.

Report actual peak resources and elapsed times, unresolved dependencies and negative
results. An unqualified visual predictor leaves the stronger-predictor requirement
open. Follow the existing G0–G5 acceptance boundaries; do not label a smoke run or
an incomplete campaign as research completion. Include this handoff and the review
when transferring the checkout; generated inputs/artifacts are ignored by Git and
must be copied separately if already prepared.
