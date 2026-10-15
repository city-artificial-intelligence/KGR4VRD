# Implementation and verification report

Date: 15 October 2026. This report distinguishes executable software, synthetic
verification, external integration acceptance and the unperformed research campaign.

## Node-review follow-up — 16 October 2026

The [four review findings](implementation-review-2026-10-15.md) received focused
software corrections on `codex/dev-node-experiments`: explicit RDF formats,
structural Boolean/RL admission with selective unknown outcomes, reproducible
CUDA dependency/configuration profiles, and actual-device visual extraction with
environment-bound caches. The [node handoff](compute-node-handoff.md) defines the
remaining integration and pilot sequence; the five specifications are unchanged.

The public VRD-World snapshots can be structurally inventoried without deleting
axioms. Their union expressions require the documented partial RL/RDF policy;
this does not prove general OWL consistency or recover historical variants.
GraphDB checks Boolean fixtures and samples fresh queries before campaign
admission. Unknown coverage must be measured on the acquired ontologies.

The two Linux CUDA locks retain the registered torch/torchvision versions and
the common CPU-lock dependencies. Cross-platform resolver dry runs pass for
CUDA 11.8 and 12.1; no GPU binaries were executed on this Intel macOS host.
The CUDA base expands to the CPU base with only `training.device` changed.
Opt-in node tests exercise actual crop kernels, both FP32 heads and exact resume
across semantic activation under the production numerical flags and the documented
cuBLAS workspace. These checks remain skipped locally, alongside real GraphDB and
the acquired encoder check. Their existence is not GPU/backend acceptance.

Preservation was rechecked against the starting inventory: all 32 specification
and research files and all 53 archived scripts are unchanged. The 25 research
files (excluding `.DS_Store`) are now included in the requested Git handoff.
The dated evidence below remains the original implementation record.

Follow-up verification: **106 passed, 7 explicit integration skips** (five CUDA,
one real GraphDB, one acquired encoder); Ruff lint/format, mypy over all 22 source
files, locked dependency checks and whitespace checks on the corrections pass.
Archived research files retain their original bytes, including existing whitespace.
The 611 warnings
are dependency deprecations from RDFLib/Matplotlib's Pyparsing calls. Both CUDA
locks and the clean-wheel Linux CPU install command resolve in target-platform
dry runs. The latter now explicitly selects the CPU PyTorch index in CI.

A fresh wheel environment completed the offline four-condition smoke from
`/private/tmp` with Internet socket connections blocked. All four run manifests,
four saved-score artifacts and eight evaluations verified; regenerating the report
preserved the records/values in all seven CSV tables and figure-data files.
Wheel sources match the checkout. Wheel/sdist builds pass, and the source archive
includes the CUDA locks while excluding research files and generated artifacts.

The active package is now **5,494 nonblank Python lines**, still 19 substantive
modules and five direct core dependencies. The existing size rationale below
continues to apply; these corrections add no second trainer or reasoning engine.
Local verification records are under `artifacts/verification/review-2026-10-16/`;
generated artifacts remain outside Git.

## Implemented software

The installable `src/kgr4vrd` package provides canonical NeSy4VRD conversion,
explicit partition loaders, the registered geometry features, two relation heads,
stable multi-label BCE, persistent semantic activation, one trainer and one
validation-only checkpoint selector. It saves resumable optimizer/RNG/mask state,
source/environment identities, complete logits in bounded shards, and structured
evaluation artifacts. Visual pair features are loaded one image at a time.

Preparation provides per-image OWL-RL augmentation, typed GraphDB decisions,
asserted-only masks, frequency-matched positive supervision and the bounded shuffle
fallback. Training and evaluation use offline knowledge artifacts. No active code
imports the historical scripts. Unknown reasoning remains distinct from logical
incompatibility and operational failure.

The explicit campaign has 40 core final jobs, 10 control jobs, eight bounded
development jobs and 110 primary evaluation views. Freeze checks actual input
artifacts and development decisions, then binds complete checkpoint groups.
Reporting retains failed/missing cells, computes paired uncertainty, and exports
CSV data and SVG/PDF/PNG figures without retraining or GraphDB.

## Evidence and limitations

The verified entry points and invocation examples are in
[running experiments](running-experiments.md). Input formats and prerequisites are
in [data preparation](data-preparation.md). Scientific differences and the starting
inventory are in [historical differences](historical-differences.md).

The CPU regression suite checks T01–T15 and the software portions of T17–T24.
Tests use hand-calculated results, an extracted legacy feature oracle, actual
PyTorch optimization, actual OWLRL closure, deterministic service/encoder fakes,
saved artifacts and a four-condition end-to-end example. Independent review
verified the T16 claim boundaries and prompted fixes for visual-memory use,
score-shard validation and frozen-input binding. Report figures were inspected.

The resume test compares actual parameters, predictions, optimizer tensors,
selected epoch, update count and persistent masks after interruption at epoch 3.
CPU floating comparisons use `atol=1e-6`, `rtol=1e-5`; identities, masks and counts
are exact. A test-target sentinel remains unread during training and validation
prediction. Re-evaluation and report regeneration use saved artifacts.

The original 32 specification/research files were hash checked after migration;
all were unchanged. All 53 moved historical files are byte-identical to the
starting Git objects. The README was updated to identify the supported package;
the manuscript and review files remain untouched and untracked.

## Gate status

| Gate | Status and acceptance boundary |
| --- | --- |
| G0 | Checkout/public-source inventory recorded; actual dataset, mapping and ontology acquisition remains unresolved. Historical checkpoints/logs/judgments are absent. |
| G1 | Offline scientific kernels and defect checks pass. Real GraphDB configuration, ontology capability and sampled reference conformance remain unverified. |
| G2 | Local CPU package, resume, raw-score regeneration and report paths verified. Clean-wheel and final suite evidence is recorded below. Linux reference execution remains to be run in CI. |
| G3 | Incomplete: no NeSy4VRD preparation, pilot, frozen development choices or five-seed geometry results. |
| G4 | Incomplete: no acquired real encoder/cache, visual qualification or final visual/control runs. Fake-encoder checks do not qualify a stronger predictor. |
| G5 | Reporting software is implemented; the research evidence package is incomplete because G3/G4 results do not exist. |

The registered empty campaign and its explicit missing-cell report can be generated
with the documented `plan`/`report` commands. Synthetic checkpoints/evaluations are
stored separately from that campaign and do not satisfy its expected seed rows.
No historical numerical result has been reproduced. No positive experimental gain,
general OWL completeness, independent-dataset generalisation or submission
readiness is claimed.

## Environment and remaining dependencies

The measured local platform is Intel macOS CPU, Python 3.11.3, PyTorch 2.2.2,
NumPy 1.26.4, RDFLib 7.1.4, OWLRL 7.1.4 and TorchVision 0.17.2. The committed lock
also selects explicit Linux CPU wheels. CI installs the lock, runs lint/format/type
and scientific checks, builds a wheel, and invokes the offline example from `/tmp`.
Linux execution has not occurred in this session. CUDA and MPS are not validated.

Outstanding external inputs are NeSy4VRD 1.1 files and license provenance;
measured image/source hashes; the exact asserted augmentation/checker ontologies
and mapping; a dedicated versioned GraphDB repository with verified consistency
checking; pinned ResNet-50 V2 weights; and the real-data pilot/resource record.
These dependencies prevent real experimental acceptance, rather than being replaced
with invented artifacts. Perceptual duplicate adjudication is not automated.

Scientific limitations remain the retrospective benchmark/ontology relationship,
incomplete annotations, the selected rule profile and fixed lambda, supported
frequency-control mass, and typed-relation rather than whole-image consistency.
Fewer epochs alone do not establish runtime savings; filtered zero violations do
not establish learned correctness; annotation AP is not audited visual precision.

## Final verification record

The final local suite passed **76 tests**, with **two explicit integration skips**:
real GraphDB fixtures/200 fresh conformance queries and the acquired ResNet-50
encoder/cache check. The 14 warnings are Matplotlib/Pyparsing deprecations.
Ruff lint, Ruff format, mypy (22 source files), locked dependency resolution,
installed dependency compatibility, and Git whitespace checks passed.

The wheel and source distribution built successfully. The source archive excludes
untracked research materials, generated artifacts and environments. The wheel's
22 Python source files were compared byte-for-byte with the active checkout.
A separate environment installed the wheel with locked dependencies and ran from
`/private/tmp`, using a socket guard that rejects Internet connections. It completed
four conditions, four epochs each, four selected checkpoints, four raw-score
artifacts, eight evaluation views and the synthetic report. All four run manifests
verified. Re-evaluation reused the saved logits, and all seven table/figure-data
CSV files regenerated with identical records and values (row ordering follows the
explicit evaluation input sequence).

Local evidence is in `artifacts/verification/preservation-and-package.json`,
`artifacts/verification/installed-wheel.json`, and
`artifacts/verified-wheel-smoke/report/report.json`. The smoke report's tables and
SVG/PDF/PNG figures are beside that manifest; its actual selected checkpoints are
under `artifacts/verified-wheel-smoke/artifacts/runs/*/selected.json`.
The registered real campaign in `artifacts/registered-plan/campaign.json` has no
completed experiments; its generated report retains all 110 missing evaluation
cells. These generated artifacts are intentionally ignored by Git.

Independent final review added checks that bind visual caches to configured weights
and source image inventories, validate augmentation schema/profile digests, and
bind inferred targets to training object classes as well as raw relations. A real
campaign cannot freeze with synthetic or insufficient GraphDB conformance.
The documented preparation/freeze snippet was executed against synthetic artifacts.

## Size and simplification review

The active package has **5,183 nonblank Python lines**, including docstrings,
across **19 substantive modules** (22 Python files including package entry points).
There are **five direct core dependencies** and two optional runtime dependencies.
This exceeds the 3,000–4,500-line target but remains below all mandatory growth-review
thresholds. The additional code implements artifact verification, complete campaign
admission, and the two external preparation adapters; it is exercised by current
callers. No production implementation is hidden in tests or shell scripts.

`reasoning`, `training`, `campaign`, `reporting`, `prediction` and the dataset adapter
exceed the 350-line module guideline. Their longer functions keep one preparation
sequence, one stateful epoch/checkpoint loop, explicit freeze checks, saved-evidence
aggregation, shard validation or the canonical conversion contract together.
Splitting those sequences into forwarding modules would add navigation without
creating reusable scientific operations. Small shared kernels remain separate:
geometry, targets, models, losses, controls, metrics and artifact persistence.
There is one supported trainer, selector and evaluator; model/intervention variants
reuse them. The visual index is read once per execution view and vectors are loaded
one image at a time, avoiding duplicate encoder work and an in-memory dataset-sized
pair-feature tensor.
