# Architecture and code quality

## Design objective

Build a small research library with a command-line interface. The useful separation is between scientific functions, experiment execution, and external formats/services. Use explicit arguments and ordinary Python modules. This project does not need enterprise service layers, an application server, or a general experiment framework.

The active path must be easy to trace: configuration -> prepared scenes/features/knowledge -> model training -> validation selection -> saved scores -> evaluation -> report. A reader should be able to follow one run without navigating multiple generic managers or dynamically registered implementations.

## Proposed layout

```text
pyproject.toml
uv.lock
README.md
src/kgr4vrd/
    __init__.py
    __main__.py
    types.py
    config.py
    data.py
    features.py
    models.py
    losses.py
    reasoning.py
    metrics.py
    training.py
    prediction.py
    artifacts.py
    reporting.py
    cli.py
    adapters/
        __init__.py
        nesy4vrd.py
        graphdb.py
        vision.py
configs/
    smoke.toml
    nesy4vrd.toml
    campaign.toml
tests/
    fixtures/
    test_data.py
    test_reasoning.py
    test_losses.py
    test_metrics.py
    test_selection.py
    test_artifacts.py
    test_pipeline.py
docs/
    specs/
    running-experiments.md
    data-preparation.md
    historical-differences.md
```

This tree is a responsibility map, not an instruction to create empty files. Add a module only with implemented behaviour. There is one dataset adapter in this phase. Do not build alternative dataset adapters or generic dataset registries in anticipation of future work. A fallback visual model may add one narrowly scoped adapter after its decision is recorded.

During migration the old directories remain available as references. The new package must not import them. After equivalent behaviour and intentional differences have been documented, retire the obsolete active entry points in one migration step; use Git history to preserve tracked source, or a clearly labelled `legacy/` snapshot when needed for historical reproduction. Do not maintain two supported training/evaluation pipelines. Preserve untracked research files separately and never sweep them into a cleanup deletion.

## Responsibility and dependency rules

| Module | Owns | Must not own |
| --- | --- | --- |
| `types` | Small dataclasses/enums for scenes, vocabularies, decisions, and manifests | Filesystem access, neural models, service clients |
| `config` | TOML loading, explicit defaults, validation, resolved configuration | Training or implicit downloads |
| `data` | Prepared-record reading, split enforcement, deterministic pair/target construction | Raw-format exceptions or ontology adaptation |
| `features` | Versioned geometric features and feature-array assembly | Training loops or model selection |
| `models` | Two concrete relation heads and model construction by a small explicit branch | Dataset parsing, callbacks, experiment scheduling |
| `losses` | Ordinary BCE and semantic penalty functions | Network calls, file writes, mutable global caches |
| `reasoning` | Profile construction, per-scene closure, immutable compatibility tables, mask lookup | Gradient computation or test-label selection |
| `metrics` | Pure matching, counting, metric aggregation | Checkpoint loading, file discovery, rounding before selection |
| `training` | Optimisation, validation, stopping, checkpoint selection | Reading test relation labels or implementing metrics again |
| `prediction` | Model forward passes and deterministic candidate ordering | Choosing checkpoints by test scores |
| `artifacts` | Hashing, atomic writes, run state, checkpoint persistence | Scientific choices or a custom database |
| `reporting` | Saved-result aggregation, uncertainty, tables and figures | Training or silently dropping failed runs |
| `cli` | Argument parsing and calls into the modules above | Losses, feature formulas, inference rules, or a workflow engine |
| `adapters` | NeSy4VRD conversion, GraphDB protocol, pretrained encoder access | Shared experiment semantics or duplicated evaluators |

Domain records do not depend on adapters. Computational modules can use NumPy/PyTorch where their mathematics requires it. Experiment functions receive prepared inputs rather than constructing external services themselves. Only preparation invokes GraphDB; ordinary training, prediction, and evaluation must work offline with prepared artifacts.

Avoid importing plotting and visual-encoder dependencies when importing the base library. No module performs work on import. All executable entry points call a `main()` function under the normal module guard.

## Small interfaces

The exact names may be refined before their first implementation, but these responsibilities and input boundaries are fixed. Use dataclasses or ordinary typed records; do not add a wrapper class for each function.

| Operation | Inputs | Output/invariant |
| --- | --- | --- |
| `load_config` | One TOML path plus documented path/device overrides | Validated resolved configuration; unknown fields fail |
| `prepare_dataset` | Raw source paths, adapter choice, split recipe | Canonical dataset manifest and partitioned records |
| `build_pairs` | One scene's supplied objects and vocabulary | Deterministic pair array and stable pair IDs |
| `build_targets` | Pair IDs and permitted training relations | Multi-hot targets and ordinary-loss row mask |
| `augment_scene` | One training scene, frozen augmentation profile | Deduplicated targets and derivation metadata |
| `prepare_compatibility` | Vocabulary/mappings, ontology, query universe, backend | Typed-pair decisions with complete provenance |
| `lookup_mask` | Pair class IDs, predicate order, immutable table | Incompatibility/coverage masks; independent of query history |
| `update_active_mask` | Current confidences, incompatibility mask, previous per-cell activation, epoch | Persistent loss activation, separate from the global classification cache |
| `compute_loss` | Logits, targets, masks, lambda, reduction | Total, ordinary, and semantic losses; no side effects |
| `train` | Train inputs, validation inputs, model config, prepared knowledge, output path | Completed run manifest and a single selected checkpoint |
| `predict` | Frozen checkpoint and object/image inputs | Raw scores aligned to stable pairs and predicate order |
| `evaluate` | Explicit expected image IDs, raw targets, scores, policy, decision table | Per-image counts, per-predicate counts, aggregate metrics |
| `summarise` | Campaign manifest and explicit evaluation artifacts | Complete result table, contrasts, uncertainty, and figure data |

Training does not receive a test-scene collection or a dataset object that automatically loads all relation partitions. Prediction can receive test images and supplied objects but no relation labels. Evaluation owns access to the target relation partition. This is an API/data-loading boundary, not an operating-system security product.

The only external-service interface initially needed is a small typed compatibility query, implemented by a deterministic test fake and the GraphDB adapter. Prefer a callable or one `Protocol`; do not create an abstract hierarchy of reasoners, repositories, stores, clients, and services around it.

## Configuration and command line

Use the standard library's TOML parser and `argparse`. Configuration contains scientific choices, not executable Python imports, shell commands, or class paths. The implementation has a small explicit allowlist of dataset/model/profile names. Do not use Hydra, a plugin registry, nested configuration inheritance, or arbitrary expression evaluation.

The following are proposed commands; they are not available before implementation:

```text
kgr4vrd inspect --config configs/nesy4vrd.toml
kgr4vrd prepare --config configs/nesy4vrd.toml --stage data
kgr4vrd prepare --config configs/nesy4vrd.toml --stage reasoning
kgr4vrd prepare --config configs/nesy4vrd.toml --stage features
kgr4vrd plan --campaign configs/campaign.toml --output artifacts/plan
kgr4vrd train --config artifacts/plan/run-configuration.json
kgr4vrd train --config artifacts/plan/run-configuration.json --resume
kgr4vrd freeze --campaign artifacts/plan/campaign.json
kgr4vrd freeze --campaign artifacts/plan/campaign.json --group geometry-core
kgr4vrd predict --selection artifacts/runs/RUN/selected.json --split test --campaign artifacts/plan/campaign.json
kgr4vrd evaluate --predictions artifacts/predictions/PRED/manifest.json --campaign artifacts/plan/campaign.json
kgr4vrd report --campaign artifacts/plan/campaign.json --output artifacts/report
kgr4vrd verify --run artifacts/runs/RUN
```

Human-authored input is TOML; planner-produced resolved run configuration is JSON using the same validated fields. `train` accepts these two explicit formats by suffix. Do not invent another configuration language or let arbitrary file content decide parser behaviour.

`inspect` is read-only and reports missing inputs, label/profile coverage, split counts, and expected work. `prepare` is an explicit local artifact build, with data downloads performed only by a documented opt-in acquisition step. `plan` expands the finite named groups and deduplicates identical runs. It does not execute jobs. A plain serial loop or a small existing scheduler wrapper is sufficient to execute a plan when authorised; no scheduler is part of the package.

`freeze` without a group freezes the completed scientific analysis plan after all development decisions. With `--group`, it binds that group's selected checkpoints to the already frozen plan. Test prediction/evaluation resolves and verifies the matching group selection; it cannot silently use a later checkpoint. `verify` checks a run's manifest, checksums, expected outputs, and reproducibility invariants. Neither command contacts an approval service or creates a separate user-confirmation workflow.

Exit codes: 0 success; 2 invalid configuration/data; 3 missing external prerequisite; 4 failed execution or invalid numerical/service outcome. Explain the affected path/image/type/run and the next actionable correction. Do not catch all exceptions and return an empty result.

## Dependencies and installation

Target Python 3.11 initially. Establish one tested dependency resolution rather than promising compatibility with all versions. Use `pyproject.toml` for metadata and dependencies, a committed lock for the tested environment, and standard wheel/editable installation. `uv` may resolve the lock; installed library use must not require an active `uv` process.

Core dependencies should be limited to PyTorch, NumPy, Pillow, RDFLib, and OWLRL. Put TorchVision in a `vision` extra and Matplotlib in a `report` extra. Use the standard library for JSON, CSV, TOML, paths, logging, hashes, and simple HTTP where practical. A small HTTP library is acceptable if it reduces GraphDB code and its use is justified. Do not introduce pandas solely for simple record counting or CSV output.

Development dependencies are pytest, Ruff, and one type checker. Pin the resolved stack and record platform-specific accelerator wheels outside portable scientific configuration. No new deep-learning training framework is needed.

The required reference platform is Linux with CPU tests and a declared CUDA training environment when hardware exists. macOS/MPS is supported only after its tensor/device and smoke checks pass; it must not be claimed from import success. CPU fixtures are mandatory everywhere. Exact cross-device numerical equality is not promised.

Install the locked dependencies from their package sources or a pre-provisioned wheel cache. After installation, the synthetic smoke example must execute without credentials, network access, GraphDB, downloaded weights, or the full dataset. Real-data acquisition and real-reasoner conformance are separately documented checks.

## Limits on code growth

A01. Target 3000–4500 nonblank production Python lines for the active package. Crossing 6000 lines, 22 substantive modules, or 6 direct core dependencies triggers a written simplification review in the implementation report. These are review thresholds, not reasons to compress readable code or omit required behaviour. Count docstrings as lines; do not hide runtime code in notebooks, shell scripts, generated templates, or tests to satisfy the threshold.

A02. Aim for modules below 350 lines and cohesive functions below 60 lines. A longer numerical routine can be clearer than a chain of trivial forwarding functions; explain exceptions rather than splitting mechanically. No arbitrary line limit is a correctness gate by itself.

A03. Keep one ordinary loss, one semantic penalty, one training loop, one checkpoint selector, and one canonical evaluator. Model and intervention changes are data/configuration branches through those functions. An external reference evaluator is an explicitly named conformance adapter, not a second silent source of headline results.

A04. Do not create numbered utility modules, one-method service classes, empty interfaces, metaclass registration, dependency-injection containers, or a class per experiment. Neural network modules and stateful backend handles are legitimate classes.

A05. Avoid copying whole legacy files and renaming them. Extract only the behaviour required by the registered conditions. Remove unused experiment dimensions, commented-out alternatives, debugging branches, personal machine paths, and log-scraping stages from the active package.

A06. Tests should cover scientific invariants, observed defects, serialization, and one integrated run. Do not mirror every implementation function with a trivial test, snapshot entire logs, or chase a coverage percentage through tests that restate the code.

A07. Every new dependency and production module must have a named responsibility and a current caller. Review whether an existing standard-library or installed dependency already handles the need. Do not wrap stable library operations merely to conceal their names.

## Coding conventions

Use descriptive names such as `predicate_ids`, `validation_image_ids`, and `incompatible_mask`. Avoid unexplained numeric treatment IDs in normal APIs. Keep legacy IDs only in a provenance mapping.

Use `pathlib.Path`, explicit UTF-8, typed public signatures, and concise docstrings stating semantics, shapes, and non-obvious assumptions. Comments should explain why a choice matters, especially scientific limitations; remove line-by-line narration. Public computational functions document tensor axes and units.

Use keyword arguments where several adjacent values have the same type or where a choice changes scientific semantics. Validate shapes and vocabulary hashes at module boundaries. Derive class/predicate counts from vocabulary records; no literal 71/109 in general algorithms.

Use stable ordering explicitly. Do not depend on incidental dictionary order, filesystem listing order, or stringified tuple keys. Never use `eval` to parse annotations or configuration. Do not mutate caller-owned scenes or target tensors to change a condition.

Allocate masks and constants on the correct tensor device/dtype. Use inference/no-gradient mode for evaluation. Reject non-finite logits, losses, and metrics with context. Keep the initial training implementation in full precision; mixed precision is a named later change with numerical checks.

Use standard logging for concise progress and structured JSONL/CSV for results. Log one record per completed epoch, clear startup provenance, and actionable failures. Avoid per-cell dumps during ordinary training. Logs are diagnostics, not the database from which metrics must be reconstructed.

## Legacy migration map

| Existing code | New responsibility | Migration treatment |
| --- | --- | --- |
| `vrd_ppnn_training_7.py`, `vrd_utils13.py` | `training`, `artifacts`, `metrics` | Extract the training/validation path; remove test-based selection and repair split handling. |
| `vrd_ppnn_models.py` | `models` | Retain one geometry head; add one visual head. |
| `vrd_ppnn_dataset.py`, `vrd_utils8.py`, `vrd_utils5.py` | `data`, `features`, NeSy adapter | Preserve the registered target and feature profile; replace string keys/eval with typed IDs. |
| `vrd_ppnn_loss.py` | `losses` | Replace custom BCE machinery with tested framework operations; retain only the selected no-relation policy. |
| `vrd_utils17.py`, `vrd_utils15.py` | `reasoning`, GraphDB adapter | Separate decisions, activation state, and transport errors; move queries out of training. |
| `knowledgeGraphMgmt/*` | `reasoning`, preparation CLI | Reuse per-image OWL-RL closure logic with explicit profiles and immutable outputs. |
| Inference, selection, matching, and performance scripts | `prediction`, `metrics` | Save raw scores once; apply named ranking/filter policies; validate metrics against examples. |
| Consolidation and text-to-CSV scripts | `artifacts`, `reporting` | Read structured run artifacts; remove log parsing and test-score maxima. |
| `experimentSpaceModels/*` | Small TOML configs and historical mapping | Port selected conditions only, not the full grid of unused dimensions. |
| `predicatePredictionAnalysis/*` | Historical reference; small required count summaries in `reporting` | Do not port deferred inverse/symmetry mechanism studies. |
| `objectDetection/*` | Historical reference | Not part of the new active task. |

Do not modify the submitted manuscript while implementing the package. Record proposed claim corrections in the evidence package; manuscript rewriting follows the corrected experiments.
