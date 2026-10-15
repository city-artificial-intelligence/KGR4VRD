# Data and artifact contracts

## Contract principles

Use plain JSON/JSONL for metadata and records, NumPy NPZ for numerical arrays, CSV for result tables, and framework checkpoints for model/optimiser state. Do not create a schema-generation service or a custom binary format. Validate the fields below with small typed loaders and focused tests.

Each persisted artifact has a `schema_version`. Initial versions use descriptive names such as `kgr4vrd.dataset/1`, `kgr4vrd.knowledge/1`, and `kgr4vrd.run/1`. Reject unsupported major versions and unknown configuration fields. Optional descriptive metadata may be preserved under an explicit `metadata` object; it must not change scientific behaviour.

Content digests use SHA-256. Hash files by bytes and structured identity objects by UTF-8 JSON with sorted keys, compact separators, and no non-finite numbers. Exclude timestamps, output locations, and credentials from scientific identity. Record them separately where appropriate. Arrays have explicit dtypes, shapes, and axis identities; never rely on pickle-loaded object arrays.

## Canonical dataset

The prepared dataset is divided into supplied objects and target relations so training and prediction do not accidentally load test relation targets.

```text
data/prepared/nesy4vrd/v1/
    dataset.json
    vocabulary.json
    splits.json
    objects/train.jsonl
    objects/validation.jsonl
    objects/test.jsonl
    relations/train.jsonl
    relations/validation.jsonl
    relations/test.jsonl
    conversion-report.json
```

Mechanical conversion may read raw test annotations to persist canonical targets and supplied objects. It must not use their predicate labels to choose mappings, constraints, split membership, thresholds, hyperparameters, or training transformations. Later stages load partitions and objects/relations explicitly.

Here `v1` is a local preparation name, distinct from public dataset version 1.1. Once accepted, its contents are immutable and identified by hashes in the manifest. Rebuilding changed inputs uses a new preparation directory; a readable directory name does not replace content verification.

### Dataset manifest

| Field | Type | Requirement |
| --- | --- | --- |
| `schema_version`, `dataset_id`, `dataset_version` | strings | Distinguish public package version from local preparation version. |
| `source_files` | list of records | Relative identifier, source URL/local provenance, publisher checksum if available, SHA-256, byte count. |
| `adapter_version`, `code_digest` | strings | Identify the conversion implementation. |
| `vocabulary_digest`, `split_digest` | strings | Bind label order and membership. |
| `partitions` | object | Exact paths/digests, image/object/pair/relation counts for each partition. |
| `coordinate_convention` | string | `xyxy_pixel_continuous`; document the raw adapter's conversion. |
| `image_manifest` | list or JSONL reference | Image ID, relative image path, dimensions, content digest, optional duplicate-group ID. |
| `exclusions` | list or JSONL reference | Every dropped/quarantined record with reason and original identity. |
| `license_provenance` | record | Source terms/reference and redistribution status; do not invent a repository license. |

Class/predicate counts are measured from the pinned vocabulary. The inspected paper describes 109 classes and 71 predicates; a different acquired resource must trigger a version discrepancy report, not silently reshape historical checkpoints.

### Vocabulary and mappings

`vocabulary.json` has ordered `object_classes` and `predicates` arrays. Each entry has `id` (dense integer starting at zero), `label` (source text), `source_id` (string), and optional `ontology_iri`. Preserve the source's intended order or publish a deterministic remapping. Store the mapping explicitly; strings are not inferred indices.

Ontology mapping records contain `label_kind`, `label_id`, `iri`, `mapping_relation` (`exact`, `broader`, or `unmapped`), `rationale`, and `provenance`. Exact mappings are the default for this aligned benchmark. Broader mappings require a documented semantics check before using them for entailment; do not treat them as equivalent mappings. The artificial auxiliary output has index P but is not a predicate, vocabulary entry, or ontology property.

The same vocabulary digest must accompany targets, features, models, knowledge tables, and predictions. A valid tensor shape alone is not enough to accept them together.

### Objects and relations

One objects record per image:

```json
{"image_id":"example-001","image_path":"images/example-001.jpg","width":640,"height":480,"objects":[{"object_id":"o0","class_id":0,"bbox":[20.0,30.0,120.0,210.0]},{"object_id":"o1","class_id":1,"bbox":[160.0,100.0,420.0,390.0]}]}
```

One relations record per image, including an explicit empty list when appropriate:

```json
{"image_id":"example-001","relations":[{"subject_id":"o0","predicate_id":0,"object_id":"o1"}]}
```

These are schema examples, not real NeSy4VRD examples. IDs are stable strings. Where the raw source lacks object IDs, deduplicate only identical `(class_id, bbox)` records within an image, sort them deterministically, and assign IDs. Do not merge merely overlapping boxes or collapse distinct native IDs without evidence.

Canonical boxes use `(xmin, ymin, xmax, ymax)` in original-image pixel coordinates, with width `xmax-xmin` and height `ymax-ymin`. The adapter must document how inclusive or integer raw bounds are interpreted. Do not add one pixel or rescale silently. Validate finite values, positive area, image bounds, object references, and dense vocabulary membership. A declared conversion correction is recorded before splitting; a malformed record is not silently skipped during training.

Relations are sets of `(subject_id, predicate_id, object_id)`. Preserve multiple predicates per pair. Self-relations are excluded from this task and counted in the conversion report. Images with fewer than two objects have no candidate pairs and remain explicit records. All pair IDs are derived from ordered distinct object IDs, sorted lexicographically by subject then object.

`splits.json` contains three ordered image-ID lists, split provenance, seed/algorithm where used, and duplicate exclusions. Sets must be disjoint, and each listed image must occur exactly once in its objects and relations records. Missing prediction records are distinguishable from legitimate empty predictions.

## Features and targets

`legacy_geometry_v1` ports feature set 2 of `vrd_utils8` and the relevant functions in `vrd_utils5`. The ordered feature names are:

1. Subject one-hot vector in object vocabulary order.
2. Subject box `xmin, ymin, xmax, ymax` after the legacy notional resizing.
3. Object one-hot vector in the same vocabulary order.
4. Object box in the same resized convention.
5. Centroid distance; subject/image and object/image area ratios; sine and cosine of the subject-to-object centroid angle; subject-in-object and object-in-subject inclusion ratios.

The dimension is `2*C + 15`. Set the notional longest image side to 8 and the shorter side proportionally; round both to two decimals. Divide those new dimensions by the original width/height and round the two scale factors to five decimals. Scale each box coordinate and round it to five decimals. Compute centroid distance from the resized boxes and round to three decimals. Compute the remaining six features from original boxes: each area divided by original image area; `dy/distance` and `dx/distance` for the original subject-to-object centroid displacement; intersection area divided by subject area and by object area. Round each to three decimals. If centroids coincide, sine and cosine are both zero. Increasing image y points downward; do not replace this convention with a mathematical upward-y angle. Use Python's existing rounding convention on double-precision intermediate values and cast the finished vector to float32.

Before accepting the port, compare with the original functions on non-degenerate fixtures and check hand-computed edge cases, including coincident centroids, reversed pairs, non-overlapping boxes, and boxes contained within each other. A correction to a numerical convention creates a new feature profile and an explicit difference entry, not an unnoticed refactor. Record if legacy scaling makes a positive-area original box degenerate after rounding; do not conceal it by changing the original box or using resized boxes for area-ratio denominators.

The visual feature manifest records encoder name, named weight revision/digest, TorchVision version, crop rules, transform parameters, normalisation, dtype, source image/box digests, and per-shard row identities. Subject/object/union vectors are cached once; they are not duplicated for every intervention or seed.

Targets have shape `[number_of_pairs, P+1]`, float32. Predicate columns are multi-hot. The auxiliary column is one only on pairs without training predicates. `ordinary_loss_rows` is a Boolean vector of length `number_of_pairs`, true exactly for pairs with at least one training predicate. The selected augmentation condition changes targets and this row mask; it never changes feature rows or object/pair IDs. Semantic masks have P columns only.

Treat target arrays as immutable inputs to loss computation. Any temporary transformation uses a new tensor. A frequency-control weight vector has length P and affects positive predicate terms only; its checksum and source counts are in the run manifest.

## Ontology, augmentation, and compatibility artifacts

### Ontology profile manifest

Record ontology file hashes, transitive import-closure hashes, label mapping digest, reasoning backend/version, ruleset/configuration digest, supported constructs, unsupported constructs, query scope, and operation (`augmentation` or `compatibility`). Imports must be resolved to pinned local files before experiments; a moving network import cannot be part of a reproducible run.

The query scope is `independent_typed_relation`: fresh individuals a and b, supplied classes Cs and Co, and one asserted predicate p(a,b), checked relative to the fixed ontology. It is not consistency checking of the entire predicted image graph. Do not assert a unique-name assumption unless it is explicitly part of the supported profile. Separate query transactions/contexts must prevent one example from contaminating another.

The initial implementation uses RDFLib/OWLRL for per-image augmentation and a thin GraphDB adapter for compatibility preparation, following the original operations. Backend equivalence is not assumed. Inventory the actual ontology constructs; verify supported positive, negative, and unknown cases. Any claimed absence of inconsistency is bounded by the tested supported profile. Do not replace an unsupported construct with a simpler one without changing the profile and report.

### Augmentation artifact

Store the raw training partition digest, augmentation ontology/profile digest, expanded relation records, original and added counts per image/predicate, newly supervised pair counts, excluded self/out-of-vocabulary consequences, timing, and a completed status. Each added relation records `origin=inferred` and its profile digest. A full proof graph for every inferred edge is not required; retain deterministic small witnesses for the explanatory examples used in a report.

Re-running preparation with identical inputs must produce the same deduplicated target relation sets and content digests. Never overwrite raw annotations. If closure exceeds a declared resource limit or fails, mark that artifact failed; do not accept its partial closure. Set resource limits from the preparation pilot and record them.

### Compatibility table

The lookup key is `(subject_class_id, predicate_id, object_class_id)`. Its identity additionally includes the complete ontology/mapping/backend/ruleset/query-scope digests. A plain numeric triple without those identities is not a reusable cache key.

Use one dense or sparse numerical table with documented axes, whichever the measured vocabulary/query universe makes simpler. Dense order is `[C, P, C]`. Suggested int8 storage is `0=compatible`, `1=incompatible`, `-1=unknown`, `-2=unresolved`. An accepted artifact has no unresolved entry within its declared query universe. Unknown entries have an explicit reason such as an unmapped label or unsupported construct. Transport errors are failed preparation events, not accepted unknown or incompatible decisions.

Derive the query universe from the declared vocabulary or supplied object class-pairs, never from held-out relation labels or observed model test errors. All train/validation/test input class-pairs required by planned evaluation must be covered by the prepared universe or declared unknown by the frozen mapping. Runtime lookup outside this universe is an error requiring a new preparation artifact; it must not query a live service during training.

Do not assume that hundreds of thousands of individual HTTP queries are inexpensive. Measure query throughput and estimate preparation cost. Reuse valid content-addressed decisions and deduplicate queries. Any accelerated local rule compilation must demonstrate equivalence for the supported profile before replacing the reference preparation path; do not implement a general reasoner to avoid this check.

For each confirmed incompatibility, preserve a structured backend reason or rule identifier and a compact response digest. Unknown status must never mean that lack of a proof was interpreted as false. Validate that the ontology alone is consistent under the supported checker before querying examples.

The separate C1 artifact has policy `simple_asserted_mask_v1`: a Boolean blocked-cell mask, the same vocabulary/mapping identity, the asserted source ontology digest, the exact extracted domain/range/disjointness facts, and counts of unsupported expressions. It does not label every retained cell compatible. Full-checker conformance is required for the full table; equality with C1 is not assumed. Both filters share the ranking function and neither invokes a service during evaluation.

### GraphDB transport contract

The adapter uses a dedicated preparation context/repository identified in configuration. It must not clear an arbitrary existing knowledge graph. Record backend/ruleset versions and confirm that consistency checking is actually enabled using known consistent and inconsistent fixtures.

Only a verified consistency-check response can become `incompatible`. HTTP 500 alone is insufficient. HTTP 401/403/404/429/502/503, timeouts, malformed responses, or an unexpected 500 produce a typed operational error. Retry only explicitly transient failures: at most three attempts, with a 30-second timeout per attempt and delays of 1 then 2 seconds by default. Honour a longer server retry requirement by stopping with a retryable failure instead of retrying too soon. Before retrying a request that may have modified the context, confirm cleanup or fail. Exhaustion fails the artifact build. Authentication/configuration failures fail immediately. Never persist a failed request as a logical decision.

Resume an interrupted table build from completed decisions with matching hashes, validating context cleanup between queries. Accepted decisions are immutable. Do not log credentials or store them in manifests; store an endpoint identifier and environment-variable names, with secret values excluded.

## Human configuration

This TOML illustrates a training configuration stored at `configs/nesy4vrd.toml`, not an executable configuration before data preparation. Preparation inputs are specified immediately below:

```toml
schema_version = "kgr4vrd.config/1"
protocol_id = "nesy4vrd-corrected-v1"

[paths]
data_root = "../data"
artifact_root = "../artifacts"

[dataset]
name = "nesy4vrd"
manifest = "prepared/nesy4vrd/v1/dataset.json"
split = "prepared/nesy4vrd/v1/splits.json"

[model]
name = "geometry_mlp"
feature_profile = "legacy_geometry_v1"
auxiliary_output = true

[intervention]
condition = "baseline"
augmentation_profile = "S_TEU"
augmentation_manifest = "knowledge/augmentation/S_TEU/manifest.json"
knowledge_manifest = "knowledge/compatibility/manifest.json"
semantic_policy = "gated_type_mask"
semantic_lambda = 1.0
semantic_start_epoch = 3
activation_threshold = 0.5

[training]
seed = 11
learning_rate = 0.0001
weight_decay = 0.001
max_epochs = 100
patience = 25
batch_images = 1
loss_reduction = "image_sum"
device = "cpu"
precision = "float32"
num_workers = 0
selection_metric = "recall_image_mean_at_50"

[evaluation]
ranking = "multi_label"
top_k = [25, 50, 100]
threshold_policy = "none"
matching = "exact_instance"
simple_filter_manifest = "knowledge/simple-filter/manifest.json"
diagnostic_threshold = 0.5
diagnostic_top_k = 999
```

Paths in `paths` resolve relative to the configuration file, with explicit command-line overrides permitted. Manifest and source paths resolve against `data_root`; run/report output paths against `artifact_root`. Prepared manifests use paths relative to their own directory for component files, and image paths relative to the recorded relocatable image root. Planner-produced JSON writes absolute execution roots so moving the generated configuration into `artifacts/plan` cannot reinterpret `../data`. Scientific identities remain location-independent; relocation explicitly rebinds roots and verifies hashes. The running guide must give a consistent example directory layout; no path can depend on the caller's working directory.

The same TOML may include the following explicit preparation fields. Populate real filenames from G0; the specification does not invent internal paths for an archive that has not been acquired. Validate required fields per stage, so data preparation can run before its output manifest exists.

| Configuration fields | Required use |
| --- | --- |
| `preparation.source_version`, `preparation.sources.object_vocabulary`, `.predicate_vocabulary`, `.train_annotations`, `.test_annotations`, `.images` | Pin the acquired resource and raw inputs for `prepare --stage data`. These are local paths under `data_root`, not download commands. |
| `preparation.sources.historical_split` | Optional recovered split; otherwise use the registered split recipe. |
| `preparation.output_directory` | New prepared dataset directory under `data_root`; its component names follow the canonical layout. |
| `preparation.augmentation_ontology`, `.compatibility_ontology`, `.mapping` | Source paths and mapping needed by `prepare --stage reasoning`; resolve pinned ontology imports before this stage. |
| `preparation.closure_timeout_seconds`, `.closure_max_triples` | Positive resource limits measured in the preparation pilot; exceeding either fails the artifact. |
| `graphdb.endpoint`, `.repository`, `.ruleset`, `.expected_version` | Dedicated preparation destination and verified backend identity; include timeout/retry defaults above. |
| `graphdb.username_env`, `.password_env` | Optional names of environment variables containing credentials, never their values. |
| `visual.weights_manifest`, `.features_manifest`, `.crop_batch_size` | Pinned encoder/transform identity, feature-cache location, and positive extraction batch bound for the visual model. |

Only stages requiring an input load it. For example, a baseline trainer does not open augmented targets or C1 masks merely because a shared configuration lists their locations. Preparation produces the declared artifact paths and hashes; later stages reject missing or mismatched outputs.

Allowed conditions are `baseline`, `augment`, `semantic_loss`, `combined`, and `frequency_control`. The conditional replacement `shuffled_augmentation_control` is admitted only by the protocol's coverage trigger and uses its frozen control-target artifact with no semantic loss. `condition` determines whether augmentation/penalties are active; do not create contradictory independent Boolean switches. A baseline's resolved effective lambda is zero even though the reusable intervention default is one. The resolved manifest records both requested defaults and effective values, and hashes the effective training configuration.

For `visual_mlp_v1`, additionally require the encoder weight/transform manifest and visual feature manifest. The fixed head widths are part of the model definition. Do not expose every internal constant as a tuning parameter. Unknown fields, unsupported profile names, conflicting dimensions, and missing required input digests are errors.

## Campaign, run, and checkpoint identity

`campaign.json` is the frozen scientific analysis plan: protocol digest, final/development seeds, explicit named run groups, expected comparisons, metric policies, effective recipes, input digests, and development-selection records. A draft may change before any test use. Freezing writes its immutable version and digest. Each `selections/<group>.json` separately binds the complete expected run list, selected checkpoint digests, and that plan digest before the group's test evaluation. A mutable `status.json` lists successful, failed, and not-started runs and inventory paths; it cannot alter scientific choices or selection files. Campaign completion requires the whole corresponding inventory, not just a successful process exit. This is a handful of files managed by the existing artifact functions, not a workflow-state framework.

A run ID is a digest of protocol/schema version, effective training configuration, training/validation data and split identities, feature/vocabulary identities, training-relevant knowledge/control identities, seed, source-code digest, and tested environment fingerprint. Evaluation-only filtering variants do not create additional training IDs. Keep a human-readable label beside the digest.

For code provenance, record the Git commit and any tracked/untracked implementation changes included in the actual execution. A commit hash with unrecorded dirty source is insufficient. Clean committed source is preferred for the frozen campaign; if a dirty snapshot is used, save its patch/source manifest and digest. Do not include raw datasets, credentials, or unrelated paper files in a source snapshot.

```text
artifacts/runs/<run-id>/
    run.json
    resolved-config.json
    environment.json
    epochs.jsonl
    events.jsonl
    checkpoints/latest.pt
    checkpoints/best.pt
    selected.json
```

`run.json` includes status, start/end UTC timestamps, identity inputs, effective device, parameter count, completed epochs/updates, failure information where applicable, and output checksums. States are `prepared`, `running`, `interrupted`, `completed`, and `failed`. A failed/interrupted run is not included as a successful replicate.

`epochs.jsonl` contains one final record per completed epoch: core/semantic loss, validation counts and unrounded metrics, selected-best indicator, current learning rate, update/empty-image counts, active penalty cells, and separated timing. Incomplete epoch diagnostics go in `events.jsonl`; they are not treated as completed epoch statistics.

Checkpoints contain model/optimiser state, completed epoch, best epoch/metric, early-stopping counter, model/config/input digests, Python/NumPy/framework RNG states, data-order generator state, and persistent semantic activation masks keyed by stable image/pair identities. Store supported accelerator RNG state when relevant. Save at epoch boundaries; resuming after interruption replays only the unfinished epoch from the latest completed boundary.

`selected.json` includes the chosen checkpoint path/digest, selection metric/value, selected epoch, validation split/target/policy digests, and the earliest-tie rule. It contains no test score. The checkpoint's vocabulary/model/feature identities must match before inference.

Writes use a temporary file followed by an atomic rename in the same directory. Never overwrite a completed accepted run silently. A deliberate repeat is a separately recorded attempt; the reporting plan identifies which attempt represents a seed before comparing outcomes. Do not choose the better attempt retrospectively. Resume verifies the inputs and environment, restores all state, and refuses incompatible changes.

## Prediction and evaluation artifacts

Save raw logits once for the selected checkpoint, before any thresholding, top-K truncation, or ontology filtering. Each score shard contains `image_ids`, `pair_offsets`, `pair_indices`, and `logits`; indices align with canonical objects and predicate order. Numeric arrays use NPZ with `allow_pickle=False`; UTF-8 image IDs have an explicit representation. Use bounded shards so the complete dataset need not fit in memory.

```text
artifacts/predictions/<prediction-id>/
    manifest.json
    index.jsonl
    scores-00000.npz
artifacts/evaluations/<evaluation-id>/
    evaluation.json
    ranked_predictions.jsonl
    per_image.csv
    per_predicate.csv
    metrics.json
```

The prediction manifest binds the checkpoint, dataset input partition, pair ordering, feature/vocabulary hashes, logits dtype, expected image IDs, shard checksums, and completion status. `index.jsonl` locates each image's rows. An image with no pairs has an explicit zero-length range. Do not substitute a missing image with an empty one.

The evaluation ID additionally binds raw target partition, ranking/threshold/matching policy, filter mode, knowledge table, and evaluator code version. `ranked_predictions.jsonl` records each emitted triple's rank, score, and decision status for the named view. Diagnostic uncapped results may be stored separately, but their counts must be regenerable from raw logits. The authoritative prediction artifact is the complete raw score manifest, not a truncated ranked file.

Filter modes are `none`, `ontology`, and `simple_asserted`. The last is registered only for C1 baseline evaluations. Bind the filtering artifact and the full diagnostic checker's artifact separately; their purposes and hashes differ even when their decisions happen to agree.

`per_image.csv` stores image ID, ground-truth count, eligible pair count, emitted/compatible/incompatible/unknown counts, and hit/AP sufficient statistics by K. `per_predicate.csv` stores fixed vocabulary IDs and ground-truth/hit counts by K. `metrics.json` stores metric keys, unrounded fractions, numerator/denominator where applicable, exclusions, and null reasons. Every table row has run and evaluation IDs.

`evaluation.json` records the knowledge scope, candidate/filter policy, target hash, selected checkpoint, timestamps, expected/actual image counts, and completed status. Evaluation must reject NaN logits, duplicate image records, vocabulary mismatches, unresolved required decisions, incomplete shards, and non-frozen test selection.

## Reports, storage, and relocation

The report contains `summary.csv`, `paired_effects.csv`, `figure_data/`, exported figures, and a short result/limitation narrative. Each summary row has dataset, model, condition, seed or aggregate statistic, filter/view, metric, value, support, and input evaluation IDs. Missing/failed cells are visible. Generated values must never be copied manually into source plotting scripts.

Keep raw scores for accepted final runs so ranking/filter changes and reporting corrections can be inspected without retraining. Keep best/latest checkpoints, manifests, logs, and prepared-input identities. Retaining every epoch checkpoint is optional and must have a storage estimate; the protocol does not require 35 test-evaluated checkpoints.

Ignore local `data/`, `artifacts/`, caches, credentials, downloaded weights, and large generated arrays in Git. Track small synthetic fixtures, configurations, source, specifications, and lock files. Dataset licenses determine whether any real-data fixture can be redistributed; synthetic fixtures are the default.

Relocating a prepared dataset/artifact tree must not change scientific identity. A verifier checks component hashes after relocation. A report must be regenerable from the saved campaign and evaluation artifacts without GPU access, original stdout logs, or GraphDB. Re-evaluation from raw scores additionally requires the declared raw targets and knowledge artifact.
