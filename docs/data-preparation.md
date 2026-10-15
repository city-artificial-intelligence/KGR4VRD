# Data, ontology and encoder preparation

The implementation accepts NeSy4VRD's JSON mapping from image filename to relation
records. A record contains `subject` and `object` with `category` and `bbox`, and an
integer `predicate`. Public vocabulary files are ordered JSON arrays of labels or
records with dense IDs. Raw boxes `[ymin,ymax,xmin,xmax]` become continuous
`[xmin,ymin,xmax,ymax]` without adding one pixel. Native object IDs are preserved;
without them only exact class/box duplicates are merged. Malformed boxes fail.

Acquire [NeSy4VRD version 1.1](https://zenodo.org/records/17076303) explicitly.
The record advertises `NeSy4VRD.zip` with MD5
`4c83779814209077e25a7a707f441eed`. Verify the archive checksum, record retrieval
date and source terms, unpack locally, and compute SHA-256. The package never
downloads data or weights implicitly. Repository licensing does not establish
rights to redistribute source images. Historical ontology filenames are not
treated as recovered artifacts.

Add this table to a copy of `configs/nesy4vrd.toml`, replacing the illustrative
paths with actual inventory entries:

```toml
[preparation]
source_version = "1.1"
output_directory = "prepared/nesy4vrd/v1"
augmentation_ontology = "raw/asserted-augmentation.ttl"
compatibility_ontology = "raw/asserted-checker.ttl"
mapping = "raw/mapping.json"
closure_timeout_seconds = 30
closure_max_triples = 1000000

[preparation.license_provenance]
reference = "REPLACE_WITH_SOURCE_TERMS_REFERENCE"
redistribution_status = "REPLACE_WITH_VERIFIED_STATUS"

[preparation.sources]
object_vocabulary = "raw/objects.json"
predicate_vocabulary = "raw/predicates.json"
train_annotations = "raw/train.json"
test_annotations = "raw/test.json"
images = "raw/images"
# historical_split = "raw/recovered-splits.json"

[graphdb]
endpoint = "http://localhost:7200"
repository = "kgr4vrd-preparation"
ruleset = "owl2-rl"
expected_version = "REPLACE_WITH_MEASURED_VERSION"
# username_env = "GRAPHDB_USERNAME"
# password_env = "GRAPHDB_PASSWORD"
```

The raw filenames above are examples, not claims about the archive's contents.
Measure closure limits and query throughput in the preparation pilot. Imports
must be flattened into a pinned local ontology before preparation; moving remote
imports fail. Keep separate asserted augmentation/checker files and record their
identity even if initially identical. Mapping records require explicit label kind,
ID, IRI, relation, rationale and provenance. Broader mappings are not treated as
equivalence; unsupported/unmapped cases must stay unknown.

Conversion separates objects from relation labels and exports exact split lists.
It preserves official test membership, excludes exact copies of test images from
the training pool, and keeps remaining exact-image groups together. Without a
recovered valid split, it uses the registered SHA-256/1729 whole-group 80/20 rule.
Perceptual duplicates are not automatically adjudicated: the conversion report
states this limitation. The raw format supplies no separate native grouping field.
The expected historical 109 classes/71 predicates are checked as a discrepancy
report, not forced onto another vocabulary.

Only mechanical conversion reads raw test relations. Augmentation, control counts,
features, training and compatibility preparation subsequently load their allowed
partitions explicitly. Compatibility queries cover supplied class-pairs from all
partitions, without loading held-out relation targets. Raw training positives that
contradict the checker fail preparation and require a new adjudicated input version.

The GraphDB adapter requires an exclusive, dedicated repository with consistency
checking enabled. It verifies the server/ruleset, consistent and inconsistent
fixtures, and background ontology before typed queries. It never clears arbitrary
graphs. Only verified consistency responses become incompatibility; service errors
fail the artifact, with bounded retries. Successful decisions are resumable only
under matching identities. A fresh stratified conformance sample is required for
real acceptance. Offline test fakes do not establish real GraphDB conformance.

Frozen visual features use TorchVision ResNet-50 `IMAGENET1K_V2` only, with the
pinned transform. Supply a local state dictionary and its measured digest in a
weights manifest; do not use the moving `DEFAULT` alias. Extraction uses stored RGB
orientation, floor/ceil clipped boxes, subject/object/union crop order and L2
normalisation. It caches deterministic rows once across interventions/seeds. The
real encoder integration test is opt-in and remains unverified until those weights
and images are supplied. Fake-encoder tests verify crop and row semantics only.

The local weights manifest has the following shape. Replace the digest placeholder
with the SHA-256 of the acquired state dictionary; its path is relative to this
manifest. Preparation checks the complete weight file before loading it.

```json
{
  "schema_version": "kgr4vrd.encoder/1",
  "encoder": "resnet50",
  "weights": "IMAGENET1K_V2",
  "torchvision_version": "0.17.2",
  "weights_file": {"path": "resnet50-11ad3fa6.pth", "sha256": "MEASURE_LOCALLY"},
  "transform": {
    "resize_size": 232, "crop_size": 224, "interpolation": "bilinear",
    "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "antialias": true
  }
}
```

Configure the cache paths relative to `paths.data_root`:

```toml
[visual]
weights_manifest = "weights/resnet50-v2.json"
features_manifest = "features/resnet50-v2/manifest.json"
crop_batch_size = 32
```

`prepare --stage features` verifies every source image hash and writes one NPZ
shard per image. Shards store unique crop vectors and pair-to-crop indices, so a
repeated object or union is encoded once. An identical completed cache is verified
and reused. Interrupted extraction leaves unpublished shards; choose a fresh cache
directory after investigating the failure. No incomplete cache is admitted to training.

Keep `data/`, downloaded weights, caches and generated `artifacts/` outside Git.
An accepted preparation is immutable; rebuild changed inputs into a new directory.
The final run report must include source, ontology, mapping and weight identities,
preparation timings, feature size, one-epoch time/memory and campaign projections.
