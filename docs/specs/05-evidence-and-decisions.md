# Evidence and decisions

## Evidence boundary

The repository audit used tracked code at commit `08bb8e5f79e5c0737b2c45334ef5745e17c06006` and the manuscript/review files present in `paper/`. The paper directory was untracked at inspection. Preserve it independently of Git-based migration or cleanup.

The user does not currently have the original experiment artifacts. They may be obtainable from the original experimenter. No numerical result from the submitted paper has been independently reproduced in this assessment.

## Findings that the implementation must address

Line numbers below identify the inspected version. Function names and the commit remain the stable reference if files later move.

| ID | Evidence | Consequence | Verification |
| --- | --- | --- | --- |
| E01 | `paper/JAIR_paper/main.tex:613` describes choosing among 35 checkpoints using test recall; `predicatePrediction/vrd_consolidate_job_results_3.py:328` marks maximum test scores | Replace selection with a validation-only rule; historical tables are exploratory | T01, T02 |
| E02 | `predicatePrediction/vrd_utils17.py:440` excludes already cached types from the loop that updates each image's loss mask | A cached incompatible type can escape penalties in a later image | T03, T04 |
| E03 | `predicatePrediction/vrd_utils13.py:190` passes the full annotation dictionary to `vrd_utils12.evaluate_performance`; the evaluator counts images without predictions as zero | Validation scores can include non-validation images in their denominator | T05 |
| E04 | `predicatePrediction/vrd_utils15.py:630` treats HTTP errors with response bodies as failures; `vrd_utils17.py:327` classifies failures as incompatible | A service error can be cached as a logical judgment | T06, T07 |
| E05 | `predicatePrediction/vrd_ppnn_models.py:44` defines the small geometry/class-label MLP; manuscript `main.tex:593` explains the moderately capable baseline choice | A visual predictor and simpler controls are needed to establish scope | W3/W4 |
| E06 | Manuscript `main.tex:934` reports four instances per scenario, but seeds, raw predictions, and training histories are absent locally | Repetitions were reported; their provenance and uncertainty are not recoverable here | T18, G3 |
| E07 | Manuscript `main.tex:862` uses 25 images, exposes hit labels during review, and groups borderline predictions with correct ones | Historical adjusted precision is exploratory; a new audit is deferred | Reporting scope, T16 |
| E08 | Manuscript `main.tex:1255` reports baseline invalidity 0.002904 and augmentation invalidity 0.000214, with different prediction volumes | Report counts, absolute rates, and coverage together | T15 |
| E09 | Active CSV cells 460, 2470, 3001, and 5500 use feature set 2, output mode 2, and no-predicate target policy 1 | Preserve or explicitly change the auxiliary output and masked unannotated-pair behaviour | T08, T09 |
| E10 | The code monitors global validation recall in `vrd_ppnn_training_7.py:1497`; manuscript `main.tex:609` says image-mean recall | Freeze one explicit metric; do not describe the old implementation as matching the prose | T01, T12 |
| E11 | `vrd_utils8.extend_image_and_object_ord_pair_features` uses maximum image dimension 8 and several rounding stages; feature columns depend on insertion order | Feature names, order, scale, and precision require a versioned contract | T08 |
| E12 | Manuscript `main.tex:259` says the experiments use OWL 2 RL; README describes general SROIQ soundness/completeness | Limit guarantees to the supported operations, ontology profile, and tested backend | T07, G5 |

The small diagnostics reproduced E02 with real CPU tensors and a mocked reasoner, E03 with exact extracted evaluation functions and two synthetic images, and E04 with a synthetic HTTP 503 response. They establish code behaviour, not the historical frequency or numerical impact of those defects. Recreate them as maintainable tests in the new package; temporary audit files are not a dependency.

## Historical configuration mapping

| Historical cell/treatment | Intended new condition | Caveat |
| --- | --- | --- |
| Cell 460, baseline | `baseline` with `legacy_geometry_v1` | Output includes an auxiliary no-relation column in the active grid; verify prediction policy from any recovered manifest. |
| Cell 2470, t09 | `augment` with the S/T/E/U profile | The CSV refers to an ontology v1_1p variant; recover and hash the actual axioms rather than assuming that a filename defines the profile. |
| Cell 3001, t21 | `semantic_loss` | The CSV refers to v1_2_disjoint; preserve intended confidence-gated persistent activation while fixing cache/error handling. |
| t22 | `semantic_loss` evaluated with filtering | Reuse the corresponding trained checkpoint. |
| Cell 5500, t30 | `combined` | Both ontology artifacts and both intervention definitions must be recorded. |

The main campaign is a corrected study, not a bit-for-bit reproduction of the submitted tables. No result may be labelled a reproduction solely because its condition has a matching treatment name.

## Proposed decisions

| Decision | Default | Rationale and limit |
| --- | --- | --- |
| Task | Predicate classification with supplied objects | Matches the manuscript's evaluated setting and permits a small shared implementation. |
| Initial data | NeSy4VRD package version 1.1, subject to ontology inventory | Public resource; exact internal annotation/ontology variants must be verified locally. |
| Visual baseline | Frozen TorchVision ResNet-50 `IMAGENET1K_V2` features with a compact relation head | Concrete, reusable, and inexpensive relative to end-to-end training; not automatically a strong or state-of-the-art SGG model. |
| Checkpoint criterion | Unfiltered image-mean recall@50 on fixed raw validation annotations | Directly specified and shared across conditions; report global and predicate-mean recall separately. |
| Ordinary targets | Multi-label BCE on annotated pairs, excluding wholly unannotated pairs from ordinary BCE | Retains the active legacy target policy; missing relation labels within annotated pairs remain an acknowledged negative-label assumption. |
| Semantic loss | Confidence-gated persistent mask after a two-epoch warm-up, lambda 1 | Preserves intended legacy semantics while separating reusable classifications from per-cell activation. |
| Reasoning deployment | Prepare immutable decisions/augmentations before training | Keeps network failures and GraphDB state out of the training loop. |
| Runtime | Python package, PyTorch, plain files, standard command line | Reusability without an orchestration framework. |

## External inputs and bounded resolution

| ID | Missing input | Required resolution and gate |
| --- | --- | --- |
| D01 | Original logs, checkpoints, predictions, judgments, and exact code revision | Inventory if recovered at G0. Their absence does not block a fresh corrected study. Never fabricate historical seeds or reconstruct judgments from aggregate means. |
| D02 | Exact ontology variants and label-to-IRI mappings | Inspect the acquired NeSy4VRD package and author resources, hash files, compare axiom sets with active CSV comments, and record differences before G1 real-data conformance. If historical variants cannot be recovered, name the new profile explicitly. |
| D03 | Exact dataset/preprocessing file hashes | Obtain from the selected source, verify publisher checksums where provided, then calculate SHA-256 and record the source and retrieval date before training. |
| D04 | Supported dependency versions | Resolve one compatible Python/PyTorch/TorchVision/RDFLib/OWLRL stack, test it, and commit a lock at G2. Do not advertise an untested platform range. |
| D05 | Execution hardware and storage allowance | Record a one-epoch pilot, feature-cache size, reasoning throughput, and projected matrix cost before each full campaign. No remote machine, GPU availability, or paid service is assumed. |
| D06 | Visual baseline adequacy | Apply the W3 validation gate. If it fails, record the result and choose one established reference predictor through the bounded fallback procedure. Do not repeatedly try architectures against test scores. |
| D07 | Manual annotation effort | Deferred, not a gate in this phase. Do not promote the old adjusted precision into a validated headline claim. |

Operational defaults can be resolved by the implementing colleague from these rules. If an external input prevents a requirement from being met, identify the exact dependency and complete independent work. Do not mark the whole project complete or quietly substitute a weaker acceptance criterion.

## Source references

- [Submitted manuscript](../../paper/JAIR_paper/main.tex) and [JAIR review letter](../../paper/Reponse%20letter%20-%20JAIR%202026.pdf) are local research evidence. The links in this list are relative to this specification directory and are checked during documentation validation.
- [NeSy4VRD version 1.1](https://zenodo.org/records/17076303) identifies the public dataset package and companion ontology. The record reports an archive checksum; local file hashes remain to be measured.
- [Neural Motifs author implementation](https://github.com/rowanz/neural-motifs) provides a published reference. Its original software stack is old; it is not a dependency to vendor into the new package.
- [HIERCOM paper](https://arxiv.org/html/2311.12889v3) and [author repository](https://github.com/bowen-upenn/scene_graph_commonsense) establish closely related commonsense filtering/distillation work.
- [TorchVision ResNet-50 documentation](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html) defines the named weights and transforms. Pin the implementation and weight digest used; do not depend on a moving `DEFAULT` alias.
- [OWL 2 profiles](https://www.w3.org/TR/owl2-profiles/) defines the standards context. Backend conformance and the actual ontology's supported constructs must still be established.

External sources were inspected on 12 October 2026. They motivate choices and identify resources; they do not validate this repository's results.
