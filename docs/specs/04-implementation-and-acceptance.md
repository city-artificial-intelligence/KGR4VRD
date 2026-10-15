# Implementation and acceptance

## Working rules

Read this specification set before implementation. Deliver small, working slices. Do not generate an entire architecture as empty files or begin by rewriting every legacy script. Establish the shared data and metric contracts first, then add only code needed by the five workstreams.

Preserve the submitted manuscript, reviews, existing untracked files, and any recovered original artifacts. Record a starting Git commit/status. Implementation changes should be reviewable separately from experiment outputs and later paper edits. Do not mix historical results with newly generated results in the same run directory.

The requested implementation is complete only when its code and required checks pass. The full experimental work is complete only when the registered runs and evidence package are available. Neither state implies a positive finding, publication readiness, or journal acceptance.

## Delivery slices

### Slice 0 Inventory and scientific differences

Inventory the public dataset package, available ontology variants, current CSV cells, and any artifacts recovered from the original experimenter. Write a short historical-differences record covering preserved behaviour, confirmed defects, and prospective choices. Include the auxiliary output, masked unannotated pairs, feature rounding/order, validation metric discrepancy, test-based selection, and confidence-gated semantic activation.

Requesting original artifacts is useful but not a prerequisite for a fresh corrected study. The useful package is the exact code revision/environment, split lists, ontology files/mappings, all retained checkpoints, epoch logs, predictions, configurations, and original manual judgments. Without those, do not claim to recreate the historical run or estimate the magnitude of its bugs.

Resolve source/ontology identities before a real-data campaign. The NeSy4VRD source is the only required dataset. The artifact inventory and source mapping complete G0; they are not a mandate to acquire additional datasets.

### Slice 1 Scientific kernels and defect tests

Implement the smallest canonical record/pair/target types, the selected loss, lookup/activation semantics, checkpoint selector, and evaluator. Start with hand-checkable fixtures. Recreate the three diagnostic failures from the audit and establish the intended corrected outcomes. Implement feature ordering and source-to-canonical conversion before model-scale work.

Add per-image RDFLib/OWLRL augmentation and the thin GraphDB preparation adapter. Keep their conformance fixtures separate from normal offline unit tests. An available GraphDB service is needed for real backend acceptance, not for importing or smoke-testing the package. G1 requires the scientific unit checks and a recorded status for real backend conformance; absence of that service does not justify claiming it passed.

### Slice 2 End-to-end package and provenance

Implement one geometry training run, validation-only selection, saved raw logits, evaluation, and a report. Then exercise all four conditions on the tiny synthetic fixture. Add resume, run identity, atomic persistence, and completion validation through this real execution path rather than a separate generic run-management framework.

Provide a clean-install command, CLI help, and a smoke command in the main README. The smoke example uses a tiny fixed vocabulary/ontology and embedded synthetic decision table, four short epochs to exercise the epoch-3 semantic gate, and no network or downloaded model weights. It is clearly labelled synthetic integration evidence.

G2 requires the clean-install smoke path, resume comparison, saved-score re-evaluation, and report regeneration to pass. At this point the reusable package exists; the full experiment results do not.

### Slice 3 Corrected geometry experiment

Prepare canonical NeSy4VRD data, fixed split lists, the S/T/E/U augmentation artifact, and the compatibility artifact. Check ontology capabilities, conflicts in training positives, expected vocabulary, candidate coverage, and real service behaviour. Record time/storage estimates from preparation and one training epoch.

Run the bounded development checks and register the four conditions with five final seeds. Before any geometry test scoring, complete the visual model/qualification work and control-preparation decisions described in Slice 4, using training and validation only; freeze these shared scientific choices in the analysis plan. Final training may then proceed in groups. Retain lambda 1 and the corrected gated semantic policy. Freeze the geometry group's selected checkpoints before scoring its planned test views. Compare the shared expected seed list against completed artifacts; display failed/missing runs.

G3 requires complete geometry results for the four conditions before/after filtering, with validation selection and provenance verified. If a run fails, repair the implementation or retry the identical configuration with a recorded reason; do not replace it with a different seed because its score is poor.

### Slice 4 Visual predictor and simple controls

Implement frozen visual feature extraction and the compact visual relation head through existing interfaces. Do not create a second trainer. Perform the W3 validation adequacy check and follow its bounded fallback only if needed; these development decisions precede the first campaign test evaluation, as specified above. Repeat the four-condition final comparison with the qualified predictor.

Build the simple asserted-constraint mask from the fixed ontology, and evaluate it on baseline scores alongside the existing unfiltered/full-filtered views. It requires no training. Add the frequency-matched positive-supervision condition, validate its realised counts/weights, and run five final seeds per predictor. Choose the single fallback augmentation control only when its documented matching-coverage trigger fires before the first campaign test evaluation. Freeze each remaining group's selected checkpoints before its test evaluation.

G4 requires both predictor comparisons, the required control results, and disclosure of the visual baseline's qualification evidence. A failure to obtain a genuinely stronger predictor is reported explicitly; more image features alone cannot satisfy that claim.

### Slice 5 Consolidation and retirement of duplicate paths

Generate the result table, effect estimates, uncertainty, diagnostic counts, and figures from saved artifacts. The report must state single-dataset scope, historical test exposure, ontology/profile scope, incomplete annotations, fixed-lambda scope, and control limitations alongside relevant conclusions.

Finish the running guide and data-preparation guide using commands verified from a fresh environment. Remove duplicate active entry points and unused imported helpers once migration checks pass. Keep the code-growth report and explain exceptions to the architecture thresholds. Ensure readers can locate the only supported training/evaluation path from the main README.

G5 requires the evidence package, reproduction instructions, and completed verification table. A later manuscript should use these results and their limitations; rewriting the submitted paper is not a hidden deliverable of this implementation.

## Required verification

The tests below are behavioural acceptance criteria. Several can share one compact fixture or test file. Do not create a separate testing framework, a file per requirement, or a suite of trivial implementation-mirroring tests.

| ID | Check | Acceptance evidence |
| --- | --- | --- |
| T01 | Validation checkpoint selection | A synthetic history with a validation tie selects the earliest best validation epoch even when test scores favour another epoch. All K/filter views reference that same checkpoint. Non-finite validation scores fail. |
| T02 | Partition and target isolation | Training cannot access a test-target sentinel. All conditions validate against the same raw targets. Split overlaps, missing IDs, and accidental augmented validation targets fail. |
| T03 | Cached invalid type across images | Two images with the same incompatible type and above-threshold score both receive active penalties, including when the decision table is already populated. No extra classification call is necessary. |
| T04 | Persistent semantic activation | Epochs 1–2 have no active penalties; a score above 0.5 at epoch 3 activates an incompatible cell; a later lower score leaves it active. Exactly 0.5 does not activate. A compatible cell never activates. |
| T05 | Validation denominator regression | A perfect prediction for the only validation image scores 1.0. Adding a training image to a wider source dictionary cannot dilute the score; the loader/evaluator must select exactly the expected validation IDs or reject the extra target input. |
| T06 | Reasoner operational errors | HTTP 503, timeout, authentication failure, malformed body, and an unrelated HTTP 500 never become incompatible decisions. Only eligible transient failures retry, with a fixed maximum; failed builds remain incomplete. |
| T07 | Supported reasoning semantics | Hand-built subclass/domain/range/disjointness fixtures distinguish compatibility, incompatibility, and unknown; an inconsistent background ontology fails preparation. A real configured backend passes the same cases before campaign acceptance. |
| T08 | Features and ordering | The geometry feature vector matches the registered legacy profile on fixed examples, has dimension `2*C+15`, preserves orientation and rounding, and does not depend on dictionary insertion order. Source differences are explicit. |
| T09 | Multi-label targets and auxiliary column | Multiple predicates survive conversion. Unannotated pairs contribute no ordinary BCE. The auxiliary column never becomes an ontology query or prediction. Augmentation changes targets without changing pair/feature identity. |
| T10 | Stable losses and gradients | At large positive/negative logits, all defined losses/gradients remain finite. A semantic-only gradient step lowers an incompatible logit. Lambda zero removes the semantic contribution. Masks have the correct device/dtype, without modifying input targets. |
| T11 | Ranking and filtering | Logit ordering, deterministic ties, multiple predicates per pair, auxiliary exclusion, duplicate removal, filtering/backfill, and threshold boundaries match hand-calculated lists. Unknowns remain. K=999 remains a cap. A direct-disjointness fixture is blocked by C1; a case requiring subclass inference demonstrates its deliberate difference from the full checker. |
| T12 | Distinct recall definitions | With one image having 1/1 hits and another 0/9, image-mean recall is 0.5 and global recall is 0.1. Predicate-mean recall is checked from a separate imbalanced fixture. Rounding happens only in presentation. |
| T13 | Matching, AP, and empty cases | Duplicate predictions cannot increase hits; exact object IDs and predicate IDs determine primary matches. Truncated AP follows the declared denominator. Empty predictions score zero recall when targets exist; missing records fail; zero-support metrics are null with reasons. |
| T14 | Resume equivalence | A deterministic CPU run resumed at an epoch boundary matches uninterrupted parameters/predictions within the registered numerical tolerance, selected epoch, optimiser progression, and persistent activation state. Compare actual outputs, not just checkpoint existence. |
| T15 | Absolute consistency reporting | For 8 predictions with 4 compatible, 2 incompatible, and 2 unknown, all-prediction incompatibility is 0.25, decided-only incompatibility is 1/3, and coverage is 0.75. Zero denominators are not reported as perfect consistency. |
| T16 | Evidence-bounded reports | A report review verifies that annotation AP is not called audited precision, filtered zero violations are not called learned correctness, and no claim exceeds the supported profile or single-dataset evidence. This is a scientific review, not a string-matching unit test. |
| T17 | Artifact integrity and relocation | Mismatched vocabulary/profile/feature hashes and corrupted or partial shards are rejected. Relocation preserves scientific identity. A completed run is not overwritten silently. |
| T18 | Matrix and seed completeness | The default plan has 40 core final runs, 10 frequency-control final runs, up to 8 development runs, and 110 primary evaluation variants before secondary metric views. No inferred Cartesian product expands these groups. Proven equivalent conditions may reuse a job while retaining their named rows. Missing/failed seeds and duplicate attempts are visible; averaging only favourable completed cells fails report validation. |
| T19 | Clean-install and offline smoke | After locked dependency installation, a fresh environment runs the synthetic four-condition pipeline from a non-repository working directory, with no GraphDB/network/weights requirement and no import-time writes. |
| T20 | Visual feature identity | Crop bounds/union construction and vector ordering are checked with a deterministic fake encoder; an acquired real encoder produces a repeatable cache with pinned weights/transforms on a declared platform. Different transform/weight digests cannot reuse the cache. |
| T21 | Augmentation isolation and profile | Symmetry, inverse, transitive, equivalent-property, and subproperty examples establish expected closure and profile removal. Raw labels are preserved, duplicates removed, held-out targets untouched, and separate image graphs cannot exchange facts. |
| T22 | Compatibility table conformance | A stratified sample of at least 200 available typed queries, spanning predicates and compatible/incompatible outcomes where present, agrees exactly with fresh reference queries. Record sample membership and scope; do not claim general OWL completeness from a sample. |
| T23 | Frequency-control contract | Realised positive-loss weights match raw-plus-added predicate counts where raw support exists; negatives, auxiliary output, and raw pair mask are unchanged. Unmatched mass is explicit and triggers the declared fallback when necessary. If needed, the shuffle deterministically preserves predicate counts and added-label counts per pair, avoids duplicates/raw collisions, and reports changed-cell coverage and infeasibility. |
| T24 | Re-evaluation and reporting | Recompute metrics from saved logits without training/GraphDB, and rebuild tables/figures from the saved campaign. Outputs match within numerical tolerance and retain complete provenance and missing-run status. Check bootstrap pairing on a tiny fixed example; identical compared predictions give zero effects and intervals. |

Suggested CPU numerical tolerances are `atol=1e-6, rtol=1e-5` for floating-point calculations and exact equality for sets, IDs, counts, masks, selected epochs, and manifest identities. Record any platform-specific tolerance with a reason; do not widen it simply to make a failed check pass. Tests with independently specified expected values take precedence over legacy numerical agreement when the legacy result is demonstrably wrong.

## Reasoning fixture design

Keep the synthetic ontology small enough to inspect directly. A suitable family contains `Person` and `Horse` below `Animal`, `Car` below `Vehicle`, and `Animal` disjoint with `Vehicle`. Give a relation an `Animal` range. `Person relation Horse` is compatible; `Person relation Car` is incompatible after type inference; an unmapped predicate is unknown. Add isolated properties for each augmentation family so disabling an axiom has a predictable consequence without another axiom accidentally deriving the same edge.

Include a domain/range mismatch that is merely unproved, without a disjointness contradiction, to ensure the implementation does not confuse open-world absence with incompatibility. Include two images with fresh individuals to detect residual backend state. Real GraphDB conformance must verify the exact ruleset and error signature used, not just a fake adapter's return values.

## Continuous integration and reproducibility checks

One simple CI workflow should install the locked CPU environment, run formatting/lint/type checks on the active package, run the scientific unit tests, build/install the wheel, and execute the offline smoke example. The real-reasoner and real-encoder checks are marked integration tests with explicit prerequisites; skipped integration checks must appear as not run, not passed.

Do not run full training campaigns in ordinary CI, attach large datasets to fixtures, require credentials for the base tests, or create many platform matrices without a demonstrated need. CUDA/MPS checks are executed on available declared hardware and recorded separately.

At each slice run the checks relevant to its new behaviour. Do not repeatedly rerun expensive checks without a new change or unresolved failure. Before a campaign, all checks affecting its semantics must pass. A fixture result is never reported as a NeSy4VRD experimental outcome.

## Pilot and resource record

The pilot record includes dataset image/pair counts, expected score volume, preparation/query throughput, augmentation and feature-cache sizes, one complete training/validation epoch, peak observed memory, and estimated total run time/storage for the planned matrix. Use measured ranges and disclose that stopping epoch and image complexity affect cost.

Default execution is serial on one declared device. Do not assume access to remote GPUs or create distributed-training infrastructure. Parallel independent jobs may be launched through an existing environment only when that execution is authorised and resource limits are explicit; they share immutable prepared artifacts and have separate output directories.

If projected cost is too large, first remove duplicate work, cache frozen features, and avoid repeated inference. Changing scientific scope, seed count, or the registered comparison is a visible protocol change, not a silent optimisation. Do not substitute a tiny sample for the full benchmark without labelling it a pilot.

## Completion report

For each gate, report completed, incomplete, or failed, with evidence paths and the exact reason. The final implementation report includes:

- Public commands and one verified minimal example.
- Scientific behaviour preserved, defects corrected, and deliberate changes.
- Verification results, integration checks not run, and supported platforms.
- Source-line/module/dependency counts and any justified threshold exceptions.
- Experiment matrix status, selected checkpoints, and locations of tables/figures.
- Evidence-supported findings, negative results, and remaining claim limitations.

Do not report G3/G4 as complete from a passing smoke test, or describe G5 as a successful KBS submission. The conclusion may be that the method offers little beyond simple filtering or frequency weighting. That is a legitimate result of this protocol.

## Handoff to the implementing colleague

Implement the five workstreams in the order above, starting with G0–G2 and the scientific defect tests. Use the smallest design that satisfies the contracts. Keep one dataset adapter, two relation heads unless the documented adequacy fallback is needed, one trainer, one selector, and one evaluator. Preserve the corrected confidence-gated loss semantics. Avoid additional datasets, deferred mechanism studies, generic extension frameworks, and speculative configuration options. Run authorised verification, record exact limitations, and stop only at the acceptance boundary supported by actual evidence.
