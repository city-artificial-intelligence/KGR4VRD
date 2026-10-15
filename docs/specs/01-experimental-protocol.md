# Experimental protocol

## Research questions and claim boundaries

| ID | Question | Required evidence |
| --- | --- | --- |
| Q1 | Do inferred training annotations improve relation prediction under corrected evaluation? | Baseline versus augmentation, with identical inputs and selection rules. |
| Q2 | Does semantic regularisation change unfiltered predictions beyond what inference filtering achieves? | Baseline, loss-only, and combined models, each evaluated before and after the same filter. |
| Q3 | Do the effects persist with a stronger predictor using visual features? | W3 comparisons, with the same task definition and data. |
| Q4 | Do learned interventions add value beyond simple filtering and changed label-frequency weighting? | W4 mask/filter and frequency-matched controls. |

The paper may establish an empirical contribution even when a condition is harmful or inconclusive. The experiments do not presume that more axioms, stronger penalties, or combined interventions must help.

## Shared design

P01. The task is predicate classification over all ordered pairs of distinct supplied object instances. Object classes and boxes are inputs for every condition. Preserve all annotated predicates for a pair. Do not convert multi-label annotations into one randomly chosen predicate.

P02. Prepare one canonical dataset and a fixed train/validation/test partition before generating augmented targets. All training conditions use the same object inventory and pair order. Augmentation changes relation targets, not objects, boxes, train membership, or evaluation annotations.

P03. Validation and test scoring always use raw, unaugmented relation annotations. Do not validate an augmented condition against augmented labels while validating the baseline against raw labels.

P04. Final training seeds are `[11, 23, 37, 53, 71]`. Development seeds are `[101, 103]`; split seed is `1729`; control randomisation seed is `2027`; uncertainty seed is `2028`. They are proposed fixed defaults, not recovered historical seeds. Use separate random generators for model initialisation, data ordering, and controls. Within each predictor, matched conditions with the same seed start from identical parameters and share image order; target preparation must not consume either generator.

P05. Results are identified by dataset, split digest, model, intervention, rule profile, semantic policy, seed, checkpoint digest, and evaluation policy. A seed alone does not establish pairing if these other conditions differ.

## W1 Correctness and data separation

Use the exact recovered historical training/validation split if available and disjoint. If it is unavailable, retain the official NeSy4VRD test partition and make a new deterministic 80/20 partition of the official training images. Order image IDs by SHA-256 of `1729:<image_id>`, assign the first floor(0.8 N) to training, and the remainder to validation. Export the actual lists. State that the historical split was not recovered.

Keep native image groups together when metadata identifies duplicates or multiple versions of the same image. Audit exact image hashes across all partitions. Preserve official test membership; exclude confirmed copies of test images from the training/validation pool and record the exclusions. For remaining duplicate groups, use the lexicographically smallest member ID as the group key, sort groups by the same seeded hash, and choose the longest whole-group prefix containing at most floor(0.8 N) images for training; the rest form validation. Record the realised ratio. With singleton groups this is the image rule above. An empty training or validation partition fails preparation. A recovered split crossing confirmed groups is not accepted unchanged. Use perceptual duplicate candidates for manual checking when feasible; report the method and limits. Freeze these decisions before training, never after inspecting model performance.

No held-out relation labels may be read by training, feature extraction, augmentation, frequency estimation, mapping construction, or reasoning-table preparation. Supplied object classes/boxes and public vocabulary definitions are legitimate predicate-classification inputs. The access boundary must distinguish those from relation targets. Statistical feature fitting, if any, uses training data only.

For NeSy4VRD, document the historical ontology's close relationship with the benchmark and previous use of its test set. These corrected results remain retrospective. Independent-dataset confirmation is deferred, and must be listed as an unresolved scientific limitation rather than implied by fresh random seeds.

## W2 Core comparisons

Use these four training conditions for each predictor on NeSy4VRD:

| Condition | Training labels | Semantic loss | Evaluation variants |
| --- | --- | --- | --- |
| `baseline` | Raw annotations | Off | Unfiltered; ontology-filtered |
| `augment` | Raw annotations plus inferred labels | Off | Unfiltered; ontology-filtered |
| `semantic_loss` | Raw annotations | On | Unfiltered; ontology-filtered |
| `combined` | Raw annotations plus inferred labels | On | Unfiltered; ontology-filtered |

This is 20 final training runs and 40 evaluation variants per predictor, before secondary scoring views. The two predictors require 40 core final training runs on NeSy4VRD. The required frequency control adds five runs per predictor, giving 50 final training runs and 100 primary evaluation variants. The simple-filter control adds 10 baseline-only evaluation variants and no training, for 110 primary evaluation variants in total. The bounded learning-rate checks add up to eight development runs. Plans must show these groups explicitly and must not expand into an implicit Cartesian product. Failed adequacy checks or a separately recorded fallback model may change the count; disclose that before execution. If preparation establishes that two conditions have identical effective training inputs and objectives, one run may serve both named rows; document the equivalence and retain both comparison rows.

### Geometry baseline and ordinary loss

The geometry baseline uses the existing `PPNN_1` architecture: four ELU hidden layers with widths `2d, 2d, 2d, d`, followed by logits. The input is `legacy_geometry_v1`, whose precise feature contract is in [Data and artifacts](03-data-and-artifacts.md). The output has `P + 1` columns: P dataset predicates and the legacy auxiliary no-relation column at index P. The auxiliary column is excluded from semantic reasoning and reported relation predictions.

For a pair with one or more training labels, form a multi-hot predicate target and auxiliary target zero. For a wholly unannotated pair, set the auxiliary target to one and ordinary BCE weight to zero for the entire row. Thus unannotated pairs are not silently treated as fully observed negative examples. Unlabelled predicates within an annotated pair still enter BCE as zeros; this incomplete-annotation assumption must be disclosed.

Use the framework's numerically stable BCE-with-logits implementation. Preserve image-wise sum reduction for the initial corrected comparison: sum eligible pair/output losses within each image and average these image losses across a batch. The default batch is one image, matching the legacy update unit. Do not change to a per-cell mean or negative sampling as part of refactoring. A later normalised-loss experiment must have a separate protocol identifier.

For image i, logits z, targets y, annotated-pair mask a, and active semantic mask m defined below:

`L_i = sum_(j,p=0..P) a_ij * BCEWithLogits(z_ijp, y_ijp) + lambda * sum_(j,p=0..P-1) m_ijp * softplus(z_ijp)`.

`softplus(z)` equals `-log(1-sigmoid(z))` but remains finite for large logits. Ontology classifications are immutable data. Activation masks change by the detached Boolean rule below and have no gradients; m is the active mask, not every incompatible cell. Active penalties apply even on pairs excluded from ordinary BCE. If no eligible ordinary targets or penalties exist, record the empty update and skip it without manufacturing a loss.

### Semantic policy

The primary semantic policy is `gated_type_mask`, preserving the intended historical activation rule. Epochs start at one. From epoch 3 onward, an incompatible cell becomes active when its sigmoid probability first exceeds 0.5; it remains active for that image/pair/predicate in later epochs, including when its probability falls. Before epoch 3 all activation masks are false. Lambda is fixed at 1.0 for loss-only and combined conditions.

Classification and activation are separate. The immutable global table says whether a typed relation is incompatible. A per-image Boolean mask says whether that cell has crossed the activation threshold. For each eligible image from epoch 3, update `active = previous_active OR ((probability > 0.5) AND incompatible)`. A cached decision must work for every image, including the first call for a later image. Resuming training restores the activation masks and their pair/vocabulary identity.

Do not silently replace this with penalties on all incompatible cells or only currently above-threshold cells. Those are different experiments and are deferred. Preparing type decisions offline changes how decisions are obtained, not this loss activation rule.

Unknown or unmapped entries never attract an incompatibility penalty. Operational failures prevent the relevant artifact or run from being accepted. Incompatibility must not be inferred from absence of an entailment under the open-world assumption.

### Augmentation profiles

Use `S_TEU` as the initial profile: symmetry, transitivity, equivalent properties, and subproperties; explicit inverse-property axioms are disabled. Its origin as a favourable historical treatment must be disclosed. This phase does not rerun all ontology subsets or establish that this profile is universally best.

Keep the augmentation ontology and the compatibility-checking ontology as separate, hashed inputs. Ablating an augmentation axiom family must not silently weaken the checker used to measure incompatibility.

Construct each augmentation variant from the asserted ontology before closure. Family switches remove the corresponding asserted declarations: `owl:SymmetricProperty`, `owl:inverseOf`, `owl:TransitiveProperty`, `owl:equivalentProperty`, and `rdfs:subPropertyOf` between task properties. Preserve the documented shared class schema. Record every removed axiom. Do not run the full closure and then delete selected consequences. Indirect consequences through retained axioms remain legitimate and must be explained.

Materialise each training image separately, using only its supplied objects and training relation labels. Retain inferred relations only when both endpoints are existing distinct objects and the predicate belongs to the benchmark vocabulary. Preserve original labels, deduplicate inferred triples, and record asserted/inferred counts and the number of newly supervised pairs. No cross-image inference is permitted.

If raw training positives contradict the checker, stop preparation and record the conflict for data/ontology adjudication. A revised ontology is a new version. Do not silently remove inconvenient training examples. Test annotation conflicts are reported after evaluation and do not authorise ontology changes for that test.

### Training and selection

Defaults are Adam, learning rate `1e-4`, weight decay `1e-3`, full-precision training, maximum 100 epochs, and patience 25. These match some legacy choices but constitute a newly recorded protocol. Validate after every epoch from epoch 1. An improvement is a strictly greater unrounded selection metric; ties retain the earliest epoch. Save the earliest best checkpoint and the latest resumable checkpoint.

The selection metric is **unfiltered image-mean recall@50 on raw validation labels**, with the primary candidate policy defined below. Test performance, invalidity, runtime, and post-filter scores must not break checkpoint ties. Validation NaN or an empty eligible validation set is an error, not a non-improving epoch.

For each predictor, choose one ordinary training recipe on the raw-label baseline using the two development seeds and learning rates `[1e-4, 3e-4]`. Weight decay and other defaults remain fixed. Select the highest mean validation selection score, breaking ties by the lower learning rate. Apply that recipe to all four conditions. Do not handicap the baseline with a smaller tuning budget.

Keep lambda fixed at 1.0, as in the main historical experiments. A systematic lambda sweep is deferred. Development runs for choosing the ordinary learning rate are excluded from final seed summaries; the same selected recipe is used for baseline, augmentation, loss, combined, and the frequency control within each predictor.

The planning command must deduplicate identical training configurations. It must show final runs, development runs, controls, and inference-only variants separately, including a projected runtime/storage range from the pilot. No unbounded search or automatic retry with new hyperparameters is allowed.

## W3 Stronger visual predictor

Implement `visual_mlp_v1` through the same data, loss, training, checkpoint, and evaluation paths. Use a frozen TorchVision ResNet-50 encoder with the named `IMAGENET1K_V2` weights, in evaluation mode with gradients disabled. Hash the weights. Use the corresponding pinned preprocessing transform, not a moving default alias.

Extract the 2048-dimensional output after global average pooling, before the ImageNet classification layer, for the subject crop, object crop, and union crop. Decode images as RGB in the pixel orientation used by the annotations; no implicit EXIF rotation may change box alignment. Crop from the original image using canonical boxes, with floor on minimum coordinates and ceil on maximum coordinates, clipped to the image. Reject empty crops. The union is the smallest box containing both objects. Cache repeated object/union crops and retain their identities. No stochastic image augmentation is used in this initial feature baseline.

Normalise each visual vector as `v / max(norm(v, 2), 1e-12)`, retaining a zero vector as zero. Concatenate subject, object, union, then unchanged geometry features, and use a relation head `input -> 512 -> 256 -> P+1`, with ELU between linear layers and no output activation. This is a separate model definition, not a silent widening of `PPNN_1`. Bound crop batches by an explicit memory setting; extract features once per dataset/weight/transform combination.

A visually informed model is not automatically a stronger baseline. On the two development seeds, it qualifies as stronger only if its mean validation image-mean recall@50 exceeds the geometry baseline by at least 1 percentage point and its predicate-mean recall@50 is no more than 1 point lower. These are prospective practical criteria, not statistical significance claims. Report the measured values.

If it fails, retain and report that result. Do not weaken the qualification rule. Add at most one established predicate-classification reference predictor through a pinned external implementation, selected using published task compatibility and validation feasibility. Neural Motifs is a documented reference option; HIERCOM is relevant related work. Do not vendor their frameworks or claim their published numbers as matched local comparisons. Freeze the external model, data conventions, training recipe, and score adapter before test evaluation.

The external adapter must preserve all pair scores required by this protocol. Integrating augmentation/loss into an external trainer requires a small, documented patch with its own smoke test; a predictions-only adapter permits baseline/filtering comparisons but does not complete W3's intervention comparison. If that integration cannot be kept small, record the scope tradeoff before replacing the default design.

Calling W3 complete as a *stronger-predictor* experiment requires one validation-qualified predictor. A failed adequacy check is useful evidence but leaves that specific claim unsupported.

## W4 Simple controls

Two controls are required in this phase. Both reuse the shared scorer and prepared data; neither needs a separate experimental framework.

| Control | Construction | Comparison |
| --- | --- | --- |
| C1 Simple asserted-constraint filter | Filter baseline scores using only asserted named domain/range constraints and explicit named-class disjointness, without ontology closure | Unfiltered baseline versus simple-filtered baseline versus full ontology-filtered baseline. |
| C2 Frequency-matched supervision | Keep raw targets and the raw annotated-pair mask; multiply positive BCE terms for predicate p by `(raw_positive_count_p + inferred_positive_count_p) / raw_positive_count_p` | Baseline and augmentation versus matched positive-label-frequency weighting. |

C1 is a distinct, deliberately limited control. From the same asserted compatibility ontology and mapping, collect each task predicate's directly asserted named domains/ranges and class pairs explicitly declared disjoint by `owl:disjointWith` or named members of `owl:AllDisjointClasses`. For a typed candidate, form the subject set from its supplied class plus those domains, and the object set from its supplied class plus those ranges. Block it if either set contains an explicit disjoint pair. Do not add subclass closure, property inheritance, inverses, transitivity, or inferred constraints. Retain unmapped cases; record unsupported expressions and the control's removal coverage. A retained candidate is not thereby proven compatible.

Apply this simple filter only to the five baseline runs of each predictor. Use the full supported checker for reporting incompatibility in all three views, including simple-filtered output. If the simple mask removes nothing or coincides with the full filter, report that result rather than adding rules after seeing scores. Compare decisions, removals, recall, and remaining full-checker violations. This isolates the tested additional constraints/inferences; it does not prove the value of every reasoning operation.

The full ontology filter in W2 is itself a compiled compatibility table, reused by loss, filtering, and diagnostics. Verify exact agreement with the supported reasoning query on fixtures and the declared real-data sample. Compiling that same checker is an implementation optimisation, not an independent scientific comparator, and does not justify claims that online reasoning is necessary.

C2 weights only positive predicate terms, not negative terms or the auxiliary output. It has no semantic penalty and no inferred labels. Thus it matches the added positive supervision mass per supported predicate while preserving the raw pair mask. Export raw counts, inferred-only counts, and realised weights. This is a frequency/loss-mass control; it does not match every structural consequence of adding labels or newly annotated pairs.

For a predicate with zero raw positives and positive inferred count, no finite matching weight exists. Report the unmatched count and do not invent smoothing or claim exact matching for that predicate. A predicate with zero raw and zero inferred positives has an unused weight of one. The required matched-frequency conclusion applies only to the supported part of the intervention. If more than 5% of added positives cannot be matched, C2 is insufficient for the planned control claim. Before test evaluation, replace C2 with one deterministic added-label shuffle preserving global predicate counts and the number of added labels at each image/pair position. Start from sorted inferred-only cells; propose swaps of the predicate labels of two uniformly selected cells using seed 2027, rejecting identical labels, duplicates, and raw-label collisions. Attempt exactly `100 * max(1, number_of_added_cells)` swaps; do not search over seeds. Record realised marginals and the fraction of final added triples absent from the original inferred set. A fraction below 0.8 is weak/infeasible, not a successful control. If no labels were added, report the intervention/control as degenerate. This is a bounded fallback, not an instruction to run both controls by default.

Run the selected augmentation control with five final seeds on both predictors. Reuse the baseline/augmentation runs from W2/W3. Evaluate the control before and after filtering, using the same selected checkpoint. Report the control's limitations even if augmentation outperforms it. This comparison alone does not prove that every useful effect comes from sophisticated reasoning.

## Scope limits

The following are not completion requirements for this implementation: a second benchmark; inverse-rule or full axiom-subset studies; a systematic lambda sweep; static-versus-gated loss comparisons; a new manual precision audit; end-to-end object detection; or a broad LLM/VLM comparison campaign. Do not implement their adapters, configuration dimensions, or jobs speculatively.

Because these items are deferred, restrict the resulting claims to the tested dataset, models, rule profile, and fixed lambda. Historical adjusted precision remains exploratory. Annotation-based AP does not replace a blinded visual audit, and fewer training epochs do not establish runtime savings.

## Prediction and metrics

The primary ranking view uses every predicate logit for every distinct ordered object pair, no confidence threshold, and multiple predicates per pair. Reject non-finite logits. Sort by decreasing logit, then by subject ID, object ID, and predicate ID to break exact ties; raw-logit ordering avoids artificial ties from sigmoid saturation. Display sigmoid probabilities. Exclude the auxiliary column. Apply filtering before taking the top K, allowing lower-ranked retained candidates to fill the list. The full filter removes only confirmed incompatible candidates and retains unknowns. Report how many candidates were removed and how many slots remain empty. Cache unfiltered scores so filtering cannot change model inference.

Evaluate K in `[25, 50, 100]`. Also provide a historical diagnostic view using probability strictly greater than 0.5, plus K=999 and a genuinely uncapped count. `999` is a cap, never a synonym for all predictions. Thresholded and unthresholded results must have different policy identifiers. A one-predicate-per-pair scoring view may be added for a published comparator, but all compared methods must use that same view and it must not replace the primary multi-label view silently.

Match canonical subject/object IDs and predicate IDs for the supplied-object task. Deduplicate identical ground-truth triples and predictions. Count each ground-truth triple at most once. If reproducing an external evaluator's IoU matching, retain it as a named additional metric policy and verify it against the reference scorer; do not mix its scores with exact-instance scores.

| Metric key | Definition and denominator |
| --- | --- |
| `recall_image_mean_at_K` | Mean of `hits_i_at_K / ground_truth_count_i` over images with at least one ground-truth relation. |
| `recall_global_at_K` | Sum of hits divided by sum of ground-truth relation counts. This is micro/global recall, not predicate-macro recall. |
| `recall_predicate_mean_at_K` | For each predicate with support in the evaluated partition, divide its hits by its ground-truth count, then average predicates equally; publish support and omitted zero-support predicates. |
| `annotation_ap_image_mean_at_K` | For each eligible image, sum `precision_at_r * hit_at_r` through rank K and divide by its ground-truth count, then average images. Label this truncated, annotation-based AP, not audited visual precision or an unspecified mAP. |
| `incompatible_count` | Number of emitted predictions with a confirmed incompatibility decision under the named checker. |
| `incompatible_rate_all` | Incompatible count divided by all emitted predictions; publish unknown count alongside it. |
| `incompatible_rate_decided` | Incompatible count divided by compatible plus incompatible predictions. |
| `reasoning_coverage` | Compatible plus incompatible predictions divided by all emitted predictions. |
| `prediction_count` | Emitted predictions, eligible pairs, empty images, and predictions per image under the named view. |

Store metrics as fractions with full available precision; display percentages in reports. A zero denominator gives `null` with a reason, not zero. An empty prediction list for an eligible image has zero hits. An absent image record is an error. Images without annotated relations are excluded from recall/AP means but remain in prediction/violation counts with their count disclosed.

The same checker used to filter predictions will normally report zero remaining known incompatibilities. Treat that as a filtering property, not evidence of improved learned knowledge. The relevant learning comparison is the pre-filter output, and visual correctness requires separate evidence.

## Uncertainty and reporting

Report all five seed results, arithmetic mean, sample standard deviation, and paired per-seed differences for each preregistered contrast. Compute a 95% percentile interval from 10,000 paired crossed bootstrap draws: sample five seed indices with replacement and sample image IDs with replacement once per draw, sharing that image sample across all selected seeds and both compared methods. Recompute each seed's metric from its sampled counts, then average seed differences. Resample the fixed eligible image set for recall/AP and all images for prediction/consistency metrics. This preserves the fact that runs share a test sample. For predicate-mean recall, use a paired seed-only bootstrap of the full-test metric, with the predicate support set fixed; label that interval as variation across runs only. This avoids dropping rare predicates from image bootstrap draws when their support disappears. Undefined zero-denominator draws are counted and disclosed; do not replace them with zero or report an interval if fewer than 95% of draws are defined. These intervals describe the sampled benchmark/runs and do not establish generalisation to arbitrary datasets.

Primary contrasts are augmentation minus baseline, semantic loss minus baseline, combined minus augmentation, and combined minus semantic loss, all before filtering. Compute these within each predictor; report visual-versus-geometry baseline differences separately. Planned control contrasts are simple-filtered and full-filtered baseline minus unfiltered baseline, full-filtered minus simple-filtered baseline, augmentation minus C2, and C2 minus baseline. The latter two use unfiltered output, with filtered versions named secondary analyses. Results for the four primary contrasts form a family. Do not select the best contrast after test inspection and present it as the only planned hypothesis. Default reporting emphasises estimates and uncertainty; adding significance tests requires a predeclared multiplicity policy.

A positive average alone does not establish a reliable or useful improvement. For accuracy claims report the interval and absolute change. For consistency claims report absolute violations and coverage, along with recall change; a recall decrease larger than 1 percentage point must be explicitly described as a tradeoff. Negative or uncertain results must remain in the main comparison table.

Record preprocessing, reasoning preparation, feature extraction, training, validation, and inference wall time separately. Include cache size, peak observed memory, hardware, precision, and effective epochs/updates. Synchronise accelerators for timing. Report cold preparation and amortised reuse separately. Fewer epochs do not establish lower total runtime.

## Test freeze and release

Before the first test scoring in this campaign, finish development for both predictors, the visual qualification/fallback decision, and the C2 coverage/fallback decision. Freeze one shared analysis plan containing effective recipes, planned groups and contrasts, metric policies, ontology/mapping hashes, expected seed rows, and development-selection records. Test outcomes from an early group must not influence scientific choices for later groups.

Training and test evaluation may then finish in stages. Each test group gets an immutable selection manifest binding all its planned validation-selected checkpoints to that same analysis-plan digest. G3 may evaluate the geometry core while final visual/control training remains incomplete. Later groups add their own selection manifests; they never edit a previously frozen plan or selection. `freeze` validates these plain files, not a new infrastructure service or approval system. An incomplete group cannot be reported as complete.

Test evaluation requires the frozen analysis plan and the relevant selection manifest, and reads only their selected checkpoints. Deterministic reruns to verify saved predictions are allowed and logged. Changing a scientific choice after seeing test results creates an exploratory protocol revision. Independent confirmation then needs untouched data, not another label on the same test set.

G5 produces an absolute-results table, per-seed effect plots, a recall/violation tradeoff plot, the required control comparison, one readable method diagram, and an evidence table connecting every proposed manuscript claim to run IDs. Generate numerical figures from saved aggregate data; no hand-entered plot values or invented examples presented as model outputs.

Use one consistent condition order and colour key. Prefer seed points and absolute effect intervals to collections of bars or relative percentage reductions. Every panel names the metric, K, filter mode, units, and uncertainty definition; show absolute violation counts beside rates. Mark missing conditions explicitly. Export vector PDF/SVG and a high-resolution PNG from the same plotting function, with legible text at the intended manuscript width. The method diagram should show where training annotations, semantic loss, and inference filtering enter the shared pipeline; it must not imply that the reasoner is an online training service.
