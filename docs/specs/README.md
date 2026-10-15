# KGR4VRD experimental revision and implementation specification

Status: specification for implementation. Date: 12 October 2026. Version: 1.

The objective is to establish whether ontology-derived annotations and semantic constraints improve visual relation prediction under a defensible experimental protocol, and to make those experiments straightforward to understand, reproduce, and extend. The implementation should be a small research package, with one training path and one evaluation path shared by all conditions.

This specification defines five workstreams. It does not claim that the implementation exists, that the historical results have been reproduced, or that the proposed experiments will produce positive results.

| Workstream | Required outcome |
| --- | --- |
| W1 Correctness and frozen protocol | Repair the known defects; freeze splits, ontology versions, seeds, and validation-only selection. |
| W2 Core comparison | Run baseline, augmentation, semantic loss, and their combination, each with and without inference filtering. |
| W3 Stronger visual predictor | Repeat the core comparison with a predictor that uses pretrained visual features, and establish its strength on validation data. |
| W4 Simple controls | Compare with a simple asserted-constraint filter and a control matched for added-label frequency. |
| W5 Refactoring and reproducibility | Replace the active script pipeline with a compact, installable, tested Python package and explicit experiment artifacts. |

W1–W4 correspond to the first four rows of the experimental-priority table, as clarified by the user. W5 is the additional refactoring requirement. Independent benchmarks, inverse-rule/lambda mechanism studies, and a new manual precision audit are deferred. They must not become hidden completion requirements or automatically expanded experiment groups.

## Reading order

1. [Evidence and decisions](05-evidence-and-decisions.md): what was observed, what is proposed, and which external inputs remain unavailable.
2. [Experimental protocol](01-experimental-protocol.md): research questions, models, conditions, selection, metrics, and interpretation.
3. [Data and artifact contracts](03-data-and-artifacts.md): canonical records, configuration, provenance, checkpoints, and results.
4. [Architecture](02-architecture.md): module boundaries, interfaces, dependencies, and limits on complexity.
5. [Implementation and acceptance](04-implementation-and-acceptance.md): ordered delivery slices, meaningful tests, and completion evidence.

The documents have distinct authority: the protocol defines scientific semantics; the contracts define persisted data; the architecture defines ownership and dependencies; acceptance defines verification. If two requirements conflict, resolve the conflict in these specifications before implementing both behaviours. Do not create a compatibility framework to accommodate an accidental contradiction.

## Scope and terminology

The initial task is predicate classification with supplied object classes and bounding boxes. Objects are ground truth inputs to this task. Predicting object classes, detecting boxes, end-to-end scene graph detection, open-vocabulary prediction, and downstream visual question answering are outside this implementation.

An annotation is an observed dataset label. An inferred annotation follows from a specified ontology and rule profile. An incompatible prediction contradicts the supported ontology constraints. Compatibility is not a claim that a relation is visually true. An unknown reasoning result is neither compatible nor incompatible.

This phase uses NeSy4VRD. Results can establish effects within that benchmark and across the two tested predictors; they cannot establish independent-dataset generalisation. Additional datasets are outside the implementation scope.

## Requirements that govern every workstream

| ID | Requirement |
| --- | --- |
| S01 | Test relation labels must not participate in training, feature fitting, ontology adaptation, hyperparameter selection, or checkpoint selection. |
| S02 | Every reported metric for a trained run must use its single validation-selected checkpoint. Filtering variants reuse the same checkpoint and scores. |
| S03 | Baseline and intervention conditions must share splits, candidate pairs, label order, evaluation targets, and scoring policy. Differences must be named factors. |
| S04 | Unknown reasoning, service failure, missing predictions, malformed data, and numerical failure must be represented explicitly. None may become a favourable score or an incompatibility judgment silently. |
| S05 | Results must be recoverable from immutable predictions and manifests without retraining or parsing console output. |
| S06 | Failed or negative experiments remain visible. Completion means the registered comparison was performed correctly, not that a desired gain appeared. |
| S07 | The active package must not import the legacy scripts, read global configuration at import time, or require a particular working directory. |
| S08 | Prefer one readable function or module to a hierarchy, factory, registry, service, or plugin system. Add an abstraction only for an existing, demonstrated second use. |
| S09 | Refactoring must distinguish preserved behaviour, corrected defects, and deliberate methodological changes. Tests must not preserve a known defect as the new scientific contract. |
| S10 | A working smoke run, a full research run, and a submission-ready evidence package are separate completion states. |

MUST denotes a requirement. SHOULD permits a documented, evidence-based exception. MAY denotes optional work. Routine implementation decisions do not require repeated user confirmation. The readiness gates are executable checks and evidence reviews, not a new approval system.

## Deliverables and boundaries

The later implementation must deliver one `src/kgr4vrd` package, a command-line entry point, a small set of experiment configurations, tests, and practical run documentation. Dataset downloads, model weights, prepared tensors, checkpoints, and predictions live outside tracked source files. The paper and review files must be preserved.

Do not build a web interface, REST API, experiment database, distributed execution service, generic workflow engine, plugin marketplace, custom configuration language, or general OWL reasoner. Do not copy an external scene graph framework into this repository. Do not port every historical experiment dimension.

The first complete slice is deliberately small: synthetic input to training to checkpoint selection to prediction to evaluation to a reproducible result table. Real data and additional research conditions are added only after this path works.

## Completion order

| Gate | Evidence | Workstreams enabled |
| --- | --- | --- |
| G0 Source and data inventory | Historical/code differences, available resources, licenses, schemas, and unresolved artifacts recorded | Preparation |
| G1 Scientific regression checks | Known defects covered; reasoning and metric semantics verified on hand-checkable examples | W1/W5 foundations |
| G2 Reusable package | Clean install, tiny end-to-end run, resume equivalence, and artifact regeneration pass | Real-data pilots |
| G3 Corrected NeSy4VRD core | Frozen protocol and complete five-seed geometry-model results | W1/W2 |
| G4 Stronger predictor and controls | Validation-qualified visual model, core comparisons, and the two required controls | W3/W4 |
| G5 Research evidence package | Traceable tables/figures, uncertainty, limitations, and reproduction instructions | Manuscript assessment |

Negative intervention effects do not prevent completion. A required comparison must still exist: for example, an unqualified visual model leaves W3/G4 incomplete as a stronger-predictor study. Missing data or unavailable hardware must be reported as a specific incomplete dependency, not replaced with synthetic results described as experimental evidence.

## Fixed defaults and unresolved inputs

Defaults in this version are implementation choices to make the project concrete: five final seeds; a frozen ResNet-50 visual encoder as the compact visual baseline; validation image-mean recall at 50 for checkpoint selection; lambda fixed at 1; and a bounded development check for the learning rate. Their scientific limitations and fallback procedures are specified in the protocol.

External facts that cannot yet be fixed, such as downloaded file hashes, the exact historical ontology variants, the execution hardware, and recovered author checkpoints, are recorded in the [decision register](05-evidence-and-decisions.md). Resolve them at the named readiness gate. An unresolved hash is not permission to invent a value or silently use whichever file is present.

The current task ends with these specifications. Their proposed commands and package paths describe the future implementation.
