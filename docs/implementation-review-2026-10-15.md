# Implementation review before compute-node execution

Reviewed on 15 October 2026 against commit `5bf238e`. This is a source review,
local CPU verification and public-ontology compatibility probe, not a real-data
experiment. The implementation is substantial, but the full campaign should wait
for the four issues below to be resolved. Production code was not changed during
this review. The [node handoff](compute-node-handoff.md) gives the execution order.

## Follow-up corrections — 16 October 2026

The original findings below are retained as the review record. The development
branch `codex/dev-node-experiments` applies these focused corrections:

| Finding | Implemented correction | Remaining acceptance |
| --- | --- | --- |
| R1 | Structural inventory of named-member Boolean expressions and inert metadata; original axioms retained. Verified contradictions remain incompatible; an unwitnessed union is unknown. Focused rule-engine and adapter fixtures exercise the boundary. | Real GraphDB Boolean fixtures, actual ontology provenance, fresh conformance and unknown coverage must pass before accepting the acquired campaign profile. This is partial RL/RDF semantics, not complete OWL consistency. |
| R2 | Explicit `augmentation_format` and `compatibility_format` (`turtle`, `xml`, `nt`), defaulting to Turtle and recorded with source hashes. | Inventory the actual acquired files; do not infer their syntax or historical identity from `.owl`. |
| R3 | Separate hash-locked CUDA 11.8 and 12.1 dependency profiles retain torch 2.2.2 / torchvision 0.17.2; a CUDA base/campaign configuration selects the device before planning. | Choose from measured driver inventory, install on Linux, and pass actual allocation, registered-head and resume checks. Local resolution is not GPU execution. |
| R4 | Frozen encoder and bounded crop tensors move to the selected device; persisted vectors remain CPU FP32. Cache identity records actual extraction hardware/runtime/settings; unavailable devices fail. | Run repeatability with acquired ResNet-50 V2 weights/images on the node before full cache preparation. |

Use [the CUDA environment guide](cuda-environment.md) and updated
[node handoff](compute-node-handoff.md). Local verification and remaining gates
are recorded in [the implementation report](implementation-report.md).

## Findings

### R1 — P1: the supported ontology profile excludes the public VRD-World files

[`ontology_inventory`](../src/kgr4vrd/reasoning.py), lines 508–567, admits a narrow
named-class profile. It rejects `owl:unionOf`, `owl:intersectionOf`, anonymous
equivalent-class/domain expressions, and some metadata declarations. Preparation
then stops at lines 694–697. Parsing the public ontologies explicitly as Turtle
produced the following results with the current implementation:

| Public reference file | Triples | Rejected triples |
| --- | ---: | ---: |
| `vrd_world_v1.owl` | 995 | 60 |
| `vrd_world_v1_2.owl` | 3,039 | 204 |

Sources: [v1](https://github.com/djherron/NeSy4VRD/blob/main/ontology/vrd_world_v1.owl)
and [v1.2](https://github.com/djherron/NeSy4VRD/blob/main/ontology/vrd_world_v1_2.owl),
retrieved 15 October 2026. Their SHA-256 hashes were respectively
`c1148a54d4a075100d0d47cccf579abce776af08b327cf21eef32bfa9fc65466` and
`5d8634646309a062c396db729a0129f1d44ea8eadc0a56e09c731aea44fd7c8e`.
These are reference snapshots, not evidence that the paper's exact augmentation
and checker variants have been recovered.

Inventory the acquired campaign ontologies before running the expensive preparation.
Distinguish inert annotations from logical axioms, and establish the actual bounded
reasoning profile with focused entailment/contradiction fixtures and the real
GraphDB backend. Use the existing reasoning libraries. Merely extending a whitelist,
deleting rejected axioms, or converting them all to unknown would not establish
correctness. If the required semantics cannot be supported, record the unresolved
dependency and its scientific consequence before proceeding.

### R2 — P2: ontology parsing depends on a misleading file extension

[`load_ontology`](../src/kgr4vrd/reasoning.py), line 150, calls `Graph.parse` without
an explicit format. Both public `.owl` files contain Turtle; the inferred XML parser
raises `SAXParseException` at line 1. Explicit `format="turtle"` succeeds.

Accept and record an explicit serialization format, or document a byte-preserving
`.ttl` preparation step with original and prepared identities. Resolve R1 as well:
correcting the extension alone does not make the ontology admissible. Cover the
actual acquired serialization in a regression check.

### R3 — P1 for the GPU launch: the Linux installation selects CPU-only PyTorch

[`pyproject.toml`](../pyproject.toml), lines 46–53, binds both PyTorch and TorchVision
to the CPU index on Linux. The documented locked installation therefore cannot use
the RTX 2080. The base configuration also defaults to `device="cpu"`.

Provide a reproducible CUDA installation profile while retaining the CPU test
environment. Keep the registered PyTorch 2.2.2/TorchVision 0.17.2 versions unless a
change is justified and revalidated. Their official binaries include CUDA 11.8 and
12.1 variants; choose against the node's driver and record the resolved environment.
See [PyTorch's version-specific installation instructions](https://pytorch.org/get-started/previous-versions/#v222).
Prove an actual CUDA allocation and model forward/backward pass on the node.

Set the intended device in the base configuration before planning/development.
Changing it only through a later training override can make the selected run differ
from the planned configuration. Also avoid a subsequent `uv run --locked` silently
restoring the current CPU wheels after a manual CUDA installation.

### R4 — P2: visual feature preparation ignores the selected device

[`vision.py`](../src/kgr4vrd/adapters/vision.py), lines 75–96 and 120–138, leaves both
the encoder and crop tensors on CPU. The cache identity hardcodes CPU at line 238.
A probe using the existing fake-encoder fixture requested `training.device="cuda"`;
every observed input batch and the recorded device remained `cpu`. This probe
establishes the routing defect, not real CUDA acceptance.

Move the frozen encoder and bounded crop batches onto the explicitly selected
device, keep persisted vectors FP32 on CPU, and record the actual extraction
environment. Preserve crop order, normalization, weight identity and inference
mode. Validate actual encoder repeatability and GPU execution before building the
full cache. Installing CUDA wheels alone does not repair this path.

## Verification and remaining uncertainty

The review reran the full suite: **76 passed, 2 skipped**, with 14 existing
Matplotlib/Pyparsing deprecation warnings. Ruff lint/format checks and mypy passed.
The skipped checks require real GraphDB and locally acquired encoder weights/images.
No CUDA execution, full dataset preparation or research run was performed here.

The shared trainer, validation-only selector, immutable score artifacts, separate
semantic activation state, distinct simple-filter control, and campaign freeze
checks are present. The package remains small enough to follow; these findings
call for focused corrections and integration evidence, not another architecture.

Two operational limits belong in the pilot: training retains two boolean
pair/predicate masks for the entire training set on the chosen device, and current
feature extraction cannot resume unpublished shards. Reasoning queries and much
of ranking are serial CPU work. These affect capacity and time estimates even when
the model itself is small. Historical checkpoints and exact ontology variants
remain external inputs; passing local tests does not resolve their provenance.
