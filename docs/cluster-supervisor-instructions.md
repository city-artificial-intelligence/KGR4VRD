# KGR4VRD cluster continuation

The user requested preparing and queuing the experiments in
`docs/compute-node-handoff.md`, then transferring their execution to the Exact-OM
supervisor procedure. Routine preparation, finite queue submission, minimal
correctness fixes, tests and compatible checkpoint recovery are authorized.
Do not babysit a healthy worker; finish each intervention after verified detached
submission or a precise blocker.

Read the supplied registry and local campaign `handoff.json` first, then the
compute-node handoff and relevant specifications. The scientific contract remains
`docs/specs/01-experimental-protocol.md`; no extra datasets, seeds, sweeps or new
model families are authorized beyond its bounded fallback. Do not change the paper.

All jobs belong to retained allocation **14482 on liseda-t2**. Preserve its
interactive shell. Use detached numeric `srun --jobid=14482 --overlap --exact`
steps. Never create/cancel allocations, use sbatch, modify another allocation,
change cluster services or run scientific work directly in an intervention.
The controller and registry are independent of existing Exact-OM campaigns.
Use the frozen Exact-OM dispatcher and the same-user extern-owned tmux socket
recorded in this registry. Do not create a tmux server inside a numeric step.

Experiment capacity is six CPU threads, one RTX 2060 SUPER (8 GiB), and
49,152 MiB RAM. The supervisor reserves one CPU and 8,192 MiB. Run one scientific
worker at a time; GraphDB is a child of its preparation step with an 8 GiB heap,
within that worker's reservation. Keep FP32, one image/update, num_workers=0,
the registered seeds and selection criteria. Do not introduce AMP or alter the
study to fit memory. Use local scratch only for disposable caches; retain data,
checkpoints, scores and receipts in durable repository storage.

The pinned supervisor checks every 300 seconds, dispatches ready descriptors every
15 seconds, and uses the existing ChatGPT login with gpt-6-astra/xhigh only for
confirmed incidents or genuinely unprepared batches. It permits two unsuccessful
attempts per incident and no daily cap. Do not change authentication, limits,
policy, retry state or the supervisor's own step. This campaign sends no external
messages; record actionable blockers and intervention outcomes locally.

## Required continuation

1. Complete data and node acceptance. Verify NeSy4VRD 1.1's published MD5 and
   measured SHA-256, exact source files, mapping provenance, ontology identities,
   and original terms. Keep the official test membership and deterministic
   1729 split. Resolve malformed input visibly; never silently drop annotations.
   The archive includes author-provided training/validation split lists; use them
   if they pass the duplicate and membership audit. Original checkpoints and exact
   historical ontology variants are not recovered. The acquired public profile
   must be named as a fresh corrected study.
2. Pass the full offline suite and lint/type checks, actual CUDA head/backward/
   resume checks, and real encoder/GraphDB integration with **zero skips**.
   `scripts/prepare_node.py acceptance` creates an exclusive repository and a
   three-training-image encoder fixture without held-out relation labels.
   Repository conflicts and unsupported semantics remain failures, not unknowns.
3. Prepare features and reasoning with the acquired configuration. The frozen
   `scripts/prepare_node.py reasoning` owns its GraphDB lifetime and never clears
   an existing repository. Preserve completed compatible typed decisions; do not
   replace GraphDB with a local approximation. Measure query throughput, unknown
   coverage and training conflicts. Check S_TEU, C1 and C2 provenance/conformance.
   Feature extraction cannot resume unpublished shards: investigate first and
   choose a new cache directory if necessary. Reuse accepted immutable caches.
4. Run isolated resource pilots for both registered heads, with complete train/
   validation epochs and the combined semantic gate exercised at epoch 3. Collect
   actual pair/crop counts, largest-image CUDA allocated/reserved peaks, host RSS,
   preparation/cache/checkpoint/score sizes and measured projections. Do not use
   pilot checkpoints as development/final results. Freeze the measured pilot record
   before releasing the eight registered development jobs.
5. Run all eight development rows serially with `scripts/run_planned.py`, retaining
   max_epochs=100 and patience=25. Record every selection in the existing inventory.
   Use the running guide's evidence-population procedure, assess visual adequacy
   and the C2 trigger, and call the package's `freeze`. If the visual model fails,
   preserve that result and follow only the bounded fallback; do not invent an
   unavailable dependency to bypass it or claim the stronger-model gate passed.
6. Re-read final config paths after freeze. Run the four groups serially: geometry
   core (20), visual core (20), geometry control (5), visual control (5). Keep all
   rows and failed seeds visible. Each group requires its own complete selection
   binding before test prediction. Produce 110 registered views from 50 raw-score
   artifacts: none/full for all, simple_asserted for the ten baselines only.
7. Verify every completed run and generate the report. Check 8 development, 50 final
   and 110 evaluation inventory rows explicitly. Regenerate evidence from saved
   artifacts; report negative outcomes, partial semantics and retrospective scope.
   Mark remaining_work_status complete only when this accounting is complete.

## Ownership and recovery

Check supervisor and campaign STOP/PAUSE markers before any mutation/submission.
Do not clear intentional pauses. A quiet log does not prove failure; inspect live
Slurm steps, descendants, status, exit and completion receipts. Never duplicate a
live or uncertain launch. Retain failure receipts and cumulative time across retries.
Do not terminate healthy work merely because a projection was exceeded.

Use `scripts/queue_step.py` to publish exact command specs to the dispatcher.
Descriptors bind nonce, argv, source/config hashes, resources and real receipt
paths. A pending row without a launch is a preparation obligation, not a running
job. Ready descriptors have priority; never submit them manually as well.
Final configs cannot be bound until development freeze has resolved them.

Every registry write must hold exclusive flock on `registry.json.lock`, reread the
registry inside the lock, and atomically replace only that registry. Preserve
concurrent additions, dependency links and registered history. On recovery keep
the failed predecessor disabled with superseded_by/reason, register the actual new
numeric step, and preserve incident retry accounting. Resolve uncertain dispatch
reservations only after proving whether the original worker exists. Never invent
a step/PID or completion receipt. Unprepared remaining work must stay represented
in pending_batches with remaining_work_status=pending.

Preserve frozen running sources and scientific identities. Develop fixes separately,
test them, commit only owned files and freeze a successor snapshot. Review affected
artifacts before reuse; changed source/environment can invalidate development and
resume identities. Never alter hashes merely to admit incompatible outputs.

Bubblewrap currently fails before command execution on this node. Request
require_escalated for necessary authorized commands and let automatic approval
review evaluate them. Do not disable sandboxing or retry rejected actions through
another tool. A rejected necessary action remains a precise recorded blocker.

Return the Exact-OM result schema with outcome repaired/submitted/no_change/needs_user,
summary, actual steps, commits and local handoff path. An unresolved scientific or
resource decision must state the exact action needed. No automatic email is configured.
