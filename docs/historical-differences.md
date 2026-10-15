# Corrected implementation and source inventory

Starting revision: `08bb8e5f79e5c0737b2c45334ef5745e17c06006`.
The initial working tree contained a modified README and untracked `docs/` and
`paper/`. Their original bytes were recorded in
`artifacts/inventory/starting-files.json` before implementation. Research materials
and the specification source are preserved. Historical code is retained as reference;
only the `kgr4vrd` package is supported for new training and evaluation.

## Inventory (15 October 2026)

The checkout contains the historical scripts, experiment-space CSVs, manuscript,
and review letter. It does not contain NeSy4VRD images/annotations, ontology variants,
label-to-IRI mappings, recovered split lists, model weights, checkpoints, predictions,
or original manual judgments. No GraphDB service or CUDA device is configured.

The public [NeSy4VRD 1.1 record](https://zenodo.org/records/17076303) advertises a
2.0 GB `NeSy4VRD.zip`, publisher MD5 `4c83779814209077e25a7a707f441eed`.
The archive has not been acquired. Local SHA-256 and internal ontology variants
remain unresolved; filenames and hashes are not inferred from historical comments.
The existing repository has an MIT license. Dataset/image redistribution terms must
be recorded from the acquired resource separately.

## Preserved behaviour

- Geometry feature set 2, explicit `2*C+15` order and legacy rounding, `PPNN_1`
  widths/ELUs, auxiliary output at P, multi-label targets, and image-wise sum loss.
- Wholly unannotated pairs are masked from ordinary BCE. Missing labels within an
  annotated pair still act as negatives; annotations are incomplete.
- Semantic activation begins at epoch 3, strictly above 0.5, persists per cell,
  and is distinct from immutable type classifications. Lambda remains 1.
- Historical cells 460, 2470, 3001 and 5500 map to baseline, augmentation,
  semantic loss and combined conditions. They are provenance, not run identities.

## Corrected defects and deliberate prospective changes

- Cached incompatible types apply independently to every image, including later
  images. Transport errors are operational failures, never logical judgments.
- Validation denominators use exactly the expected partition. One earliest best
  checkpoint is selected by unrounded, unfiltered image-mean recall@50; neither
  test scores nor filtered scores select or break ties.
- The historical implementation monitored global recall while its prose described
  image-mean recall. The new metric is an explicit prospective protocol choice.
- Frozen split recipe, five final seeds, bounded development learning rates,
  pinned visual encoder, controls and artifact contracts are new choices.
- Compatibility concerns independent typed relations under a declared supported
  profile, not unrestricted OWL completeness or whole-image graph consistency.

Historical results have not been reproduced, and the effect sizes of the old bugs
cannot be reconstructed without original artifacts. Synthetic checks establish
software behaviour only. NeSy4VRD results remain retrospective because the ontology
and historical research have already used this benchmark and its test partition.
