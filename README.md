# KGR4VRD: Knowledge Graph Reasoning for Visual Relationship Detection

Research code accompanying the paper:

> David Herron, Ernesto Jiménez-Ruiz, and Tillman Weyde (2026).
> **Logical Reasoning with OWL Knowledge Graphs for Learning and Validating Visual Relationship Detection.**

KGR4VRD studies how symbolic knowledge and OWL reasoning can guide neural networks
that predict relationships between objects in images, such as **(person, ride,
horse)**. It combines a neural predicate classifier with **VRD-World**, the
companion ontology of the **NeSy4VRD** dataset.

The Python package focuses on **predicate classification**: given object classes
and bounding boxes, predict the relationships between ordered pairs of objects.
It provides data preparation, ontology reasoning, training, evaluation, and report
generation through the `kgr4vrd` command-line interface.

## Approach

The research investigates two uses of OWL reasoning:

- **Annotation augmentation:** infer additional relationships from the training
  annotations and ontology to enrich the supervision available to the network.
- **Prediction validation:** identify relationships that conflict with the
  ontology, using these judgments as semantic loss penalties during training and
  as filters during inference.

The package supports four training conditions—baseline, annotation augmentation,
semantic loss, and their combination—with optional inference filtering. Geometry
and visual-feature predictors share the same training, checkpoint selection, and
evaluation pipeline. Checkpoints are selected on validation data, and saved
prediction scores can be evaluated under different filtering conditions without
retraining.

The [experimental protocol](docs/specs/01-experimental-protocol.md) describes the
comparisons, controls, and evaluation measures. The
[reasoning documentation](docs/data-preparation.md) defines the supported ontology
profile and the treatment of unknown reasoning outcomes.

## Dataset

[NeSy4VRD](https://github.com/djherron/NeSy4VRD) combines visual relationship
annotations with **VRD-World**, an OWL ontology describing the object classes and
relationships in the images. The dataset and ontology are available from
[Zenodo](https://doi.org/10.5281/zenodo.7916355).

Follow the [data preparation guide](docs/data-preparation.md) to acquire the
inputs, configure local paths, and prepare the dataset, ontology artifacts, and
visual features. Dataset files, pretrained weights, and generated experiment
artifacts are stored locally outside version control.

## Installation and quick start

Use **Python 3.11** and [uv](https://docs.astral.sh/uv/) to install the dependencies
from the committed lockfile:

```sh
git clone https://github.com/city-artificial-intelligence/KGR4VRD.git
cd KGR4VRD
uv sync --locked --all-extras --group dev --python 3.11
uv run --locked kgr4vrd --help
uv run --locked kgr4vrd smoke --output artifacts/smoke
```

The smoke example runs all four training conditions for four epochs on synthetic
data, then saves checkpoints, predictions, evaluation tables, and report figures
under `artifacts/smoke/`. It runs offline after installation and requires a fresh
output directory each time. Use it to check the installation and exercise the
complete pipeline before preparing NeSy4VRD experiments.

This installation selects CPU PyTorch on Linux. For GPU execution, follow the
[CUDA environment guide](docs/cuda-environment.md).

## Running experiments

The [experiment guide](docs/running-experiments.md) covers input preparation,
experiment planning, training, checkpoint selection, prediction, evaluation, and
report generation. Example configurations are provided in [`configs/`](configs/).

Training records the configuration, input identities, and resumable state for
each run. Evaluation saves raw scores and structured results, allowing tables and
figures to be regenerated from saved artifacts. Ontology artifacts are prepared
before training; GraphDB is required for compatibility preparation, while training
and evaluation use the saved artifacts.

The [protocol and implementation specifications](docs/specs/README.md) document
the experimental design and artifact formats. The
[implementation history](docs/historical-differences.md) records methodological
changes from the original scripts.

## Repository structure

| Path | Contents |
| --- | --- |
| [`src/kgr4vrd/`](src/kgr4vrd/) | Python package and command-line interface |
| [`configs/`](configs/) | Dataset, model, and experiment configurations |
| [`scripts/`](scripts/) | Input preparation and experiment orchestration |
| [`tests/`](tests/) | Scientific regression, pipeline, and integration tests |
| [`docs/`](docs/) | Installation, experiment instructions, and technical specifications |
| [`paper/JAIR_paper/`](paper/JAIR_paper/) | Manuscript sources and figures |
| [`legacy/`](legacy/) | Original scripts retained for reference |

## Citation

If you use this code in your research, please cite the accompanying paper:

```bibtex
@misc{herron2026kgr4vrd,
  title  = {Logical Reasoning with {OWL} Knowledge Graphs for Learning and Validating Visual Relationship Detection},
  author = {Herron, David and Jiménez-Ruiz, Ernesto and Weyde, Tillman},
  year   = {2026}
}
```

## License

The code is available under the [MIT License](LICENSE).
