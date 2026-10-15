# KGR4VRD: Knowledge Graph Reasoning for Visual Relationship Detection

## Corrected research package

The supported implementation is [`src/kgr4vrd`](src/kgr4vrd), with one trainer,
validation-only checkpoint selector and evaluator shared by the geometry and visual
heads. The [five specifications](docs/specs/README.md) define its scientific contract.

With Python 3.11 and [uv](https://docs.astral.sh/uv/):

```sh
uv sync --locked --all-extras --group dev --python 3.11
uv run --locked kgr4vrd --help
uv run --locked kgr4vrd smoke --output artifacts/smoke
```

The four-epoch, four-condition smoke example is offline after installation and
creates synthetic checkpoints, predictions, evaluations and report figures. Use a
fresh output directory. Synthetic success is not a completed NeSy4VRD experiment.

See [running experiments](docs/running-experiments.md),
[data preparation](docs/data-preparation.md),
[historical differences](docs/historical-differences.md), and
[implementation and verification status](docs/implementation-report.md).
Real-data, GraphDB and encoder acceptance remain explicit requirements.

## Historical research description

The description below records the original research context. Its broad statement
about OWL soundness/completeness is not a guarantee made by this package: the
corrected implementation is bounded by the declared supported reasoning profile,
backend conformance and independent-typed-relation query scope. Historical scripts
are preserved under `legacy/` and are not supported entry points for new studies.

## KGR4VRD
The **KGR4VRD** repository hosts code associated with aspects of our **neurosymbolic AI research**. This research explores **combining symbolic knowledge and symbolic reasoning with deep (subsymbolic) learning** in neurosymbolic systems directed at the computer vision task of **detecting visual relationships in images**. An example visual relationship for a hypothetical image is, say, **(person, ride, horse)**. 

For the symbolic elements (knowledge and reasoning) of our neurosymbolic systems, we rely entirely on **Semantic Web** technologies. That is, to represent symbolic knowledge, we use **OWL** (the Web Ontology Language), **OWL ontologies** (descriptions of domains of interest that have associated inference semantics), and **OWL knowledge graphs** (ontologies together with asserted facts in RDF triple stores).  For symbolic reasoning, we use **OWL reasoning**, or OWL knowledge graph reasoning. Since OWL is the Semantic Web incarnation of the formal Description logic $\mathcal{SROIQ}$, OWL reasoning is really $\mathcal{SROIQ}$ reasoning. Hence, it is guaranteed to be both logically sound and logically complete. Description logics are decidable fragments of first-order logic, and $\mathcal{SROIQ}$ is a particularly expressive fragment.

We describe this research in the (forthcoming) paper:
> **Herron, D., Jimenez-Ruiz, E., Weyde, T.** (2026). `Logical Reasoning with OWL Knowledge Graphs for Learning and Validating Visual Relationship Detection.` (under review)


## NeSy4VRD

Our research, and hence our code, uses the **NeSy4VRD** dataset.
The acronym NeSy4VRD stands for "Neurosymbolic AI for Visual Relationship Detection". 
NeSy4VRD is a unique **`image dataset plus OWL ontology' resource** that we created to facilitate our neurosymbolic AI research.

NeSy4VRD has two components: the NeSy4VRD dataset package, and the NeSy4VRD GitHub repository.

The **NeSy4VRD dataset package** contains *(i)* the NeSy4VRD dataset, consisting of an image dataset with high-quality visual relationship annotations, and *(ii)* the NeSy4VRD OWL ontology, **VRD-World**.  VRD-World is a custom-designed, closely-aligned, common sense, companion ontology to the NeSy4VRD dataset.  It describes the world reflected in the visual relationships annotated for the NeSy4VRD images (their object classes and predicates) in a common sense way. The close alignment of VRD-World OWL ontology with the NeSy4VRD dataset ensures that the results of symbolic OWL reasoning are pertinent to the deep (subsymbolic) learning task at hand: visual relationship detection.
The NeSy4VRD dataset package is available on the open-access digital repository [Zenodo](https://doi.org/10.5281/zenodo.7916355).

The **NeSy4VRD GitHub repository** hosts *(i)* code and related infrastructure that supports extensibility of NeSy4VRD, *(ii)* extensive documentation describing how to use the extensibility support features made available, and *(iii)* sample code.
The NeSy4VRD GitHub repository is on [GitHub](https://github.com/djherron/NeSy4VRD).



