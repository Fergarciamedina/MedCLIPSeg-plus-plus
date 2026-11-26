# System architecture

Location `docs/architecture/` · [Up to docs/](../DOCUMENTATION_MAP.md)

MedCLIPSeg++ connects a breast ultrasound image and a request to a segmentation overlay, a review decision and, when eligible, similar reference cases. The six-stage diagram describes the overall system. The web application serves precomputed segmentation artifacts. The [system design rationale](DESIGN_RATIONALE.md) explains the interfaces and engineering choices.

![Six-stage MedCLIPSeg++ architecture](figures/block_diagram/complete_system_architecture.png)

[Editable diagram](figures/block_diagram/complete_system_architecture.drawio) · [PDF](figures/block_diagram/complete_system_architecture.pdf) · [SVG](figures/block_diagram/complete_system_architecture.svg)

## Follow a case through the system

| Stage | Input | Output |
| --- | --- | --- |
| 1 · Inputs | Ultrasound image and text request. Offline model runs also use a lesion-description prompt. | Decoded image and the information needed to route the request. |
| 2 · Intent routing | Request text and optional result count. | Segmentation, retrieval or combined intent, plus the requested number of neighbors. The router also reports ambiguous requests. |
| 3 · Segmentation | Image and offline lesion-description prompt. | Frozen MedCLIPSeg prediction and a pixelwise entropy map. The dashboard looks up saved outputs by decoded image content. |
| 4 · Selective prediction | Binary predicted mask and predictive entropy, or their catalogued scores. | Accepted by policy, needs review, or not evaluated, with the relevant score and cutoff when available. |
| 5 · Similarity retrieval | Retrieval intent, Stage 4 status, query embedding and indexed reference embeddings. | Up to the requested number of similar, eligible references, with the query itself excluded. |
| 6 · Output | Source image, mask, policy outcome and optional reference matches. | Dashboard response with an overlay, status and case matches. |

The [application stage guides](../../application/stages/APPLICATION_STAGES.md) explain the settings and software checks for each stage. The [research stage guides](../../research/RESEARCH_GUIDE.md#explore-the-six-stages) follow the experiment from its input audit to its results. For the mathematical rule and evidence, see the [selective prediction guide](SELECTIVE_SEGMENTATION_ARCHITECTURE.md). For cosine similarity, filtered vector search and the limits of approximate lookup, see the [retrieval rationale](DESIGN_RATIONALE.md#vector-search-and-predicate-aware-block-pruning).

## How the repository runs it

The dashboard is **React → FastAPI → shared application services**. React renders the current request and case gallery. FastAPI validates input, matches scans and calls the catalog, triage and retrieval services. The same services are available through the CLI and Streamlit interface. The six-stage notebook calls the stage interfaces explicitly so readers can inspect intermediate results. Its segmentation adapter generates synthetic outputs for the walkthrough.

A dashboard upload can reuse a stored result when the decoded image matches a catalog entry. A new scan without stored outputs is marked **Not Evaluated**. Accepted query cases can retrieve accepted reference cases. The index is built in memory from the catalog and keyed embeddings. See the [application guide](../../application/APPLICATION_GUIDE.md) to configure data and launch the dashboard.

## Where to look next

| Reader task | Reference |
| --- | --- |
| Prepare an image catalog and embeddings | [Application setup](../../application/APPLICATION_GUIDE.md) |
| Inspect each stage in Python | [Six-stage software notebook](../../research/notebooks/01_complete_system_pipeline.ipynb) |
| Review the uncertainty experiment | [Stage 4 overview and reproduction](../../research/stage_4/SELECTIVE_PREDICTION_STUDY.md) |
| Find editable drawings and exports | [figures/](figures/ARCHITECTURE_FIGURES.md) |
| Study the visual specifications | [Application flowchart](figures/flowcharts/application_operational_flowchart.pdf), [selective decision flowchart](figures/flowcharts/selective_decision_flowchart.pdf), and [PABP retrieval diagram](figures/block_diagram/pabp_accepted_case_retrieval.pdf). |
