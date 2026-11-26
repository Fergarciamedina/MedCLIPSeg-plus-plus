# System and component block diagrams

Location `docs/architecture/figures/block_diagram/` · [Up to figures/](../ARCHITECTURE_FIGURES.md)

Use these figures to inspect component boundaries and the data passed between them. The [system architecture guide](../../SYSTEM_ARCHITECTURE.md) walks through each stage.

| Diagram | What it shows | Files |
| --- | --- | --- |
| Complete system | The six-stage architecture from input to system output. This is the editable source referenced by the repository overview. | [Draw.io](complete_system_architecture.drawio) · [PDF](complete_system_architecture.pdf) · [SVG](complete_system_architecture.svg) · [PNG](complete_system_architecture.png) · [Page PNG export](complete_system_architecture-Editable%20six-stage%20architecture.drawio.png) |
| Scientific functional architecture | An alternate editable view of the system. | [Draw.io](complete_system_architecture_VvvV.drawio) |
| Dual embedding modes | How precomputed and locally produced embeddings enter the retrieval path. | [Draw.io](dual_embedding_execution_modes.drawio) |
| Accepted-case retrieval | How the eligibility filter and PABP index determine which candidates are compared. | [Draw.io](pabp_accepted_case_retrieval.drawio) · [PDF](pabp_accepted_case_retrieval.pdf) · [SVG](pabp_accepted_case_retrieval.svg) |
| Selective segmentation | The path from the pretrained model's mask and uncertainty to the review rule. | [Draw.io](selective_segmentation_architecture.drawio) · [PDF](selective_segmentation_architecture.pdf) · [SVG](selective_segmentation_architecture.svg) |

[Figure collection](../ARCHITECTURE_FIGURES.md)
