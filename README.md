# MedCLIPSeg++

MedCLIPSeg++ combines text-guided breast ultrasound segmentation, uncertainty-based quality triage, and retrieval of comparable reference cases. Its research contribution is **Stage 4**, a post-hoc selective prediction policy that turns dense predictive entropy from a frozen MedCLIPSeg model into an image-level decision.

| Start here | Contents |
| --- | --- |
| [Research](research/README.md) | Six-stage map, notebooks, experiment protocol, validation selection, locked BUSI test, and external-cohort results |
| [Application](application/README.md) | Run the React and FastAPI dashboard against your own precomputed case collection, or prepare and reuse a retrieval index |
| [Architecture](docs/architecture/SYSTEM_ARCHITECTURE.md) | System interfaces and the [editable six-stage diagram](docs/architecture/figures/block_diagram/complete_system_architecture.drawio) |
| [Attribution](ATTRIBUTION.md) | Upstream MedCLIPSeg, UniMedCLIP, PABP, and dataset sources |

The web dashboard runs locally and evaluates catalogued images using precomputed segmentation artifacts. An uploaded image without an exact match in the configured collection is reported as **Not Evaluated**. This system is a research prototype for evaluating segmentation quality and retrieval.
