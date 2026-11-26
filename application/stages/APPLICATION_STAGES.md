# Application stages

Location `application/stages/` · [Up to application/](../APPLICATION_GUIDE.md)

Follow an image from the dashboard request to the displayed result. The web path is **React → FastAPI → shared application services**. The Python interfaces next to these documents also support the [six-stage notebook](../../research/notebooks/01_complete_system_pipeline.ipynb), where the segmentation stage uses synthetic outputs. Each guide identifies the configuration and checks that apply to the running application.

[__init__.py](__init__.py) exposes the stage interfaces through one import path. Open the relevant stage guide to find the symbols and code used by the walkthrough.

| Stage | What it does | Where to go for details |
| --- | --- | --- |
| [1 · Receive inputs](stage1_inputs/APPLICATION_INPUTS.md) | Read a scan and the user request, then match the scan to the configured gallery. | File limits, decoded-image matching, gallery fields and input checks. |
| [2 · Route the request](stage2_intent_router/INTENT_ROUTER.md) | Recognize segmentation or retrieval intent and extract the requested result count. | Implementation credit, methodological references, supported requests and tests. |
| [3 · Load segmentation](stage3_segmentation_backbone/SEGMENTATION_ARTIFACTS.md) | Retrieve precomputed outputs for a matched scan. | Offline model settings, stored artifact types and notebook adapter. |
| [4 · Apply the review rule](stage4_selective_prediction/RUNTIME_TRIAGE.md) | Use the applicable foreground or empty-mask score to decide if the query is eligible. | Frozen cutoffs, output states and policy checks. |
| [5 · Find similar cases](stage5_similarity_retrieval/SIMILARITY_SEARCH.md) | Search eligible gallery embeddings when the request and triage allow it. | Embedding contract, IVF/PABP settings and search tests. |
| [6 · Present the result](stage6_system_output/DASHBOARD_OUTPUTS.md) | Return the source scan, overlay, decision and any reference matches. | Response fields, display geometry and rendering checks. |

[Install and launch](../APPLICATION_GUIDE.md#install-and-launch) · [System architecture](../../docs/architecture/SYSTEM_ARCHITECTURE.md) · [Sources by stage](../../ATTRIBUTION.md#sources-by-stage) · [Research guide](../../research/RESEARCH_GUIDE.md)
