# Stage 6 · Present the output

Location `application/stages/stage6_system_output/` · [Up to stages/](../APPLICATION_STAGES.md)

React presents the response assembled by the [FastAPI backend](../../medical_dashboard/backend/main.py). The interface keeps the original scan, mask overlay, quality-screening decision and similar cases distinct so the reader can see the source image and understand the basis for the result.

| Output | What the reader sees |
| --- | --- |
| Matched case | Original ultrasound and stored mask overlay. When available, the gallery can also show a saved uncertainty image. |
| Review decision | Policy state, reason, the applied cutoff and score when available. |
| Reference matches | Similar eligible cases and their similarity scores when retrieval ran. That score describes embedding similarity. |
| Unmatched upload | Preview and **Not Evaluated**, because the configured collection has no matching precomputed result. |

For an overlay, nearest-neighbor resizing places the binary mask on the original image grid without creating fractional label values. This assumes that the mask and source image cover the same frame. The API includes optional Dice and NSD fields when they are present in the catalog. Annotations are unnecessary to serve a result.

The dashboard layout and mask-overlay rendering are implemented in this repository. Keeping the original scan visible while presenting the decision up front allows a reader to inspect the result before opening detailed scores or reference cases. This progressive presentation follows the information-visualization principle of an overview with details available on demand. Nearest-neighbor display resizing selects an existing label value and preserves the binary label vocabulary. [Shneiderman, *The Eyes Have It*](https://doi.org/10.1109/VL.1996.545307) · [React dashboard](../../medical_dashboard/frontend/src/main.tsx) · [Mask response implementation](../../medical_dashboard/backend/main.py).

The [API tests](../../medical_dashboard/backend/tests/test_api.py) cover case lists, mask responses, aligned binary overlays and retrieval gating. The [visualization tests](../../uncertainty_aware_retrieval/tests/test_visualizer.py) exercise real and synthetic overlays and uncertainty display. The [research results](../../../research/stage_4/experiments/RESULTS_GUIDE.md) provide the separate cohort-level tables and figures.

## Notebook files in this folder

[visual_composer.py](visual_composer.py) assembles the walkthrough figure from scans, masks, entropy, decisions and retrieved neighbors. [triage_reporter.py](triage_reporter.py) turns case telemetry into a summary DataFrame. [__init__.py](__init__.py) exports these two presentation helpers.

[Application stage map](../APPLICATION_STAGES.md) · [Dashboard guide](../../medical_dashboard/DASHBOARD_GUIDE.md) · [Study outputs](../../../research/stage_6/STUDY_OUTPUTS.md)
