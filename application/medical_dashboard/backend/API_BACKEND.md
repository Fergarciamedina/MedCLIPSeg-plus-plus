# Dashboard API backend

Location `application/medical_dashboard/backend/` · [Up to medical_dashboard/](../DASHBOARD_GUIDE.md)

This folder exposes the web workflow as an HTTP API. The [shared services](../../uncertainty_aware_retrieval/RETRIEVAL_SERVICES.md) own gallery parsing, embeddings, triage and search.

| File or folder | Role |
| --- | --- |
| [main.py](main.py) | FastAPI entry point. Initializes the gallery and index on demand, resolves uploads to stored cases, and combines routing, triage and search into the response. It also exposes health, summary, case and image endpoints. |
| [requirements.txt](requirements.txt) · [requirements-encoder.txt](requirements-encoder.txt) | Base API requirements and the optional local embedding encoder. |
| [tests/](tests/API_TESTS.md) | End-to-end API contract checks using gallery fixtures and synthetic uploads. |

Use the [dashboard launcher](../DASHBOARD_GUIDE.md) to start the API with the frontend. A running API exposes its schemas at [local OpenAPI docs](http://127.0.0.1:5173/docs).
