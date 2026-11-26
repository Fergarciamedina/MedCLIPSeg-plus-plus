# Dashboard API tests

Location `application/medical_dashboard/backend/tests/` · [Up to backend/](../API_BACKEND.md)

[test_api.py](test_api.py) checks the HTTP interface through FastAPI's test client. It exercises upload matching, unknown uploads, mask alignment, gallery validation and association of vectors by case key. Temporary fixtures make the custom-gallery cases independent of a user's own collection.

From the repository root, run:

```bash
.venv/bin/python -m unittest discover -s application/medical_dashboard/backend/tests -v
```

See the [API backend](../API_BACKEND.md) for the code behind these requests.
