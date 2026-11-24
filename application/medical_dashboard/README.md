# MedCLIPSeg++ dashboard

See the [application guide](../README.md) for setup, the gallery contract, and embedding preparation.

From the repository root, start the React frontend and FastAPI backend on the loopback interface with the launcher below.

```bash
bash application/medical_dashboard/run_local.sh
```

The [dashboard](http://127.0.0.1:5173) accepts one prompt and up to 20 image uploads. An exact gallery match returns its precomputed result. If the request includes retrieval and the matched case passes the uncertainty policy, the response also includes similar accepted cases. Results are request-scoped.

## Verify

```bash
npm --prefix application/medical_dashboard/frontend run build
.venv/bin/python -m unittest discover -s application/medical_dashboard/backend/tests -v
```

The [API documentation](http://127.0.0.1:8000/docs) describes available endpoints and request schemas.
