# MedCLIPSeg++ dashboard

Location `application/medical_dashboard/` · [Up to application/](../APPLICATION_GUIDE.md)

See the [application guide](../APPLICATION_GUIDE.md) for setup, the gallery contract, and embedding preparation. The [six-stage guide](../stages/APPLICATION_STAGES.md) traces what happens after a user submits an image and an instruction.

See the [dashboard preview](../../README.md#see-the-dashboard) for the gallery layout with synthetic cases.

From the repository root, install and serve the React build and API together on the loopback interface. Use `--setup` for the first launch.

```bash
bash application/medical_dashboard/run_local.sh --setup \
  --gallery "/path/to/gallery.csv" --embeddings "/path/to/gallery_embeddings.npz"
```

The [dashboard](http://127.0.0.1:5173) accepts one prompt and up to 20 image uploads. An exact gallery match returns its precomputed result. If the request includes retrieval and the matched case passes the uncertainty policy, the response also includes similar accepted cases. Results are request-scoped.

| File or folder | Role |
| --- | --- |
| [run_local.sh](run_local.sh) | Installs and builds on request, then serves the dashboard and API from one local port. |
| [backend/](backend/API_BACKEND.md) | FastAPI endpoints, uploaded-image matching, result assembly and API tests. |
| [frontend/](frontend/FRONTEND_BUILD.md) | React source and the npm, TypeScript and Vite build configuration. |

## Verify

```bash
npm --prefix application/medical_dashboard/frontend run build
.venv/bin/python -m unittest discover -s application/medical_dashboard/backend/tests -v
```

The [API documentation](http://127.0.0.1:5173/docs) describes available endpoints and request schemas.
