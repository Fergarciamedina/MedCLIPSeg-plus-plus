# Application

The React dashboard and FastAPI backend take an image and a text request, look up precomputed segmentation artifacts, apply the published uncertainty policy and return eligible similar cases. For a new collection, export a CSV catalog from your data source and supply its image and model artifacts. The application searches the resulting gallery with an in-memory FAISS/PABP index.

## Run with your own collection

Use Python 3.9 or newer, Node.js 20 or newer, npm, Bash and curl. The browser and API run on your machine. From the repository root, install the Python and frontend dependencies with these commands.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r application/medical_dashboard/backend/requirements.txt
npm --prefix application/medical_dashboard/frontend ci
```

Export your case collection as [a CSV catalog](#gallery-catalog). If your database stores image blobs or artifact URLs, materialize them as files and point the CSV to them. The same contract works for a filesystem, SQL export or object-store snapshot. Expose only the case fields listed below and use identifiers suitable for display. Use file paths readable by the backend process.

```bash
export MEDCLIPSEG_GALLERY=/absolute/path/to/gallery.csv
export MEDCLIPSEG_EMBEDDINGS=/absolute/path/to/gallery_embeddings.npz
bash application/medical_dashboard/run_local.sh
```

Open [the local dashboard](http://127.0.0.1:5173). Upload a PNG or JPEG from the catalog and enter, for example, `Segment and find 3 similar cases`. A matching scan gets its stored mask and triage decision. Retrieval requires both a retrieval request and an accepted query, and excludes the query itself. An unmatched scan shows **Not Evaluated**. Model inference for new scans runs offline before onboarding them into the catalog.

[API documentation](http://127.0.0.1:8000/docs) lists the HTTP operations. [Gallery summary](http://127.0.0.1:8000/api/summary) reports the number of loaded cases and their selective statuses and surfaces initialization errors. The [command-line interface](uncertainty_aware_retrieval/cli.py) also accepts the same environment settings.

## Gallery catalog

If your source is a SQLite database with file paths and matching column names, export it with the following query. Adapt the table name and column aliases to your schema.

```bash
sqlite3 -readonly -csv -header /path/to/cases.db \
  'SELECT dataset, image_id, image_path, seg_mask_path, unc_npy_path, u_fg, u_mean
   FROM cases ORDER BY dataset, image_id;' > /path/to/gallery.csv
```

Use UTF-8 CSV, one row per case. Each path may be absolute or relative to the CSV file. Quote values that contain commas. The complete header and two illustrative rows are below.

```csv
dataset,image_id,image_path,seg_mask_path,unc_npy_path,unc_png_path,u_fg,u_mean,dsc,nsd
site-a,case-01,images/case-01.png,masks/case-01.png,entropy/case-01.npy,,,,,
site-a,case-02,images/case-02.jpg,masks/case-02.png,,,0.042,0.018,,
```

| Column | Type and meaning |
| --- | --- |
| `dataset`, `image_id` | Required strings. The pair uniquely identifies a case. Use distinct IDs within a dataset and keep both fields free of `@`, slashes and backslashes. Use descriptive identifiers instead of `.` or `..`. |
| `image_path` | Required PNG or JPEG. Web uploads match this image by decoded RGB pixels. Resizing, cropping, or lossy re-encoding can prevent a match. |
| `seg_mask_path` | Required binary PNG aligned to the image, with foreground encoded as 1 or 255 and background as 0. A full-frame model-grid mask such as 224×224 is also accepted. The dashboard resizes the display mask to the original grid with nearest-neighbor interpolation. |
| `unc_npy_path` | Optional dense floating-point predictive entropy array in nats, with values in `[0, ln(2)]` and the same height and width as the mask. When present, the application computes any missing scores from it. |
| `unc_png_path` | Optional display image of the uncertainty map for clients that request it. |
| `u_fg`, `u_mean` | Optional if a raw entropy map supplies them. Use the mean predictive entropy over the predicted foreground and over the full image, respectively. Provide `u_fg` for a nonempty mask or `u_mean` for an empty mask. Leave undefined values blank. |
| `dsc`, `nsd` | Optional reference segmentation scores in `[0, 1]`. Ground-truth masks are needed only to compute these metrics offline. |

The app derives foreground size and triage status from these fields. Stage 4's frozen thresholds apply to the entropy output and preprocessing specified in the [experiment protocol](../research/stage_4/EXPERIMENTAL_PROTOCOL.md). Collections produced with different models or acquisition settings require a separate evaluation of that operating point.

To create masks and raw entropy for the supported MedCLIPSeg dataset format, follow [upstream dataset preparation](https://github.com/HealthX-Lab/MedCLIPSeg/blob/main/assets/DATASETS.md) and the [local inference runner](../shared/scripts/run_inference.sh). The relevant cohort YAMLs under [shared/configs](../shared/configs) select the model, image size, and Monte Carlo sample count. Set your raw-data and output directories, select the compute device in the YAML, and provide the corresponding MedCLIPSeg segmentation checkpoint and text prompts. The runner writes masks and raw entropy arrays to the chosen output tree. Export their paths into the CSV above when they are ready. The following commands prepare the original BUSI layout.

```bash
python -m pip install -r requirements.txt
export MEDCLIPSEG_DATA=/absolute/path/to/raw
export MEDCLIPSEG_OUTPUT=/absolute/path/to/outputs
# Place the original BUSI segmentation checkpoint at:
# $MEDCLIPSEG_OUTPUT/BUSI/trained_models/seed666/MedCLIPSeg_unimedclip_ViT-B-16_latest.pth
bash shared/scripts/run_inference.sh BUSI 666
```

The supplied evaluation runner reads prepared `Test_Folder/img` and `Test_Folder/label` directories plus prompts under `Prompts_Folder`. For a collection without annotations, export precomputed masks and entropy from your inference pipeline and use the same catalog contract. The dashboard itself does not require annotations.

## Embeddings and search index

**Supply precomputed vectors.** Point `MEDCLIPSEG_EMBEDDINGS` at a NumPy `.npz` archive containing a one-dimensional Unicode `keys` array and a floating-point `embeddings` matrix with shape `(N, 512)`. The key for `case-01` from `site-a` is `case-01@site-a`. Supply every catalog key exactly once. The loader aligns the vectors by key and normalizes them for cosine similarity. All vectors must come from the same encoder and preprocessing configuration. This path needs no model weights.

```python
# Export rows from your embedding store as (image_id, dataset, vector) tuples.
import numpy as np

keys = np.asarray([f"{image_id}@{dataset}" for image_id, dataset, _ in rows], dtype=str)
vectors = np.stack([vector for _, _, vector in rows]).astype(np.float32)
np.savez("gallery_embeddings.npz", keys=keys, embeddings=vectors)
```

**Compute vectors locally.** Configure the [UniMedCLIP ViT-B/16 checkpoint](https://huggingface.co/UzairK/unimed-clip-vit-b16) and run the preparation command. This encoder checkpoint serves retrieval. Segmentation uses the full MedCLIPSeg checkpoint separately.

```bash
unset MEDCLIPSEG_EMBEDDINGS
export MEDCLIPSEG_CHECKPOINT=/absolute/path/to/unimed-clip-vit-b16.pt
export MEDCLIPSEG_RUNTIME=/absolute/path/to/writable/gallery-cache
# Enable downloads for first-time initialization if the BiomedBERT config is absent.
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0
.venv/bin/python application/uncertainty_aware_retrieval/cli.py --prepare
bash application/medical_dashboard/run_local.sh
```

Preparation validates the catalog and computes or reuses a keyed `gallery_embeddings-*.npz` file in `MEDCLIPSEG_RUNTIME`. The default runtime directory is `application/uncertainty_aware_retrieval/runtime/`. Subsequent starts reuse matching vectors. The API builds the in-memory search index on the first gallery request. After catalog updates, refresh externally supplied vectors when case membership or images change, and restart the service so it reloads the catalog and rebuilds its index.

For the original benchmark layout, unset `MEDCLIPSEG_GALLERY`. By default the app looks in the sibling `../data/raw` and `../data/outputs` folders. Set `MEDCLIPSEG_DATA` to the raw tree or its parent and `MEDCLIPSEG_OUTPUT` to the inference output tree to use another location. This path expects the five evaluated cohorts, their Stage 4 aggregation CSVs and corresponding images and masks at the original [experiment paths](../research/stage_4/EXPERIMENTAL_PROTOCOL.md).

## Layout

- [medical_dashboard/frontend](medical_dashboard/frontend) hosts the React interface.
- [medical_dashboard/backend](medical_dashboard/backend) hosts the FastAPI service.
- [uncertainty_aware_retrieval](uncertainty_aware_retrieval) provides the shared gallery, triage and retrieval services, a CLI, and an alternate Streamlit interface.
- [stages](stages) contains the executable components used by the six-stage walkthrough.
