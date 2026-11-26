# Run the dashboard

Location `application/` · [Up to repository root](../README.md)

The dashboard lets you upload a breast ultrasound image, review its precomputed segmentation and uncertainty decision, and request similar reference cases. It uses a local React frontend, FastAPI backend and an in-memory FAISS/PABP search index. To use your own collection, supply a case catalog and its saved model outputs. The [study overview](../research/stage_4_selective_prediction/SELECTIVE_PREDICTION_STUDY.md) explains where the review rule comes from.

[Setup](#install-and-launch) · [Gallery inputs](#gallery-catalog) · [Embeddings](#embeddings-and-search-index) · [Six-stage walkthrough](stages/APPLICATION_STAGES.md) · [Software verification](#verification)

See the [dashboard preview](../README.md#see-the-dashboard) or follow the [interactive walkthrough](../docs/learning/LEARNING_GUIDE.md) from the uploaded image to the final response.

## How a request is processed

You provide an ultrasound image and an instruction. The instruction chooses segmentation, retrieval or both. The API looks up a matching case and its precomputed outputs, then applies the uncertainty policy. If the query passes and retrieval was requested, the backend searches the eligible gallery entries. The dashboard presents the source image, overlay, decision and any matching references. The [six-stage guide](stages/APPLICATION_STAGES.md) expands each step into inputs, configuration, rationale and checks.

The text box controls the running application. Offline MedCLIPSeg inference uses the lesion descriptions recorded with the study data to produce the stored masks. The [Stage 1 study guide](../research/stage_1_study_inputs/STUDY_INPUTS.md) describes the cohorts, validation and test splits, and both kinds of text.

## Install and launch

Use Python 3.9 or newer, Bash, and Node.js 20 or newer with npm for the first build. Setup needs access to a Python and npm package registry. With precomputed vectors, subsequent runs work offline. Run from the repository root. Prepare a [catalog CSV](#gallery-catalog) and either a [keyed embedding archive or the UniMedCLIP checkpoint](#embeddings-and-search-index).

```bash
bash application/medical_dashboard/run_local.sh --setup \
  --gallery "/absolute/path/to/gallery.csv" \
  --embeddings "/absolute/path/to/gallery_embeddings.npz"
```

The launcher creates `.venv`, installs the [backend requirements](medical_dashboard/backend/requirements.txt), builds the frontend from its npm lockfile, and starts one loopback service. The precomputed path uses neither PyTorch nor model weights. For later runs, use the same command without `--setup`. The command also accepts the [equivalent environment variables](../.env.example) and `--port` to change the default port. Use `--help` for syntax.

Open [the local dashboard](http://127.0.0.1:5173), upload a file from the catalog and enter `Segment and find 3 similar cases`. The application returns a stored mask and triage decision for a matching scan. Retrieval requires an accepted query and excludes that query from its references. Use [the local API documentation](http://127.0.0.1:5173/docs) for request schemas and [the gallery summary](http://127.0.0.1:5173/api/summary) to confirm that the configured cases loaded.

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

The app derives foreground size and triage status from these fields. Stage 4's frozen thresholds apply to the entropy output and preprocessing specified in the [experiment protocol](../research/stage_4_selective_prediction/study_reference/EXPERIMENTAL_PROTOCOL.md). Collections produced with different models or acquisition settings require a separate evaluation of that operating point.

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

**Compute vectors locally.** Download the [UniMedCLIP ViT-B/16 checkpoint](https://huggingface.co/UzairK/unimed-clip-vit-b16). With `--checkpoint`, setup adds the [encoder requirements](medical_dashboard/backend/requirements-encoder.txt). The first access to the gallery computes and caches vectors. Use the `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` settings below for first-time initialization if the BiomedBERT configuration is absent from the local cache.

```bash
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 \
  bash application/medical_dashboard/run_local.sh --setup \
    --gallery "/absolute/path/to/gallery.csv" \
    --checkpoint "/absolute/path/to/unimed-clip-vit-b16.pt"
```

To warm the cache separately, export `MEDCLIPSEG_GALLERY` and `MEDCLIPSEG_CHECKPOINT` and run `.venv/bin/python application/uncertainty_aware_retrieval/cli.py --prepare`. The archive is stored in `MEDCLIPSEG_RUNTIME` or the default service runtime directory. The application builds its IVF/PABP index in memory. Restart after replacing the catalog or embeddings so the service reloads the assets and rebuilds the index. Keep all vectors in the same embedding space.

For the original benchmark layout, unset `MEDCLIPSEG_GALLERY`. By default the app looks in the sibling `../data/raw` and `../data/outputs` folders. Set `MEDCLIPSEG_DATA` to the raw tree or its parent and `MEDCLIPSEG_OUTPUT` to the inference output tree to use another location. This path expects the five evaluated cohorts, their Stage 4 aggregation CSVs and corresponding images and masks at the original [experiment paths](../research/stage_4_selective_prediction/study_reference/EXPERIMENTAL_PROTOCOL.md).

## Layout

The web path is **React → FastAPI → shared application services**. The notebook uses the six-stage interfaces to make intermediate outputs visible.

| Directory | What it does | Why it is separate |
| --- | --- | --- |
| [`medical_dashboard/`](medical_dashboard/DASHBOARD_GUIDE.md) | The [React frontend](medical_dashboard/frontend/FRONTEND_BUILD.md) and the [FastAPI backend](medical_dashboard/backend/API_BACKEND.md). | The browser presents results. The Python API owns gallery files and the search index and can be tested independently of the interface. |
| [`uncertainty_aware_retrieval/`](uncertainty_aware_retrieval/RETRIEVAL_SERVICES.md) | Case loading, precomputed masks, triage, embeddings and PABP search. Also contains a CLI for preparing and testing the gallery. | Shared services keep the case, score, and search behavior consistent across entry points. |
| [`stages/`](stages/APPLICATION_STAGES.md) | Six modular interfaces used by the [software walkthrough](../research/notebooks/01_complete_system_pipeline.ipynb). | The notebook can inspect each stage and demonstrate it with a synthetic segmentation adapter. The application consumes precomputed model results through its services. |

The [system design rationale](../docs/architecture/DESIGN_RATIONALE.md) covers the component boundaries, numerical contracts, and vector index. The [experimental rationale](../research/stage_4_selective_prediction/study_reference/EXPERIMENTAL_RATIONALE.md) covers the cohorts, hardware and Stage 4 methods.

## Verification

The web API tests use synthetic image and gallery fixtures. The full shared service suite also checks the original 542-case catalog and expects the locally prepared benchmark artifacts.

```bash
.venv/bin/python -m unittest discover -s application/medical_dashboard/backend/tests -v
.venv/bin/python -m unittest discover -s application/uncertainty_aware_retrieval/tests -v
```

For the statistical evaluation and its separate 31-check regression suite, see the [Stage 4 validation guide](../research/stage_4_selective_prediction/SELECTIVE_PREDICTION_STUDY.md#validation-and-test-design).
