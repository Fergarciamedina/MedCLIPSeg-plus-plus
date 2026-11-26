# Stage 1 · Application inputs

Location `application/stages/stage1_inputs/` · [Up to stages/](../APPLICATION_STAGES.md)

The dashboard receives a **breast ultrasound image** and a **user instruction**. The image identifies a case in the configured collection. The instruction describes which operation to perform, for example `find 3 similar cases`. The gallery provides the saved mask and uncertainty artifacts required to process the request.

## Notebook files in this folder

[input_loader.py](input_loader.py) reads an image path, Pillow image or NumPy array and packages the resized image, original dimensions and request text for the walkthrough. Its default grid is 256 × 256 with bilinear resampling. [__init__.py](__init__.py) exposes the loader and its input record. The [dashboard API](../../medical_dashboard/backend/API_BACKEND.md) owns the web upload flow described below.

## Dashboard input contract

| Input | Configuration and checks |
| --- | --- |
| Upload | A request accepts 1–20 PNG or JPEG files, each up to 20 MiB. FastAPI checks the declared image media type, payload size and readability before processing the image with Pillow. |
| Instruction | One text string for the request batch. The backend sends it to the [Stage 2 intent router](../stage2_intent_router/INTENT_ROUTER.md). The original MedCLIPSeg lesion description comes from the separate offline workflow. |
| Gallery | A CSV keyed by `dataset` and `image_id`, with the original image, a binary mask, and raw entropy or the applicable precomputed score. In addition, retrieval uses a keyed vector archive or embeddings prepared by the application. The custom loader checks case-key uniqueness, file existence, formats, and array and score validity. |
| Benchmark layout | The default collection contains 542 test images from five breast-ultrasound cohorts. Set `MEDCLIPSEG_GALLERY` to use a custom CSV. For the original directory layout, `MEDCLIPSEG_DATA` and `MEDCLIPSEG_OUTPUT` configure the raw and generated data roots. |

The API matches files using a hash of the original decoded RGB raster and its dimensions. A renamed file or a lossless re-encoding can still match, and edits to the raster can change the match. A scan absent from the gallery is reported as **Not Evaluated**. The notebook's [input loader](input_loader.py) uses a 256 × 256 display-oriented input container for its walkthrough. The API matches the unresized image, and the [offline model](../stage3_segmentation_backbone/SEGMENTATION_ARTIFACTS.md) uses its own 224 × 224 grid.

The catalog uses a stable case key to pair image and model artifacts before the search index is built. The upload matcher instead uses dimensions and decoded RGB values to compare image identity across filenames and encoding containers. This avoids perceptual matches that could attach a stored mask to a different image. The digest mechanism is SHA-256; the choice of decoded input and matching policy is local to the project. The file and batch limits are application defaults for request handling. [CSV format reference](https://www.rfc-editor.org/rfc/rfc4180) · [NIST SHA-256 specification](https://csrc.nist.gov/pubs/fips/180-4/upd1/final) · [Upload matcher and limits](../../medical_dashboard/backend/main.py).

The [manifest tests](../../uncertainty_aware_retrieval/tests/test_manifest.py) check the 542-case catalog, stored paths and exclusion of demo queries from their reference set. The [API tests](../../medical_dashboard/backend/tests/test_api.py) exercise exact-content matching and a custom gallery. The [620-case study input audit](../../../research/stage_1/STUDY_INPUTS.md#what-was-validated-on-the-inputs) covers the separate research data.

[Application stage map](../APPLICATION_STAGES.md) · [Gallery schema](../../APPLICATION_GUIDE.md#gallery-catalog) · [Next · Intent routing](../stage2_intent_router/INTENT_ROUTER.md)
