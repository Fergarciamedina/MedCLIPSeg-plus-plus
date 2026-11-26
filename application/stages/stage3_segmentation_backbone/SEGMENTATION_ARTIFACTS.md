# Stage 3 · Load the segmentation result

Location `application/stages/stage3_segmentation_backbone/` · [Up to stages/](../APPLICATION_STAGES.md)

The dashboard [segmentation service](../../uncertainty_aware_retrieval/uncertainty_aware_retrieval/segmentation_service.py) looks up artifacts for a matching catalog entry. It supplies the predicted mask, paths for optional uncertainty displays, and score fields for Stage 4. Serving saved outputs keeps model inference out of interactive requests and lets a given scan reuse the same mask and entropy.

The study artifacts were prepared with the frozen [MedCLIPSeg](https://github.com/HealthX-Lab/MedCLIPSeg) UniMedCLIP ViT-B/16 model. Its reference settings are a **224 × 224** model grid, **30** stochastic predictions and a binary mask from mean foreground probability **above 0.5**. The input description comes from the offline dataset workbook. The output entropy is raw float32 in nats, aligned with the mask. A colorized uncertainty PNG is optional for viewing.

The segmentation method and released model weights belong to the original MedCLIPSeg project. The local study reuses the weights and input configuration and writes mask and entropy artifacts. The artifact lookup is application code. Precomputing the outputs allows the same uncertainty evidence to be used by every score and avoids repeated stochastic inference during web requests. [Koleilat et al., CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Koleilat_MedCLIPSeg_Probabilistic_Vision-Language_Adaptation_for_Data-Efficient_and_Generalizable_Medical_Image_CVPR_2026_paper.html) · [BUSI YAML](../../../shared/configs/BUSI.yaml) · [Offline inference runner](../../../shared/medclipseg/test.py).

The web dashboard reports an unmatched upload as **Not Evaluated**. To add a case, prepare its mask and entropy offline and register them in the gallery. The [portable gallery API tests](../../medical_dashboard/backend/tests/test_api.py) verify lookup and delivery of a stored result. The notebook's [synthetic adapter](model_adapter.py) demonstrates the interface without frozen weights. The [study input audit](../../../research/stage_1/STUDY_INPUTS.md#what-was-validated-on-the-inputs) verifies model-grid pairs in the research artifacts.

## Notebook files in this folder

[model_adapter.py](model_adapter.py) defines the mask-and-entropy contract and the synthetic implementation used to exercise downstream stages. Its live adapter is an integration stub that expects an inference interface beyond the current precomputed-artifact service. To generate real outputs, use the [offline commands](../../../shared/scripts/SEGMENTATION_COMMANDS.md). [__init__.py](__init__.py) exports the adapter classes for the walkthrough.

[Application stage map](../APPLICATION_STAGES.md) · [Prepare saved artifacts](../../APPLICATION_GUIDE.md#gallery-catalog) · [Research model settings](../../../research/stage_3/FROZEN_SEGMENTATION.md) · [Next · Selective prediction](../stage4_selective_prediction/RUNTIME_TRIAGE.md)
