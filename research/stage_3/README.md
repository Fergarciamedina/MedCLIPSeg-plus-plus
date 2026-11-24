# Stage 3 · Segmentation

The backbone is MedCLIPSeg by Koleilat et al. (CVPR 2026), using the UniMedCLIP ViT-B/16 variant. The frozen backbone supplies a binary mask and pixelwise predictive entropy for each input. Stage 4 uses these two artifacts for all seven candidate scores. See the [MedCLIPSeg paper](https://arxiv.org/abs/2602.20423), [upstream implementation](https://github.com/HealthX-Lab/MedCLIPSeg), and [checkpoint documentation](https://huggingface.co/TahaKoleilat/MedCLIPSeg).

See [attribution and local adaptations](../../ATTRIBUTION.md) and the [experiment's checkpoint provenance](../stage_4/EXPERIMENTAL_PROTOCOL.md).
