# Encoder architecture profiles

Location `shared/medclipseg/open_clip_lib/model_configs/` · [Up to open_clip_lib/](../OPEN_CLIP_COMPONENTS.md)

The [OpenCLIP factory](../factory.py) discovers these JSON records by model name. The profiles supply feature dimensions and vision and text tower settings, while compatible checkpoints supply weights. Groups below distinguish the encoder architectures available in the bundled source. The breast ultrasound study uses [frozen UniMedCLIP ViT-B/16](../../../../research/stage_3_medclipseg/FROZEN_SEGMENTATION.md).

| Encoder family | Profiles |
| --- | --- |
| ResNet | [RN101-quickgelu](RN101-quickgelu.json) · [RN101](RN101.json) · [RN50-quickgelu](RN50-quickgelu.json) · [RN50](RN50.json) · [RN50x16](RN50x16.json) · [RN50x4](RN50x4.json) |
| ViT Base | [ViT-B-16-plus-240](ViT-B-16-plus-240.json) · [ViT-B-16-plus](ViT-B-16-plus.json) · [ViT-B-16-quickgelu](ViT-B-16-quickgelu.json) · [ViT-B-16](ViT-B-16.json) · [ViT-B-32-plus-256](ViT-B-32-plus-256.json) · [ViT-B-32-quickgelu](ViT-B-32-quickgelu.json) · [ViT-B-32](ViT-B-32.json) |
| ViT Large | [ViT-L-14-280](ViT-L-14-280.json) · [ViT-L-14-336-quickgelu](ViT-L-14-336-quickgelu.json) · [ViT-L-14-336](ViT-L-14-336.json) · [ViT-L-14-quickgelu](ViT-L-14-quickgelu.json) · [ViT-L-14](ViT-L-14.json) · [ViT-L-16-320](ViT-L-16-320.json) · [ViT-L-16](ViT-L-16.json) |
| ViT Huge and giant | [ViT-H-14-quickgelu](ViT-H-14-quickgelu.json) · [ViT-H-14](ViT-H-14.json) · [ViT-H-16](ViT-H-16.json) · [ViT-bigG-14-quickgelu](ViT-bigG-14-quickgelu.json) · [ViT-g-14](ViT-g-14.json) |
| timm-backed encoders | [timm-efficientnetv2_rw_s](timm-efficientnetv2_rw_s.json) · [timm-resnet50d](timm-resnet50d.json) · [timm-resnetaa50d](timm-resnetaa50d.json) · [timm-resnetblur50](timm-resnetblur50.json) · [timm-swin_base_patch4_window7_224](timm-swin_base_patch4_window7_224.json) · [timm-vit_base_patch16_224](timm-vit_base_patch16_224.json) · [timm-vit_base_patch32_224](timm-vit_base_patch32_224.json) · [timm-vit_small_patch16_224](timm-vit_small_patch16_224.json) |

[OpenCLIP components](../OPEN_CLIP_COMPONENTS.md)
