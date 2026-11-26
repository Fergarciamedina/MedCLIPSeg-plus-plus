# Bundled BiomedCLIP components

Location `shared/medclipseg/biomedclip/` · [Up to medclipseg/](../BACKBONE_SOURCE.md)

These modules support the MedCLIPSeg BiomedCLIP variant. The shared project structure retains them with the source architecture variants. The reported breast ultrasound study used UniMedCLIP ViT-B/16. [Backbone source map](../BACKBONE_SOURCE.md).

| File or folder | Purpose |
| --- | --- |
| [biomedclip.py](biomedclip.py) | Defines the BiomedCLIP model wrapper. |
| [text_modules.py](text_modules.py) | Provides BERT text encoder blocks. |
| [text_module_utils.py](text_module_utils.py) | Provides chunking and attention-head pruning helpers. |
| [transformer.py](transformer.py) | Implements attention, feed-forward and transformer blocks. |
| [vision_modules.py](vision_modules.py) | Defines vision attention and transformer blocks. |
| [layers/](layers/VISION_LAYERS.md) | Bundled vision operators, layer constructors and helper functions. |
