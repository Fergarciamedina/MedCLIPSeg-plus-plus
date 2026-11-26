# Bundled OpenCLIP components

Location `shared/medclipseg/open_clip_lib/` · [Up to medclipseg/](../BACKBONE_SOURCE.md)

This folder contains the model construction, tokenization and preprocessing infrastructure referenced by the [UniMedCLIP segmentation builder](../trainers/medclipseg_unimedclip.py). The repository preserves additional upstream variants for their checkpoint definitions.

| File or folder | Purpose |
| --- | --- |
| [__init__.py](__init__.py) · [factory.py](factory.py) | Expose the package interface and construct models and transforms from a registered configuration. |
| [model.py](model.py) · [transformer.py](transformer.py) | Define image and text towers and their transformer components. |
| [hf_model.py](hf_model.py) · [hf_configs.py](hf_configs.py) | Adapt Hugging Face text models and define their architecture and pooling metadata. |
| [timm_model.py](timm_model.py) | Adapts timm vision backbones for the CLIP image tower. |
| [openai.py](openai.py) · [pretrained.py](pretrained.py) | Load OpenAI-compatible weights and resolve pretrained model metadata and downloads. |
| [tokenizer.py](tokenizer.py) · [BPE vocabulary](bpe_simple_vocab_16e6.txt.gz) | Provide byte-pair and Hugging Face tokenization options and the merge table. |
| [transform.py](transform.py) · [pos_embed.py](pos_embed.py) | Define image preprocessing and positional embedding operations. |
| [loss.py](loss.py) | Contains contrastive training losses kept with the upstream model code. |
| [utils.py](utils.py) · [version.py](version.py) | Provide shared utilities and the bundled package version. |
| [model_configs/](model_configs/ENCODER_PROFILES.md) | JSON architecture definitions read by the model factory. |

[Backbone source map](../BACKBONE_SOURCE.md) · [UniMedCLIP and MedCLIPSeg credit](../../../ATTRIBUTION.md)
