# Bundled CLIP components

Location `shared/medclipseg/clip/` · [Up to medclipseg/](../BACKBONE_SOURCE.md)

This folder supports the upstream MedCLIPSeg variants based on the CLIP implementation and its checkpoint conventions. Consult the [backbone source map](../BACKBONE_SOURCE.md) to find which variants use each encoder.

| File | Purpose |
| --- | --- |
| [__init__.py](__init__.py) | Exposes the CLIP loading and tokenization functions. |
| [clip.py](clip.py) | Loads checkpoints and supplies image transforms and text tokenization. |
| [model.py](model.py) | Defines the model architecture and helpers for rebuilding it from checkpoint weights. |
| [interpolate.py](interpolate.py) | Resizes positional embeddings when the vision token grid differs from the checkpoint. |
| [simple_tokenizer.py](simple_tokenizer.py) · [BPE vocabulary](bpe_simple_vocab_16e6.txt.gz) | Implements byte-pair tokenization and provides its merge vocabulary. |
