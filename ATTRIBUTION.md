# Attribution

- The frozen segmentation backbone and dataset configurations are adapted from [MedCLIPSeg](https://github.com/HealthX-Lab/MedCLIPSeg) by Taha Koleilat, Hojat Asgariandehkordi, Omid Nejati Manzari, Berardino Barile, Yiming Xiao, and Hassan Rivaz. The local copy lives in [shared/medclipseg](shared/medclipseg) and [shared/configs](shared/configs). Local adaptations include configurable paths, raw entropy export and evaluation runners. See the [MedCLIPSeg paper](https://arxiv.org/abs/2602.20423) and [upstream Apache-2.0 license](https://github.com/HealthX-Lab/MedCLIPSeg/blob/main/LICENSE). A [local copy of Apache-2.0](shared/LICENSE-MedCLIPSeg) is included.
- The visual embedding workflow uses [UniMedCLIP](https://github.com/mbzuai-oryx/UniMed-CLIP). Original model checkpoint terms apply to downloaded weights.
- Predicate-Aware Block Pruning is attributed to the [pabp-filtered-ivf project](https://github.com/Fergarciamedina/pabp-filtered-ivf). The application uses a local integration.
- The five ultrasound datasets are described in the [source dataset guide](https://github.com/HealthX-Lab/MedCLIPSeg/blob/main/assets/DATASETS.md). Follow each provider's data access and redistribution terms.
- The [Stage 4 research](research/stage_4/README.md) documents the contribution and its relationship to prior work.

Third-party notices in the vendored source continue to apply.
