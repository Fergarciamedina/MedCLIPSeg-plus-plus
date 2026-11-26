# MedCLIPSeg dataset configurations

Location `shared/configs/` · [Up to shared/](../SHARED_COMPONENTS.md)

These YAML profiles tell the frozen MedCLIPSeg inference runner where to find images and prompts, which model settings to load, and how to sample predictive uncertainty. Start here when preparing offline masks and entropy for the [Stage 4 experiment](../../research/stage_4/SELECTIVE_PREDICTION_STUDY.md) or the [application gallery](../../application/APPLICATION_GUIDE.md).

## Choose a profile

| Profile | Role in MedCLIPSeg++ |
| --- | --- |
| [BUSI](BUSI.yaml) | Source checkpoint, validation and locked in-domain test. |
| [UDIAT](UDIAT.yaml), [BUID](BUID.yaml), [BUSUC](BUSUC.yaml), [BUSBRA](BUSBRA.yaml) | External breast ultrasound test cohorts evaluated with the frozen BUSI checkpoint. |
| [BKAI](BKAI.yaml), [CVC300](CVC300.yaml), [ClinicDB](ClinicDB.yaml), [ColonDB](ColonDB.yaml), [Kvasir](Kvasir.yaml) | Retained colon endoscopy profiles. |
| [BRISC](BRISC.yaml), [BTMRI](BTMRI.yaml) | Retained brain MRI profiles. |
| [ISIC](ISIC.yaml), [UWaterlooSkinCancer](UWaterlooSkinCancer.yaml) | Retained skin-imaging profiles. |
| [Covid19](Covid19.yaml) | Retained chest X-ray profile. |
| [EUS](EUS.yaml) | Retained endoscopic ultrasound profile. |

The five study profiles expand `${MEDCLIPSEG_DATA}` to the directory containing cohort folders. Each cohort follows the upstream `Train_Folder/`, `Val_Folder/`, `Test_Folder/` and `Prompts_Folder/` layout. The inherited profiles with `./data/...` resolve paths relative to the launch directory. See the [upstream dataset guide](https://github.com/HealthX-Lab/MedCLIPSeg/blob/main/assets/DATASETS.md).

## Read the settings

| Field | Purpose |
| --- | --- |
| `DATASET.*_PATH` | Split folders with `img/` and `label/` and the separate prompt folder. Use the paths for the split being run. |
| `DATASET.SIZE` | Model input grid. The study uses 224 × 224. |
| `MODEL.CLIP_MODEL`, `MODEL.BACKBONE` and adapter settings | Define the model expected by the checkpoint. The study uses UniMedCLIP ViT-B/16. |
| `TEST.NUM_SAMPLES` | Number of stochastic predictions per image. The study averages 30 predictive samples before computing entropy. |
| `TEST.USE_LATEST` | Selects the `latest` checkpoint variant when enabled and `best_dice` otherwise. |
| `MODEL.DEVICE` | Inference backend. The study profiles use `mps` for Apple Silicon. Set the device for your environment when running a local copy. |
| `TRAIN` | Upstream settings retained where present. The evaluation scripts load a checkpoint and run inference. |

The model settings control image-to-mask inference. The [frozen Stage 4 policy](../../research/stage_4/experiments/RESULTS_GUIDE.md#how-to-read-the-frozen-yaml) controls the later review decision. See the [experimental rationale](../../research/stage_4/study_reference/EXPERIMENTAL_RATIONALE.md) for the model grid, sample count and hardware trade-offs.
