# Stage 1 · Inputs

The system receives breast ultrasound images and a text request. Offline segmentation also uses a lesion-description prompt in the format expected by MedCLIPSeg. Reference masks are used for the research metrics.

| Cohort | Role | Scans in the evaluated split |
| --- | --- | ---: |
| BUSI validation | Select the image-level uncertainty score and calibrate the cutoff | 78, including 65 lesion cases |
| BUSI test | Locked in-domain evaluation | 78, including 65 lesion cases |
| UDIAT | External evaluation | 25 |
| BUID | External evaluation | 35 |
| BUSUC | External evaluation | 122 |
| BUSBRA | External evaluation | 282 |

These counts refer to the evaluated splits. Dataset sources and preparation are listed in the [MedCLIPSeg dataset guide](https://github.com/HealthX-Lab/MedCLIPSeg/blob/main/assets/DATASETS.md). See the [experimental protocol](../stage_4/EXPERIMENTAL_PROTOCOL.md) for cohort and split criteria and the [application data contract](../../application/README.md) for onboarding your own collection.
