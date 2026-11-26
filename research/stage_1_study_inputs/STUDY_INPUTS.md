# Stage 1 · Study inputs and data checks

Location `research/stage_1_study_inputs/` · [Up to research/](../RESEARCH_GUIDE.md)

The study evaluates an uncertainty rule on **2D breast ultrasound images** from five cohorts. Stage 1 establishes which cases enter each split, how images match their reference annotations, and which text is used for frozen model inference. Once the predictions are saved, Stage 4 reads the resulting masks and entropy maps.

[Research overview](../RESEARCH_GUIDE.md) · [Stage 3 model settings](../stage_3_medclipseg/FROZEN_SEGMENTATION.md) · [Stage 4 validation and test](../stage_4_selective_prediction/SELECTIVE_PREDICTION_STUDY.md) · [Application inputs](../../application/stages/stage1_inputs/APPLICATION_INPUTS.md)

## Which inputs were used

| Input | Where it is used |
| --- | --- |
| Original ultrasound scan | Image files from the prepared cohort's `Val_Folder/img/` or `Test_Folder/img/`. MedCLIPSeg uses the image to generate the frozen prediction. |
| Reference annotation | A mask from the corresponding `label/` folder. Dice and normalized surface Dice use it as a reference. Stage 4 defines its foreground from the **predicted** mask when computing uncertainty. |
| Per-image text description | The upstream `Prompts_Folder/` workbook maps `Image` and `Ground Truth` filenames to a `Description`. Descriptions condition the frozen MedCLIPSeg model during offline inference. They are fixed provenance for this aggregation comparison. |
| Saved prediction and uncertainty | A binary PNG mask and an aligned `224 × 224` float32 entropy `.npy` array in nats. These two artifacts are the direct inputs for the seven Stage 4 scores. The per-image table joins those scores to reference Dice and NSD for evaluation. |

The word *prompt* has two meanings in this repository. In the offline workflow, it is a **model description** associated with an image in an upstream workbook. In the dashboard, it is a **request to the intent router**, such as `Find 3 similar cases`. The web request does not enter the Stage 4 study, alter the lesion description used to generate stored masks, or select a threshold. See the [application routing guide](../../application/stages/stage2_intent_router/INTENT_ROUTER.md).

The prepared BUSI validation data use `Val_text.xlsx`, which is the validation runner's fallback when `Val_text_original.xlsx` is absent. The five prepared test splits have `Test_text_original.xlsx`. An audit of the available descriptions found terms such as *benign* and *malignant*. They are part of the evaluated upstream prompt workflow. A diagnosis-blind prompting study would require its own test design. [Upstream data guide](https://github.com/HealthX-Lab/MedCLIPSeg/blob/main/assets/DATASETS.md) · [Experimental rationale](../stage_4_selective_prediction/study_reference/EXPERIMENTAL_RATIONALE.md#backbone-and-digital-image-processing).

## Dataset sources

The original publications describe the images and annotations. The MedCLIPSeg release supplies the prepared splits and text workbooks used here. Use its manifest to reproduce the partition membership and each provider's publication for data attribution. [MedCLIPSeg dataset guide](https://github.com/HealthX-Lab/MedCLIPSeg/blob/main/assets/DATASETS.md) · [Prepared archives](https://huggingface.co/datasets/TahaKoleilat/MedCLIPSeg).

| Prepared cohort | Original publication or release |
| --- | --- |
| BUSI | Al-Dhabyani et al., [*Dataset of breast ultrasound images*](https://pubmed.ncbi.nlm.nih.gov/31867417/). |
| UDIAT | Yap et al., [*Automated Breast Ultrasound Lesions Detection Using Convolutional Neural Networks*](https://pubmed.ncbi.nlm.nih.gov/28796627/), describes the UDIAT data. |
| BUID | Ardakani et al., [*An open-access breast lesion ultrasound image database*](https://pubmed.ncbi.nlm.nih.gov/36535208/). |
| BUSUC | Iqbal and Sharif, [*Memory-efficient transformer network with feature fusion for breast tumor segmentation and classification task*](https://doi.org/10.1016/j.engappai.2023.107292) · [BUS_UC data release](https://data.mendeley.com/datasets/3ksd7w7jkx/1). |
| BUSBRA | Gómez-Flores et al., [*BUS-BRA: A breast ultrasound dataset for assessing computer-aided diagnosis systems*](https://pubmed.ncbi.nlm.nih.gov/37937827/). |

## How many cases were used

| Prepared split | Images checked | Empty reference masks | Empty predicted masks | Role |
| --- | ---: | ---: | ---: | --- |
| BUSI validation | 78 | 13 | 13 | Compare scores and calibrate the operating point. The primary analysis uses 65 nonempty predictions. |
| BUSI held-out test | 78 | 13 | 13 | Test the frozen rule on 65 nonempty predictions and examine the empty-mask branch on the full split. |
| UDIAT test | 25 | 0 | 0 | External test. |
| BUID test | 35 | 0 | 0 | External test. |
| BUSUC test | 122 | 0 | 0 | External test. |
| BUSBRA test | 282 | 0 | 0 | External test. |
| **Total** | **620** | **26** | **26** | **78 validation images and 542 test images, including 464 external images.** |

There is **one calibration validation set**, BUSI validation. The five test sets assess the frozen rule. The four external sets do not set new thresholds. The counts are from the prepared splits used by this study, so the BUSI validation denominator here is distinct from the smaller training-fraction subsets in the upstream data-efficiency table. The five test counts match the upstream dataset guide. The study retained breast ultrasound across sources and reused the frozen checkpoint and inference artifacts to keep computation tractable. [Upstream dataset guide](https://github.com/HealthX-Lab/MedCLIPSeg/blob/main/assets/DATASETS.md) · [Cohort and hardware rationale](../stage_4_selective_prediction/study_reference/EXPERIMENTAL_RATIONALE.md#cohort-scope-and-sample-sizes).

**How the hardware limited the scope.** The workflow concentrates on one modality, a single frozen source checkpoint, and the available prepared splits. It saves one mask and one entropy map per image so the seven score comparisons and bootstrap analyses can reuse the outputs on CPU. With the configured 30 samples at 224 × 224 in float32, the sampled output tensor alone requires about 45.9 MiB at validation batch size 8 and 183.8 MiB at test batch size 32, before model weights and activations. These estimates explain why batching and caching matter. The exact cohort counts come from the prepared partitions, and the study has no measured maximum feasible sample size or peak-memory benchmark. [Hardware and execution budget](../stage_4_selective_prediction/study_reference/EXPERIMENTAL_RATIONALE.md#hardware-and-execution-budget) · [PyTorch MPS backend](https://docs.pytorch.org/docs/stable/notes/mps.html).

## What was validated on the inputs

A read-only audit checked all six prepared splits and their stored outputs. The counts below describe these files and pairings. They are separate from the calibration split size and from the software regression suite.

| Check | Result |
| --- | --- |
| Image integrity | 620/620 source image files decoded. |
| Workbook and per-image table alignment | The image IDs matched exactly among the image folders, the workbook rows and the uncertainty score tables for all six splits. Each workbook had unique image IDs and nonmissing descriptions, and all 620 referenced label paths existed. |
| Split separation | Zero matching image filenames across the 78 BUSI validation and 78 BUSI test images. Patient-level overlap could not be established from filename checks. |
| Mask and entropy pairing | 620/620 predicted masks were binary, and all corresponding arrays were finite float32 and matched the mask grid at 224 × 224. Observed entropy ranged from zero to the float32 representation of `ln(2)`. |
| Reference quality fields | The 620 Dice and NSD values were within `[0, 1]`. Predicted foreground pixel counts agreed with the CSV for all 620 cases. The two BUSI score tables each had 65 valid foreground cases and the external tables had 464 in total. The same 26 cases had empty reference and predicted masks. |

These checks verify input consistency and the expected split membership. The [experimental protocol](../stage_4_selective_prediction/study_reference/EXPERIMENTAL_PROTOCOL.md) covers test separation, prompt and checkpoint provenance, and evaluation limits. The [Stage 4 validation guide](../stage_4_selective_prediction/SELECTIVE_PREDICTION_STUDY.md#validation-and-test-design) lists the seven score comparisons, six coverage targets and software tests. The [configuration guide](../../shared/configs/DATASET_CONFIGURATIONS.md) explains the cohort YAMLs and data-folder layout.
