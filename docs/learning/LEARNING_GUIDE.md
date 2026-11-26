# From Ultrasound to Mask Review

Location `docs/learning/` · [Up to docs/](../DOCUMENTATION_MAP.md)

This interactive walkthrough follows one synthetic ultrasound-style image through MedCLIPSeg++. It introduces the segmentation model, develops the mathematics behind the uncertainty-based review rule, then shows what the rule changes in the dashboard.

[Download the standalone HTML](https://raw.githubusercontent.com/Fergarciamedina/MedCLIPSeg-plus-plus/main/docs/learning/from_ultrasound_to_mask_review.html) and open it in a browser. Use the browser's **Save As** command if the raw link displays text. From a local clone, open [from_ultrasound_to_mask_review.html](from_ultrasound_to_mask_review.html) directly. The interactions work offline with JavaScript enabled. Source links open separately.

| Chapter | What you can inspect | Where to go deeper |
| --- | --- | --- |
| 1 · Study inputs | The image, reference annotation, region overlap and contour agreement | [Dataset counts, provenance and integrity checks](../../research/stage_1_study_inputs/STUDY_INPUTS.md) |
| 2 · Intent routing | What happens when the user asks for a mask, similar cases or both | [Rules, examples and upstream attribution](../../application/stages/stage2_intent_router/INTENT_ROUTER.md) |
| 3 · Probabilistic segmentation | How image and lesion description lead to a probability field, mask and entropy map | [Frozen MedCLIPSeg configuration](../../research/stage_3_medclipseg/FROZEN_SEGMENTATION.md) |
| 4 · Selective segmentation review | Pixel error and Dice, seven entropy summaries, validation, the frozen cutoff and held-out results | [Study overview](../../research/stage_4_selective_prediction/SELECTIVE_PREDICTION_STUDY.md) · [Theory](../../research/stage_4_selective_prediction/study_reference/THEORY_AND_METRICS.md) · [Results](../../research/stage_4_selective_prediction/experiments/RESULTS_GUIDE.md) |
| 5 · Similarity retrieval | IVF list probing, PABP block skipping and search over eligible cases | [Embeddings, index and retrieval tests](../../application/stages/stage5_similarity_retrieval/SIMILARITY_SEARCH.md) |
| 6 · System outputs | What the dashboard returns for an accepted, deferred or unmatched scan | [Dashboard setup](../../application/APPLICATION_GUIDE.md) · [Notebooks](../../research/notebooks/NOTEBOOK_GUIDE.md) |

The interactive HTML and synthetic examples were created with generative AI assistance for teaching. The validation correlations, bootstrap intervals, AURC and operating points come from the repository's archived result tables.

The presentation was inspired by [arXivisual](https://github.com/rajshah6/arXivisual#creators), created by Armaan Gupta, Raj Shah, Nikhil Hooda and Ajith Bondili. Its [AI-generated MedCLIPSeg explainer](https://www.arxivisual.org/abs/2602.20423) accompanies the [original paper by Koleilat et al.](https://arxiv.org/html/2602.20423v1). The draggable controls were inspired by [Vincent D. Warmerdam's WigglyStuff TangleLatex](https://koaning.github.io/wigglystuff/reference/tangle-latex/). See [attribution and scientific sources](../../ATTRIBUTION.md) for the model, study and application.
