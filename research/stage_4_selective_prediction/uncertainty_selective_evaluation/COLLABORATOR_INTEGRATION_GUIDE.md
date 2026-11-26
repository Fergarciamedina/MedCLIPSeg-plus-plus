# Integrate selective prediction into an evaluation workflow

Use the toolkit after your segmentation model has produced predictions. The minimum CSV input is one uncertainty score and one reference-quality value per image. The optional helper functions can compute both from a predicted mask, a raw entropy map and a reference mask. The [toolkit guide](EVALUATION_TOOLKIT.md) gives CLI commands and output formats.

## Produce per-image scores

Run Python with `research/stage_4_selective_prediction` on `PYTHONPATH`, or import the toolkit from its enclosing research package when working from the repository root.

```python
from research.stage_4_selective_prediction.uncertainty_selective_evaluation import compute_dice, compute_u_fg

score = compute_u_fg(entropy_map, predicted_mask)
dice = compute_dice(predicted_mask, reference_mask)
```

The entropy map and predicted mask must share a grid. The entropy map uses raw values in nats. Use the reference mask to measure Dice. The predicted mask defines the foreground used by `U_fg`.

## Move from validation to test

Save the scores and reference quality as a CSV with columns such as `image_id`, `U_fg`, and `DSC`. Fit the cutoff on validation and carry the same numeric value to test. The [project protocol](../study_reference/EXPERIMENTAL_PROTOCOL.md) shows this process for BUSI and the external cohorts.

If you use the optional two-stage example in the toolkit, include `U_mean` and use a missing foreground score for empty predictions. The [frozen policy guide](../experiments/RESULTS_GUIDE.md#how-to-read-the-frozen-yaml) describes the observed empty-mask analysis. The [application guide](../../../application/APPLICATION_GUIDE.md) describes the separate catalog and runtime contract.

## Verify the setup

From the repository root, the following command tests the synthetic workflow and compares the saved external cohorts when their local case-level CSVs are available.

```bash
python research/stage_4_selective_prediction/uncertainty_selective_evaluation/test_reproducibility.py
```

Set `MEDCLIPSEG_OUTPUT` to the directory that holds the original per-cohort output trees. Use the [synthetic demo](EVALUATION_TOOLKIT.md#quick-start) to try the toolkit with the checkout alone.
