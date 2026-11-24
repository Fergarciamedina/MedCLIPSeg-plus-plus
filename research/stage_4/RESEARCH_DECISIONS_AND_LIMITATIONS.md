# Research Decisions and Limitations: Cumulative Scientific Rationale and Experimental Audit Trail

**MedCLIPSeg++ Research Methodology & Boundaries**  
**Study:** Uncertainty-Based Selective Prediction for Text-Guided Medical Image Segmentation Under Domain Shift (frozen MedCLIPSeg)  
**Version:** 2.0, 2026-11-07 · **Status:** Historical Decisions D-001–D-134 preserved in English; new consolidation Decisions D-135–D-156 through Step 11C.  
**Companion documents:** [Theory and Metrics](THEORY_AND_METRICS.md) · [Experimental Protocol](EXPERIMENTAL_PROTOCOL.md) · [Scientific Evidence and Claims](SCIENTIFIC_EVIDENCE_AND_CLAIMS.md).

> **Evidence and authorship convention:** The text is written in impersonal scientific language. “The study executed/reported” means the project's recorded experimental runs, not a claim that the author of this Markdown independently reran Mac-side code. All internal results were obtained from the project's experiment and audit reports and should be checked against original per-image CSVs/configurations in the final release. Published citations establish theoretical/methodological antecedents only. Internal implementation choices are not automatically reproduced published methods.

## 0. Research purpose, progression and governing rules

The central research question is whether **one scalar measure of segmentation uncertainty per image** can identify unreliable MedCLIPSeg predictions and support abstention without retraining, particularly after the input distribution changes from BUSI to other breast-ultrasound cohorts. It is a **post-hoc quality-control study**, not a change to MedCLIPSeg weights, a learned quality-estimation model, an uncertainty-map/GT-error equivalence study, or a clinically validated diagnostic device.

The research deliberately began with simple aggregators, identified plausible geometric weaknesses algebraically, tested them in BUSI validation, chose a method under a controlled procedure, locked the rule, and then examined independent BUSI test and external domains. This sequence is important: **the selected method is not self-evidently “better”**, and its selection cannot rest only on a visually attractive entropy map or an unexamined headline number.

**Mandatory decision record:** problem → formal hypothesis → relevant external work → possible mechanism → exact formula → image/label access rules → comparison cohort → numerical result → uncertainty → counterexample → conclusion permissible for the paper → unresolved threat → reproducible artifact. The corresponding theory, protocol and evidence IDs are cross-referenced below.

### 0.1. Three levels of claims

1. **Mathematics and definitions:** e.g. `U_mean=αU_fg+(1−α)U_bg` follows from partitioning pixels, without a medical-image experiment.
2. **Internal observations:** e.g. `U_fg` has smaller AURC than `U_mean` on reported BUSI validation/test/OOD cohorts. Such statements require their exact CSV, denominator and code convention.
3. **External/clinical generalization:** e.g. “safe for any hospital” or “guaranteed failure detection” is **not supported** by either level above and must not appear as a finding.

The literature source list is maintained in [Theory and Metrics](THEORY_AND_METRICS.md) §12 and the detailed claim ledger in [Scientific Evidence and Claims](SCIENTIFIC_EVIDENCE_AND_CLAIMS.md) §9.

## 1. Current authoritative study summary

| Phase | Reference sample | Principal quantitative finding | Decision and scope |
|---|---|---|---|
| Frozen model and input check | BUSI 78 test images | 78 raw entropy arrays; matching PNG masks; raw `[0,ln2]` | Use **raw `.npy` float32** only for measurement |
| Seven aggregators | BUSI test + val | Valid `U_fg` on 65 lesion cases; 13 true-normal predicted-empty cases in each BUSI split | Undefined regional scores are NaN, not zero |
| BUSI validation | 65 lesion cases | `U_fg` rho vs DSC `−0.7202`; `U_mean −0.0451` | Rank association suggests foreground conditioning useful |
| BUSI method selection | Same N65 | AURC `U_fg 0.0839`, `U_std_fg 0.0850`, `U_mean 0.1438`, Random 0.1434 | Freeze `U_fg` primary; do not claim superiority over `U_std_fg` |
| Validation operating point | Same N65 | `τ=0.064058324` retained 52/65 (80%) with accepted DSC 0.8910 | Fixed BUSI-calibrated cutoff, six-point sensitivity ladder |
| Pre-test audit | Validation/synthetic only | Common AURC convention, hashes, null/empty-mask checks, 23 tests | Explicit primary 80% flag before opening test |
| Locked BUSI test | N65 with lesion | 47 accepted (72.3%); DSC 0.8286→0.9031 among retained; AURC `U_fg 0.0790` | Confirmation with lower realized coverage and one severe accepted failure |
| All-image BUSI test | N78 incl. 13 normals | Two-stage accepted 60/78; accepted DSC 0.9241 | Separate Setting B; 13 easy normal cases affect mean |
| Four external OOD sets | N=25+35+122+282=464 | 238 accepted (51.3%); pooled baseline 0.7780; accepted 0.8718; rejected 0.6791 | Generalization of ranking/selection utility at high rejection cost |
| Final Step 11C reconciliation | N464 and BUSI N65 | Corrected gain +0.0939, pooled AURC `U_fg 0.1267`, corrected BUSI Random 0.1687 | Supersede transcription/cohort-mixing errors; no retuning |

**Outstanding safety limitation:** a segmentation with serious error may receive low predictive entropy. One BUSI test malignant image (`116_malignant.png`, DSC=0.2193) was **accepted**. A predicted-empty mask with `U_mean≤10^-4` is also accepted under `two_stage` even when a lesion is present in a synthetic counterexample. In the 464 external images, no predicted-empty cases occurred; that failure branch therefore remains untested empirically in OOD.

## 2. Foundational rationale and why the study did not stop at an uncertainty image

### 2.1. From probabilistic pixels to an image-level question (D-001, D-002, D-015, D-016)

MedCLIPSeg generates pixelwise entropy `H(E[p])` from 30 MC predictions. It is expected to be high in ambiguous areas, frequently around boundaries, but **is not an error map**. A local high value does not prove wrong delineation, and a local low value does not prove correctness. Image-level rejection requires an additional function `s(U,M)` and a measurable criterion `Q(Y,M)` for evaluation. This distinction underlies all subsequent decisions. The quantitative entropy source is the **raw float32 `.npy`**, not a colorized PNG that may be individually normalized. Earlier PNG-derived correlations are invalid as confirmatory findings and are retained only as a historical methodological failure.

### 2.2. From unconditional averaging to conditional averaging (D-005, D-010, D-055–D-058, D-071–D-072)

A natural starting baseline was the simplest mean over all image pixels (`U_mean`). Its limitations are testable, not merely intuitive: partitioning the pixel set exactly yields

\[
U_{mean}=\alpha U_{fg}+(1-\alpha)U_{bg},\qquad\alpha=|F|/|\Omega|.
\]

With near-zero background uncertainty, global mean is approximately `α U_fg`. Thus different foreground sizes can change the score even when intralesion entropy is comparable. This yields the **specific hypothesis** that `U_fg` may offer a stronger quality ranking on lesion-positive cases. The validity of the hypothesis was **measured**: BUSI validation `ρ(U_fg,DSC)=−0.7202` vs global `−0.0451`; AURC 0.0839 vs 0.1438. Repetition on BUSI test and four external sets supports a reproducible advantage **for these data**, not for arbitrary settings.

The example `120_benign.png`, with predicted area `753/50,176≈1.50%`, DSC 0.3737, `U_mean=0.003271`, `U_fg=0.102933`, is consistent with dilution. However, high `ρ(area,U_mean)=0.9148` on validation and low `ρ(area,U_fg)=−0.0848` alone cannot establish causal attribution or statistical independence after conditioning on other factors. The study retains this mechanistic claim as **plausible and evidence-consistent**, not conclusively isolated.

### 2.3. Why simpler controls and geometric alternatives were necessary (D-006–D-011)

To evaluate whether any foreground benefit was accidental, seven scores were compared: global average, strict-positive average, predicted-foreground mean, predicted-boundary-band mean, high-entropy conditional mean, foreground spatial standard deviation, and foreground/global ratio. These are **the study's seven operational comparison procedures**. External papers have relevant mean/threshold/structure concepts but do not all define exactly those implementations.

- **Global `U_mean`:** lower AURC than `U_fg` was not observed; mixture with background can be area-sensitive. Its failure is an observed difference, not proof global averaging never works.
- **Strict-positive `U_positive`:** removing only true zeros does not necessarily remove background because floats can be tiny positive numbers. BUSI val AURC `0.1455`; test `0.1282`, similar to `U_mean`.
- **Boundary `U_boundary`:** a two-sided 5×5 **rectangular morphological** band reflects contour uncertainty but not necessarily contour *error*. Validation Spearman NSD (`−0.4174`) was more negative than DSC (`−0.2273`), while DSC AURC (`0.1302`) remained worse than `U_fg` (`0.0839`). It is not intrinsically pointless; it answered a different geometric question.
- **Threshold `U_threshold`:** using `mean(U|U>0.1)` emphasizes high values but discards their count. Scores were narrowly distributed in validation; DSC AURC was `0.1750`, worse than Random `0.1434`. This does not refute other threshold definitions or Guarino's ATA under other empty-set policies.
- **Dispersion `U_std_fg`:** tests spatial variation rather than level. It almost matched `U_fg`, with unresolved paired differences. It remains the prespecified close comparator, **not** an established inferior method.
- **Ratio `U_ratio`:** because `U_mean≈αU_fg` when background entropy is negligible, `U_ratio≈1/α`; the validation rank correlation with area (`−0.9957`) demonstrates a serious potential size-proxy effect. It was a useful negative ablation rather than a successful correction.

**Attribution correction:** `U_fg` shares a per-structure mean operation with Roy et al., 2019, not a full Bayesian QuickNAT pipeline. Guarino et al., 2026 define AVG and ATA; their ATA returns **zero** if no pixel exceeds threshold, whereas this project marks the conditional mean **undefined**. PLM and QFR are not implemented. `U_std_fg` is not reproducibly identified as Jungo's exact formula; similarly `U_boundary` should not be labeled Mehrtash's or Wang's formula based on relevance alone. The detailed source/equation audit is in [Scientific Evidence](SCIENTIFIC_EVIDENCE_AND_CLAIMS.md) §2.

### 2.4. Why method selection needed AURC, not Spearman alone (D-062–D-070, D-087–D-097)

Spearman measures **rank association** between uncertainty and observed segmentation quality; risk–coverage instead models the **operational consequence** of retaining top-ranked cases. A method can have modest Spearman and still help at a particular coverage, or strong association with poor clinical behavior at a fixed threshold. AURC summarizes the quality–coverage tradeoff across the empirical curve. Every comparison must be on the **same cases and evaluation convention**, using DSC loss as primary and NSD loss as secondary.

Oracle sorts by **ground-truth loss**; the top-k exchange argument shows it is the **minimum possible empirical risk/AURC**, not an upper bound. Random is an uninformed selection control. Neither is a deployable method or a published system reproduced here. Validation `U_fg` AURC 0.0839 was better than `U_mean` 0.1438, but the difference from `U_std_fg` 0.0850 was not statistically resolved (`Δ=−0.0011`, CI crosses zero, p=0.294). This justified selecting a **simple, interpretable** primary score while retaining `U_std_fg` as a serious comparator.

### 2.5. Why abstention changes a quality distribution rather than the segmentation (D-098–D-105)

The selector accepts a predicted-nonempty case when `U_fg≤τ_img`. In the reported validation experiment, `τ_img=0.064058324` gave 52/65 accepted (80%); accepted DSC 0.8910 versus full lesion cohort DSC≈0.8520. This is **selective retention**. It changes the mean over the accepted population, not any image's predicted mask. The mixture identity `R_all=cR_accept+(1−c)R_reject` makes that clear. The image-level threshold is distinct from `t_pixel=0.1`, `τ_neg=1e−4`, and the NSD boundary tolerance.

## 3. Data integrity and nonstandard edge conditions

### 3.1. Pairing, split identity and source of numbers

The source pipeline generates raw predictive entropy arrays, binary predicted masks and official DSC/NSD CSV rows. Matching is by image ID, split, seed and prompt, not by incidental file order. BUSI validation (78) and test (78) were reported to share no image IDs. This does **not** guarantee distinct patients. The checkpoint's training provenance must be traced before stating that the evaluation is wholly independent of training patients.

### 3.2. Normal scans and missing metric regions (D-008, D-025–D-029, D-035–D-039)

Each BUSI evaluation split contains 13 GT-empty and predicted-empty normal cases, DSC/NSD=1 by the source evaluator's explicit empty–empty convention. `U_fg`, `U_boundary`, `U_std_fg` and `U_ratio` are undefined for empty predicted foreground; `U_threshold` is also undefined if no pixel has entropy above `0.1`. These are not “zero uncertainty” estimates. Tiny numerical entropy values (`~1e−18`) explain why strict `U>0` is not an anatomically meaningful pixel selection. Normal inclusion changes the statistical population: BUSI validation `Spearman(U_mean,DSC)=−0.4449` at N78 versus `−0.0451` at N65; comparing these without denominators would be misleading.

**BUSI validation empty/empty case IDs:** `2_normal`, `31_normal`, `36_normal`, `43_normal`, `78_normal`, `106_normal`, `107_normal`, `108_normal`, `120_normal`, `121_normal`, `124_normal`, `126_normal`, `132_normal`.  
**BUSI test empty/empty case IDs:** `12_normal`, `13_normal`, `38_normal`, `41_normal`, `49_normal`, `53_normal`, `55_normal`, `60_normal`, `66_normal`, `71_normal`, `73_normal`, `103_normal`, `133_normal`.

### 3.3. OpenCV morphology at image edges (D-006/D-007/D-030)

With `boundary_width=2`, the morphological kernel is a 5×5 **rectangle**, not a circular distance radius of two pixels. Default OpenCV erosion/dilation behavior near the frame can produce nonintuitive counts. A 2×2 foreground mask touching a corner had 9 dilated pixels, 1 eroded pixel and 8 band pixels in a test case. The BUSI test predictions `129_malignant`, `426_benign` and `96_malignant` were recorded as frame-touching; a QC visualization of `129_malignant` was generated. Changes to kernel, padding, spacing or boundary extraction would alter `U_boundary` and require a new protocol, not a silent paper correction.

### 3.4. DSC and NSD are different and require conventions

DSC measures region overlap. NSD compares tolerant surface match; the evaluator was reported to use surfel/marching-cubes machinery and a surface tolerance described as **two pixel units**, not verified physical millimeters. Both-empty gets 1.0; one-sided empty gets 0.0, per source evaluation report. Physical pixel spacing must be verified before reporting surface tolerance in `mm`. The full equations, units and interpretation are in [Theory](THEORY_AND_METRICS.md) §5.

## 4. Experimental phase records and decisions

### 4.1. Steps 1–5: raw entropy audit and exploratory work

78 BUSI test raw maps were described as float32 finite within the theoretical range `[0,ln2]` and dimension 224×224; predicted masks were binary, with official DSC recomputation within approximately `5e−5`. The original exploratory figures and scores were generated **before** a validation-only selection was fully organized and therefore cannot be treated as test-based confirmation of a chosen score. Invalid preliminary pseudo-color-entropy results were excluded. An evaluation protocol must disclose this chronology; test peeking is a potential design limitation even if the later selection script did not read test data.

**BUSI test uncertainty distributions (seven aggregators):**

| Score | Valid N | Mean | Std | Min | Max |
|---|---:|---:|---:|---:|---:|
| `U_mean` | 78 | 0.01018 | 0.01347 | 0 | 0.07608 |
| `U_positive` | 78 | 0.01098 | 0.01419 | 0 | 0.07634 |
| `U_fg` | 65 | 0.05925 | 0.03229 | 0.01974 | 0.17911 |
| `U_boundary` | 65 | 0.27943 | 0.11738 | 0.08807 | 0.51174 |
| `U_threshold` | 65 | 0.39594 | 0.01072 | 0.37120 | 0.43147 |
| `U_std_fg` | 65 | 0.14876 | 0.03088 | 0.09226 | 0.23766 |
| `U_ratio` | 65 | 13.21863 | 16.67820 | 1.22134 | 100.76916 |

These are **descriptive scales**, not evidence that a larger metric value has better failure detection power. Scale differences and missing values motivated same-cohort rank-based evaluation.

### 4.2. Steps 6–7: frozen BUSI validation inference and data checks

The study generated 78 validation predicted masks, 78 raw maps, 78 visualizations and 78 official segmentation quality results using the frozen MedCLIPSeg checkpoint, `seed=666`, `T=30`, original prompts and MPS. BUSI validation baseline DSC=`0.8767` and NSD=`0.9040` across all 78; 65 lesion and 13 normal cases were individually cross-matched for empty/nonempty presence. The reported validation and test IDs did not overlap. Two early MPS runner attempts failed and the corrected inference runner completed; exact original error traceback and changed lines remain provenance gaps. No source weights were retrained in this pipeline.

**Validation score distribution** (mean; valid N): `U_mean 0.008225 (78)`, `U_positive 0.008977 (78)`, `U_fg 0.051638 (65)`, `U_boundary 0.257360 (65)`, `U_threshold 0.395187 (65)`, `U_std_fg 0.141477 (65)`, `U_ratio 16.038413 (65)`.  
**Two most severe validation failures:** `45_malignant` DSC 0.0491, NSD 0.0566, `U_mean 0.021035`, `U_fg 0.067293`, `U_boundary 0.411354`; and `120_benign` DSC 0.3737, NSD 0.4127, `U_mean 0.003271`, `U_fg 0.102933`, predicted area 753px. There were only **two cases with DSC<0.5**, so validation failure AUROC under that cutoff is fragile.

### 4.3. Step 8: statistical screening without using test to select

BUSI validation lesion-only N65 identical sample set: `U_fg` Spearman vs DSC `−0.7202` and vs NSD `−0.6925`; `U_std_fg −0.7068/−0.6768`; `U_boundary −0.2273/−0.4174`; `U_mean −0.0451/−0.2570`; `U_positive −0.0207/−0.2360`; `U_ratio −0.1874/+0.0499`; `U_threshold +0.0178/+0.0186`. The 1,000-resample intervals for all values are preserved in [Theory](THEORY_AND_METRICS.md) §6. The result favors region-level summaries over global or conditional-threshold averages for **the observed segmentation quality ranking**, not pixelwise error localization.

Validation AURC(1−DSC) ranks: Oracle `0.0686` (ideal), `U_fg 0.0839`, `U_std_fg 0.0850`, `U_boundary 0.1302`, `U_ratio 0.1368`, Random `0.1434`, `U_mean 0.1438`, `U_positive 0.1455`, `U_threshold 0.1750`. For NSD the ordering changes in detail; `U_fg 0.0563`, `U_std_fg 0.0573`, `U_boundary 0.0848`, `U_mean 0.0969`, Random `0.1116`. The study did **not** define an independent learned quality predictor; its principal change is a deterministic image-level score and abstention rule.

### 4.4. Step 9: validation-only comparison, threshold selection and empty policy

With N65 lesion cases held constant, paired AURC bootstrap gave `U_fg−U_std_fg=−0.0011`, CI `[−0.0033,+0.0008]`, p=`0.294`; `U_fg−U_mean=−0.0599`, CI `[−0.0871,−0.0320]`; and `U_fg−U_boundary=−0.0462`, CI `[−0.0711,−0.0221]`. The scientific decision is not “foreground std failed,” but “no resolved improvement over the simpler foreground mean on the preselected validation criterion.” `U_fg` was frozen as primary, `U_std_fg` frozen as comparator.

**Calibrated validation threshold ladder:**

| Nominal coverage | `τ_fg` | Accepted N65 | Realized coverage | Accepted DSC |
|---:|---:|---:|---:|---:|
| 95% | 0.10193 | 61 | 93.8% | 0.8622 |
| 90% | 0.08346 | 58 | 89.2% | 0.8680 |
| 85% | 0.07270 | 55 | 84.6% | 0.8753 |
| **80% primary** | **0.064058324** | **52** | **80.0%** | **0.8910** |
| 75% | 0.05969 | 49 | 75.4% | 0.8909 |
| 70% | 0.05621 | 45 | 69.2% | 0.8942 |

A two-stage decision for empty predictions was selected because five foreground-derived aggregators are undefined there. When `F=∅`, accept if `U_mean≤10^-4`, otherwise reject. For nonempty `F`, accept if `U_fg≤τ_fg`. This is **not a diagnosis of a healthy patient**. In BUSI validation all 13 predicted-empty cases were GT empty, but the synthetic GT-positive / prediction-empty / low-entropy input proves the gate can accept a missed lesion. An all-image AURC of 0.0496 was reported for N78, but comparison with N65 AURC 0.0839 would confound method with population composition.

### 4.5. Step 9.5: mathematical and provenance gate, including literature corrections

The pre-test audit reported YAML–JSON semantic equality, code/config hashes, 23/23 tests, and synthetic numerical AURC checks. The initial validation Random AURC discrepancy was explained under an unanchored empirical integration domain (approximately `R_all×(1−2/65)=0.143433`), not by using a distinct risk definition. The audit discovered that an explicit primary coverage flag was needed and designated **80%** before test execution. The literature attribution audit corrected “full reproduction of Roy et al.” to “shared local structure-mean equation,” and clarified that Guarino's ATA uses a different empty-set rule. It also cautioned against mislabeling Oracle as an **upper** AURC bound and against claiming a confidence threshold verifies absence of disease.

At the moment of Step 10 lock, reported frozen YAML SHA256 was `6e24ddb14baac0049b79f1499c9b281f988b07b9e375747b02b8f596e392382e`; JSON SHA256 `ddbea219e53c6a2163fae0e04b98b4d747873764e6198e66c614e2a56d3f4806`. These values should be checked against release artifacts; they are historical provenance, not newly recomputed here. The difference from earlier Step 9.5 hashes is consistent with adding the explicit primary operating point and must be traceable in version history.

### 4.6. Step 10: independent locked BUSI test and its counterexample

BUSI test lesion-positive N65, frozen threshold `τ_fg=0.064058324`: accepted `47` (72.3%), rejected `18`. Baseline DSC=`0.8286`, accepted=`0.9031` (gain `+0.0745`, CI `[0.0349,0.1235]`), rejected=`0.6341`. Baseline NSD=`0.8602`, accepted=`0.9349` (gain `+0.0747`), rejected=`0.6651`. Target coverage was 80%; realized test coverage was **not** 80%. For all N78 images including 13 normal cases, two-stage accepted 60 and rejected 18 (76.9%), with retained DSC=`0.9241` versus full DSC=`0.8572`.

**AURC on identical N65 BUSI test cases:** `U_fg 0.0790`, `U_std_fg 0.0793`, `U_boundary 0.1113`, `U_mean 0.1262`, `U_positive 0.1282`, `U_ratio 0.1499`, Random `0.1687`, `U_threshold 0.2045`, Oracle `0.0692`. Paired comparison `U_fg−U_std_fg=−0.0003` CI `[−0.0024,+0.0014]`, p=`0.786`: still unresolved, not statistically superior. `U_fg−U_mean=−0.0472` CI `[−0.0690,−0.0277]` supports the validation-selected direction. Four of the five lowest DSC cases were rejected, but **`116_malignant.png` DSC=0.2193**, `U_fg=0.06279`, **was accepted**. Claiming perfect failure detection or clinical readiness would be contradicted by the own locked data.

**Subgroups:** benign N44 baseline DSC 0.8655; 37/44 accepted (84.1%); malignant N21 baseline DSC 0.7514; 10/21 accepted (47.6%). These are descriptive differences in performance and coverage; no causal fairness or scanner explanation is established.

### 4.7. Steps 11A–11B: OOD question, dataset choice, frozen external evaluation

The study deliberately used four **breast ultrasound** external test sets from the MedCLIPSeg official data manifest, rather than adding a thyroid dataset with a second simultaneous anatomy shift. Near-domain is descriptive: the four sets are external and same-organ, but their scanner, label and source differences are not experimentally isolated. Official test sizes: UDIAT 25, BUID 35, BUSUC 122, BUSBRA 282. The BUSI-trained checkpoint, seed 666, 30 MC samples and BUSI validation thresholds remained frozen; no OOD re-training, recalibration or test-driven prompt optimization was reported. External test splits should be independently verified by hashes of images, masks and exact original prompt tables before final release.

| External cohort | N | DSC full | Accepted / rejected | Coverage | DSC accepted | DSC rejected | Selective gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| UDIAT | 25 | 0.7467 | 14 / 11 | 56.0% | 0.8747 | 0.5837 | +0.1281 |
| BUID | 35 | 0.7900 | 20 / 15 | 57.1% | 0.8818 | 0.6675 | +0.0919 |
| BUSUC | 122 | 0.8430 | 78 / 44 | 63.9% | 0.8982 | 0.7453 | +0.0551 |
| BUSBRA | 282 | 0.7511 | 126 / 156 | 44.7% | 0.8536 | 0.6683 | +0.1025 |

The external datasets had **464/464 nonempty predictions** per audit report; thus the fallback for predicted-empty lesions was not exercised and its false-negative safety cannot be inferred. Most importantly, OOD fixed-threshold coverage ranged from **44.7% to 63.9%**, substantially below the nominal BUSI validation 80%. The method maintained *selection utility* but did not maintain the intended workload/retention rate. The tradeoff must appear in every external results table and high-level conclusion.

### 4.8. Step 11C: reconciliation of pooled calculations and legacy discrepancies

The first cross-domain draft contained discrepancies caused by approximate manual transcription and mixing BUSI N78/N65. The reconciliation recomputed reported micro statistics from individual case-level CSVs, distinguished macro and pooled AURC, and restored the BUSI test Random reference to the **N65** cohort. The correction is a research finding about reproducibility and must not be silently removed from the log.

| Quantity | Superseded draft | Authoritative Step 11C corrected |
|---|---:|---:|
| OOD pooled full DSC | 0.7711 | **0.7780** |
| OOD pooled retained DSC | 0.8732 | **0.8718** |
| OOD pooled rejected DSC | 0.6635 | **0.6791** |
| OOD pooled retained-minus-full DSC | +0.1021 | **+0.0939**, bootstrap CI `[+0.0775,+0.1113]` |
| OOD pooled global-ranking `U_fg` AURC | 0.1242 | **0.1267** |
| BUSI test Random AURC in draft cross-domain summary | 0.1406 (N78 contamination) | **0.1687** on lesion-only N65 |

**Pooled micro case count:** 464 images, 238 accepted and 226 rejected; `DSC_all=0.7780`, `DSC_accepted=0.8718`, `DSC_rejected=0.6791`.  
**Macro arithmetic mean of dataset results:** full 0.7827, accepted 0.8771, rejected 0.6662, gain 0.0944.  
**Pooled case-level AURC, N464:** `U_fg 0.1267`, `U_mean 0.1864`, Random `0.2216`, Oracle `0.1046`.  
**Sample-size-weighted average per-domain AURC:** 0.1314, 0.1849, 0.2202, 0.1070.  
**Unweighted macro per-domain AURC:** 0.1168, 0.1825, 0.2127, 0.0978, respectively.

These three AURC summaries are not interchangeable: pooling sorts **all 464 cases together**, while averaging dataset-specific AURCs summarizes **within-domain** sorting. AURC is not additive under pooling due to cross-domain rank interactions. The primary external conclusion is not that a model gained 9.39 points of segmentation ability; it is that **the accepted subset had 0.0939 higher mean Dice than the full external set, at a cost of rejecting 48.7%**. The Step 11C report recorded 31/31 passing regression tests. This does not establish prospective clinical safety, validity for arbitrary hospitals, or patient-group independence.

## 5. Explicit literature verification and novelty boundary

- **Koleilat et al., CVPR 2026:** original frozen probabilistic model. The post-hoc rejection is an extension to its inference outputs, not a claim of new MedCLIPSeg training or a new segmentation architecture.
- **Roy et al., NeuroImage 2019:** corresponding per-structure uncertainty **mean** provides an antecedent for `U_fg`. The present implementation is not a reproduction of QuickNAT, MRI, or its full four-metric suite.
- **Guarino et al., CVPR 2026:** establishes the scientific relevance of comparing aggregation strategies and explicitly defines AVG, ATA, PLM, QFR and others. `U_mean` matches global AVG operation, conditional `U_threshold` differs from ATA on no selected pixels, and PLM/QFR are **not implemented**.
- **Jungo et al., 2020; Mehrtash et al., 2020:** provide the evidence-based motivation that spatial uncertainty is imperfect at error localization and may support subject-level failure recognition; neither source proves that `U_std_fg` or `U_boundary` is their specific algorithm.
- **DeVries & Taylor, 2018:** two-stage neural quality estimator is a conceptually related trained system; the study's `two_stage` is a deterministic **empty-prediction policy**, not their architecture.
- **Geifman & El-Yaniv, 2017:** selective risk and coverage motivates the evaluation, but the study does **not** inherit theoretical guaranteed risk control from its empirical cutoff.

**Contribution to state precisely:** an empirically evaluated, **training-free segmentation-quality selection rule** based on raw MedCLIPSeg uncertainty aggregated over predicted foreground, including fair lesion-cohort comparisons, a frozen BUSI validation cutoff, locked BUSI test validation, four same-anatomy external cohorts, quantitative risk/coverage, and explicit failure modes. The simple `U_fg` mathematical operation itself is **not novel**. Evidence of novelty in the *application/protocol and evaluation* must be argued carefully and with citations, not assumed from attaching the method to a new baseline.

## 6. Risks and unresolved scientific limitations, ordered by severity

| Priority | Risk | Mechanism and measured evidence | Status / appropriate response |
|---|---|---|---|
| **P0** | Severe confident errors escape | `116_malignant` DSC 0.2193 accepted; synthetic GT-positive empty prediction with low entropy accepted | Explicit negative result; never claim comprehensive detection |
| **P0** | Coverage transfer instability | 80% validation nominal → 72.3% BUSI test → 51.3% external pooled | Report workload cost; no distribution-free coverage guarantee |
| **P0** | Historical test exploration | Test-side exploratory figures existed before validation selection was formalized | Disclose chronology; retain frozen selection logs/hashes |
| **P0** | Nonindependent patients or overlapping source images | Zero overlapping image IDs does not show distinct patients | Seek patient-level IDs/split metadata and exact file fingerprints |
| **P0** | Prompt or GT label leakage | Text prompts may contain diagnosis/class not available at deployment | Audit exact prompt sheets and their source for each domain |
| **P0** | Model checkpoint provenance | Test-time frozen file may have incomplete original training lineage | Archive model hash, original training dataset/split/checkpoint identity |
| **P0** | Missing raw-release replication | This documentation records audit reports; raw Mac-side CSV/code not mounted here | Re-run authoritative tables from final files and bundle manifests |
| **P1** | Conditional size dependence | Global `U_mean` strong area rho; `U_fg` low rho but not zero conditional dependence | Describe as association; consider regression/stratification if separate study planned |
| **P1** | Two-stage empty-mask policy blind spot | Empty wrong segmentation can have `U_mean≈0`; 464 OOD cases nonempty | Limit claims; require new independent sample to estimate rate |
| **P1** | OOD domain heterogeneity | Different case mix, scanner, annotations and prompts mixed | Per-domain analysis; no single isolated scanner effect claim |
| **P1** | Thin validation failure sample | Only two cases with DSC<0.5, AUROC unstable | Prefer continuous risk/AURC primary endpoint |
| **P1** | Statistical uncertainty | B=1000 paired bootstrap, possible patient clustering, multiplicity, p construction | Report assumptions and effect size+CIs; do not overclaim significance |
| **P1** | NSD spacing and surface tolerance | Numerical tolerance reportedly 2 pixel units, physical mm not verified | Verify spacing/distance evaluator prior to publication |
| **P1** | AURC endpoint/tie convention | Unanchored integration and tied scores alter numeric area | Keep shared grid and exact estimator; release code plus synthetic tests |
| **P1** | No GT-independent quality guarantee | Predictive entropy is not an error probability; acceptance thresholds are heuristic | Keep claim confined to observed predictive ranking |
| **P2** | Missing published advanced comparison methods | PLM/QFR/Soft Dice Confidence not executed | Do not claim complete literature state-of-the-art ranking |
| **P2** | Clinical effectiveness and intervention cost | Rejecting 226/464 requires reviewer throughput | Mark as future prospective research, not accomplished |

## 7. Artifact and code provenance map

| Stage | Reported project artifact | Why it matters |
|---|---|---|
| Raw output | `data/outputs/BUSI/seg_results/`, `unc_results/`, validation `seed666/` | Derived scores must use raw entropy |
| Aggregation | `MedCLIPSeg-plus-plus/research/stage_4/src/uncertainty/aggregation.py` | Exact seven formulas, morphology and NaN behavior |
| Statistical comparison | `.../src/uncertainty/metrics.py`, `selective_prediction.py` | Spearman, bootstrap, AURC, empty policies |
| BUSI validation scores | `data/outputs/BUSI/validation/seed666/aggregation_results/uncertainty_scores_BUSI_val_seed666.csv` | Selection population, N65/78 |
| Validation stats | `data/outputs/BUSI/validation/seed666/statistical_analysis/tables/` | ρ, AURC, area, failure AUROC |
| Frozen config | `data/outputs/BUSI/validation/seed666/selective_prediction/frozen_selective_prediction_config.{yaml,json}` | No changes after selection, primary τ |
| Pre-test scientific audit | `.../selective_prediction/PRE_TEST_AUDIT.md` | Numerical/tie/bootstrap/no-leakage checks |
| Locked BUSI test | `data/outputs/BUSI/test_evaluation/seed666/LOCKED_TEST_EVALUATION_REPORT.md` | Independent confirmation and severe failure |
| OOD execution | `.../research/stage_4/scripts/run_ood_inference_and_aggregation.sh` | Frozen external inference workflow |
| OOD stats/summary | `data/outputs/cross_domain_evaluation/seed666/cross_domain_summary.csv` | Pooled and domain-level metrics |
| OOD evaluation report | [Cross-domain results](results/cross_domain_evaluation/seed666/cross_domain_summary.csv) | Historical external results, superseded aggregate correction |
| Step 11C reconciliation | [Cross-domain reconciliation](#48-step-11c-reconciliation-of-pooled-calculations-and-legacy-discrepancies) | Authoritative corrected pooled and N65 values |
| Regression tests | `.../research/stage_4/scripts/run_tests.sh` | Final reported 31/31 status |

Absolute paths on the source Mac are not portable. Preserve the repository-relative paths and exact hashes before final paper/GitHub release. Raw medical image data must not be uploaded to public GitHub in violation of licensing or privacy terms; publish code, scripts, provenance and permitted derived results only.

## 8. Historical decision ledger D-001–D-134 (English, archival IDs maintained)

The ledger below preserves **all 134 decision identifiers** from the earlier research log, in English. The narrative sections above retain details and numerical evidence; a short ledger entry is an index, **not** a replacement for the experimental context. IDs must not be reused for new decisions. The **most recent status** (Step 11C) supersedes any earlier “pending test/OOD” stage note in the archived log.

| Decision ID | Historical decision (English) | Scientific reason or safeguard |
|---|---|---|
| **D-001** | Use raw predictive entropy .npy, not pseudo-color PNG, for numerical scores | Quantitative scale and cross-image comparability |
| **D-002** | Keep entropy in absolute nats with theoretical maximum ln2 | Per-image visualization scaling is nonquantitative |
| **D-003** | Pair data by exact image ID, split, prompt and seed | Prevent accidental mismatched quality labels |
| **D-004** | Distinguish 224x224 uncertainty lattice from the evaluator mask space | Avoid unannounced resolution effects |
| **D-005** | Use predicted foreground, never GT foreground, when deriving inference scores | Prevent target leakage |
| **D-006** | Check OpenCV morphological behavior at image borders | Frame-touching masks change erosion/band pixels |
| **D-007** | Interpret width=2 as square 5x5 kernel, not exact Euclidean thickness | Morphological geometry must be specified |
| **D-008** | Represent undefined-region scores as NaN and valid=False | Undefined is not zero uncertainty |
| **D-009** | Keep pixel cutoff t separate from image abstention cutoff tau | Different levels and units of decision |
| **D-010** | Treat U_ratio as potentially area-dependent | Exact foreground/background mixture yields reciprocal-area limit |
| **D-011** | Use population spatial standard deviation for U_std_fg | Do not call it MC-sample dispersion |
| **D-012** | Obtain independent BUSI validation inference for selection | Do not optimize method using test outcomes |
| **D-013** | Exclude early pseudo-color/test exploratory numbers from confirmation | Avoid contaminated/invalid numerical evidence |
| **D-014** | Evaluate rejection using coverage, Random/Oracle, and validation-calibrated tau | Make selection operational and falsifiable |
| **D-015** | Do not treat boundary uncertainty as delineation error or diagnosis | Confident errors remain possible |
| **D-016** | Identify H(mean probability) as predictive entropy, not pure epistemic variance | Taxonomy requires careful interpretation |
| **D-017** | Keep seed/config, private data, environment and artifacts traceable | Reproducible pipeline without sharing restricted data |
| **D-018** | Expand synthetic/integration regressions throughout stages | Test count advanced 15,20,23,27,31 |
| **D-019** | Distinguish unimplemented PLM/QFR/SDC from actual study baselines | Exact published method requires code/input equivalence |
| **D-020** | Keep BUSI cutoff fixed during domain shift evaluation | Report realized external coverage honestly |
| **D-021** | Separate prompt-robustness collaboration from uncertainty rejection | Avoid coupled experiments and double-use |
| **D-022** | Correct obsolete handoff paths, early values and historical provenance | Prevent superseded figures being used in final paper |
| **D-023** | Use same-subject Spearman, bootstrap, AURC and cautious significance | Association and operational utility differ |
| **D-024** | Treat PLUS A batch triage and PLUS B FAISS retrieval as future independent systems | Avoid expanding the course experiment without evidence |
| **D-025** | Process all 78 BUSI test cases with raw entropy and seven scores | One keyed quantitative row per image |
| **D-026** | Record 13 predicted-empty BUSI test cases and undefined region metrics | They are GT-empty normal cases, not errors by presence |
| **D-027** | Record separate threshold-region emptiness where no U>0.1 | Different cause of NaN from predicted-empty foreground |
| **D-028** | Maintain explicit missing-value and abstention policy | Silent zero replacement changes ranking |
| **D-029** | Cross-match predicted-empty and GT-empty cases individually | Equal empty counts alone do not prove same cases |
| **D-030** | Check three test masks touching image frame; produce QC panel | OpenCV boundary effects occur on real predictions |
| **D-031** | Document metrics' disparate ranges and outliers | Raw magnitudes cannot be compared across definitions |
| **D-032** | Label early test scatterplots and heatmaps exploratory | Do not choose metric from them |
| **D-033** | Include mask/entropy/score validity and metadata in reproducibility CSV | A bare score is insufficient for method reproduction |
| **D-034** | Keep generated tables/plots in private output store, not Git | Dataset licensing and privacy |
| **D-035** | Verify 13 BUSI test empty/empty normal cases | Case-level source/ground-truth matching |
| **D-036** | Retain explicit evaluator convention DSC=NSD=1 for both-empty | Empty-empty is not defined by ordinary 0/0 algebra |
| **D-037** | Recognize positive floating-point entropy residuals | Strict >0 is not clinically meaningful support |
| **D-038** | Design explicit undefined-foreground fallback for deployment | Single scalar F-based score cannot rank empty mask alone |
| **D-039** | Do not use correct BUSI normals to assert universal empty-mask safety | False negatives may occur under shift |
| **D-040** | Verify no BUSI validation/test image-ID overlap | Patient-level uniqueness remains unverified |
| **D-041** | Match validation 78 images, labels and original prompt spreadsheet | Prompt and ID integrity |
| **D-042** | Run frozen existing checkpoint in eval/no_grad mode | No retraining during evaluation |
| **D-043** | Use original prompt, MC30, seed666 and MPS in validation | Match stochastic inference protocol |
| **D-044** | Document corrected MPS runner after failed initial runs | Trace causal implementation changes and error logs |
| **D-045** | Preserve separate validation masks, raw uncertainty, visualizations and CSV | No overwriting BUSI test outputs |
| **D-046** | Report full BUSI validation baseline DSC0.8767 NSD0.9040 | These are raw model, all-image statistics |
| **D-047** | Audit RAW shapes/dtype, masks, bounds, and tests in validation | Ensure suitable input for aggregation |
| **D-048** | Cross-tab validation 13 both-empty and 65 both-nonempty by image ID | Presence matching is not delineation accuracy |
| **D-049** | Keep selection, tau, and empty-policy choices in validation stage | Avoid later retrospective selection |
| **D-050** | Do not infer checkpoint was trained in this study | Checkpoint training provenance still requires audit |
| **D-051** | Require exact run manifest: checkpoint/code/config hash, seed, prompts, environment | Needed for independent replication |
| **D-052** | Create full 78-row validation CSV with seven scores and quality | Basis of method selection |
| **D-053** | Separate mask-presence agreement from DSC/NSD accuracy | Correct positive presence can still have bad contour |
| **D-054** | Mark five region-dependent scores undefined in 13 normal cases | Missingness must be visible |
| **D-055** | Document small-foreground severe validation failure 120_benign | Illustrates global mean dilution without proving causation |
| **D-056** | Document severe 45_malignant failure with higher global uncertainty | Failure modes are not uniform |
| **D-057** | Derive and test area-fraction mixing mechanism | Use exact mixture identity plus correlation evidence |
| **D-058** | State threshold score and ratio output ranges separately | Ratio not bounded by ln2; conditional mean narrow |
| **D-059** | Treat validation failure AUROC with two positive failures as exploratory | Very sparse binary endpoint |
| **D-060** | Disclose N65 lesion versus N78 all-image estimands | Do not compare methods across different samples |
| **D-061** | Treat test-suite logs as implementation checks, not clinical validation | Scope of evidence and release provenance |
| **D-062** | Run validation statistical analysis and keep artifacts | 20/20 regression stage |
| **D-063** | Use identical N65 lesion cohort for all seven Spearman comparisons | Fair ranking comparison |
| **D-064** | Bootstrap Spearman intervals with B1000 and note inference assumptions | Nonindependence and ties may bias CIs |
| **D-065** | Quantify all-image rank mixing by 13 true normals | Full-cohort U_mean rho differs sharply |
| **D-066** | Avoid inferring global score is inherently strong from N78-only correlation | Composition effect changes estimand |
| **D-067** | Construct risk-coverage curves and empirical AURC with common endpoints | Primary operational ranking measurement |
| **D-068** | Use DSC loss as primary and NSD loss as secondary | Avoid endpoint switching after test |
| **D-069** | Compare with ideal GT-informed Oracle | Lower AURC bound and interpretive reference |
| **D-070** | Compare with no-information Random baseline | Detect whether ranking beats chance |
| **D-071** | Measure association between score and predicted lesion area | Identify geometric proxy risk |
| **D-072** | Do not treat low area Spearman as independence proof | Nonmonotone and conditional effects remain |
| **D-073** | Compute binary-failure AUROC only as sensitivity analysis | Just two severe validation failures |
| **D-074** | Avoid data-driven optimistic claims from multiple exploratory endpoints | Statistical multiplicity |
| **D-075** | Define deployable behavior for predicted-empty foreground | F-based score cannot exist there |
| **D-076** | Contrast global fallback, drop-empty, conservative abstention and two-stage | Different policies and populations |
| **D-077** | Use predicted-mask emptiness, never GT presence, in runtime selector | Prevents evaluation label leakage |
| **D-078** | Distinguish pixel t, image tau and empty-branch tau_neg | Avoid invalid threshold substitutions |
| **D-079** | Create validation statistical CSV tables and figures | Auditable numeric claims |
| **D-080** | Retain cases/score validity when calculating Spearman and AURC | Avoid hidden N changes |
| **D-081** | Add statistical regression tests, including ties and degenerate inputs | Protect numerical conventions |
| **D-082** | Keep all results under private project/data/outputs | Source data not pushed to public repo |
| **D-083** | Investigate apparent Random AURC versus mean-risk difference | Integration domain/tie issue later reconciled |
| **D-084** | Avoid concluding from validation that method generalizes | Freeze then evaluate locked test/OOD |
| **D-085** | Treat all chosen metrics as study-specific measures, not universal optimum | Dataset/task dependence in literature |
| **D-086** | Require pre-test independent audit of code, stats and freeze | Scientific gate prior to confirmatory evaluation |
| **D-087** | Start Step9 with validation-only method comparison | Single designated selection population |
| **D-088** | Use N65 identical lesion images for all candidates | Avoid comparing NaN-dropped and full cohorts |
| **D-089** | Select from fixed primary DSC AURC with secondary NSD | Predetermined method-selection rule |
| **D-090** | Record near-identical leading AURCs for U_fg and U_std_fg | Small point difference is uncertain |
| **D-091** | Define Oracle as GT-informed ideal ranking, not database | Risk/AURC lower bound; evaluation-only |
| **D-092** | Define Random as no-information quality baseline | Expected risk; discrete integration domain required |
| **D-093** | Avoid claiming significant U_fg superiority over U_std_fg | Validation confidence interval includes zero |
| **D-094** | Use identical cohort and coverage integration for all AURCs | N65 versus N78 cannot be compared directly |
| **D-095** | Freeze U_fg primary and U_std_fg comparator | Simplicity and validation point estimate, not novel formula |
| **D-096** | Audit bootstrap p-value logic, paired cases and tie handling | CI does not automatically validate reported p |
| **D-097** | Disclose multiplicity and possible validation optimism | Many candidate comparisons |
| **D-098** | Choose deterministic two-stage policy for predicted-empty masks | Operational fallback with known false-negative hazard |
| **D-099** | Freeze empty-case U_mean threshold at 1e-4 | Distinct from foreground tau and pixel t |
| **D-100** | Record synthetic overconfident lesion-miss acceptance | No normality verification via entropy |
| **D-101** | Do not interpret N78 low AURC as policy effect relative to N65 | Thirteen normal perfect scores change quality mixture |
| **D-102** | Check ranking construction when empty and nonempty branches use different rules | No natural single scalar ordering guaranteed |
| **D-103** | Freeze primary 80%-target tau from BUSI validation | tau≈0.064058324, 52/65, accepted DSC0.8910 |
| **D-104** | Persist frozen machine-readable YAML and JSON | Integrity and audit before test |
| **D-105** | Add explicit single primary operating-point designation | Six candidate cutoffs alone insufficient |
| **D-106** | Do not call internally specified aggregators reproduced published baselines | Citation is not equation-code equivalence |
| **D-107** | Demand equation, hyperparameter and input match before method attribution | Prevent inaccurate literature novelty claims |
| **D-108** | Keep Soft Dice Confidence as unimplemented related method | Probability map prerequisites differ from entropy-only artifacts |
| **D-109** | Retain implementation outputs, script names and provenance paths | Enable reproducibility |
| **D-110** | Report Step9 23/23 passing tests without overstating coverage | Pass does not guarantee correctness for all inputs |
| **D-111** | Separate git-status checks from proof of commits or file modifications | Do not fabricate repository changes |
| **D-112** | Keep private data and prompt robustness work isolated | Project scope and confidentiality |
| **D-113** | Pre-test audit common coverage grid, endpoints, ties, Random AURC | Resolve numerical convention before test |
| **D-114** | Pre-test audit paired bootstrap, 1000 draws, multiplicity and patient independence | Prevent invalid inferential interpretation |
| **D-115** | Pre-test audit YAML/JSON hashes, frozen parameters and primary point | Enforce lock before test exposure |
| **D-116** | Verify runtime acceptance cannot read GT/DSC/NSD/category | Prevent target leakage |
| **D-117** | Prove synthetic confident false negative passes empty-branch gate | Expose safety limit before OOD |
| **D-118** | Audit N78 two-stage AURC calculation separately from N65 | Different populations and branch ranking |
| **D-119** | Map every literature claim to actual equation; verify NSD tolerance/units | Method integrity and publication correctness |
| **D-120** | Record Step9.5 pre-test audit without touching BUSI test | Report plus 23/23 tests |
| **D-121** | Record initial Step9.5 YAML/JSON semantic equality and SHA hashes | Before explicit primary operating-point flag |
| **D-122** | Reproduce two synthetic AURC reference examples exactly | 0.1000000 and 0.0833333 in audit |
| **D-123** | Explain Random validation AURC as unanchored empirical domain | 0.147986*(1-2/65)=0.143433, subject to code convention |
| **D-124** | Require paired case-level bootstrap triples and careful null p interpretation | Dependence/p-values still matter |
| **D-125** | Correct Roy attribution to local structure-mean equation only | No full QuickNAT reproduction |
| **D-126** | Correct Guarino AVG/ATA attribution and empty-policy discrepancy | ATA zero empty, study NaN empty; PLM/QFR not run |
| **D-127** | Separate Jungo/Mehrtash/DeVries contextual literature from implemented algorithms | Do not claim equivalence from related work |
| **D-128** | Correct Oracle bound direction | Lower risk/AURC, upper retained quality |
| **D-129** | Retain non-significant U_fg versus U_std_fg validation comparison | Δ−0.0011, CI crosses 0, p0.294 |
| **D-130** | Reject claims of complete area independence from small Spearman rho | Unmeasured conditional/nonlinear dependence |
| **D-131** | Keep overconfident predicted-empty acceptance documented | Synthetic counterexample |
| **D-132** | Do not convert NSD δ=2 to millimeters without spacing | Unit provenance is required |
| **D-133** | Mark validation 80%-target cutoff as primary before test | Explicit flag and full-precision tau |
| **D-134** | Treat Step9.5 ready verdict as conditional on locked common-grid/primary config | No BUSI test tuning |

## 9. New decisions D-135–D-156: completed BUSI test, OOD and final reconciliation

| Decision ID | Decision adopted | Scientific support / limitation |
|---|---|---|
| **D-135** | BUSI locked-test selection remained frozen | U_fg at tau0.064058324, N65; no test-driven cutoff change |
| **D-136** | BUSI test realized coverage did not match validation target | 47/65=72.3% versus 80% in validation; report actual workload |
| **D-137** | BUSI test retained quality improved through selection only | DSC accepted 0.9031 versus all 0.8286; raw masks unchanged |
| **D-138** | Keep serious accepted malignant failure visible | 116_malignant DSC0.2193, U_fg0.06279 accepted despite cutoff |
| **D-139** | Maintain U_std_fg as unresolved statistical comparator on locked test | AURC 0.0793 versus U_fg0.0790, CI for difference includes zero |
| **D-140** | Track pathology-specific coverage disparities | Benign37/44 accepted versus malignant10/21, observational not causal |
| **D-141** | Distinguish full BUSI two-stage Setting B from lesion-only Setting A | Normal empty/empty cases inflate retained quality and change denominators |
| **D-142** | Prioritize four external same-organ breast ultrasound cohorts | UDIAT, BUID, BUSUC, BUSBRA; avoid simultaneous thyroid-anatomy shift |
| **D-143** | Retain official test split rather than entire dataset | External N=25+35+122+282=464 from original manifest |
| **D-144** | Keep frozen model, prompts and tau for all external evaluations | No OOD re-training/recalibration; prompt provenance must still be verified |
| **D-145** | Conclude OOD ranking utility only within studied cohorts | AURC advantage observed but no arbitrary-hospital guarantee |
| **D-146** | Report fixed-cutoff coverage loss as major result | 51.3% pooled OOD, 44.7% BUSBRA; manual review cost |
| **D-147** | Record absence of OOD predicted-empty samples | 464/464 nonempty, so two-stage false negative hazard untested OOD |
| **D-148** | Reconcile pooled OOD results by per-case micro weights | Baseline 0.7780, accepted 0.8718, rejected 0.6791 |
| **D-149** | Replace incorrect preliminary aggregate gain without erasing history | +0.0939 CI [0.0775,0.1113] supersedes +0.1021 |
| **D-150** | Restore BUSI test Random AURC to N65 cohort | 0.1687, not N78-mixed 0.1406 |
| **D-151** | Report pooled AURC separately from weighted and macro AURC | U_fg 0.1267 pooled, 0.1314 weighted, 0.1168 macro |
| **D-152** | Avoid unqualified claims of eliminating lesion-size confounding | Near-zero BUSI rank correlation does not exclude nonmonotone/OOD effects |
| **D-153** | Avoid claims of clinical safety from 31 regression tests | Synthetic checks and image-level metrics do not validate prospective care |
| **D-154** | Use literature only for exact concepts and equation-matched relationships | U_fg/Roy local mean; AVG/ATA caveats; unimplemented PLM/QFR |
| **D-155** | Write all project code/documentation in English and third-person scientific voice | Publication-ready evidence hierarchy; no tool/agent narration |
| **D-156** | Freeze authoritative Step11C numbers for paper drafting; make no new selection | Documented, falsifiable outcomes with provenance gaps explicit |
| **D-157** | Correct MedCLIPSeg CVPR 2026 Table 2 published baseline comparison | Authoritative published values: BUSI 85.72%, BUSBRA 75.06%, BUSUC 84.37%, BUID 78.99%, UDIAT 74.64%. All five cohorts classified as Category A (replicated lineage equivalence, difference <= 0.07%) |
| **D-158** | Resolve full BUSI test (Setting B) accepted DSC discrepancy | Canonical per-image CSV confirms accepted DSC = 0.924127 (0.9241), gain +0.0669, accepted NSD 0.9490; supersedes draft 0.9193 transcription |
| **D-159** | Reconcile patient identifier hierarchy and clustering proxies | Distinguish verified clinical IDs (absent across all sets), filename grouping proxies (BUSBRA bus_XXXX: 262 clusters across 282 cases), and unknown identity (BUSI, UDIAT, BUID, BUSUC). State bootstrap as image-level resampling |
| **D-160** | Formal manuscript structure and IEEE Computer Society 10-page format selection | Complete 10-page IEEEtran manuscript with 5 publication figures and comprehensive empirical/theoretical analysis |
| **D-161** | Refine background dilution wording | State explicitly that Eq. (10) explains why U_mean is confounded by area, while U_fg superiority is an empirical finding across multi-center cohorts rather than an a priori mathematical guarantee for arbitrary metrics |
| **D-162** | Bootstrap empirical resolution reporting convention | Standardize all bootstrap p-values to p < 0.001 corresponding to B = 1,000 paired resamples empirical resolution (1/B = 0.001), avoiding uncalibrated asymptotic p < 0.0001 assertions |
| **D-163** | Codify empty mask evaluation policy | Formally define DSC = 1.0 for both-empty (pred and GT empty) and DSC = 0.0 for false-positive predictions on empty GT in Setting B |


## 10. Publication-facing decision synthesis

**Why `U_fg` and not `U_mean`?** An exact foreground/background decomposition predicts that a global mean can mix lesion uncertainty with foreground fraction. On BUSI validation, test and all four external test sets, the reported same-cohort AURC for `U_fg` was lower than global mean entropy. This is a defensible empirical selection argument; it does not prove that foreground conditioning is the only causal driver.

**Why `U_fg` and not `U_std_fg`?** Both showed similar quality-rank discrimination, and paired validation/test CIs of their AURC differences included zero. The simpler first-moment score was prespecified as primary after validation. It would be inaccurate to claim the second-moment score failed or was proven inferior.

**Why not `U_boundary`?** Its spatial focus can correspond to contour ambiguity and thus the NSD endpoint, yet uncertainty on a contour is not the same as contour error. Under the study's primary DSC risk metric it had higher AURC, while morphology introduced shape/padding sensitivity. It may remain useful for other tasks.

**Why not `U_threshold`?** The fixed high-entropy *conditional mean* discards count and spatial organization of high-entropy pixels and was poorly discriminative in the observed data. This does not show all threshold-based aggregation is invalid; the empty-set policy also differs from published ATA.

**Why not `U_ratio`?** Under a plausible low-background regime it approximates `1/foreground_fraction`; its near-perfect inverse area rank association in validation makes quality claims vulnerable to geometric proxies.

**Why a separate `two_stage` rule?** Foreground region scores are undefined for empty predicted masks. A fallback is needed for complete workflow evaluation. However, a confident missed lesion can be accepted, so this is an operational convenience and explicitly **not a diagnostic safety guarantee**.

**Why Oracle/Random?** The first is a mathematically ideal, GT-dependent lower risk bound at each coverage; the second is an uninformed ranking reference. They establish scale and operational meaning, not a new algorithm or a commercial system.

**Why a fixed external threshold?** It makes the domain-shift hypothesis falsifiable. Recalibrating independently for each OOD test set would confound reported performance with additional exposure to that dataset and obscure the observed coverage collapse.

**What the results actually establish:** In the recorded BUSI and four external cohorts, the selected image-level entropy score ranks segmentation quality well enough that the retained subset has higher mean DSC while a sizable fraction is rejected. The result is conditional on the model, prompts, source data, scoring definition, case mix and quality evaluator. It does not establish perfect failure detection, a calibrated error probability, clinical impact, or generalization to arbitrary centers.

## 11. Documentation and verification release checklist

- [ ] Verify full-resolution original per-image CSVs for BUSI validation, BUSI locked test and all four OOD datasets against all paper tables.
- [ ] Confirm freeze hashes, model provenance, original prompt table hashes, file manifests and code commit hashes; do not fabricate missing values.
- [ ] Retain source-level AURC grid/tie conventions and synthetic reference tests.
- [ ] Confirm NSD spacing/tolerance units and both-empty behavior with source code.
- [ ] Confirm whether patient-group independence and cross-set duplicate checks are possible; document limits if not.
- [ ] Report threshold coverage target separately from realized coverage, rejected counts and rejected DSC.
- [ ] Keep no-leakage code audit and the confident false-negative counterexample in the release.
- [ ] Verify every external bibliography entry at the equation/application level, especially claims of published baseline replication.
- [ ] Include negative and unresolved findings: `U_std_fg` non-significance, severe accepted BUSI failure, absent empty-mask OOD cases, and coverage collapse.
- [ ] Use English-only project artifacts; write figures, tables, names and captions in third-person scientific style.

## 12. Changelog and archive integrity

- **Historical v1.0–v1.7:** Method definitions, raw entropy audit, exploratory BUSI test, validation runs, Step8 statistics, Step9 selection and Step9.5 scientific audit. Earlier Spanish versions remain as a nonrelease source archive, not part of the English public document set.
- **v2.0 (2026-11-07):** Reconstructed the entire document in English; preserved all D-001–D-134 identifiers with an English decision ledger; added D-135–D-156 for Steps10–11C; integrated corrected case-level pooled OOD values, explicit numerical caveats, formal method selection rationale, published-claim corrections, and release-verification limits.

**Versioning rule:** Future entries receive new IDs rather than overwriting historical decisions. When a prior numeric assertion is corrected, retain the old value with a clearly named superseding value, cohort and root cause. All scientific numbers and tables should ultimately resolve to a machine-readable case-level record and a specific code configuration.
