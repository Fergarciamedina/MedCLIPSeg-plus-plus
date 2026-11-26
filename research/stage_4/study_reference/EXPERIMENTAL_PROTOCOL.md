# Stage 4 experimental protocol

Use this document to trace the study from frozen MedCLIPSeg inference through validation-only score selection, held-out BUSI testing and external testing. The [reproduction guide](../SELECTIVE_PREDICTION_STUDY.md#reproduction) lists the executable commands, and the [experimental rationale](EXPERIMENTAL_RATIONALE.md) explains the choice of cohorts, hardware and metrics. A later rerun from the saved per-image inference outputs reproduced the principal calibration, locked-test and external results. The full image-to-mask workflow and exact historical checkpoint and prompt provenance have separate verification requirements.

## 0. Prespecified scientific questions, hypotheses, and endpoints

| ID | Question and falsifiable alternative | Primary evidence | Predetermined decision relevance |
|---|---|---|---|
| **H1** | Does a predicted-foreground entropy mean rank segmentation failures better than a global pixel mean? | Same-cohort `ΔAURC(U_fg,U_mean)` on BUSI validation; independent locked BUSI test; external OOD comparisons | Main method rationale |
| **H2** | Does contour-band entropy provide a stronger predictor for boundary quality (NSD) than region quality (DSC)? | Spearman against both endpoints; same-cohort NSD/DSC AURC | Explain role and limitations of `U_boundary` |
| **H3** | Does foreground spatial dispersion improve on foreground mean? | `U_std_fg` versus `U_fg`, paired ΔAURC CI | Maintain comparator; avoid false uniqueness claims |
| **H4** | Does freezing a validation-derived threshold produce useful selective retention on untouched test cases? | Retained DSC/NSD, rejected DSC/NSD, realized coverage, CIs | Primary operational study |
| **H5** | Does the rejection rule retain ranking/selection utility on different breast-ultrasound datasets without calibration? | Four OOD cohorts, full per-domain and pooled data, target-versus-realized coverage | Main domain-shift study |
| **H6 (safety limitation)** | Can a confident, wrong/empty prediction bypass the uncertainty gate? | Synthetic false-negative counterexample; observed severe accepted failure | Defines non-guaranteed failure detection |

**Primary quality loss:** `L_DSC=1−DSC`; secondary `L_NSD=1−NSD`. **Primary method selection:** lower lesion-cohort AURC on BUSI validation, with equal cases and numerical definitions. **Operational evaluation:** threshold `U_fg≤0.064058324`, selected as BUSI-validation 80% target coverage. **External evaluation:** no post-hoc method/threshold/prompt selection. Hypotheses and directions are explicit; the tests do not imply guaranteed superiority on future hospitals.

## 1. Data lifecycle and strict split separation

### 1.1. Model and environment

- Source architecture: MedCLIPSeg, Koleilat et al., CVPR 2026. Original paper and implementation: https://openaccess.thecvf.com/content/CVPR2026/html/Koleilat_MedCLIPSeg_Probabilistic_Vision-Language_Adaptation_for_Data-Efficient_and_Generalizable_Medical_Image_CVPR_2026_paper.html ; https://github.com/HealthX-Lab/MedCLIPSeg .
- Checkpoint path in project workspace: `project/data/outputs/BUSI/trained_models/seed666/MedCLIPSeg_unimedclip_ViT-B-16_latest.pth`. Frozen during evaluation; exact original training/checkpoint lineage still needs release-level provenance.
- Execution hardware described in project logs: Apple Silicon M4, PyTorch MPS backend; seed=`666`, MC samples=`30`, batch configuration subject to runner, output masks and entropy arrays approximately 224×224.
- Project-defined modification happens **after inference**. No training of MedCLIPSeg, secondary RF, quality-regression network, or fine-tuned calibrator is part of the uncertainty-rejection contribution.
- Probabilistic output: mean sigmoid probability over 30 stochastic passes; pixelwise entropy in **nats** stored raw float32 `.npy`, mask is a binarized prediction (`0.5` threshold per evaluation log). Colorized PNGs are visualization artifacts only.

### 1.2. Cohorts and denominators

| Cohort | Images | Lesion-positive reference | Empty predicted masks | Role and interpretation |
|---|---:|---:|---:|---|
| BUSI validation | 78 | 65 | 13 | **Only** method selection and threshold calibration |
| BUSI locked test | 78 | 65 | 13 | Independent in-domain evaluation after freeze |
| UDIAT official test | 25 | 25 | 0 reported | OOD near-domain, same organ |
| BUID official test | 35 | 35 | 0 reported | OOD, same organ |
| BUSUC official test | 122 | 122 | 0 reported | OOD, same organ |
| BUSBRA official test | 282 | 282 | 0 reported | OOD, same organ |
| **External pool** | **464** | **464** | **0 reported** | Cross-domain analysis, 4 heterogeneous external cohorts |

BUSI test and validation image IDs were reported non-overlapping; **this does not establish patient-level independence** without patient identifiers. No separate external OOD dataset was used to recalibrate the selection threshold. Official test split counts correspond to the MedCLIPSeg dataset preparation manifest: https://github.com/HealthX-Lab/MedCLIPSeg/blob/main/assets/DATASETS.md . Full-dataset sizes from unrelated papers must not be substituted for the **official test subsets** above.

**Domain shift definition:** external images differ in source distribution, scanner/clinical collection/protocol and possibly label protocols. Using only breast ultrasound reduces an obvious *anatomy* confound relative to switching to thyroid, but it does **not** isolate a single causal factor. DDTI is not part of the final four-dataset study. Avoid declaring verified zero patient overlap solely from institutional names; use known patient IDs or declare overlap not exhaustively excluded.

### 1.3. Prompt handling and label leakage

- The source MedCLIPSeg datasets include prompt files. The evaluated runs should be checked against the exact **prompts used per image**, prompt-table hashes, and filename→prompt mappings.
- Any class-conditioned prompt that uses a GT diagnosis (benign/malignant) must be labeled as an **oracle-information input** unless diagnosis was genuinely known in the intended inference workflow. It is **not** automatically deployment-equivalent to a class-agnostic prompt.
- No test outcome can be used to select a better external prompt after seeing DSC/AURC; that would confound domain transfer with prompt optimization.
- The execution summaries assert frozen original prompts. **Release gate:** verify actual prompt files and their SHA256s for BUSI and all four external sets; the summarized text alone does not prove this.

## 2. Inference-to-evaluation separation

**Inference-only input:** image, available text prompt, frozen checkpoint, seed/config, masks/probabilities/entropy. **Evaluation-only input:** GT segmentation, DSC, NSD, diagnostic category, subgroup and failure flags. No evaluation-only variable may appear inside an acceptance decision.

`U_fg`, `U_std_fg`, `U_mean`, `U_boundary`, `U_positive`, `U_threshold`, `U_ratio` are defined in [Theory](THEORY_AND_METRICS.md) §4. Source implementation map from project execution reports:

| Stage | Main project file | Output contract |
|---|---|---|
| Source inference | `shared/medclipseg/test.py`; `run_validation_inference.py` | Binary PNG masks, raw entropy `.npy`, official segmentation CSV |
| Aggregation | `research/stage_4/src/uncertainty/aggregation.py` | Seven scores; valid flags; mask areas and score metadata |
| Aggregation runner | `research/stage_4/experiments/prepare_uncertainty_scores/compute_uncertainty_csv.py` | Image-keyed CSV with scores plus DSC/NSD joined for **evaluation only** |
| Statistical methods | `research/stage_4/src/uncertainty/metrics.py` | Spearman, bootstrapped CIs, failure analysis |
| Selective decisions | `research/stage_4/src/uncertainty/selective_prediction.py` | Score-based ordering, coverage, risk, empty-mask policies, threshold application |
| Validation selection | `research/stage_4/experiments/validation_selection/run_validation_selection.sh` | Frozen YAML/JSON and decision report |
| Locked BUSI test | `research/stage_4/experiments/locked_busi_test/run_locked_test_evaluation.py` | Test-level tables/figures, no recalibration |
| OOD study | `research/stage_4/experiments/external_test/run_external_inference_and_aggregation.sh`, `run_ood_selective_prediction_evaluation.py` | External per-dataset and pooled tables, figures |

**Interpretation invariant:** `U_fg` is computed from predicted foreground, never GT foreground. Empty predicted areas must be represented as undefined for region-based metrics, then handled by a **separately documented** deployment policy. The grayscale/color PNG must never be used as a quantitative entropy source.

## 3. Historical execution sequence, preserving decisions and evidence status

| Step | Purpose | Completed deliverable / critical outcome |
|---|---|---|
| 1 | Data/model read-only audit | 78 BUSI test RAW float32 maps; matching predictions and GT; baseline DSC check |
| 2 | Seven aggregators | `aggregation.py` and explicit missing-region conventions |
| 3 | Synthetic tests | Morphological border and undefined-region tests; initial 15/15 pass |
| 4–5 | Exploratory BUSI-test extraction/figures | Exploratory output generated before validation work; **not** legitimate for tuning/selection |
| 6 | Frozen BUSI validation inference | 78 validation images/GT/prompts; baseline mean DSC=`0.8767`, NSD=`0.9040` |
| 7 | Validation aggregation | 78 matched rows, 65 lesion cases, 13 true normals, two severe DSC failures |
| 8 | Statistical screening | Spearman, risk–coverage, lesion area, sparse failure AUROC, 20/20 tests |
| 9 | Validation-only selection | `U_fg` selected, `U_std_fg` comparator; rejection thresholds and empty policy frozen; 23/23 tests |
| 9.5 | Pre-test audit | YAML/JSON equivalence, AURC/ties/bootstrap review, primary 80% operation designated |
| 10 | Locked BUSI test | Primary `U_fg` acceptance at frozen threshold: `47/65`, DSC accepted `0.9031`, baseline `0.8286`; 27/27 tests |
| 11A | External dataset readiness | Four original breast-ultrasound OOD datasets chosen; DDTI optional/out-of-scope |
| 11B | Frozen external test execution | UDIAT+BUID+BUSUC+BUSBRA, 464 samples, 31/31 tests |
| 11C | Numerical reconciliation | Corrected pooled versus macro DSC/AURC, restored consistent BUSI `N=65` Random AURC=`0.1687` |
| 12A | Scientific documentation | This consolidation; **not** a new experiment or threshold search |

**History note:** some early test outputs existed before validation calibration. The confirmatory selection was reported to have been performed strictly with BUSI validation. The independence and pre-registration claim is supported by execution chronology and frozen configuration records, not by a mathematical guarantee; the release should preserve timestamps, logs, hashes, and commit history wherever available. Any early exploratory inspection must remain disclosed rather than retrospectively hidden.

## 4. Validation evidence: how candidates were selected

### 4.1. Descriptive score distributions, BUSI validation (`N=78`)

| Score | Valid N | Mean | Std | Min–Max |
|---|---:|---:|---:|---|
| `U_mean` | 78 | 0.008225 | 0.009616 | 0 to 0.039888 |
| `U_positive` | 78 | 0.008977 | 0.010427 | 0 to 0.041934 |
| `U_fg` | 65 | 0.051638 | 0.023894 | 0.020599 to 0.145610 |
| `U_boundary` | 65 | 0.257360 | 0.112729 | 0.082326 to 0.518154 |
| `U_threshold` | 65 | 0.395187 | 0.009608 | 0.363855 to 0.431394 |
| `U_std_fg` | 65 | 0.141477 | 0.026860 | 0.094623 to 0.222098 |
| `U_ratio` | 65 | 16.038413 | 21.719865 | 1.488932 to 120.995205 |

These scores have different ranges and interpretations; one may compare **ranks and rejection performance**, not raw magnitudes across metric definitions. `U_ratio` is unitless but can exceed 100; `U_std_fg` does not mean uncertainty *across stochastic runs*. The two severe validation cases are `45_malignant.png` (DSC=`0.0491`, `U_mean=0.021035`, `U_fg=0.067293`) and `120_benign.png` (DSC=`0.3737`, `U_mean=0.003271`, `U_fg=0.102933`, predicted area 753 px). The latter supports the hypothesis that global averaging can obscure high uncertainty in a small foreground.

### 4.2. Validation Spearman and hypotheses

For **identical 65 lesion-present BUSI validation images**, Spearman rho against DSC was: `U_fg −0.7202`, `U_std_fg −0.7068`, `U_boundary −0.2273`, `U_mean −0.0451`, `U_positive −0.0207`, `U_ratio −0.1874`, `U_threshold +0.0178`. Against NSD: `−0.6925`, `−0.6768`, `−0.4174`, `−0.2570`, `−0.2360`, `+0.0499`, `+0.0186`, respectively. See [Theory](THEORY_AND_METRICS.md) §6 for reported CIs and sign interpretation.

**Why look at area?** The exact decomposition `U_mean=αU_fg+(1−α)U_bg` implies a potential foreground-size mixing effect. In validation, `ρ(area,U_mean)=+0.9148`, `ρ(area,U_positive)=+0.9271`, `ρ(area,U_boundary)=+0.7987`, `ρ(area,U_ratio)=−0.9957`, `ρ(area,U_fg)=−0.0848`, `ρ(area,U_std_fg)=−0.1118`. These observations are **not a formal multivariable causal adjustment**. They justify cautious size-stratified sensitivity analysis and do not establish that `U_fg` is area-independent in all data.

### 4.3. Same-sample validation AURC, primary DSC risk

| Rank | Method | AURC(1−DSC) | AURC(1−NSD) | Scientific decision |
|---:|---|---:|---:|---|
| Reference | Oracle | 0.0686 | 0.0412 | GT-informed unattainable minimum risk |
| 1 | **`U_fg`** | **0.0839** | **0.0563** | Lowest measured uncertainty-score AURC; selected |
| 2 | `U_std_fg` | 0.0850 | 0.0573 | Prespecified close comparator; unresolved difference |
| 3 | `U_boundary` | 0.1302 | 0.0848 | Better than random for these cohorts; less predictive of region DSC |
| 4 | `U_ratio` | 0.1368 | 0.1159 | Size-driven ratio concerns |
| Reference | Random | 0.1434 | 0.1116 | No-information expected risk reference |
| 5 | `U_mean` | 0.1438 | 0.0969 | Not useful versus random on DSC lesion-cohort AURC; more favorable on NSD |
| 6 | `U_positive` | 0.1455 | 0.0984 | Strict positivity filtering did not rescue the DSC ranking |
| 7 | `U_threshold` | 0.1750 | 0.1412 | Conditional high-entropy mean was worst by AURC |

**Selection logic.** `U_fg` was not selected simply because `0.0839<0.0850`: that comparison had a paired bootstrap 95% interval crossing zero (`Δ=−0.0011`, CI `[−0.0033,+0.0008]`, reported p=`0.294`). It was chosen as a simple/no-training primary measure with a slight point-estimate advantage on both DSC and NSD and less observed rank dependence on lesion area; **`U_std_fg` remains scientifically competitive**. In contrast, `U_fg−U_mean` validation `ΔAURC=−0.0599`, CI `[−0.0871,−0.0320]`; versus `U_boundary`, `−0.0462`, CI `[−0.0711,−0.0221]`. Inferential significance needs the paired bootstrap assumptions discussed in the theory file.

### 4.4. Published-work comparison protocol

No published paper receives credit for an internally defined formula without equation-to-code verification. The method provenance is: `U_fg` mean-on-predicted-segment operation closely corresponds to Roy et al. (2019) per-structure aggregation; `U_mean` is global AVG used in Guarino et al. (2026); `U_threshold` is an ATA-like adaptation with different empty-region behavior; `U_boundary`, `U_std_fg`, `U_positive`, and `U_ratio` are internally specified aggregation procedures motivated by general literature, **not full replications** of Jungo, Mehrtash, DeVries, or Eaton-Rosen. PLM, QFR and Soft Dice Confidence are **not implemented**. See [Evidence](SCIENTIFIC_EVIDENCE_AND_CLAIMS.md) for bibliography and exact scope.

## 5. Frozen validation-derived operating specification

### 5.1. Identical numeric cutoff under test and OOD

**Selected method:** `U_fg` (mean raw predictive entropy on the predicted foreground). **Secondary method:** `U_std_fg`. **Primary operation:** `coverage_80`, threshold `0.064058324` (full precision as reported; YAML/JSON authoritative). All cases are accepted iff their computed score passes the applicable branch and cutoff.

| Validation target coverage | Frozen `U_fg` threshold | Realized BUSI val | Accepted val |
|---:|---:|---:|---:|
| 95% | 0.10193 | 93.8% | 61/65 |
| 90% | 0.08346 | 89.2% | 58/65 |
| 85% | 0.07270 | 84.6% | 55/65 |
| **80% (primary)** | **0.064058324** | **80.0%** | **52/65** |
| 75% | 0.05969 | 75.4% | 49/65 |
| 70% | 0.05621 | 69.2% | 45/65 |

The [paired comparison table](../experiments/validation_selection/results/paired_bootstrap_aurc_ci.csv) is the statistical record for the comparison with foreground dispersion. The narrative strings in the archived YAML/JSON contain a different interval and p-value. The numeric operating-point fields define the deployed policy.

Additional frozen parameters: `T=30`, `seed=666`, pixel `U_threshold` `t=0.1`, `U_boundary` width `w=2` with square 5×5 kernel, `U_ratio` epsilon `10^-6`, two-stage empty branch `τ_neg=10^-4`, mask binarization reported `0.5`. Distinguish source mask threshold, pixel entropy threshold, image threshold, empty branch threshold, and NSD distance tolerance.

### 5.2. Artifact authority and integrity checks

At end of Step 9.5 / immediately before Step 10, the frozen YAML and JSON were reported semantically equivalent. After recording the explicit primary operating point, Step 10 reported SHA256 hashes:

| Artifact | SHA256 captured in Step 10 report |
|---|---|
| Frozen YAML | `6e24ddb14baac0049b79f1499c9b281f988b07b9e375747b02b8f596e392382e` |
| Frozen JSON | `ddbea219e53c6a2163fae0e04b98b4d747873764e6198e66c614e2a56d3f4806` |
| `aggregation.py` | `4563d2b18fe5c82705209ccafe5b939f11829cdb5e9f59a542059f4ee915b9eb` |
| `metrics.py` | `9c0f18b3e974c939cd6d40554849b2ee4e21f9d9e715b1e870a2ae08c9699098` |
| `selective_prediction.py` | `2fc490932fc2c346a210ba143c9fe9990b6ea5377cb1be450bf8f95b44c8c3f2` |
| `eval.py` | `85245ee04a5f49124e0e7d515e5ccaf470f63ac49e1f39af7d9a2fd9a3470a96` |
| `SurfaceDice.py` | `e5349b0f18112b0319d80f1b626590db02fcfaf4773cec2474e6a8747973fb2f` |

These are **reported historical hashes**, not newly measured on the attached machine. Regenerate and compare against actual release files before claiming bitwise provenance. Checkpoint, raw data and prompt manifest hashes must also be recorded for complete reproducibility.

### 5.3. Empty mask fallback and policy limits

On predicted-nonempty cases: select by `U_fg≤τ_img`. On predicted-empty cases: two-stage rule accepts if `U_mean≤τ_neg`, otherwise rejects. A deployment selector receives **no GT, DSC/NSD, or diagnostic filename substring**. This rule is motivated by a need for a defined decision on otherwise NaN scores; it **does not** prove that an accepted empty mask is a true negative. The synthetic false-negative counterexample remains open clinically, because the external study reported no predicted-empty cases. With different denominators, **Setting A (lesion-positive)** and **Setting B (all images)** must not be compared as if they differed only by algorithm.

## 6. Locked BUSI test results (not used to choose the method)

Primary operation, reference lesion-positive `N=65`: `U_fg τ=0.064058324` accepted `47/65=72.3%`, rejected `18/65=27.7%`. Mean DSC baseline=`0.8286`; accepted=`0.9031`; rejected=`0.6341`; accepted-vs-full gain=`+0.0745`, reported 95% bootstrap CI `[+0.0349,+0.1235]`. Mean NSD baseline=`0.8602`; accepted=`0.9349`; rejected=`0.6651`; NSD gain=`+0.0747`.

| Test method | AURC 1−DSC on same `N=65` | Interpretation |
|---|---:|---|
| Oracle | 0.0692 | GT-dependent lower bound |
| `U_fg` | **0.0790** | Selected method |
| `U_std_fg` | 0.0793 | Nearly identical; paired delta CI includes zero |
| `U_boundary` | 0.1113 | Better than global in this test cohort |
| `U_mean` | 0.1262 | Higher risk than `U_fg` across curve |
| `U_positive` | 0.1282 | No improvement over global mean |
| `U_ratio` | 0.1499 | Worse ranking than foreground methods |
| Random | 0.1687 | Corrected `N=65` reference |
| `U_threshold` | 0.2045 | Worse than uninformed random under this empirical convention |

Paired BUSI-test AURC contrasts reported: `U_fg−U_std_fg=−0.0003`, CI `[−0.0024,+0.0014]`, p=`0.786`; `U_fg−U_mean=−0.0472`, CI `[−0.0690,−0.0277]`; `U_fg−U_boundary=−0.0324`, CI `[−0.0440,−0.0198]`. These comparisons should not be converted into claims about a patient-level or cross-hospital guarantee.

**Severe failure retained:** `116_malignant.png` had DSC=`0.2193`, `U_fg=0.06279`, therefore passed the `0.064058324` cutoff. Four of the five lowest-DSC listed BUSI-test cases were rejected, one was retained. This specifically falsifies any claim of perfect failure detection. Benign baseline DSC=`0.8655` with 37/44 accepted (84.1%); malignant baseline DSC=`0.7514` with 10/21 accepted (47.6%). Subgroup comparisons are descriptive and may depend on case mix; do not claim diagnosis-aware fairness.

**All-image deployment Setting B, BUSI N=78:** two-stage accepted 60 (47 lesion-predicted cases plus 13 true-normal empty predictions), rejected 18, realized coverage 76.9%, accepted DSC=`0.9241` versus full N78 baseline=`0.8572`. This result is highly sensitive to 13 empty/empty normal cases whose DSC=1 and should not be interpreted as a like-for-like improvement over lesion-only Setting A.

## 7. Frozen OOD transfer: four breast-ultrasound test splits

**Scope:** exact same frozen `U_fg` method/threshold and BUSI checkpoint on the original MedCLIPSeg external breast-ultrasound test splits, without retraining or cross-domain threshold tuning. All four reported OOD cohorts were GT lesion-positive; their predicted foreground masks were nonempty (`464/464`). That means the two-stage empty-prediction safety branch **was not empirically challenged** on these OOD cases.

### 7.1. Per-dataset locked threshold outcome

| Dataset | Test N | Baseline DSC | Accepted / rejected | Realized coverage | Accepted DSC | Rejected DSC | Gain vs full DSC [95% CI] |
|---|---:|---:|---:|---:|---:|---:|---|
| UDIAT | 25 | 0.7467 | 14 / 11 | 56.0% | 0.8747 | 0.5837 | +0.1281 [0.0325,0.2569] |
| BUID | 35 | 0.7900 | 20 / 15 | 57.1% | 0.8818 | 0.6675 | +0.0919 [0.0339,0.1644] |
| BUSUC | 122 | 0.8430 | 78 / 44 | 63.9% | 0.8982 | 0.7453 | +0.0551 [0.0363,0.0758] |
| BUSBRA | 282 | 0.7511 | 126 / 156 | 44.7% | 0.8536 | 0.6683 | +0.1025 [0.0789,0.1278] |
| **Pooled case-level** | **464** | **0.7780** | **238 / 226** | **51.3%** | **0.8718** | **0.6791** | **+0.0939 [0.0775,0.1113]** |

All gains compare **selected sample mean DSC** with **full sample mean DSC**. This is a selection result, not a raw model-performance improvement. **External distribution shift impaired coverage transfer:** despite 80% nominal validation target, only 44.7–63.9% were accepted on individual OOD datasets. Absolute external mean error on retained cases may therefore be better only because substantially more cases were deferred to human review. Any practical utility statement must disclose that cost.

### 7.2. All-dataset AURC and comparison reference

| Dataset | `U_fg` | `U_mean` | Random | Oracle |
|---|---:|---:|---:|---:|
| BUSI locked test N65 | 0.0790 | 0.1262 | 0.1687 | 0.0692 |
| UDIAT N25 | 0.1133 | 0.1503 | 0.2432 | 0.0944 |
| BUID N35 | 0.1094 | 0.2411 | 0.2040 | 0.0935 |
| BUSUC N122 | 0.0916 | 0.1365 | 0.1557 | 0.0834 |
| BUSBRA N282 | 0.1529 | 0.2020 | 0.2480 | 0.1200 |
| **Pooled OOD case-level N464** | **0.1267** | **0.1864** | **0.2216** | **0.1046** |
| **Weighted mean of per-OOD AURCs** | **0.1314** | **0.1849** | **0.2202** | **0.1070** |
| **Macro mean of four OOD AURCs** | **0.1168** | **0.1825** | **0.2127** | **0.0978** |

The latter three rows are **different estimands** and must remain separately labeled. The historical error that listed BUSI Random AURC as `0.1406` was due to evaluating all 78 BUSI cases instead of the N65 lesion subset; the corrected figure is `0.1687` when comparing lesion-cohort rankings. The old OOD aggregate `DSC baseline=0.7711`, accepted=`0.8732`, gain=`0.1021`, rejected=`0.6635` was a draft transcription, **superseded** by the micro results above.

### 7.3. Exact micro/macro definitions and numerical reconciliation

Let datasets `d=1,...,4`; `n_d` = number of test images; `a_d` = number accepted; `q_{di}` = per-image Dice. Pooled micro means weight each **case** equally:

\[
\bar Q_{\text{all,micro}}=\frac{\sum_d\sum_{i=1}^{n_d}q_{di}}{\sum_d n_d},\quad
\bar Q_{\text{acc,micro}}=\frac{\sum_d\sum_{i:a_{di}=1}q_{di}}{\sum_d a_d}.
\]

Macro means weight each **dataset** equally:

\[
\bar Q_{\text{all,macro}}=\frac14\sum_{d=1}^4\bar Q_{\text{all},d},\quad
\bar Q_{\text{acc,macro}}=\frac14\sum_{d=1}^4\bar Q_{\text{acc},d}.
\]

The final audit reported pooled means `(0.7780,0.8718,0.6791)` for full/accepted/rejected and macro means `(0.7827,0.8771,0.6662)`. Their differences arise because BUSBRA has many more samples than UDIAT. Using the rounded per-domain values reproduces micro means to four decimals. **The pooled gain** is `0.8718−0.7780≈+0.0939`, not a weighted mean of the four separate gains. With only **four** source domains, pooled image bootstrap intervals do not estimate uncertainty over a random population of future hospitals.

### 7.4. Additional transfer limitations

- No cross-domain recalibration was performed; therefore retention fell from 80% nominal to 51.3% pooled OOD. This is evidence of **threshold/coverage shift**, not evidence of stable coverage.
- External test cases can share patient provenance, scanner setups, labeling conventions or calibration artifacts; independent image-level observations cannot be assumed without metadata.
- `U_mean` correlates strongly with predicted lesion area in the study, but Spearman does not prove a singular mechanism or complete removal by `U_fg`. The audited cross-domain analysis noted possible residual correlations up to approximately `|rho|=0.31` in some cohorts.
- Paired bootstrap `B=1000` and positive CIs for observed selected-vs-full gain describe **within-cohort** uncertainty under assumptions, not cross-hospital guarantees.
- Zero predicted-empty lesions in the 464 OOD cases means this external suite provides no empirical rate for *confidently missed empty-mask tumors*; a synthetic counterexample remains important.
- Model-origin differences in anatomy, acquisition, case mix, annotations, and prompts can interact; the design supports an **external test of usefulness**, not a causal isolation of scanner type.

## 8. Statistical methods and quality checks

1. Compute per-image raw entropy in original nats; verify finite values, shape, dtype, one-to-one mask pairing.
2. Derive seven image scores only from predictions, not GT. Keep `NaN` and explicit valid flags for undefined regional scores.
3. For method comparisons, take **identical sample IDs** and the same quality metric; report sample size and empty-mask handling.
4. Compute Spearman on rank-transformed scores and DSC/NSD, with documented ties; do not infer causation.
5. Compute AURC with the same coverage grid, endpoints, and tie rule. Report lower-is-better risk and Oracle/Random values on the **same cohort**.
6. Bootstrap comparisons on paired case indices. Report CI construction and a defensible null hypothesis. With repeated patients, consider cluster bootstrap if IDs available.
7. Freeze thresholds on BUSI validation only, write YAML/JSON/manifest and code hashes; confirm no BUSI-test or OOD result affected selection.
8. Evaluate original frozen BUSI test and OOD sets once as confirmatory, record realized coverage rather than renormalizing thresholds to hit targets.
9. Report both selected and rejected quality, not only the improvement among accepted cases.
10. Inspect accepted severe failures and patient/pathology subgroup behavior, especially missed lesions and different lesion sizes.
11. Enforce synthetic edge tests for ties, empty masks, morphological padding, zero entropy, false-negative confidence and reproducibility.
12. Retain raw reports **without silently replacing erroneous historical tables**; corrected versions include reconciliation notes and authoritative output paths.

The final test suite reportedly reached **31/31 passing** after OOD-specific regressions. A passing suite confirms test-covered implementation properties; it is not proof that the deployed system is clinically safe or that all source data are independently verified.

## 9. Authoritative artifact paths for final replication

Paths below are relative to the project workspace (some path segments are based on recorded output manifests, not independently mounted here).

| Artifact | Reported relative project location |
|---|---|
| BUSI validation inference | `data/outputs/BUSI/validation/seed666/` |
| BUSI validation 7-score CSV | `data/outputs/BUSI/validation/seed666/aggregation_results/uncertainty_scores_BUSI_val_seed666.csv` |
| BUSI validation statistical analysis | `data/outputs/BUSI/validation/seed666/statistical_analysis/` |
| Frozen selection YAML/JSON | `data/outputs/BUSI/validation/seed666/selective_prediction/frozen_selective_prediction_config.{yaml,json}` |
| Step 9 decision report | `data/outputs/BUSI/validation/seed666/selective_prediction/DECISION_REPORT_STEP9.md` |
| Pre-test audit | `data/outputs/BUSI/validation/seed666/selective_prediction/PRE_TEST_AUDIT.md` |
| BUSI locked test report | `data/outputs/BUSI/test_evaluation/seed666/LOCKED_TEST_EVALUATION_REPORT.md` |
| BUSI locked test score tables | `data/outputs/BUSI/test_evaluation/seed666/tables/` |
| Four OOD per-set outputs | `data/outputs/{UDIAT,BUID,BUSUC,BUSBRA}/test_evaluation/seed666/` |
| Cross-domain summary and AURC | `data/outputs/cross_domain_evaluation/seed666/cross_domain_summary.csv` and companion figures |
| Final OOD evaluation report | [Cross-domain results](../experiments/external_test/results/cross_domain_summary.csv) |
| Corrective audit | [Cross-domain reconciliation](RESEARCH_DECISIONS_AND_LIMITATIONS.md#48-step-11c-reconciliation-of-pooled-calculations-and-legacy-discrepancies) |

**Recommended provenance file for release:** `experiment_manifest.json` including dataset names/test split hashes, checkpoint hash, prompt-file hash, script commits/hashes, environment versions, seed, `T`, mask binarization rule, entropy formula, area convention, NSD tolerance/spacing, tie handling, AURC integration convention, bootstrap RNG seed and exact denominator of each table. A manifest template cannot substitute for values measured from the actual working tree.

## 10. Threats to validity and what data would resolve them

| Threat | Why it matters | Current mitigation | Remaining evidence request |
|---|---|---|---|
| Earlier BUSI-test exploratory inspection | Could indirectly influence hypothesis choices | Confirmatory selection reported on validation only; explicitly disclosed history | Full timestamps/version history and exact selection code freeze |
| Shared patient images | Effective N may be smaller than image count | No patient-level claims | IDs and patient-group split audit |
| Prompt leakage or prompt differences | Text model depends on prompt | Predeclared fixed original-prompt approach | Exact per-image prompt logs/hashes and GT availability check |
| Checkpoint origin | A frozen file may have uncertain training provenance | Frozen test-time weights | Checkpoint metadata/hash and original training split manifest |
| No calibrated probability of failure | Entropy is a score, not a failure probability | Evaluate ranking/rejection empirically | Reliability/conditional-risk study, not in current scope |
| Empty-mask blind spot | Confident missed lesion may be accepted | Synthetic counterexample and explicit warning | Prospective/OOD cohorts containing predicted-empty false negatives |
| Inconsistent AURC conventions | Can change ranking area numerically | Step 9.5 and 11C audit reports | Exact code and synthetic reference outputs bundled |
| Class imbalance / sample selection | All OOD test cases lesion-positive | Separate Setting A/B and subgroup reporting | External negative cases, balanced triage evaluation |
| Only four OOD domains | No inference over arbitrary future sites | Report per-domain results rather than claiming universality | Additional hospitals/independent cohorts |
| Endogenous acceptance | Accepted gain is selection, not treatment effect | Report accepted/rejected/full quality | Prospective utility and actual review costs |
| Selected threshold coverage shift | 80% BUSI validation target became 51.3% OOD | Report realized coverage and reject fraction | Explicit review-capacity/coverage utility model |
| Multiple statistical comparisons | Exploratory tests inflate false positives | Primary comparison declared and caveats | Multiplicity-aware or preregistered confirmatory plan |
| Visualized entropy maps | Per-image normalization destroys absolute scale | Raw `.npy` only for metrics | Reproducible code/data validation |

## 11. Bibliography and document links

Primary publication and method references are maintained with complete URLs and attribution scope in [Theory and Metrics](THEORY_AND_METRICS.md) §12 and [Scientific Evidence and Claims](SCIENTIFIC_EVIDENCE_AND_CLAIMS.md). This protocol cites MedCLIPSeg (Koleilat et al., CVPR 2026), Roy et al. (NeuroImage 2019), Guarino et al. (CVPR 2026), Geifman and El-Yaniv (NeurIPS 2017), Jungo et al. (Frontiers 2020), Mehrtash et al. (TMI 2020), and Efron and Tibshirani (1993) for **the concepts those sources actually establish**, not as proof that all their systems were implemented here.

## 12. Publication gate (after documentation, before submission)

- Confirm every headline number against the final corrected *per-image* tables and reproduce all weighted means and AURCs.
- Use **BUSI validation** only for the selected method and operating-point story; treat BUSI test and OOD as locked evaluation, not opportunities to improve the method.
- Label **raw model segmentation** versus **quality among retained predictions** unambiguously in abstract, tables and figure captions.
- For each claim, cite a literature source for external method/theory and a specific artifact for own experimental evidence (see claim ledger).
- Preserve `U_std_fg` as statistically unresolved comparator rather than a definitively inferior method.
- Report 238/464 accepted and 226/464 rejected under OOD; no safety claim without the full coverage cost and the confident failure counterexample.
- Use third-person scientific prose (“The study evaluates…” / “The results indicate…”), not AI-agent narration and not unsupported first-person anecdotes.
- Record unresolved data/provenance gaps transparently; do not fabricate raw CSV inspection, paper equations, source commits, confidence intervals, or clinical validation.
