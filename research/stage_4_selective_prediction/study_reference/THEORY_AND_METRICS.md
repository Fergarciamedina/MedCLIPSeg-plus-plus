# Theory and metrics

This reference defines the predicted mask, pixelwise entropy, the seven scan-level scores, and the metrics used to evaluate selective prediction. Start with the [Stage 4 overview](../SELECTIVE_PREDICTION_STUDY.md) for the research question and the [experimental rationale](EXPERIMENTAL_RATIONALE.md) for why these choices were made. The [results guide](../experiments/RESULTS_GUIDE.md) shows how to read the tables and the frozen policy.

## 1. Research question

**Central question.** Can a fixed, image-level summary of MedCLIPSeg's spatial predictive entropy rank segmentation failures and support useful selective rejection, without retraining, on BUSI and on four independent breast-ultrasound evaluation sets? The problem is **ranking segmentation quality**, not making an uncertainty image visually resemble a segmentation error map.

Three independently testable statements are relevant:

1. **Score-quality association:** larger uncertainty scores are associated with lower ground-truth Dice/NSD (Spearman's rho); association alone does not measure operational usefulness.
2. **Rejection utility:** retaining cases with scores at or below a threshold reduces *mean loss among retained cases*, at a measurable coverage cost (risk–coverage/AURC and a frozen operating point).
3. **Transfer:** a threshold selected only on BUSI validation maintains selective utility on independent BUSI test and external breast-ultrasound datasets, even if **coverage itself does not transfer**.

These are not guaranteed by predictive entropy, the original MedCLIPSeg publication, or general results on selective classification. The experiments test them. The statements do **not** assert clinical safety guarantees.

## 2. Symbol dictionary and exact units

| Symbol | Meaning | Units, domain, or source |
|---|---|---|
| `i`, `j`, `s` | Image, pixel, Monte Carlo sample indices | Integer; `s=1,…,T` |
| `x_i`, `t_i` | Input ultrasound image and text prompt | Image tensor and text |
| `Ω_i` | Pixel lattice for the entropy map | Usually 224×224 = 50,176 pixels |
| `Y_i`, `M_i` or `Ŷ_i` | Ground-truth mask, **predicted** binary mask | Values 0/1; GT is evaluation-only |
| `F_i={j:M_ij=1}` | **Predicted** foreground, not ground-truth lesion area | Pixel set |
| `p_ij^(s)`, `p̄_ij` | Sample-wise sigmoid probability and MC mean probability | `[0,1]` |
| `U_ij` | **Raw predictive entropy** | Nats, theoretical `[0,ln(2)]` |
| `T` | Monte Carlo samples | `T=30` |
| `B_w(F)` | Morphological band around the predicted boundary | Pixels; `w=2`, square 5×5 structuring element |
| `t_pixel` | Within-map threshold for `U_threshold` | `0.1` nats, **not** rejection threshold |
| `ε` | Denominator stabilizer for `U_ratio` | `10^-6`, same numeric entropy scale |
| `τ_img` | Image-level `U_fg` acceptance cutoff | `0.064058324` nats, from BUSI validation |
| `τ_neg` | Background mean threshold when predicted mask empty | `10^-4` nats |
| `δ_surface` | NSD surface tolerance | Reported as `2` pixel units in project evaluator; spacing metadata must be checked |
| `Q_i`, `L_i` | Segmentation quality and loss/risk | `Q∈[0,1]`, `L=1−Q` |
| `a_i(τ)` | Acceptance decision | `1` accept; `0` abstain/refer |
| `C`, `R(C)` | Coverage and selective risk | Proportion and mean loss |
| `ρ`, `ΔAURC` | Spearman coefficient, paired AURC contrast | Dimensionless |
| `N_val`, `N_ID`, `N_OOD` | Samples | BUSI validation 78; BUSI test 78; OOD 464 |

**Terms frequently confused:** `foreground` is the predicted set `F`, not the annotated lesion; `normal` is a dataset's diagnostic/reference category, not a diagnosis made from entropy; `rejection` means withholding a segmentation from automatic retention, not detecting disease; `confidence` is not guaranteed correctness; `domain shift` is a distributional change, not necessarily a demonstrated causal effect of a scanner.

## 3. Frozen probabilistic segmentation output

The source model is *MedCLIPSeg: Probabilistic Vision-Language Adaptation for Data-Efficient and Generalizable Medical Image Segmentation* [R01], evaluated without weight updates. It receives the image and associated prompt and produces a lesion probability field. In the project's inference code, multiple stochastic predictions were generated (`T=30`) and the **predictive entropy of the mean probability** was stored as a raw float32 `.npy` array.

For each pixel `j`:

$$
\bar p_j=\frac{1}{T}\sum_{s=1}^{T}p_j^{(s)},\qquad
U_j=H(\bar p_j)=-\bar p_j\ln\bar p_j-(1-\bar p_j)\ln(1-\bar p_j).
$$

`0 ln 0 := 0` by continuity. Entropy is measured in **nats**, because logarithms are natural. Examples: `H(0)=0`, `H(0.1)≈0.3251`, `H(0.5)=ln 2≈0.6931`, `H(0.9)≈0.3251`, and `H(1)=0`.

**Proof of the maximum.** `H'(p)=ln((1-p)/p)` and `H''(p)=−1/[p(1-p)]<0` for `0<p<1`. Thus the unique maximum is at `p=1/2`. The number does **not** equal the probability of a wrong segmentation. Two different pixel configurations can have identical entropies and different ground-truth errors.

**What this uncertainty actually captures.** `H(E[p])` is total predictive entropy. It is not a direct measurement of geometric error and is not, by itself, a pure epistemic/aleatoric decomposition. With a suitable distribution over stochastic predictions one may formally write

$$
H(\mathbb E[p])=\mathbb E[H(p)]+\left(H(\mathbb E[p])-\mathbb E[H(p)]\right),
$$

where the bracketed difference is a mutual-information-like epistemic contribution under the appropriate probabilistic interpretation. **That decomposition was not measured in these experiments.** Kendall and Gal [R06] and Gal and Ghahramani [R07] provide background, but the stochastic MedCLIPSeg adapter must not automatically be described as a standard MC-dropout implementation unless the actual code proves that equivalence.

**Spatial interpretation.** High entropy frequently occurs along predicted lesion boundaries because probabilities are less decisive there. A high boundary score identifies model ambiguity, *not* a boundary misdelineation unless compared with reference annotations. Conversely, a wrong but confident prediction can have low uncertainty. This distinction motivated a move from visual inspection to an empirical ranking test [R01,R04,R05].

**Source integrity.** Use the raw float32 entropy arrays, not color-mapped visualization PNGs. Per-image min–max scaling of a visualization changes absolute magnitudes and can invalidate cross-image ranking. Earlier estimates derived from pseudo-color PNGs are **excluded from confirmatory evidence**; see [Research Decisions](RESEARCH_DECISIONS_AND_LIMITATIONS.md).

## 4. Seven aggregation functions: operational definitions and anticipated failure modes

Let `avg_U(A) = (Σ_{j∈A}U_j)/|A|` for any nonempty pixel subset `A`. The seven implemented quantities were defined **before test selection**.

| Metric | Exact function | Intended test | Mathematical/operational limitation | Publication relationship |
|---|---|---|---|---|
| **`U_mean`** | `avg_U(Ω)` | Simple global comparator | Dominance of low-entropy background; size-mixture effect | Same local formula as global AVG of Guarino et al. [R03] |
| **`U_positive`** | `avg_U({j:U_j>0})` | Remove exact zeros | Floating-point tiny positives are retained; distribution changes | Internal ablation; not claimed as published baseline |
| **`U_fg`** | `avg_U(F)`, with `F={M=1}` | Contextual lesion uncertainty | Undefined for predicted-empty masks; can miss confident errors | Matches the **structure-level mean** operation in Roy et al. [R02], not the full QuickNAT model |
| **`U_boundary`** | `avg_U(B_w(F))` | Are uncertain borders predictive of quality? | Shape, perimeter, rectangular morphology and padding alter the band | Own contour-band construction; literature-motivated, **not** a verified exact reproduction |
| **`U_threshold`** | `avg_U({j:U_j>0.1})` | Focus on high entropy | Conditional mean discards uncertain-pixel count; undefined if no selected pixels | Related to ATA in Guarino [R03], but **our empty-set rule is NaN**, whereas their ATA returns zero |
| **`U_std_fg`** | `sqrt(|F|^-1 Σ_{j∈F}(U_j−U_fg)^2)` | Does intraregion heterogeneity add information? | Measures spatial dispersion, not stochastic variance between MC passes; undefined if `F=∅` | Internal adaptation; do **not** attribute its exact formula to Jungo |
| **`U_ratio`** | `U_fg / (U_mean + 10^-6)` | Does normalized contrast remove global scale? | Reciprocal size dependence, instability at tiny denominator; undefined if `F=∅` | Internal ablation; **not** QFR [R03] |

**Empty regions.** `U_fg`, `U_boundary`, `U_std_fg`, and `U_ratio` yield `NaN`/`valid=False` when their region is undefined. `U_threshold` also yields `NaN` when no pixel exceeds its threshold. The all-image `U_mean` remains computable. Do not equate undefined with `0`: zero means the defined region had precisely zero entropy, a different mathematical statement.

### 4.1. Derivation: why global uncertainty can be diluted by foreground area

Let `α=|F|/|Ω|`, `U_fg=mean(U|F)`, and `U_bg=mean(U|Ω\F)`. Partitioning the sum gives the **exact identity**

$$
U_{\rm mean}=\alpha U_{\rm fg}+(1-\alpha)U_{\rm bg}.
$$

If `U_bg≈0`, then `U_mean≈α U_fg`. Example: when a lesion occupies `α=0.015` and `U_fg=0.10`, the global score is approximately `0.0015`, even though lesion-region uncertainty is substantial. This is not a theorem that `U_mean` *always* fails: its behavior depends on foreground, background, their relative area and uncertainty. It is a **mechanistic reason to hypothesize** a size-sensitive ranking bias and test it.

**Experimental cross-check.** In BUSI validation, `120_benign.png` had predicted foreground 753/50,176 pixels (~1.50%), `U_mean=0.003271`, `U_fg=0.102933`, DSC=0.3737. The example illustrates the mechanism; a single case cannot prove the dataset-wide correlation is caused solely by dilution. Across BUSI validation lesion cases, `Spearman(U_mean,predicted area)=+0.9148`, versus `Spearman(U_fg,predicted area)=−0.0848`. The corresponding BUSI-test coefficients were reported as `+0.8960` and `−0.0328`. **Near-zero marginal rho is not proof of statistical independence or elimination of all confounding.**

### 4.2. Derived concern: ratio and inverse-size instability

When background uncertainty is negligible and the stabilizer is small relative to the global mean,

$$
U_{\rm ratio}\approx\frac{U_{\rm fg}}{\alpha U_{\rm fg}}=\frac1\alpha.
$$

Thus `U_ratio` can become a disguised inverse-foreground-area measure. This explains why **the formula itself** is problematic as an unadjusted quality score; the observed validation `ρ(area,U_ratio)=−0.9957` is consistent with the mechanism. Cases with very small `U_mean` may also generate large ratios. A “normalization” label does not establish removal of bias.

### 4.3. Boundary measurement and OpenCV edge conventions

For predicted foreground `F` and square structuring element `K_w` of side `2w+1`:

$$
B_w(F)={\rm dilate}(F,K_w)\setminus {\rm erode}(F,K_w),\quad w=2.
$$

This is a **two-sided, rectangular morphological band**, not a Euclidean distance transform, not exclusively an interior contour, and not the ground-truth error contour. In OpenCV, erosion/dilation at image borders depends on default padding and morphological border values. Synthetic tests found a 2×2 corner mask with dilated area 9, eroded area 1, and band area 8. The example should remain in regression tests. A fixed pixel-wide ring is **not** an invariant physical distance across resolutions/scanners.

### 4.4. Why `U_threshold` may lose ranking power

A *conditional average* over pixels above a threshold says how large the selected values are on average; it does **not** count how many pixels passed the threshold. Two images with one versus ten thousand high-uncertainty pixels can have the same score. On BUSI validation, `U_threshold` varied narrowly around `0.395`, whereas its region size could vary. This loss of count information is a plausible **interpretation**, not a proved causal explanation of the observed AURC disadvantage. Guarino et al. define ATA with an explicit zero value for the empty set; that is a reproducibility-relevant difference [R03].

### 4.5. Why `U_fg` and `U_std_fg` should not be conflated

`U_fg` measures **level** (first moment). `U_std_fg` measures **dispersion** (second central moment). A spatially constant field of `0.15` across `F` gives high mean `0.15` and std `0`; a mixture of near-zero and high entropy values can have lower mean and higher std. No mathematical ordering guarantees that either score better detects failure. Their performance must be compared **on identical images** using the same quality/risk function. The BUSI validation AURC values (`0.0839` vs `0.0850`, paired `p=0.294`) and BUSI test AURC (`0.0790` vs `0.0793`, paired `p=0.786`) support **no resolved superiority between the two**, despite point estimates favoring `U_fg`.

## 5. Segmentation quality: Dice versus boundary tolerance

### 5.1. Dice similarity coefficient (DSC)

$$
DSC(Y,M)=\frac{2|Y\cap M|}{|Y|+|M|}\quad (|Y|+|M|>0).
$$

DSC is region overlap. If `|Y|=100`, `|M|=80`, intersection=60, then DSC=`120/180=0.6667`. The project's evaluation convention is `DSC(∅,∅)=1`, and one-sided empty/nonempty gives `0`. The empty–empty convention is an explicit choice, **not** implied by the algebraic `0/0`. DSC may be high despite a clinically consequential localized contour deviation; this motivates NSD as secondary evidence.

### 5.2. Normalized Surface Dice (NSD, surface Dice at tolerance)

Given predicted/reference surfaces `S_M,S_Y`, surfel weights `w_s`, distance `d(s,S_other)` and tolerance `δ`:

$$
NSD_\delta=\frac{\sum_{s\in S_M}w_s\mathbf1[d(s,S_Y)\le\delta]+\sum_{s\in S_Y}w_s\mathbf1[d(s,S_M)\le\delta]}{\sum_{s\in S_M}w_s+\sum_{s\in S_Y}w_s}.
$$

The paper/project evaluator reportedly uses surface-distance machinery and a nominal 2-pixel tolerance. **Do not silently rewrite `2 pixels` as `2 mm`:** conversion requires verified physical spacing. No evidence in the supplied summaries establishes patient-specific physical calibration. If both surfaces are empty, the evaluation's reported NSD=1; if only one is empty, the reported NSD=0. NSD cannot be described as boundary sensitivity without specifying its tolerance and weighting implementation [R10].

## 6. Statistical association: Spearman, bootstrap and failure AUROC

**Spearman's rho** is Pearson correlation computed on the ranks; with no ties it equals `1−6Σ_i d_i²/[N(N²−1)]`. With ties, average-rank handling is necessary. Negative rho between uncertainty and quality is directionally desirable: higher uncertainty tends to imply lower DSC. It does **not** prove a calibrated probability of failure, a causal relationship, or the ability to reject the lowest-quality cases at every coverage.

**Validation lesion-cohort Spearman (`N=65`, 1,000 bootstrap samples):**

| Metric | rho vs DSC [95% CI] | rho vs NSD [95% CI] | Interpretation bounded to this cohort |
|---|---:|---:|---|
| `U_fg` | `−0.7202 [−0.833,−0.557]` | `−0.6925 [−0.814,−0.525]` | Strongest observed negative rank association |
| `U_std_fg` | `−0.7068 [−0.824,−0.540]` | `−0.6768 [−0.800,−0.504]` | Similar association; not a demonstrated worse method |
| `U_boundary` | `−0.2273 [−0.468,+0.028]` | `−0.4174 [−0.601,−0.184]` | More related to NSD than DSC in these data |
| `U_mean` | `−0.0451 [−0.285,+0.203]` | `−0.2570 [−0.472,−0.018]` | Near-zero DSC rank association in lesion subset |
| `U_positive` | `−0.0207 [−0.259,+0.219]` | `−0.2360 [−0.447,+0.001]` | Small association |
| `U_ratio` | `−0.1874 [−0.410,+0.046]` | `+0.0499 [−0.192,+0.267]` | Weak/unstable; geometric dependence |
| `U_threshold` | `+0.0178 [−0.256,+0.270]` | `+0.0186 [−0.243,+0.264]` | No useful monotone association observed |

**Cohort-mixture warning.** On all 78 BUSI validation images, global mean entropy had DSC rho=`−0.4449`; this is substantially stronger than `−0.0451` on the 65 lesion cases. The 13 true-normal empty/empty cases have almost zero uncertainty and DSC=NSD=1, changing the rank distribution. Results from 78 and 65 have different estimands; neither should be substituted for the other.

**Bootstrap.** Paired resampling resamples the **same case indices** for both candidate scores and the same corresponding DSC/NSD values. This preserves covariance for the difference estimator. A percentile confidence interval and a bootstrap tail-based p-value are **not identical inferential procedures**, and the latter requires a defensible null-centering/assumption treatment to support formal type-I-error claims. `B=1000` also limits tail resolution; if no replicate crosses a null boundary, `p<0.001` should be treated as a simulation-resolution statement, not infinite certainty. Repeated images from one patient can invalidate independent-case bootstrap assumptions.

**Failure-detection AUROC.** For binary failures `Z_i=1[DSC_i<q_crit]`, AUROC measures how often failure scores exceed nonfailure scores. At `DSC<0.5`, BUSI validation contained only **two** positive failures; an AUROC point estimate such as `0.8968` for `U_fg` is descriptive and statistically fragile. Failure AUROC is not the primary selection objective.

## 7. Selective prediction: acceptance, coverage, risk, and utility

Define a high-uncertainty-is-bad selector

$$
a_i(\tau)=\mathbf1[s_i\le\tau],\qquad C(\tau)=\frac1N\sum_i a_i(\tau),
$$

where `s_i` is the chosen image-level score; `τ` is frozen on validation. For accepted cases,

$$
R(\tau)=\frac{\sum_i a_i(\tau)(1-Q_i)}{\sum_i a_i(\tau)}=1-\overline Q_{\rm accepted},
$$

when at least one image is accepted. Coverage measures the fraction **not** rejected. Rejection changes the population being summarized; it does not modify the original segmentation, the CNN weights, or individual `Q_i` values.

**Exact mixture identity.** For full cohort risk `R_all`, accepted risk `R_A`, rejected risk `R_R`, and accepted fraction `c`:

$$
R_{all}=cR_A+(1-c)R_R.
$$

This identity explains how average accepted DSC may exceed the full-cohort DSC when rejected cases have lower quality. It is **selection**, not repaired predictions. For BUSI test lesion cases: `c=47/65≈0.7231`, accepted DSC=`0.9031`, rejected DSC=`0.6341`; weighting those observed means reconstructs baseline DSC≈`0.8286` (rounding differences only).

**Coverage calibration is not risk calibration.** The BUSI validation target `80%` produced `52/65` accepted there. Applying the frozen threshold to BUSI test gave `47/65=72.3%`; pooled external test gave `238/464=51.3%`. This is not a contradiction: an absolute threshold selects different proportions when uncertainty-score distributions differ. A validation quantile is not a distribution-free coverage guarantee and certainly not a clinical risk guarantee [R08].

## 8. Risk–Coverage curve and AURC: exact estimand and caveats

Sort cases by increasing uncertainty, `s_(1) ≤…≤ s_(N)`, breaking or grouping ties according to a prespecified convention. For each retained prefix `k`, `c_k=k/N` and `R_k=k^-1 Σ_{i=1}^k (1-Q_(i))`. Lower risk at the **same coverage** is better. The project's discrete empirical estimator was described as trapezoidal area along the observed curve:

$$
\widehat{AURC}=\sum_{k=k_0}^{N-1}(c_{k+1}-c_k)\frac{R_{k+1}+R_k}{2}.
$$

Here `k_0` depends on how the implementation groups identical scores. This estimator does **not** integrate `[0,1]` unless a convention for the zero-coverage endpoint is separately imposed. The empirical domain and tie grouping **must be identical for fair method comparison**; the pre-test audit and later cross-domain audit report that the comparison convention was aligned. Because only summaries, not source CSV/code, are present here, that verification retains an artifact-level provenance dependency.

### 8.1. Oracle: definition, role, and proof of its bound

**Oracle is an evaluation-only ideal ranking that knows the ground-truth quality** `Q_i`. Sort cases by **increasing true loss** `L_i=1−Q_i` (best quality first), then compute its risk–coverage curve. It is not Oracle Database, a commercial model, a deployable score, or a prediction of ground truth.

**Why it is optimal at each fixed integer `k`.** For any chosen set of `k` cases, the smallest possible sum of nonnegative losses is the sum of the `k` smallest losses. If a retained set includes a higher-loss case while excluding a lower-loss one, swapping them decreases or preserves mean retained loss. Repeating exchanges obtains the Oracle prefix. Consequently, for any coverage grid with equal valid conventions, `R_oracle(c) ≤ R_method(c)` and hence `AURC_oracle ≤ AURC_method`. Oracle is a **lower bound on risk/AURC** and an **upper bound on attainable retention quality**, never the reverse. It is unattainable at deployment without GT labels.

### 8.2. Random rejection: what it is and what it is not

Random is an **uninformed selection reference**; under uniformly random order, the expected mean risk of every nonempty prefix is the population mean risk `R_all`. For a constant-risk reference integrated by the trapezoid over `[1/N,1]`, the area is

$$
AURC_{rand}=R_{all}(1-1/N).
$$

For BUSI validation lesion cases with `N=65` and baseline DSC≈`0.852014`, `R_all≈0.147986`; an implementation whose first distinct tied-score position is `2/65` may yield `0.147986(1−2/65)=0.143433`. Thus `AURC_random` need not numerically equal `R_all` under an unanchored empirical convention. Do **not** change convention silently between methods or datasets. The Step 10 BUSI test `N=65` Random AURC=`0.1687`; an earlier OOD summary mistakenly mixed in 13 normal images and showed `0.1406`. The corrected comparison uses `0.1687`.

**Random is not a patient-level safety floor.** One realization of random selection fluctuates; an analytic expectation is a benchmark, not an actionable rejection policy.

### 8.3. AURC versus E-AURC and pooled versus per-dataset AURC

AURC is an area over a specified discrete coverage domain. **Excess-AURC (E-AURC)** subtracts an appropriate Oracle reference when the same cohort/convention is used. E-AURC was **not** the project's prespecified primary evaluation objective; do not label a raw AURC as E-AURC.

Three cross-domain summaries answer **different questions**:

1. **Pooled case-level AURC (`N=464`)**: rank all external cases together using one common score; operationally relevant for a shared cross-site queue.
2. **Sample-size-weighted mean of dataset-specific AURCs**: summarize within-dataset rankings, emphasizing large datasets.
3. **Unweighted macro mean**: each external dataset receives equal weight independent of sample size.

Because AURC depends on ordering and partial-retention sets, **pooled AURC is not algebraically the weighted mean of separate AURCs**. For `U_fg`, the audited values were `0.1267` (pooled), `0.1314` (weighted), and `0.1168` (macro). Their disagreement is mathematically legitimate, not an inconsistency to erase.

## 9. Empty predicted masks and two-stage deployment policy

`F=∅` makes `U_fg` undefined. For all-image assessment the frozen rule uses only inference-time information:

$$
s_i=\begin{cases}
U_{fg,i}, & |F_i|>0;\\
\text{accept if }U_{mean,i}\le 10^{-4},\text{ otherwise abstain}, & |F_i|=0.
\end{cases}
$$

The actual implementation may encode this as ranking with `±infinity`, so the exact ordering/coverage policy must be read from its saved code. This is *not* DeVries and Taylor's trained two-stage neural architecture [R09]. In BUSI validation/test, the 13 empty predicted masks corresponded to annotated normal cases with DSC=NSD=1. But a synthetic lesion-positive case with predicted-empty mask and near-zero entropy demonstrates a counterexample: the rule **accepts a missed lesion**. Therefore, low background entropy cannot establish clinical normality. The four reported OOD test cohorts contain **464/464 nonempty predictions**, so they do not empirically validate this particular risk mode.

**Setting A**: lesion/reference-positive cohort for quality analysis (N=65 BUSI test; OOD cohorts all lesion-positive). In the OOD protocol also report predicted-positive counts, and do not hide GT-positive/predicted-empty cases if they arise. **Setting B**: full cohort and the deployable two-stage decision; performance comparisons must disclose different sample composition. Ground truth must never be used to choose the runtime decision branch.

## 10. Measurement thresholds: three different roles

| Name | Value | Governs | Does **not** govern |
|---|---:|---|---|
| `t_pixel` | `0.1` nats | Which pixels enter `U_threshold` | Rejection of entire images |
| `τ_img` | `0.064058324` nats | Which *nonempty-prediction images* are retained by `U_fg` | Pixel segmentation threshold |
| `τ_neg` | `10^-4` nats | Empty-prediction branch of two-stage rule | Certainty that no tumor exists |
| `p_mask` | Reported `0.5` | Binarization of probability mask | Quality or calibrated false-negative probability |
| `δ_surface` | Reported `2` pixels | NSD surface match tolerance | Abstention or entropy |

## 11. Documented numerical outcomes and correct interpretations

| Evaluation | Cohort | Fixed `U_fg` AURC | `U_mean` AURC | Additional conclusion |
|---|---:|---:|---:|---|
| BUSI validation | 65 lesion | `0.0839` | `0.1438` | Used for selection; `U_std_fg=0.0850` statistically unresolved versus `U_fg` |
| BUSI locked test | 65 lesion | `0.0790` | `0.1262` | Replication of ordering advantage under frozen threshold; no post-test selection |
| UDIAT | 25 lesion | `0.1133` | `0.1503` | Smaller cohort; wide uncertainty |
| BUID | 35 lesion | `0.1094` | `0.2411` | Global mean can underperform uninformed random baseline (`0.2040`) |
| BUSUC | 122 lesion | `0.0916` | `0.1365` | Fixed threshold still filters lower-quality predictions |
| BUSBRA | 282 lesion | `0.1529` | `0.2020` | Most stringent relative retention under fixed threshold, `44.7%` coverage |
| Pooled OOD | 464 lesion | `0.1267` | `0.1864` | Pooled ranking, not mean of four separate AURCs |

**Why smaller AURC is preferable:** at the same coverage, the ranking tends to retain lower-loss segmentations; the area integrates this preference across operating points. A single small difference in AURC is not evidence of superiority without paired uncertainty analysis. **Why `U_fg` was selected:** lowest observed AURC on the prespecified BUSI validation primary DSC risk, simple/no-training implementation, smaller observed size-rank association than global averaging, and comparable behavior on NSD. **Why `U_std_fg` remains a valid comparator:** the differences were below statistical resolution on both validation and test. These are selection criteria and empirical observations, not a claim that the formula of `U_fg` is novel.

## 12. References (primary literature and technical specifications)

- **[R01]** Koleilat, T., et al. (2026). *MedCLIPSeg: Probabilistic Vision-Language Adaptation for Data-Efficient and Generalizable Medical Image Segmentation*. CVPR, 1406–1417. https://openaccess.thecvf.com/content/CVPR2026/html/Koleilat_MedCLIPSeg_Probabilistic_Vision-Language_Adaptation_for_Data-Efficient_and_Generalizable_Medical_Image_CVPR_2026_paper.html
- **[R02]** Roy, A. G., Conjeti, S., Navab, N., and Wachinger, C. (2019). *Bayesian QuickNAT: Model uncertainty in deep whole-brain segmentation for structure-wise quality control*. NeuroImage, 195, 11–22. https://doi.org/10.1016/j.neuroimage.2019.03.042
- **[R03]** Guarino, V. E., et al. (2026). *Better than Average: Spatially-Aware Aggregation of Segmentation Uncertainty Improves Downstream Performance*. CVPR, 13145–13156. https://openaccess.thecvf.com/content/CVPR2026/html/Guarino_Better_than_Average_Spatially-Aware_Aggregation_of_Segmentation_Uncertainty_Improves_Downstream_CVPR_2026_paper.html
- **[R04]** Jungo, A., Balsiger, F., and Reyes, M. (2020). *Analyzing the Quality and Challenges of Uncertainty Estimations for Brain Tumor Segmentation*. Frontiers in Neuroscience, 14:282. https://doi.org/10.3389/fnins.2020.00282
- **[R05]** Mehrtash, A., et al. (2020). *Confidence Calibration and Predictive Uncertainty Estimation for Deep Medical Image Segmentation*. IEEE TMI, 39(12), 3868–3878. https://doi.org/10.1109/TMI.2020.3006437
- **[R06]** Kendall, A., and Gal, Y. (2017). *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?* NeurIPS 30. https://proceedings.neurips.cc/paper/2017/hash/2650d6089a6d640c5e85b2b88265dc2b-Abstract.html
- **[R07]** Gal, Y., and Ghahramani, Z. (2016). *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning*. ICML. https://proceedings.mlr.press/v48/gal16.html
- **[R08]** Geifman, Y., and El-Yaniv, R. (2017). *Selective Classification for Deep Neural Networks*. NeurIPS. https://arxiv.org/abs/1705.08500
- **[R09]** DeVries, T., and Taylor, G. W. (2018). *Leveraging Uncertainty Estimates for Predicting Segmentation Quality*. https://arxiv.org/abs/1807.00502
- **[R10]** Nikolov, S., et al. (2018). *Deep Learning to Achieve Clinically Applicable Segmentation of Head and Neck Anatomy for Radiotherapy*. https://arxiv.org/abs/1809.04430 ; surface Dice implementation and tolerance must be read from the evaluated code.
- **[R11]** Efron, B., and Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall/CRC.
- **[R12]** MedCLIPSeg original dataset and split manifest. https://github.com/HealthX-Lab/MedCLIPSeg/blob/main/assets/DATASETS.md
- **[R13]** OpenCV morphology reference. https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html
- **[R14]** SciPy Spearman implementation reference. https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html

**Citation scope caution:** R02 supports a comparable per-structure mean aggregation, **not** replication of the QuickNAT architecture; R03 defines alternative aggregators, **not** evidence that PLM/QFR were executed; R04/R05 contextualize failure detection and uncertainty limitations, **not** authorship of `U_std_fg` or `U_boundary`; R08 discusses selective classification but provides **no distribution-free guarantee** for this project's empirical-quantile threshold.

## 13. Common questions about the results

1. **What is Oracle?** Ground-truth-informed best possible ranking, a lower bound on risk/AURC, unusable during inference.
2. **Why compare with Random?** Separates a meaningful ranking from rejecting cases without quality information.
3. **Why `U_fg`?** Avoids including vast background by conditioning on predicted lesion; tested, not universally optimal.
4. **Is `U_fg` new?** No; its spatial-mean operation has a published antecedent. The project tests training-free ranking/abstention on this frozen text-guided model and domains.
5. **Why can `U_mean` fail?** Exact mixture with foreground fraction `α`, plus case-dependent background uncertainty.
6. **Does low `rho(area,U_fg)` prove independence?** No; it rules out a strong sample-wide *monotonic* association in that cohort only.
7. **Why different quality metrics?** DSC measures region overlap, NSD boundary agreement within a tolerance.
8. **What changed at +0.0939 DSC?** The selected **population**, not the original pixel predictions.
9. **What does 51.3% coverage mean?** Nearly half the external segmentations were rejected under the original BUSI-calibrated threshold.
10. **What happens to a confident false negative?** It may pass; entropy does not certify lesion absence.
11. **What does `seed=666` do?** Controls pseudorandom runs where supported, but does not ensure bitwise reproducibility across hardware.
12. **How to challenge a claim?** Identify its formula, cohort, output artifact, uncertainty interval, literature scope and failure case in the evidence registry.
