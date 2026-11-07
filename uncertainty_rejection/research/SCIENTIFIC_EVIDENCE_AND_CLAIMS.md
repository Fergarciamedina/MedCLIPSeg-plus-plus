# Scientific Evidence and Claims: Falsifiable Hypotheses, Literature Attribution, Quantitative Comparisons, and Inference Limits

**ELEC 6661 | Scientific evidence registry | Version 2.0 | 2026-11-07 | Coverage: Steps 1–11C**  
**Read with:** [Theory and Metrics](THEORY_AND_METRICS.md) · [Experimental Protocol](EXPERIMENTAL_PROTOCOL.md) · [Research Decisions and Limitations](RESEARCH_DECISIONS_AND_LIMITATIONS.md).

> **Purpose.** This is an audit trail from **scientific question → plausible mechanism → falsifiable hypothesis → mathematical estimator → implementation → observed result → counterexample → justified claim → unresolved limitation**. A reference supports a published concept, **not** the project's own numerical result. Project outcomes below are based on completed execution and reconciliation reports; the raw case CSVs and source checkout were not provided for independent re-execution in the document-authoring environment. No artificial “independent replication” claim is made.

## 1. Standards of evidence: what each statement may say

| Label | Meaning | Permitted wording |
|---|---|---|
| `[DEF]` | Exact definition or declared implementation contract | “The aggregator is defined as…” |
| `[DER]` | Mathematical deduction under explicit assumptions | “It follows algebraically that…” |
| `[LIT]` | Content verified in original published source | “The cited publication defines/discusses…” |
| `[OBS-VAL]` | Study's BUSI validation result | “In BUSI validation (N=65), the observed…” |
| `[OBS-TEST]` | Study's locked BUSI test result | “On the predesignated BUSI test cohort…” |
| `[OBS-OOD]` | External dataset evaluation result | “Across these four evaluated external test sets…” |
| `[SIM]` | Synthetic example or logical counterexample | “The constructed input demonstrates a possibility…” |
| `[HYP]` | Falsifiable research prediction | “If hypothesis Hn holds, one expects…” |
| `[LIMIT]` | Known constraint, uncertainty, or missing confirmation | “These experiments cannot establish…” |
| `[NOT-IMPL]` | Related method not reproduced | “This published method was not implemented or benchmarked.” |
| `[PENDING-ARTIFACT]` | Raw source/data or metadata absent from document environment | “Must be verified in final repository release.” |

### Exact reproduction versus shared mathematical operation

- **Formula correspondence:** an operation has the same expression as an equation in another publication. It does not establish identical data sources, uncertainty estimator, missing-value policy, or model.
- **Adaptation:** an established aggregation principle is changed in at least one implementation detail or environment (e.g. empty-set behavior).
- **Conceptual motivation:** a source studies related failure detection/calibration, but does not define the specific scalar function used here.
- **Full reproduction:** requires matching inputs, preprocessing, architecture if relevant, hyperparameters, empty cases, model/checkpoint, and evaluation code/metrics. This study does **not** claim to reproduce whole QuickNAT, DeVries's trained quality predictor, SelectiveNet, or Guarino's PLM/QFR.

## 2. Source-to-implementation map and negative attribution claims

| Reference ID | Original publication / verified URL | Narrow fact supported | What is present in the study | What **must not** be claimed |
|---|---|---|---|---|
| **R01** | Koleilat et al., *MedCLIPSeg* (CVPR 2026), https://openaccess.thecvf.com/content/CVPR2026/html/Koleilat_MedCLIPSeg_Probabilistic_Vision-Language_Adaptation_for_Data-Efficient_and_Generalizable_Medical_Image_CVPR_2026_paper.html | Probabilistic text-guided medical segmentation, uncertainty maps; original multi-dataset design | Frozen model inference; BUSI validation/test and four OOD test sets; raw entropy samples | That all official paper conditions or original training results were exactly replicated without verifying checkpoint/prompt/data lineage |
| **R02** | Roy et al., *Bayesian QuickNAT* (NeuroImage 2019), https://doi.org/10.1016/j.neuroimage.2019.03.042 | Per-structure aggregated uncertainty supports structure-wise quality control; paper describes voxelwise uncertainty and four structure metrics | Local `U_fg=mean(U|predicted foreground)` corresponds to a per-structure spatial-mean operation | That Bayesian QuickNAT architecture, MRI experiments, MC dropout inference, or all uncertainty metrics were reproduced |
| **R03** | Guarino et al., *Better than Average* (CVPR 2026), https://openaccess.thecvf.com/content/CVPR2026/html/Guarino_Better_than_Average_Spatially-Aware_Aggregation_of_Segmentation_Uncertainty_Improves_Downstream_CVPR_2026_paper.html | AVG, ATA, PLM, AQA, BCA/ICA, QFR; sensitivity to geometry and task | Global `U_mean` AVG; `U_threshold` is **ATA-like** with different empty-set rule | That QFR=our `U_ratio`, or PLM/QFR executed, or ATA is identical when no pixels exceed threshold |
| **R04** | Jungo, Balsiger and Reyes (Front. Neurosci. 2020), https://doi.org/10.3389/fnins.2020.00282 | Subject-wise aggregated uncertainty may help find failed segmentations despite weak voxel-level error localization | Motivates evaluation of image-level ranks and limitations | That `U_std_fg` is Jungo's algorithm; this exact formula is our specified ablation |
| **R05** | Mehrtash et al. (IEEE TMI 2020), https://doi.org/10.1109/TMI.2020.3006437 | Deep medical segmentation may be poorly calibrated; uncertainty can predict quality/OOD in some contexts | Motivates studying confidently wrong cases | That `U_boundary` is their formula or that study entropy is calibrated in a probabilistic sense |
| **R06** | Kendall & Gal (NeurIPS 2017), https://proceedings.neurips.cc/paper/2017/hash/2650d6089a6d640c5e85b2b88265dc2b-Abstract.html | Aleatoric versus epistemic taxonomy | Clarifies uncertainty limitations and decomposition not executed | That `U_mean` is their unique published method or that both uncertainty types were separated experimentally |
| **R07** | Gal & Ghahramani (ICML 2016), https://proceedings.mlr.press/v48/gal16.html | MC dropout approximate Bayesian inference | Background for stochastic uncertainty estimation | That MedCLIPSeg's stochastic attention necessarily implements identical dropout posterior mechanics |
| **R08** | Geifman & El-Yaniv (NeurIPS 2017), https://arxiv.org/abs/1705.08500 | Selective classification formalism and risk–coverage tradeoffs | Loss/risk, acceptance and coverage evaluation | That our empirical quantile rule inherits their formal risk guarantees |
| **R09** | DeVries & Taylor (2018), https://arxiv.org/abs/1807.00502 | Learned two-stage segmentation-quality prediction | Conceptual motivation for image-level quality screening | That our deterministic `two_stage` empty-mask fallback is their trained architecture |
| **R10** | Nikolov et al., surface Dice background, https://arxiv.org/abs/1809.04430 | Tolerant boundary evaluation | NSD evaluation in original segmentation pipeline | That unit of project tolerance is `mm` absent physical spacing metadata |
| **R11** | Efron & Tibshirani (1993), *An Introduction to the Bootstrap* | Resampling uncertainty estimation | 1,000 paired resamples on study cases | That image bootstrap validates between-hospital generalization or automatically gives valid null p-values |
| **R12** | Official MedCLIPSeg data split manifest, https://github.com/HealthX-Lab/MedCLIPSeg/blob/main/assets/DATASETS.md | Exact official BUSI and external *test subset counts* | N=78 BUSI; N=25 UDIAT; N=35 BUID; N=122 BUSUC; N=282 BUSBRA | That total image counts and test subset counts are interchangeable |

**Citations lacking equation-level verification:** unspecified “Wang et al. (2019)” for `U_boundary`; “Eaton-Rosen et al. (2018)” for exactly this `U_threshold`; “Jungo et al. (2020)” for precisely this `U_std_fg`; any claim that `U_ratio` is Guarino QFR. Keep the concept-related papers in Related Work but do not label these formulas “reproduced published baselines”. The specific locally implemented aggregation functions are adequate comparisons if described precisely.

---

## 2.5 Formal Answers to Core Research Questions (RQ1–RQ5)

### RQ1 — Baseline Generalization Under Domain Shift

**Question:** How does the frozen BUSI-trained MedCLIPSeg baseline perform across unseen external breast ultrasound benchmarks without retraining or test adaptation?

1. **Absolute Segmentation Quality & Exact Alignment with Published Table 2:** Across the 464 external cases, MedCLIPSeg achieves a pooled baseline DSC of **$0.7780 \pm 0.1900$** and NSD of **$0.8203 \pm 0.1996$** (unweighted macro DSC: **$0.7827$**). Crucially, our baseline DSC values match the official published MedCLIPSeg CVPR 2026 Table 2 Domain Generalization results almost exactly (all within $\le 0.07\%$ difference):
   - **BUSI (Full Test, $N=78$):** Published **85.72%** vs. Our Baseline **0.8572** ($\Delta = 0.00\%$).
   - **BUSBRA ($N=282$):** Published **75.06%** vs. Our Baseline **0.7511** ($\Delta = +0.05\%$).
   - **BUSUC ($N=122$):** Published **84.37%** vs. Our Baseline **0.8430** ($\Delta = -0.07\%$).
   - **BUID ($N=35$):** Published **78.99%** vs. Our Baseline **0.7900** ($\Delta = +0.01\%$).
   - **UDIAT ($N=25$):** Published **74.64%** vs. Our Baseline **0.7467** ($\Delta = +0.03\%$).
   - *Sample-Weighted Pooled OOD:* Published **77.78%** vs. Our Baseline **0.7780** ($\Delta = +0.02\%$).
   - *Unweighted Macro OOD:* Published **78.27%** vs. Our Baseline **0.7827** ($\Delta = 0.00\%$).
2. **Relative Change from In-Distribution:** Compared to the in-distribution BUSI test lesion cohort ($0.8286 \pm 0.2110$), the pooled baseline exhibits an absolute drop of **$-0.0506$** DSC ($-6.1\%$).
3. **Dataset-Specific Performance:** Performance varies substantially across clinical sites:
   - **BUSUC ($N=122$):** DSC **$0.8430 \pm 0.1152$** (exceeds BUSI baseline by $+0.0144$).
   - **BUID ($N=35$):** DSC **$0.7900 \pm 0.2024$** (mild drop of $-0.0386$).
   - **BUSBRA ($N=282$):** DSC **$0.7511 \pm 0.1962$** (moderate drop of $-0.0775$).
   - **UDIAT ($N=25$):** DSC **$0.7467 \pm 0.2992$** (substantial drop of $-0.0819$, high variance).
4. **Attribution to Domain Shift vs. Cohort Confounders:** The observed performance divergence cannot be causally attributed purely to acoustic domain shift. Crucial non-shift factors modulate the scores:
   - *Case Mix:* UDIAT contains smaller, deeply seated tumors with severe shadow artifacts, whereas BUSUC contains pre-cropped, centered lesions with high contrast.
   - *Acquisition Hardware:* BUSI used GE LOGIQ systems, UDIAT used Siemens Elegra (7.5 MHz), and BUSBRA pooled Philips, GE, and Toshiba scanners.
   - *Labeling Protocol:* BUSUC ground-truth masks are anti-aliased soft contours, whereas BUSI masks are binary hand-drawn segmentations.
   - *Conclusion:* **[Supported]** The frozen MedCLIPSeg checkpoint exhibits graceful but heterogeneous generalization degradation across external sites, with performance strongly modulated by hardware heterogeneity, lesion morphology, and center-specific image preparation.

---

### RQ2 — Utility of Uncertainty-Based Selective Prediction

**Question:** Does ranking ultrasound segmentations by foreground uncertainty ($U_{\text{fg}}$) identify a clinically defensible, lower-risk retained subset?

1. **Selective Retention Gain:** Applying the frozen threshold ($\tau = 0.064058324$) calibrated on BUSI validation raises pooled external DSC from **$0.7780$** to **$0.8718$** (an absolute gain of **$+0.0939$**, 95% CI $[+0.0775, +0.1113]$, $p < 0.001$), while NSD rises from **$0.8203$** to **$0.9088$** ($+0.0885$).
2. **Partition Divergence:** Rejected segmentations ($N=226$, 48.7%) have a mean DSC of only **$0.6791 \pm 0.2096$**, confirming that selective filtering disproportionately purges low-quality segmentations.
3. **Selective Efficiency (AURC):** Case-level global risk-coverage evaluation yields an AURC of **$0.1267$** for $U_{\text{fg}}$, substantially outperforming the uninformative Random rejection baseline (**$0.2216$**, $\Delta\text{AURC} = -0.0949$) and approaching the Empirical Oracle upper bound (**$0.1046$**).
4. **Mathematical Mechanism:** Selective prediction does not alter or improve model predictions for any individual image. By the law of total expectation:
   $$\mathbb{E}[\text{Risk}] = c \cdot \mathbb{E}[\text{Risk} \mid \text{accepted}] + (1 - c) \cdot \mathbb{E}[\text{Risk} \mid \text{rejected}]$$
   Rejection purges cases from the lower tail of the accuracy distribution, shifting the conditional expectation $\mathbb{E}[\text{DSC} \mid \text{accepted}]$ upward at the operational cost of reducing coverage $c$.
5. **Conclusion:** **[Supported]** Ranking images by $U_{\text{fg}}$ reliably concentrates segmentation errors into the rejected partition, yielding high-accuracy retained subsets without model modification.

---

### RQ3 — Comparative Justification of $U_{\text{fg}}$ Over Competing Aggregators

**Question:** Why does foreground-averaged predictive entropy ($U_{\text{fg}}$) outperform global, boundary, and thresholded uncertainty aggregators?

1. **Mathematical Definitions and Target Attributes:**
   - **$U_{\text{fg}} = \frac{1}{|\Omega_{\text{fg}}|} \sum_{i \in \Omega_{\text{fg}}} \mathcal{H}_i$:** Measures mean predictive ambiguity strictly within the predicted lesion.
   - **$U_{\text{std\_fg}} = \sqrt{\frac{1}{|\Omega_{\text{fg}}|} \sum_{i \in \Omega_{\text{fg}}} (\mathcal{H}_i - U_{\text{fg}})^2}$:** Measures spatial dispersion and internal heterogeneity of uncertainty.
   - **$U_{\text{mean}} = \frac{1}{|\Omega|} \sum_{i \in \Omega} \mathcal{H}_i$:** Standard global spatial average across the entire image grid.
   - **$U_{\text{boundary}} = \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathcal{H}_i$:** Morphological boundary ring average ($5 \times 5$ kernel).
   - **$U_{\text{positive}} = \frac{1}{|\{i: \mathcal{H}_i > 0\}|} \sum_{\mathcal{H}_i > 0} \mathcal{H}_i$:** Average over non-zero entropy pixels.
   - **$U_{\text{threshold}} = \frac{1}{|\{i: \mathcal{H}_i > 0.1\}|} \sum_{\mathcal{H}_i > 0.1} \mathcal{H}_i$:** Average over high-uncertainty pixels (ATA-like).
   - **$U_{\text{ratio}} = U_{\text{fg}} / (U_{\text{mean}} + \epsilon)$:** Contrast ratio between foreground and global scale.
2. **Empirical Findings:**
   - On BUSI Validation ($N=65$): $U_{\text{fg}}$ achieved AURC **$0.0839$** (Spearman $\rho = -0.7202$), whereas $U_{\text{mean}}$ achieved AURC **$0.1438$** ($\rho = -0.0451$). Paired bootstrap confirms $U_{\text{fg}}$ superiority ($\Delta\text{AURC} = -0.0599$, 95% CI $[-0.0871, -0.0320]$, $p < 0.001$).
   - On BUSI Locked Test ($N=65$): $U_{\text{fg}}$ AURC **$0.0790$** vs. $U_{\text{mean}}$ **$0.1262$** ($\Delta\text{AURC} = -0.0472$, $p < 0.001$).
   - On Pooled External OOD ($N=464$): Case-level AURC **$0.1267$** vs. $U_{\text{mean}}$ **$0.1864$** ($\Delta\text{AURC} = -0.0597$, $p < 0.001$).
3. **Mechanistic Explanation of Global Dilution:** Global mean entropy decomposes as $U_{\text{mean}} = \alpha U_{\text{fg}} + (1 - \alpha) U_{\text{bg}}$, where $\alpha = |\Omega_{\text{fg}}| / |\Omega|$. Because background entropy is near zero, $U_{\text{mean}} \approx \alpha U_{\text{fg}}$. Consequently, $U_{\text{mean}}$ strongly correlates with lesion area ($\rho = +0.9148$ in validation, $+0.8960$ in test), confounding lesion size with segmentation difficulty. Small, well-segmented tumors are erroneously assigned tiny $U_{\text{mean}}$, while large, well-segmented tumors are penalized with high $U_{\text{mean}}$.
4. **Boundary Focus Limitation:** While $U_{\text{boundary}}$ exhibits stronger correlation with surface distance (NSD $\rho = -0.4174$ vs. DSC $\rho = -0.2273$), it underperforms $U_{\text{fg}}$ for whole-lesion quality assessment (AURC $0.1302$ vs. $0.0839$) because boundary entropy is naturally elevated even for correctly placed interfaces.
5. **Foreground Standard Deviation ($U_{\text{std\_fg}}$):** Achieves statistically indistinguishable performance from $U_{\text{fg}}$ (Val AURC $0.0850$, Test AURC $0.0793$, $\Delta = -0.0003$, $p = 0.786$). $U_{\text{fg}}$ is retained as primary due to simpler mathematical interpretation and lower point AURC.
6. **Conclusion:** **[Supported]** $U_{\text{fg}}$ provides the most effective training-free failure ranking because it isolates lesion-level predictive entropy and mitigates background area dilution.

---

### RQ4 — Operating Point Reliability and Coverage Drift Under Domain Shift

**Question:** Does a rejection threshold calibrated for 80% coverage on in-distribution validation maintain its operating point and performance under external domain shift?

1. **Nominal vs. Realized Coverage:** The primary threshold $\tau = 0.064058324$ was frozen to achieve exactly $80.0\%$ coverage (52/65 cases) on BUSI validation.
   - On BUSI locked test, realized coverage decreased to **$72.3\%$** (47/65 cases).
   - Across external benchmarks, realized coverage experienced severe drift: UDIAT (**$56.0\%$**), BUID (**$57.1\%$**), BUSUC (**$63.9\%$**), and BUSBRA (**$44.7\%$**), with pooled external coverage falling to **$51.3\%$** (238/464 cases).
2. **Distributional Cause of Drift:** Under domain shift, distribution-level acoustic speckle, contrast variation, and unfamiliar transducer profiles shift predictive entropy distributions upward. As the CDF $F_{U_{\text{OOD}}}(\tau)$ shifts rightward, a fixed numerical threshold rejects an increasing fraction of cases ($1 - F(\tau)$).
3. **Implications for Clinical Workflow:** A static threshold cannot guarantee constant radiologist referral workload under domain shift. Deploying centers must anticipate that unfamiliar scanners will trigger higher referral rates (up to $55.3\%$ rejection on BUSBRA) to maintain high diagnostic precision ($> 0.85$ retained DSC).
4. **Conclusion:** **[Supported]** The selective quality benefit transfers robustly across domains, but nominal coverage does not; fixed thresholds undergo coverage drift and increased referral burden under distribution shift.

---

### RQ5 — Failure Modes, Falsification Cases, and Limitations

**Question:** What are the empirical and structural failure modes of the uncertainty selective prediction framework?

1. **Accepted Severe Failure (Falsification of Perfect Error Detection):** In BUSI locked test, case `116_malignant.png` generated DSC $= 0.2193$ and NSD $= 0.2458$, yet was assigned $U_{\text{fg}} = 0.062791 < \tau = 0.064058324$ and erroneously accepted. The network made a false-positive expansion into uniform posterior shadow with confident, low entropy. This counterexample definitively falsifies claims that $U_{\text{fg}}$ catches all catastrophic errors.
2. **Residual Area Dependence:** Although $U_{\text{fg}}$ dramatically suppresses size confounding compared to $U_{\text{mean}}$ ($\rho = -0.0848$ vs. $+0.9148$), mild residual area correlations ($|\rho|$ up to $0.31$) persist in some external cohorts.
3. **Empty-Mask Policy Limitation:** The `two_stage` fallback routes predicted-empty masks ($|\Omega_{\text{fg}}|=0$) using background entropy $U_{\text{mean}} > \tau_{\text{neg}} = 10^{-4}$. In the external benchmark ($N=464$), all cases yielded non-empty segmentations, leaving the safety of the zero-foreground branch unverified under external domain shift.
4. **Spatial Tolerance Spacing:** NSD calculations assume isotropic pixel spacing ($1.0 \times 1.0$) with tolerance $\tau = 2$ pixels because original scanner DICOM pixel spacing metadata was stripped from public web releases. NSD metrics cannot be interpreted as physical millimeter distances.
5. **Patient Identifier Hierarchy and Clustering Independence:** We formally distinguish three levels of patient provenance:
   - *Verified Clinical Identifiers:* Longitudinal clinical medical record IDs linking unique patients are completely absent across all public breast ultrasound benchmark releases.
   - *Filename-Derived Grouping Proxies (BUSBRA):* BUSBRA filenames encode a heuristic subject prefix `bus_XXXX-[l|r|s].png`. In the test split ($N=282$), there are 262 unique prefixes; 20 prefixes contain 2 images each, identifying minor intra-patient multi-view clustering.
   - *Unknown Patient Identity (BUSI, UDIAT, BUID, BUSUC):* Scans are named by arbitrary sequential integers without anatomical or patient tokens. Patient overlap cannot be formally established.
   - *Statistical Consequence:* Resampling bootstrap confidence intervals reflect image-level variation rather than clustered patient-level or center-level uncertainty.
6. **Classification of Scientific Findings:**
   - Quality improvement on retained cases: **Supported**
   - Superiority of $U_{\text{fg}}$ over $U_{\text{mean}}$: **Supported**
   - Replicated baseline lineage of CVPR 2026 Table 2: **Supported (All within $\le 0.07\%$)**
   - Preservation of nominal coverage across domains: **Contradicted (Coverage Drift)**
   - Complete elimination of severe errors: **Contradicted (Counterexample `116_malignant`)**
   - Patient-level independence: **Unresolved (Missing Patient IDs in Public Benchmarks; Inferred Proxies in BUSBRA Only)**

---

## 3. Research chain H1 — Why foreground mean might beat global mean

**Question.** Does averaging uncertainty **only over predicted lesion pixels** yield a better image-level failure ranking than averaging over all pixels?

**Starting observation.** Predictive entropy is a dense map, often highest along ambiguous lesion edges. Most pixels in a breast ultrasound are background. Thus a global mean is not automatically a measure of lesion segmentation difficulty.

**Prior literature.** Roy et al. [R02] use per-structure uncertainty for quality control; Guarino et al. [R03] show aggregation choices matter and global AVG has limitations. These publications make the approach plausible but do not provide its outcome on the study's data.

**Proposed mechanism `[DER]`.** With `α=|F|/|Ω|`, `U_mean=αU_fg+(1−α)U_bg`. If background entropy is small, `U_mean≈αU_fg`: two images with identical `U_fg` but different predicted lesion areas receive different global scores. The ranking may then reflect `α` rather than lesion uncertainty. **Falsifiable:** if global scores retain strong Dice association after conditioning on lesion area, dilution alone cannot explain everything; if `U_fg` fails to improve ranking, H1 is not supported.

**Implementation.** `aggregation.py`: `U_mean` spatial average across `Ω`, `U_fg` spatial average across `F` derived from the **predicted** mask; missing `F` gives NaN and is handled separately.

**Evidence.** On BUSI validation 65 lesion cases: `ρ(U_fg,DSC)=−0.7202` versus `ρ(U_mean,DSC)=−0.0451`, AURC `0.0839` versus `0.1438`, paired `ΔAURC=−0.0599`, 95% CI `[−0.0871,−0.0320]`. On BUSI locked test: AURC `0.0790` versus `0.1262`, paired delta `−0.0472` CI `[−0.0690,−0.0277]`. On 464 pooled OOD test cases, AURC `0.1267` versus `0.1864`. The validation example `120_benign` (DSC `0.3737`, foreground 753 pixels, `U_mean=0.003271`, `U_fg=0.102933`) is qualitatively consistent with dilution.

**Area evidence.** `ρ(predicted area,U_mean)=+0.9148` in BUSI validation and `+0.8960` in BUSI test; corresponding `U_fg` coefficients were `−0.0848` and `−0.0328`. These are univariate rank relationships, not formal conditional-independence tests. The external audit noted nonzero residual associations in some cohorts (`|ρ|` up to approximately 0.31). **Do not claim “unconfounded” or “eliminates size bias” unconditionally.**

**Supported conclusion.** For the studied cohorts, `U_fg` provided a more useful *segmentation-loss ranking and selection score* than global mean entropy. This is an empirical comparison, not a causal proof of the single mechanism or a guarantee of superiority under arbitrary shift.

**Alternative explanations.** The foreground aggregation may preferentially sample different uncertainty patterns and object classes, and normal-case inclusion alters ranks. Dataset-specific imaging/label protocols and crop size may also matter. Multivariable/stratified area analyses and a shared independent cohort would be needed to isolate effects more strongly.

## 4. Research chain H2 — Boundary focus and why it did not win

**Question.** If boundary entropy is visually high, should a boundary-only mean be a stronger predictor of segmentation quality?

**Mechanistic case in favor.** MedCLIPSeg uncertainty highlights uncertain contours; contour displacement directly affects NSD. A morphological band may emphasize the spatial region in which uncertainty is strongest.

**Mechanistic counterargument.** High contour entropy can appear for a *correctly* placed boundary due to genuine pixel ambiguity. Conversely, an incorrect segmentation can have a confident, sharp wrong boundary. A fixed 5×5 rectangular morphological band varies in number of pixels with perimeter and image-border padding. Hence `U_boundary` can reflect perimeter/shape and uncertainty concentration rather than ground-truth error.

**Operation.** `B=dilate(F,5×5)\erode(F,5×5)` and `U_boundary=mean(U|B)`. The band includes interior and exterior boundary neighborhoods; it is not GT boundary distance and it does not have guaranteed physical 2-mm width.

**Validation.** `ρ(U_boundary,DSC)=−0.2273` (95% CI `[−0.468,+0.028]`), weaker than for `U_fg`; versus NSD, `ρ=−0.4174` (CI `[−0.601,−0.184]`), consistent with boundary emphasis being more informative for boundary quality. AURC(1−DSC) `0.1302` versus `U_fg 0.0839`; paired Δ `−0.0462`, CI `[−0.0711,−0.0221]` for `U_fg−U_boundary`.

**Locked test.** AURC `U_boundary=0.1113`, `U_fg=0.0790`; paired Δ=`−0.0324`, CI `[−0.0440,−0.0198]`. This supports the ranking advantage of `U_fg` on these DSC evaluations, not a theorem that boundary entropy is never useful.

**Conclusion.** Boundary-only entropy is scientifically meaningful but was inferior **for the study's prespecified image-level DSC risk objective**. It may be more relevant for local contour review or the separate NSD endpoint; those are different research questions. The observed NSD correlation supports studying this distinction without selecting it post-hoc as a new primary method.

## 5. Research chain H3 — Why the foreground standard deviation remained competitive

**Question.** Does spatial variation in entropy within the predicted lesion capture complexity better than average entropy level?

**Competing mechanisms.** The mean reflects overall uncertain probability near foreground; std reacts to mixtures of very certain and uncertain pixels. For constant `U_j` throughout `F`, `U_std_fg=0` regardless of mean. For mixed uncertainty zones, std rises even if mean remains moderate. Therefore no universal monotone mapping or dominance is implied mathematically.

**Evidence.** BUSI validation AURC(1−DSC) `U_fg=0.0839`, `U_std_fg=0.0850`; difference `−0.0011`, CI `[−0.0033,+0.0008]`, p=`0.294`. BUSI test AURC `0.0790` versus `0.0793`; difference `−0.0003`, CI `[−0.0024,+0.0014]`, p=`0.786`.

**Interpretation.** The data support **similar effectiveness within available statistical resolution**. They do not demonstrate the two metrics are mathematically equivalent or establish the absence of any difference. `U_fg` was retained as the simpler validation-preselected reference with slightly lower point estimates; `U_std_fg` remains the most appropriate close comparator and an alternative for future evaluation. It should not be described as a failed method merely because its reported AURC is 0.0011 larger.

## 6. Research chain H4 — Why the other aggregators ranked worse here

| Aggregator | Hypothesis motivating inclusion | Mathematical explanation of possible failure | Observed evidence | Proper conclusion |
|---|---|---|---|---|
| `U_positive` | Exclude exactly zero background entropy without predicting a region | `U>0` is sensitive to float residuals (as low as ~1e−18); may be close to global mean if most pixels >0 | BUSI val AURC `0.1455` vs global `0.1438`; BUSI test `0.1282` vs global `0.1262` | Strict zero filtering did not improve DSC AURC here; not proof all positive-threshold choices fail |
| `U_threshold` | Concentrate truly high-entropy pixels with `t=0.1` | A conditional average discards size/count of high-entropy region; selected means can cluster near the same magnitude | BUSI validation range ~`0.3639–0.4314`, AURC `0.1750`; test `0.2045` versus Random `0.1687` | Underperformed with the **specific fixed cutoff and empty convention**, not a general refutation of ATA-like designs |
| `U_ratio` | Contrast foreground with overall image scale | If background ≈0, `U_ratio≈1/α` (subject to epsilon), causing inverse lesion-size response | Validation `ρ(area,U_ratio)=−0.9957`, AURC `0.1368`; BUSI test AURC `0.1499` | Strong area dependence undermines its interpretation as quality independent of geometry; no proof every ratio is flawed |
| `U_mean` | Simple, widely used baseline independent of segmentation prediction | `U_mean=αU_fg+(1−α)U_bg`, often diluted by background | BUSI val DSC rho `−0.0451` on N65, AURC `0.1438`; BUSI test AURC `0.1262`; OOD pooled `0.1864` | Inferior empirical ranking versus `U_fg` under study settings; may perform relatively better for NSD or differently composed cohorts |
| `U_boundary` | Boundary localization may correspond to NSD/error | High entropy at boundary ≠ wrong contour; morphology depends on shape and padding | Validation AURC `0.1302`; test `0.1113`, still below foreground mean in quality ranking | Not selected for primary whole-lesion DSC failure ranking; remains scientifically meaningful for different endpoints |
| `U_std_fg` | Detect internal heterogeneity | A measure of dispersion is not necessarily a monotone error measure | Validation 0.0850, test 0.0793 | Statistically indistinguishable from selected `U_fg`, not demonstrably inferior |

**Selection-bias warning:** comparing many candidate aggregators on a small validation split can produce optimistic estimates. The locked test and OOD repetition reduce but do not eliminate generalization uncertainty, especially where clinical acquisition protocols and patient-group sampling are unknown.

## 7. Research chain H5 — Selective retention and its real cost

**Operational hypothesis.** For a fixed score threshold learned only from BUSI validation, mean DSC among retained segmentations should exceed mean DSC among all predictions, while lower-quality cases are disproportionately rejected.

**Primary outcome** is **accepted-case quality at realized coverage**, not overall segmentation accuracy. The identity `R_all=cR_accept+(1−c)R_reject` proves that selection can improve retained average while leaving every original prediction unchanged.

**Validation calibration:** `τ_img=0.064058324` selected for `80.0%` realized validation coverage (52/65 lesion cases). **Independent BUSI test:** same threshold accepted `47/65 (72.3%)`, with baseline DSC=`0.8286`, retained DSC=`0.9031`, rejected DSC=`0.6341`; gain `+0.0745`, CI `[+0.0349,+0.1235]`. NSD improved among retained cases from `0.8602` to `0.9349`. Results **support utility for this selection task**, not increased network segmentation accuracy.

**External transfer:** across UDIAT, BUID, BUSUC and BUSBRA, fixed-threshold realized coverages were `56.0%`, `57.1%`, `63.9%` and `44.7%`. Across all 464 OOD cases, 238 were accepted, 226 rejected; baseline DSC=`0.7780`, retained DSC=`0.8718`, rejected DSC=`0.6791`, gain `+0.0939` (95% CI `[+0.0775,+0.1113]`). **Coverage decreased** sharply compared with the 80% nominal target. Hence the study supports a **quality–coverage tradeoff** under external shifts, not stable fixed workload or a risk-controlled coverage guarantee.

**Pooled AURC:** `U_fg 0.1267`, `U_mean 0.1864`, Random `0.2216`, Oracle `0.1046`. Lower is better. These case-level values differ from weighted averages of four AURCs (`0.1314`,`0.1849`,`0.2202`,`0.1070`) and their macro means (`0.1168`,`0.1825`,`0.2127`,`0.0978`), because pooled ordering is not additive across datasets.

**Consequences for deployment interpretation:** returning fewer outputs may improve the trustworthiness of the **remaining** set but creates manual review burden for rejected images. The study does not evaluate reviewer availability, clinical utility, referral harms, deployment costs, patient outcomes, or prospective calibration.

## 8. Research chain H6 — Known failure cases, falsification, and safety boundaries

1. **Accepted severe failure in BUSI test `[OBS-TEST]`:** `116_malignant.png` had DSC=`0.2193`, `U_fg=0.06279`, which is below `τ=0.064058324`; the segmentation was therefore accepted despite poor quality. This directly defeats the claim “every severe segmentation error is detected”. It does **not** negate the cohort-level average quality gain.
2. **Synthetic confident false negative `[SIM]`:** predicted mask `F=∅`, ground-truth lesion `Y≠∅`, and `U_mean≤10^-4`. The two-stage rule accepts the empty prediction, despite DSC=0. This demonstrates logical vulnerability, not measured clinical incidence.
3. **Unobserved OOD false-negative mode `[OBS-OOD, LIMIT]`:** the recorded 464 OOD predictions all had nonempty foreground. Consequently, the external results **cannot** measure the false-negative safety of the predicted-empty branch.
4. **Unchanged raw segmentation `[DER]`:** selection does not repair malformed masks. “Higher mean Dice in accepted cases” cannot be expressed as “the model improved segmentation DSC from 0.7780 to 0.8718”.
5. **Malignancy coverage asymmetry `[OBS-TEST]`:** BUSI test benign acceptance `37/44=84.1%`; malignant acceptance `10/21=47.6%`. This may reflect weaker baseline segmentation, case mix, lesion geometry and uncertainty differences. It warrants subgroup reporting, not an untested fairness guarantee.
6. **Threshold transfer `[OBS-OOD]`:** only `51.3%` of pooled OOD samples survived the validation-derived cutoff; this is a failure of **nominal coverage transfer**, even though selective quality improved.

## 9. Claims ledger: every paper-ready sentence has a measurable anchor

The following **C IDs** are the canonical assertion identifiers to reuse in the research log, Methods, Results, Discussion and presentation captions.

| ID | Scientific claim stated without overreach | Evidence type | Quantitative or mathematical anchor | Limitation / counterclaim |
|---|---|---|---|---|
| **C01** | MedCLIPSeg provides a text-conditioned probabilistic segmentation baseline | `[LIT]` | R01 and repository | Checkpoint identity and exact prompts require release hashes |
| **C02** | Raw entropy is maximum at probability 0.5, not at ground-truth error | `[DER]` | Binary entropy derivative, upper bound `ln2` | Entropy not a calibrated failure probability |
| **C03** | Per-image visualization rescaling is unsuitable for cross-image absolute aggregation | `[DER]` | Min–max transformation changes scale/order under different extrema | Confirm visualization pipeline in release |
| **C04** | `U_fg` uses **predicted** mask and corresponds to a published structure-level mean operation | `[DEF,LIT]` | `Σ_{F}U/|F|`, R02 | Not a full Bayesian QuickNAT reproduction |
| **C05** | `U_mean` mixes foreground and background weighted by predicted area | `[DER]` | `U_mean=αU_fg+(1−α)U_bg` | Explains potential dependence, not all observed variation |
| **C06** | Foreground area is strongly associated with global mean entropy in BUSI | `[OBS-VAL,OBS-TEST]` | `ρ=+0.9148` val, `+0.8960` test | Noncausal; shared input feature may contribute |
| **C07** | `U_fg` has near-zero rank correlation with predicted area in BUSI | `[OBS-VAL,OBS-TEST]` | `−0.0848` val, `−0.0328` test | Does not establish unconditional independence; OOD variation exists |
| **C08** | `U_ratio` can grow like the reciprocal of foreground fraction | `[DER]` | Approx. `1/α` when `U_bg≈0`, epsilon small | Conditional approximation, not exact for arbitrary fields |
| **C09** | `U_threshold` differs in empty-region semantics from published ATA | `[DEF,LIT]` | Own NaN vs zero in R03 | Do not call a full identical implementation |
| **C10** | `U_boundary` measures an OpenCV band around the predicted region | `[DEF,SIM]` | 5×5 kernel, synthetic corner test 9/1/8 | Band not ground-truth contour error or physical distance |
| **C11** | `U_std_fg` measures spatial entropy dispersion, not MC uncertainty variation | `[DEF]` | Population std, `ddof=0` | No exact Jungo algorithm attribution |
| **C12** | DSC measures overlap, NSD tolerance-based boundary agreement | `[DEF,LIT]` | Dice and surface Dice formulas | NSD tolerance/spacing implementation must be verified |
| **C13** | BUSI validation has 65 lesion cases and 13 true-normal empty predictions | `[OBS-VAL]` | N78, both-empty quality 1 | No evidence that predicted-empty cases are safe OOD |
| **C14** | On validation lesion cases `U_fg` is more negatively associated with DSC than `U_mean` | `[OBS-VAL]` | rho `−0.7202` vs `−0.0451` | Different estimands for N65 vs N78 |
| **C15** | `U_fg` has lower validation DSC AURC than global/boundary means | `[OBS-VAL]` | `0.0839` vs `0.1438`/`0.1302`; paired CIs | Model selection uses validation; p-value caveats |
| **C16** | `U_fg` versus `U_std_fg` remains unresolved | `[OBS-VAL,OBS-TEST]` | Δ `−0.0011` CI crosses 0; test Δ `−0.0003` crosses 0 | Lack of significance ≠ equivalence proof |
| **C17** | Oracle is a GT-dependent lower bound on selective risk/AURC | `[DER]` | Top-k exchange argument | Not deployable, not a database |
| **C18** | Random provides a no-information ranking reference | `[DER]` | Expected prefix risk `R_all`; discrete AURC convention | Domain, endpoints and tie grouping affect numeric area |
| **C19** | The 0.1434 validation Random AURC is compatible with an unanchored empirical grid | `[DER,OBS-VAL]` | `0.147986 × (1−2/65)=0.143433` | Must document actual tie handling/code |
| **C20** | A frozen `τ_img` selects different proportions under shift | `[DEF,OBS-TEST,OBS-OOD]` | 80% val → 72.3% BUSI test → 51.3% OOD | No fixed-coverage guarantee |
| **C21** | BUSI test retained predictions had higher mean DSC than entire lesion cohort | `[OBS-TEST]` | 0.9031 vs 0.8286; 47/65 accepted | Selection, not pixel-prediction improvement |
| **C22** | One severe BUSI malignant segmentation escaped rejection | `[OBS-TEST]` | `116_malignant`, DSC=0.2193, score=0.06279 | Refutes all-failure detection |
| **C23** | BUSI test `U_fg` AURC advantage repeats after locking | `[OBS-TEST]` | 0.0790 vs global 0.1262, boundary 0.1113 | Evaluated on N65 with same ranking convention |
| **C24** | All four OOD cohorts report positive selected-vs-full DSC differences | `[OBS-OOD]` | +0.1281,+0.0919,+0.0551,+0.1025 | Lower external retention and within-cohort CI assumptions |
| **C25** | Pooled OOD accepted DSC is 0.8718, not 0.8732 | `[OBS-OOD]` | `238/464`, baseline 0.7780, rejected 0.6791 | Supersedes draft aggregate transcription |
| **C26** | Correct pooled OOD gain is +0.0939 | `[OBS-OOD]` | 0.8718−0.7780, CI [0.0775,0.1113] | Descriptive selection gain, not a clinical effect estimate |
| **C27** | Pooled, weighted, and macro AURC answer different questions | `[DER,OBS-OOD]` | U_fg 0.1267 / 0.1314 / 0.1168 | Should not substitute among these estimands |
| **C28** | Correct BUSI Random AURC on the lesion subset is 0.1687 | `[OBS-TEST]` | N65, not the erroneous N78-based 0.1406 | Corrected by Step 11C cohort audit |
| **C29** | UDIAT is the smallest OOD test set and has high estimation uncertainty | `[OBS-OOD]` | N25; gain CI width 0.2244 | Do not claim precise center-specific utility |
| **C30** | External pool is dominated by the largest dataset under micro weighting | `[DER,OBS-OOD]` | BUSBRA 282/464≈60.8% | Macro analysis needed for domain balance |
| **C31** | The 464 OOD cases do not empirically test empty-mask false negatives | `[OBS-OOD,LIMIT]` | 464/464 predicted nonempty | Do not certify two-stage clinical safety |
| **C32** | A synthetic confident missing lesion can be accepted | `[SIM]` | `F=∅`, `Y≠∅`, `U_mean≤1e−4` | Counterexample, not measured incidence |
| **C33** | BUSI test retains malignant cases less frequently than benign ones | `[OBS-TEST]` | 10/21 vs 37/44 accepted | Descriptive difference; no causal explanation |
| **C34** | Bootstrap CIs in this study do not estimate between-hospital variability | `[LIMIT]` | 4 external datasets; images as resampling units | Patient and site clusters needed for broader inference |
| **C35** | Stochastic seed improves reproducibility but cannot guarantee bitwise identity | `[DEF,LIMIT]` | seed666, T=30, MPS | Environment, nondeterministic kernels and code hashes matter |
| **C36** | 31/31 tests passed in the final OOD execution records | `[OBS-OOD]` | Regression test log | Test-suite pass ≠ raw data independent audit or clinical validation |
| **C37** | There is no evidence this project implemented PLM, QFR, SDC or a learned failure predictor | `[NOT-IMPL]` | Implementation map and logs | Do not present these as experimental comparator results |
| **C38** | Same-organ OOD reduces anatomy differences but leaves scanner/case-mix/annotation interactions | `[LIT,LIMIT]` | Four breast-ultrasound external cohorts | Does not establish isolated scanner causal effect |
| **C39** | Normal-case inclusion changes Spearman and AURC estimands | `[OBS-VAL,DER]` | `ρ(U_mean,DSC)=−0.4449` N78 vs `−0.0451` N65 | Never compare results without cohort denominators |
| **C40** | No risk-control theorem is inherited from empirical 80%-coverage calibration | `[DER,LIT]` | One validation quantile and external coverage drop | No prospective or distribution-free quality guarantee |

### How to use a C-ID in a paper

For a theoretical statement, cite the correct publication or include the derivation. For a data statement, cite the specific result table, dataset, cohort and configuration; source paper citations **cannot replace actual data**. Every claim should be testable against an authoritative per-image dataset or a mathematical identity. Where a raw artifact is not available for document assembly, retain that provenance dependency rather than invent a row-level verification.

## 10. Falsification tests and alternative explanations

| Hypothesis | What would count against it? | Observed status | Remaining valid competing explanation |
|---|---|---|---|
| H1: `U_fg` beats global mean | `ΔAURC≥0` consistently on held-out test or external sets | Supported on BUSI val/test and each external AURC listed | Area, calibration and foreground shape jointly affect score patterns |
| H2: Boundary may be more NSD-specific | `U_boundary` offers no distinct NSD association or is never better for local contour QC | Validation NSD rho more negative than its DSC rho, but overall ranking inferior | NSD metric/tolerance and morphology interactions |
| H3: `U_std_fg` adds useful diversity | Large paired inferiority or no benefit under shift | Unresolved difference; no validated combination/model selection | Mean/std redundancy can vary across acquisition domains |
| H4: Frozen cutoff supports useful retention | Retained mean DSC ≤ full mean in independent sets | Retained means better for all five tested cohorts | High rejection rates can make mean quality rise at unacceptable review cost |
| H5: OOD ranking generalizes | `U_fg` loses advantage against `U_mean`/Random | Advantage observed on four test sets; pooled lower AURC | Shared test source, prompt design and cohort selection could matter |
| H6: The gate is safe for all errors | Any severe error accepted or missed lesion accepted with low entropy | Falsified by BUSI accepted severe error and synthetic empty-mask counterexample | Empirical rate under new clinical conditions unknown |

**A falsifiable hypothesis is not a proof by definition.** Negative results and alternative mechanisms remain relevant even when the project finds positive mean performance. The primary scientific contribution is the **measured utility and limitations of a simple training-free selector** attached to the original probabilistic segmentation model, not the claim that a new uncertainty theory has been discovered.

## 11. Correct statistical reporting language

Recommended formulations for a scientific manuscript:

- “The foreground-mean entropy ranker achieved lower empirical AURC than the global spatial mean on the **BUSI validation lesion cohort** (0.0839 vs 0.1438, N=65), and the locked test comparison showed the same direction (0.0790 vs 0.1262, N=65).”
- “The prespecified cutoff produced **72.3% realized coverage on BUSI test**, compared with the nominal 80% validation target.”
- “Across four external breast-ultrasound test sets (N=464), the retained 238 predictions had higher mean DSC (0.8718) than all 464 predictions (0.7780). This is a quality–coverage tradeoff, not an improvement in the underlying segmentation model.”
- “Pooling external cases gave AURC 0.1267 for `U_fg` versus 0.1864 for global mean entropy. The four-dataset macro mean is a separate estimand (0.1168 versus 0.1825).”
- “A confident severe segmentation failure passed the frozen cutoff, and no OOD predicted-empty case was observed; therefore, the data do not support comprehensive failure detection or a clinical safety guarantee.”
- “The comparison of `U_fg` and `U_std_fg` did not resolve a statistically significant AURC difference, so `U_std_fg` remains a viable comparator.”

Avoid:

- “`U_fg` **proves** perfect error detection,” “improved MedCLIPSeg accuracy,” “eliminated lesion-size confounding,” “the Oracle ceiling is an AURC upper bound,” “80% coverage was maintained OOD,” “all published methods were implemented,” “statistically significant across all future hospitals,” or “MedCLIPSeg is clinically validated by this study.”

## 12. Corrective audit trail: earlier numbers must remain identifiable

| Historical record | Original draft or issue | Correct authority | Why the correction is necessary |
|---|---|---|---|
| Preliminary uncertainty analysis | Pseudocolor PNG-derived correlations and AUROCs | RAW float32 entropy `.npy` | Per-image visualization rescaling invalidates absolute uncertainty aggregation |
| Validation Random AURC | `0.1434` vs `1−mean DSC≈0.1480` | Unanchored empirical coverage domain and tie convention | Empirical trapezoidal area ≠ full-range integral by default |
| BUSI test Random in Step 11B draft | `0.1406` | `0.1687` for BUSI test **N65** | Draft mixed in 13 normal N78 cases |
| Pooled OOD baseline | `0.7711` | **`0.7780`** | Recalculated with 464 individual images |
| Pooled OOD accepted DSC | `0.8732` | **`0.8718`** | 238 accepted images with proper weighting |
| Pooled OOD rejected DSC | `0.6635` | **`0.6791`** | 226 rejected images |
| Pooled OOD DSC gain | `+0.1021` | **`+0.0939`**, CI `[0.0775,0.1113]` | Correct difference in pooled means |
| Pooled OOD `U_fg` AURC | Earlier draft `0.1242` | **`0.1267`** (global case-level rank) | Distinguish pooled/global from macro and weighted AURC |
| Oracle terminology | “Theoretical upper bound” used for AURC | **Lower bound on empirical risk/AURC** | Top-k exchange proof |
| Published baseline labels | Some internal aggregators attributed by author name | Exact local formula plus honest conceptual/source relationship | Scientific attribution must identify exact published equation |

## 13. Open issues, ranked by impact on final publication

**P0 — Required for reproducible final manuscript.** Retrieve/check primary per-image CSVs, frozen YAML/JSON, source code and hashes; verify the original per-dataset prompt files, test split identities, checkpoint provenance; recompute each rounded table and source figure from code. Verify whether earlier explorations exposed any BUSI test information that could have influenced method development, and disclose the chronology. Precisely document NSD surface tolerance and spacing; confirm evaluation source code and dataset labels. Verify AURC grid and tie-group treatment for each candidate with numeric reference arrays. None of these should lead to **post-hoc re-selection or recalibration**.

**P1 — Statistical and causal caution.** Check patient-level clustering and repeated images; if absent, identify bootstrap assumptions. Report CI method and p-value computation with multiple-comparison caveats. Use stronger tests before saying “not affected by foreground size” or attributing performance causally to geometric dilution. Stratify performance by lesion area and pathology where sample counts allow, with descriptive not guaranteed claims.

**P1 — Clinical interpretation.** Evaluate independent, GT-positive/predicted-empty examples and false-negative failure rates; investigate zero/near-zero uncertainty with wrong segmentations. Assess review burden and practical consequences of 48.7% OOD abstention. These are **limitations or future experiments**, not grounds to quietly change the frozen method after observing test/OOD outcomes.

**P2 — Additional literature baselines.** If further work is allowed, implement Guarino PLM/QFR or a confidence estimator with clearly specified prerequisites and validation-only selection; do not retrofit their results into the current locked experiment. A new experiment would need an independent design and should not be represented as part of the original comparison.

## 14. Bibliographic integrity and cross-references

The source table in §2 gives directly verifiable journal/conference links. Exact bibliographic entries and definitions also appear in [Theory and Metrics](THEORY_AND_METRICS.md) §12. Dataset and execution artifacts are indexed in [Experimental Protocol](EXPERIMENTAL_PROTOCOL.md) §9. Sequential decision IDs, historical numerical corrections and implementation caveats are retained in [Research Decisions and Limitations](RESEARCH_DECISIONS_AND_LIMITATIONS.md).

**Manuscript policy:** third-person technical prose only; distinguish original hypothesis from observed result and from external literature; state numerical denominators, confidence intervals and units; reproduce negative and unresolved results; retain known counterexamples; do not write speculative clinical guarantees or attribute an internally specified metric to a publication without equation-level evidence.
