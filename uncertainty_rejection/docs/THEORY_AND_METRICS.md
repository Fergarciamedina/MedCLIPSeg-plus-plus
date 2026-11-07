# ELEC 6661 — Theoretical Foundations & Uncertainty Metrics Formulation

**Document:** `THEORY_AND_METRICS.md`  
**Location:** `uncertainty_rejection/docs/`  
**Date:** 2026-11-07  
**Scope:** Mathematical derivations, formal definitions of uncertainty aggregation operators, risk-coverage analysis, and statistical hypothesis testing framework.

---

## 1. Probabilistic Segmentation & Pixel Entropy

Let $x \in \mathbb{R}^{H \times W \times 3}$ denote an ultrasound input image, and let $t$ denote a clinical text prompt. MedCLIPSeg parametrizes a conditional probability map $p(y_i = 1 \mid x, t; \theta)$ for each pixel $i \in \Omega = \{1, \dots, H \times W\}$.

### 1.1. Monte Carlo Dropout Sampling
To approximate the posterior predictive distribution without training an ensemble, we apply Monte Carlo dropout (Gal & Ghahramani, 2016). Given $T = 30$ stochastic forward passes with dropout active at test time:
$$\bar{p}_i = \frac{1}{T} \sum_{t=1}^T \sigma\left( f(x, t; \theta^{(t)})_i \right)$$
where $\sigma(\cdot)$ is the sigmoid function, $\theta^{(t)}$ represents the model parameters under stochastic dropout mask $t$, and $\bar{p}_i \in [0, 1]$ is the empirical posterior mean probability.

### 1.2. Pixel-Level Predictive Entropy
The predictive entropy at pixel $i$, expressed in natural units (nats), is given by:
$$H(\bar{p}_i) = - \left[ \bar{p}_i \ln(\bar{p}_i + \epsilon) + (1 - \bar{p}_i) \ln(1 - \bar{p}_i + \epsilon) \right]$$
where $\epsilon = 10^{-8}$ prevents numerical instability. The entropy is strictly bounded:
$$0 \le H(\bar{p}_i) \le \ln 2 \approx 0.69315 \text{ nats}$$
with $H = 0$ representing complete deterministic certainty ($\bar{p}_i \in \{0, 1\}$) and $H = \ln 2$ representing maximum epistemic ambiguity ($\bar{p}_i = 0.5$).

The predicted binary segmentation mask $\hat{Y} \in \{0, 1\}^{H \times W}$ is obtained via standard thresholding:
$$\hat{y}_i = \mathbb{I}(\bar{p}_i > 0.5)$$
The predicted foreground lesion pixel set is defined as $\Omega_{\mathrm{fg}} = \{i \in \Omega \mid \hat{y}_i = 1\}$.

---

## 2. Seven Image-Level Uncertainty Aggregation Operators

To convert the $224 \times 224$ entropy map $H \in \mathbb{R}^{H \times W}$ into an actionable scalar image-level score $U$, we define seven spatial operators:

### 2.1. Global Mean Entropy ($U_{\mathrm{mean}}$)
Baseline representing full-image spatial averaging (Kendall & Gal, 2017; Lakshminarayanan et al., 2017):
$$U_{\mathrm{mean}} = \frac{1}{|\Omega|} \sum_{i \in \Omega} H(\bar{p}_i)$$
where $|\Omega| = 224 \times 224 = 50,176$.

### 2.2. Positive-Entropy Filter ($U_{\mathrm{positive}}$)
Ablation baseline filtering out completely deterministic background pixels:
$$U_{\mathrm{positive}} = \frac{1}{|\Omega_{+}|} \sum_{i \in \Omega_{+}} H(\bar{p}_i), \quad \Omega_{+} = \{i \in \Omega \mid H(\bar{p}_i) > 0\}$$

### 2.3. Foreground Mean Predictive Entropy ($U_{\mathrm{fg}}$) — PRIMARY METHOD
Normalizes predictive entropy strictly over the predicted lesion footprint (adapted from Roy et al., Bayesian QuickNAT, 2019):
$$U_{\mathrm{fg}} = \begin{cases} \frac{1}{|\Omega_{\mathrm{fg}}|} \sum_{i \in \Omega_{\mathrm{fg}}} H(\bar{p}_i), & \text{if } |\Omega_{\mathrm{fg}}| > 0 \\ \text{NaN}, & \text{if } |\Omega_{\mathrm{fg}}| = 0 \end{cases}$$

### 2.4. Morphological Boundary Band Entropy ($U_{\mathrm{boundary}}$)
Measures contour localization ambiguity within a morphological dilation/erosion band (Wang et al., 2019; Mehrtash et al., 2020):
$$\Omega_{\mathrm{boundary}} = (\Omega_{\mathrm{fg}} \oplus K_w) \setminus (\Omega_{\mathrm{fg}} \ominus K_w)$$
$$U_{\mathrm{boundary}} = \begin{cases} \frac{1}{|\Omega_{\mathrm{boundary}}|} \sum_{i \in \Omega_{\mathrm{boundary}}} H(\bar{p}_i), & \text{if } |\Omega_{\mathrm{boundary}}| > 0 \\ \text{NaN}, & \text{if } |\Omega_{\mathrm{boundary}}| = 0 \end{cases}$$
where $K_w$ is a structuring element of half-width $w = 2$ pixels.

### 2.5. High-Uncertainty Pixel Area Fraction ($U_{\mathrm{threshold}}$)
Measures the volume fraction of severely ambiguous pixels (Eaton-Rosen et al., MICCAI 2018):
$$U_{\mathrm{threshold}} = \frac{1}{|\Omega|} \sum_{i \in \Omega} \mathbb{I}(H(\bar{p}_i) > t), \quad t = 0.1 \text{ nats}$$

### 2.6. Foreground Entropy Standard Deviation ($U_{\mathrm{std\_fg}}$) — COMPARATOR
Measures internal spatial variance and contour dissonance within the predicted lesion (Jungo et al., 2020):
$$U_{\mathrm{std\_fg}} = \begin{cases} \sqrt{\frac{1}{|\Omega_{\mathrm{fg}}|} \sum_{i \in \Omega_{\mathrm{fg}}} \left( H(\bar{p}_i) - U_{\mathrm{fg}} \right)^2}, & \text{if } |\Omega_{\mathrm{fg}}| > 0 \\ \text{NaN}, & \text{if } |\Omega_{\mathrm{fg}}| = 0 \end{cases}$$

### 2.7. Foreground-to-Global Entropy Ratio ($U_{\mathrm{ratio}}$)
Measures the signal-to-background contrast ratio:
$$U_{\mathrm{ratio}} = \frac{U_{\mathrm{fg}}}{U_{\mathrm{mean}} + \epsilon_{\mathrm{ratio}}}, \quad \epsilon_{\mathrm{ratio}} = 10^{-6}$$

---

## 3. Selective Prediction & Risk-Coverage Analysis

Selective prediction (El-Yaniv & Wiener, 2010; Geifman & El-Yaniv, 2017) defines a rejection decision rule $g(x) \in \{0, 1\}$ parameterized by threshold $\tau$:
$$g(x) = \mathbb{I}(U(x) \le \tau)$$
where $g(x) = 1$ denotes retaining (accepting) the segmentation for clinical use, and $g(x) = 0$ denotes rejecting (abstaining) to route the image to expert sonographer review.

### 3.1. Empirical Coverage and Risk
For a dataset of $N$ images with segmentation quality metric $S_i \in [0, 1]$ (e.g. DSC or NSD), the realized coverage $c(\tau)$ is the fraction of accepted scans:
$$c(\tau) = \frac{1}{N} \sum_{i=1}^N g(x_i) = \frac{N_{\mathrm{accepted}}}{N}$$
The selective empirical risk $R(\tau)$ is the expected segmentation error on accepted scans:
$$R(\tau) = 1 - \frac{1}{N_{\mathrm{accepted}}} \sum_{i: g(x_i)=1} S_i = 1 - \bar{S}_{\mathrm{accepted}}$$

### 3.2. Deterministic Tie-Handling
When multiple images share identical uncertainty scores, arbitrary sorting permutations produce non-deterministic risk-coverage steps. We enforce deterministic evaluation by evaluating coverage and risk strictly at distinct score cutoff boundaries:
$$\mathcal{T} = \text{Unique}(\{U(x_i)\}_{i=1}^N)$$

### 3.3. Area Under the Risk-Coverage Curve (AURC)
The Risk-Coverage curve plots $R(c)$ as a function of coverage $c \in [c_1, 1.0]$, where $c_1 = 1/N$. The overall quality of selective ranking is quantified by the Area Under the Risk-Coverage Curve (AURC) via trapezoidal integration:
$$\text{AURC} = \int_{c_1}^{1.0} R(c) \, dc \approx \sum_{k=1}^{K-1} \frac{R(c_k) + R(c_{k+1})}{2} (c_{k+1} - c_k)$$
A lower AURC indicates superior selective prediction performance (lower error across all retention budgets).

---

## 4. Evaluation Bounds: Random Rejection & Oracle Reference

To calibrate AURC values, every evaluation is benchmarked against two theoretical bounds:

### 4.1. Uninformed Random Rejection (Null Control)
An uninformed baseline that rejects images completely at random without inspection. The expected risk at any coverage $c$ equals the overall baseline error:
$$R_{\mathrm{rand}}(c) = 1 - \bar{S}_{\mathrm{base}} = R_0$$
The empirical trapezoidal AURC over the domain $[1/N, 1.0]$ is analytically exact:
$$\text{AURC}_{\mathrm{rand}} = R_0 \times \left( 1 - \frac{1}{N} \right)$$

### 4.2. Oracle Rejection (Theoretical Performance Ceiling)
An idealized, non-deployable upper bound that sorts images by ground-truth segmentation metric $S_i$ in descending order (highest DSC accepted first). It represents the absolute minimum achievable empirical risk for that dataset.

---

## 5. Statistical Hypothesis Testing Framework

### 5.1. Bootstrap Confidence Intervals
For selective DSC gains $\Delta \bar{S} = \bar{S}_{\mathrm{accepted}} - \bar{S}_{\mathrm{base}}$, we compute empirical percentile bootstrap confidence intervals ($B = 1000$ resamples with replacement, seed 666):
$$\text{CI}_{95\%} = \left[ q_{0.025}^*, \, q_{0.975}^* \right]$$

### 5.2. Paired Bootstrap $\Delta\text{AURC}$ Test
To compare two competing uncertainty metrics $A$ and $B$, we test the null hypothesis:
$$H_0: \Delta\text{AURC} = \text{AURC}(A) - \text{AURC}(B) = 0$$
Using $B = 1000$ paired bootstrap resamples:
$$\Delta^{*(b)} = \text{AURC}^*(A) - \text{AURC}^*(B)$$
The empirical two-tailed $p$-value is calculated as:
$$p = 2 \times \min\left( \frac{1}{B} \sum_{b=1}^B \mathbb{I}(\Delta^{*(b)} \ge 0), \, \frac{1}{B} \sum_{b=1}^B \mathbb{I}(\Delta^{*(b)} \le 0) \right)$$
$p < 0.05$ indicates statistically significant superiority.

