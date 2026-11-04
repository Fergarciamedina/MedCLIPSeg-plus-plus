# Future Work & Conceptual Extensions (Untested Ideas)

**Scope Notice:** The directions documented here represent conceptual hypotheses and potential post-course extensions. None of the concepts below have been validated with raw numerical evaluation, and they must **not** be interpreted as claimed improvements, established findings, or demonstrated clinical systems.

---

## 1. Morphological and Boundary-Sensitive Metrics

- **Boundary Entropy vs. NSD:**
  While Dice Similarity Coefficient (DSC) penalizes regional overlap, Normalized Surface Distance (NSD) reflects boundary precision. Exploring thin boundary bands ($U_{\text{boundary}}$) may provide stronger correlation with boundary-sensitive metrics than global region statistics, though this remains to be empirically tested on raw entropy arrays.
- **Topological Edge Cases (Multifocal and Ring Lesions):**
  Complex lesions (e.g., central necrosis, cystic acoustic enhancement, or multifocal tumors) present disjoint foreground masks where simple mean pooling may mask localized ambiguities. Formulating connected-component or cluster-based uncertainty metrics could address these edge cases.

---

## 2. Advanced Spatial Statistics

- **Spatial Dispersion Measures:**
  Investigating higher-order spatial moments of the uncertainty distribution (e.g., spatial skewness, spatial kurtosis, or entropy gradient magnitude along the lesion perimeter) to determine if boundary sharpness correlates with clinician disagreement.

---

## 3. Dimensionality and Clinical Workflow Integration

- **Extension to 3D Automated Breast Ultrasound (ABUS):**
  Evaluating whether slice-wise 2D variational Monte Carlo passes can be integrated into volumetric consistency scores across adjacent coronal and transverse planes in 3D ultrasound volumes.
- **Real-Time Clinical Review & Active Learning:**
  Conceptualizing an active learning loop where images rejected by the selective prediction rule are submitted to expert sonographers for manual correction, whose annotations subsequently update adapter weights. (Note: Retraining or adapter fine-tuning is strictly outside the scope of ELEC 6661).

---

## 4. Multi-Modal Retrieval Coupling

- Coupling rejected high-uncertainty acquisitions with historical reference cases via multimodal feature embeddings to provide sonographers with comparative visual evidence during triage. (Explored conceptually in Level 2).
