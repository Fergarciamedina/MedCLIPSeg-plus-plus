# Scientific Manuscript: Uncertainty-Based Selective Prediction for Text-Guided Medical Image Segmentation Under Domain Shift

**Course:** ELEC 6661 -- Medical Image Processing (Fall 2026), Concordia University  
**Authors:** Fernando Garcia-Medina  
**Supervision:** IMPACT Lab (Prof. Hassan Rivaz, Omid Nejati Manzari)  
**Baseline Model:** MedCLIPSeg (Koleilat et al., CVPR 2026)  
**Status:** Step 12B Delivery -- Complete LaTeX Manuscript & Compiled PDF

---

## Directory Overview

```
uncertainty_rejection/paper/
├── main.tex                       # Primary LaTeX source (IEEE Computer Society 10pt format)
├── references.bib                 # Verified BibTeX bibliography (16 entries)
├── main.pdf                       # Fully compiled 10-page scientific publication PDF
├── generate_paper_figures.py      # Automated reproduction script for all paper figures
├── MANUSCRIPT_VERIFICATION.md     # Numerical audit, source-to-claim mapping & verification checklist
├── figures/                       # Vector PDF and 300 DPI PNG figures
│   ├── fig1_methodology.{pdf,png}
│   ├── fig2_qualitative_cases.{pdf,png}
│   ├── fig3_val_aggregators.{pdf,png}
│   ├── fig4_busi_risk_coverage.{pdf,png}
│   └── fig5_cross_domain_performance.{pdf,png}
└── tables/                        # Master data summaries
```

---

## Quick Compilation Guide

To compile `main.pdf` from source:
```bash
cd project/MedCLIPSeg-plus-plus/uncertainty_rejection/paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

To regenerate all publication figures:
```bash
cd project/MedCLIPSeg-plus-plus
MPLCONFIGDIR=/tmp/mpl .venv/bin/python3 uncertainty_rejection/paper/generate_paper_figures.py
```

