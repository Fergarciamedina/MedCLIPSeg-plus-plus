"""
Script to generate publication-quality figures for the ELEC 6661 scientific manuscript:
"Uncertainty-Based Selective Prediction for Text-Guided Medical Image Segmentation Under Domain Shift"

Figures:
1. fig1_methodology.png / .pdf - Schematic workflow of MedCLIPSeg selective prediction
2. fig2_qualitative_cases.png / .pdf - Qualitative comparison (Accepted, Rejected, and False Certainty Failure)
3. fig3_val_aggregators.png / .pdf - Validation comparison of 7 aggregators (Spearman rho & AURC)
4. fig4_busi_risk_coverage.png / .pdf - BUSI test risk-coverage curves
5. fig5_cross_domain_performance.png / .pdf - External OOD baseline vs accepted DSC & realized coverage
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Set styling
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.color'] = '#333333'
plt.rcParams['ytick.color'] = '#333333'
plt.rcParams['font.size'] = 9

OUTPUT_DIR = "project/MedCLIPSeg-plus-plus/uncertainty_rejection/paper/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------------------
# Figure 1: Methodology Pipeline Diagram
# -------------------------------------------------------------
def generate_fig1_methodology():
    fig, ax = plt.subplots(figsize=(10, 3.8), dpi=300)
    ax.axis('off')

    # Color palette
    c_blue = '#E3F2FD'
    c_blue_b = '#1976D2'
    c_gray = '#F5F5F5'
    c_gray_b = '#616161'
    c_green = '#E8F5E9'
    c_green_b = '#388E3C'
    c_orange = '#FFF3E0'
    c_orange_b = '#E65100'
    c_red = '#FFEBEE'
    c_red_b = '#D32F2F'

    # Box 1: Input
    b1 = patches.FancyBboxPatch((0.02, 0.40), 0.16, 0.48, boxstyle="round,pad=0.02",
                                facecolor=c_blue, edgecolor=c_blue_b, linewidth=1.5)
    ax.add_patch(b1)
    ax.text(0.10, 0.76, "Clinical Input", ha='center', va='center', weight='bold', color=c_blue_b, fontsize=10)
    ax.text(0.10, 0.58, "Ultrasound Image $X$\n+\nNeutral Text Prompt $T$\n('a tumor in breast')",
            ha='center', va='center', fontsize=8.5, color='#1A237E')

    # Arrow 1 -> 2
    ax.annotate('', xy=(0.23, 0.64), xytext=(0.18, 0.64),
                arrowprops=dict(facecolor=c_gray_b, edgecolor=c_gray_b, width=1.5, headwidth=6))

    # Box 2: Frozen Model
    b2 = patches.FancyBboxPatch((0.23, 0.32), 0.22, 0.64, boxstyle="round,pad=0.02",
                                facecolor=c_gray, edgecolor=c_gray_b, linewidth=1.5)
    ax.add_patch(b2)
    ax.text(0.34, 0.84, "Frozen MedCLIPSeg\n(CVPR 2026)", ha='center', va='center', weight='bold', color='#212121', fontsize=10)
    ax.text(0.34, 0.55, "Frozen UniMedCLIP\nViT-B/16 Backbone\n+\nProbabilistic PVL Adapters\n($T=30$ MC Value Sampling)\nZero-shot / No Retraining",
            ha='center', va='center', fontsize=8, color='#424242')

    # Arrow 2 -> 3
    ax.annotate('', xy=(0.50, 0.64), xytext=(0.45, 0.64),
                arrowprops=dict(facecolor=c_gray_b, edgecolor=c_gray_b, width=1.5, headwidth=6))

    # Box 3: Dense Inference Outputs
    b3 = patches.FancyBboxPatch((0.50, 0.38), 0.18, 0.52, boxstyle="round,pad=0.02",
                                facecolor='#F3E5F5', edgecolor='#7B1FA2', linewidth=1.5)
    ax.add_patch(b3)
    ax.text(0.59, 0.78, "Dense Outputs", ha='center', va='center', weight='bold', color='#7B1FA2', fontsize=10)
    ax.text(0.59, 0.56, "Predicted Mask $\hat{Y} \in \{0,1\}$\n&\nRaw Predictive Entropy\n$\mathcal{H}_i \in [0, \ln 2]$\n(float32 map)",
            ha='center', va='center', fontsize=8.5, color='#4A148C')

    # Arrow 3 -> 4
    ax.annotate('', xy=(0.73, 0.64), xytext=(0.68, 0.64),
                arrowprops=dict(facecolor=c_gray_b, edgecolor=c_gray_b, width=1.5, headwidth=6))

    # Box 4: Foreground Aggregator
    b4 = patches.FancyBboxPatch((0.73, 0.45), 0.25, 0.38, boxstyle="round,pad=0.02",
                                facecolor=c_orange, edgecolor=c_orange_b, linewidth=1.5)
    ax.add_patch(b4)
    ax.text(0.855, 0.72, "Lesion-Focused Aggregation", ha='center', va='center', weight='bold', color=c_orange_b, fontsize=9.5)
    ax.text(0.855, 0.55, r"$U_{\mathrm{fg}} = \frac{1}{|\Omega_{\mathrm{fg}}|} \sum_{i \in \Omega_{\mathrm{fg}}} \mathcal{H}_i$" + "\n(Mitigates Background Dilution)",
            ha='center', va='center', fontsize=8.5, color='#BF360C')

    # Decision Split Arrows
    ax.annotate('', xy=(0.855, 0.28), xytext=(0.855, 0.45),
                arrowprops=dict(facecolor=c_gray_b, edgecolor=c_gray_b, width=1.5, headwidth=6))

    # Box 5A: Accept (Green)
    b5a = patches.FancyBboxPatch((0.68, 0.05), 0.16, 0.22, boxstyle="round,pad=0.02",
                                 facecolor=c_green, edgecolor=c_green_b, linewidth=1.5)
    ax.add_patch(b5a)
    ax.text(0.76, 0.20, r"$U_{\mathrm{fg}} \leq \tau$", ha='center', va='center', weight='bold', color=c_green_b, fontsize=9.5)
    ax.text(0.76, 0.10, "RETAIN MASK\n(Autonomous QA)", ha='center', va='center', weight='bold', color='#1B5E20', fontsize=8)

    # Box 5B: Reject (Red)
    b5b = patches.FancyBboxPatch((0.87, 0.05), 0.16, 0.22, boxstyle="round,pad=0.02",
                                 facecolor=c_red, edgecolor=c_red_b, linewidth=1.5)
    ax.add_patch(b5b)
    ax.text(0.95, 0.20, r"$U_{\mathrm{fg}} > \tau$", ha='center', va='center', weight='bold', color=c_red_b, fontsize=9.5)
    ax.text(0.95, 0.10, "ABSTAIN / REFER\n(Expert Review)", ha='center', va='center', weight='bold', color='#B71C1C', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig1_methodology.png"), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(OUTPUT_DIR, "fig1_methodology.pdf"), bbox_inches='tight')
    plt.close()
    print("Generated Fig 1: Methodology Diagram")

# -------------------------------------------------------------
# Figure 2: Qualitative Cases (Accepted, Rejected, Failure Mode)
# -------------------------------------------------------------
def generate_fig2_qualitative_cases():
    cases = [
        {
            "title": "Case A: High-Quality Retained Segment (BUSI 186_benign)",
            "sub": "DSC = 0.9531 | U_fg = 0.0435 <= tau (0.0641) -> ACCEPTED",
            "img": "project/data/raw/BUSI/Test_Folder/img/186_benign.png",
            "gt": "project/data/raw/BUSI/Test_Folder/label/186_benign.png",
            "pred": "project/data/outputs/BUSI/seg_results/seed666/MedCLIPSeg_unimedclip_ViT-B-16_Prompt-original/186_benign.png",
            "unc": "project/data/outputs/BUSI/unc_results/seed666/MedCLIPSeg_unimedclip_ViT-B-16_Prompt-original/186_benign.npy"
        },
        {
            "title": "Case B: Low-Confidence Correctly Rejected (BUSI 169_malignant)",
            "sub": "DSC = 0.0877 | U_fg = 0.1042 > tau (0.0641) -> REJECTED",
            "img": "project/data/raw/BUSI/Test_Folder/img/169_malignant.png",
            "gt": "project/data/raw/BUSI/Test_Folder/label/169_malignant.png",
            "pred": "project/data/outputs/BUSI/seg_results/seed666/MedCLIPSeg_unimedclip_ViT-B-16_Prompt-original/169_malignant.png",
            "unc": "project/data/outputs/BUSI/unc_results/seed666/MedCLIPSeg_unimedclip_ViT-B-16_Prompt-original/169_malignant.npy"
        },
        {
            "title": "Case C: Counterexample False Certainty (BUSI 116_malignant)",
            "sub": "DSC = 0.2193 | U_fg = 0.0628 <= tau (0.0641) -> ERRONEOUSLY ACCEPTED",
            "img": "project/data/raw/BUSI/Test_Folder/img/116_malignant.png",
            "gt": "project/data/raw/BUSI/Test_Folder/label/116_malignant.png",
            "pred": "project/data/outputs/BUSI/seg_results/seed666/MedCLIPSeg_unimedclip_ViT-B-16_Prompt-original/116_malignant.png",
            "unc": "project/data/outputs/BUSI/unc_results/seed666/MedCLIPSeg_unimedclip_ViT-B-16_Prompt-original/116_malignant.npy"
        }
    ]

    fig, axes = plt.subplots(3, 4, figsize=(9.5, 7.5), dpi=300)
    col_names = ["(a) Ultrasound", "(b) Ground Truth", "(c) MedCLIPSeg", "(d) Predictive Entropy"]

    for row_idx, case in enumerate(cases):
        # Load files
        img = Image.open(case["img"]).convert('L')
        gt = Image.open(case["gt"]).convert('L')
        pred = Image.open(case["pred"]).convert('L')
        unc = np.load(case["unc"])

        axes[row_idx, 0].imshow(img, cmap='gray')
        axes[row_idx, 1].imshow(gt, cmap='gray')
        axes[row_idx, 2].imshow(pred, cmap='gray')
        im_unc = axes[row_idx, 3].imshow(unc, cmap='plasma', vmin=0, vmax=np.log(2))

        for col_idx in range(4):
            axes[row_idx, col_idx].set_xticks([])
            axes[row_idx, col_idx].set_yticks([])
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(col_names[col_idx], fontsize=9.5, weight='bold', pad=4)

        # Label each row on the left
        axes[row_idx, 0].set_ylabel(f"Row {row_idx+1}", fontsize=9, weight='bold')

    # Colorbar on right
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cb = fig.colorbar(im_unc, cax=cbar_ax)
    cb.set_label('Predictive Entropy $\mathcal{H}_i$ [nats]', fontsize=8.5)

    plt.subplots_adjust(left=0.05, right=0.90, top=0.94, bottom=0.04, hspace=0.18, wspace=0.08)
    plt.savefig(os.path.join(OUTPUT_DIR, "fig2_qualitative_cases.png"), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(OUTPUT_DIR, "fig2_qualitative_cases.pdf"), bbox_inches='tight')
    plt.close()
    print("Generated Fig 2: Qualitative Cases")

# -------------------------------------------------------------
# Figure 3: Validation Comparison of 7 Aggregators
# -------------------------------------------------------------
def generate_fig3_val_aggregators():
    methods = [r"$U_{\mathrm{fg}}$ (Primary)", r"$U_{\mathrm{std\_fg}}$", r"$U_{\mathrm{boundary}}$",
               r"$U_{\mathrm{ratio}}$", r"$U_{\mathrm{mean}}$ (Global)", r"$U_{\mathrm{positive}}$", r"$U_{\mathrm{threshold}}$"]
    aurc_val = [0.0839, 0.0850, 0.1302, 0.1368, 0.1438, 0.1455, 0.1750]
    rho_dsc = [-0.7202, -0.7068, -0.2273, 0.0812, -0.0451, -0.0398, -0.1245]
    rho_nsd = [-0.7410, -0.7285, -0.4174, 0.0945, -0.0712, -0.0655, -0.1582]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8), dpi=300)

    # Panel A: Spearman Correlation
    y = np.arange(len(methods))
    width = 0.35

    ax1.barh(y - width/2, rho_dsc, width, label=r'Spearman $\rho$ (DSC)', color='#1976D2', alpha=0.85)
    ax1.barh(y + width/2, rho_nsd, width, label=r'Spearman $\rho$ (NSD)', color='#388E3C', alpha=0.85)
    ax1.set_yticks(y)
    ax1.set_yticklabels(methods, fontsize=8.5)
    ax1.invert_yaxis()
    ax1.set_xlabel('Spearman Rank Correlation (Negative = More Predictive)', fontsize=8.5)
    ax1.set_title('(a) Rank Correlation with Segmentation Accuracy', fontsize=9.5, weight='bold')
    ax1.axvline(0, color='black', linestyle='--', linewidth=0.7)
    ax1.legend(loc='lower left', fontsize=8)
    ax1.grid(axis='x', linestyle=':', alpha=0.6)

    # Panel B: AURC
    colors = ['#1976D2', '#0288D1', '#7B1FA2', '#F57C00', '#E64A19', '#D32F2F', '#C2185B']
    bars = ax2.barh(y, aurc_val, height=0.55, color=colors, alpha=0.85)
    ax2.axvline(0.1434, color='#757575', linestyle='--', linewidth=1.2, label='Random Baseline (0.1434)')
    ax2.axvline(0.0612, color='#388E3C', linestyle=':', linewidth=1.2, label='Oracle Upper Bound (0.0612)')
    ax2.set_yticks(y)
    ax2.set_yticklabels([])
    ax2.invert_yaxis()
    ax2.set_xlabel('Area Under Risk-Coverage Curve (AURC $\downarrow$)', fontsize=8.5)
    ax2.set_title('(b) Selective Quality Efficiency (BUSI Val N=65)', fontsize=9.5, weight='bold')
    ax2.legend(loc='lower right', fontsize=8)
    ax2.grid(axis='x', linestyle=':', alpha=0.6)

    # Annotate bar values
    for i, v in enumerate(aurc_val):
        ax2.text(v + 0.003, i, f"{v:.4f}", va='center', fontsize=7.5, color='#212121')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig3_val_aggregators.png"), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(OUTPUT_DIR, "fig3_val_aggregators.pdf"), bbox_inches='tight')
    plt.close()
    print("Generated Fig 3: Validation Aggregators")

# -------------------------------------------------------------
# Figure 4: BUSI Test Risk-Coverage Curves
# -------------------------------------------------------------
def generate_fig4_busi_risk_coverage():
    # Load canonical BUSI test results directly
    df = pd.read_csv("project/data/outputs/BUSI/aggregation_results/seed666/uncertainty_scores_BUSI_test_seed666.csv")
    
    # Filter lesion cases (N=65)
    df_lesion = df[~df["image_id"].str.contains("normal")].copy()
    n = len(df_lesion)
    
    def compute_rc(scores, risks):
        # Sort by score ascending (lowest uncertainty first)
        order = np.argsort(scores)
        sorted_risks = risks[order]
        covs = np.arange(1, n + 1) / n
        cum_risks = np.cumsum(sorted_risks) / np.arange(1, n + 1)
        # Prepend (0, cum_risks[0])
        return np.insert(covs, 0, 0), np.insert(cum_risks, 0, cum_risks[0])

    risks = (1.0 - df_lesion["DSC"].values)
    cov_ufg, risk_ufg = compute_rc(df_lesion["U_fg"].values, risks)
    cov_std, risk_std = compute_rc(df_lesion["U_std_fg"].values, risks)
    cov_mean, risk_mean = compute_rc(df_lesion["U_mean"].values, risks)
    cov_oracle, risk_oracle = compute_rc(risks, risks) # Sorted by true risk

    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=300)
    ax.plot(cov_ufg, risk_ufg, label=r'$U_{\mathrm{fg}}$ (Primary, AURC = 0.0790)', color='#1976D2', linewidth=2.0)
    ax.plot(cov_std, risk_std, label=r'$U_{\mathrm{std\_fg}}$ (Comparator, AURC = 0.0793)', color='#0288D1', linewidth=1.5, linestyle='-.')
    ax.plot(cov_mean, risk_mean, label=r'$U_{\mathrm{mean}}$ (Global, AURC = 0.1262)', color='#E64A19', linewidth=1.5, linestyle='--')
    ax.axhline(0.1687, color='#757575', linestyle=':', label='Random Rejection (AURC = 0.1687)', linewidth=1.3)
    ax.plot(cov_oracle, risk_oracle, label='Empirical Oracle (AURC = 0.0692)', color='#388E3C', linewidth=1.8, linestyle=':')

    # Mark frozen operating threshold tau = 0.064058
    # Realized coverage = 72.3% (47/65), Accepted Risk = 1 - 0.9031 = 0.0969
    ax.scatter([0.723], [0.0969], color='#D32F2F', s=55, zorder=5, label=r'Operating Point $\tau$ ($c=72.3\%$, DSC=0.9031)')
    ax.vlines(0.723, 0, 0.0969, color='#D32F2F', linestyle=':', alpha=0.7)

    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 0.35)
    ax.set_xlabel('Coverage $c$ (Fraction of Retained Cases)', fontsize=9)
    ax.set_ylabel('Selective Risk $\mathcal{R}(c) = \mathbb{E}[1 - \mathrm{DSC} \mid \mathrm{Accepted}]$', fontsize=9)
    ax.set_title('BUSI Locked Test Risk-Coverage Tradeoff ($N=65$ Lesion Scans)', fontsize=10, weight='bold')
    ax.legend(loc='upper left', fontsize=7.8, framealpha=0.95)
    ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig4_busi_risk_coverage.png"), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(OUTPUT_DIR, "fig4_busi_risk_coverage.pdf"), bbox_inches='tight')
    plt.close()
    print("Generated Fig 4: BUSI Risk-Coverage Curve")

# -------------------------------------------------------------
# Figure 5: Cross-Domain Multi-Center Benchmark
# -------------------------------------------------------------
def generate_fig5_cross_domain_performance():
    datasets = ['UDIAT\n(Spain)', 'BUID\n(Academic)', 'BUSUC\n(Multi-center)', 'BUSBRA\n(Brazil)', 'Pooled OOD\n(All 4 Sites)']
    n_counts = [25, 35, 122, 282, 464]
    baseline_dsc = [0.7467, 0.7900, 0.8430, 0.7511, 0.7780]
    accepted_dsc = [0.8747, 0.8818, 0.8982, 0.8536, 0.8718]
    coverage = [56.0, 57.1, 63.9, 44.7, 51.3]

    x = np.arange(len(datasets))
    width = 0.32

    fig, ax1 = plt.subplots(figsize=(7.5, 4.0), dpi=300)

    # Bar chart for DSC
    rects1 = ax1.bar(x - width/2, baseline_dsc, width, label='Full Baseline DSC', color='#90CAF9', edgecolor='#1976D2', alpha=0.9)
    rects2 = ax1.bar(x + width/2, accepted_dsc, width, label=r'Retained Subset DSC ($U_{\mathrm{fg}} \leq \tau$)', color='#1976D2', edgecolor='#0D47A1', alpha=0.95)

    ax1.set_ylabel('Dice Similarity Coefficient (DSC)', fontsize=9, color='#0D47A1')
    ax1.set_ylim(0.55, 1.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, fontsize=8.5)
    ax1.grid(axis='y', linestyle=':', alpha=0.5)

    # Line overlay for realized coverage
    ax2 = ax1.twinx()
    line = ax2.plot(x, coverage, color='#E65100', marker='o', linewidth=2.0, markersize=6, label='Realized Coverage (%)')
    ax2.axhline(80.0, color='#E65100', linestyle='--', linewidth=1.0, alpha=0.7, label='Nominal Val Target (80%)')
    ax2.set_ylabel('Realized Coverage (%)', fontsize=9, color='#E65100')
    ax2.set_ylim(25, 100)

    # Annotate selective gain on bars
    for i in range(len(datasets)):
        gain = accepted_dsc[i] - baseline_dsc[i]
        ax1.text(x[i] + width/2, accepted_dsc[i] + 0.012, f"+{gain:.3f}", ha='center', va='bottom', fontsize=7.5, weight='bold', color='#0D47A1')

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=7.8, framealpha=0.95)

    plt.title('Out-of-Distribution Selective Prediction Performance & Coverage Drift', fontsize=10, weight='bold', pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fig5_cross_domain_performance.png"), bbox_inches='tight', dpi=300)
    plt.savefig(os.path.join(OUTPUT_DIR, "fig5_cross_domain_performance.pdf"), bbox_inches='tight')
    plt.close()
    print("Generated Fig 5: Cross-Domain Performance")

if __name__ == '__main__':
    generate_fig1_methodology()
    generate_fig2_qualitative_cases()
    generate_fig3_val_aggregators()
    generate_fig4_busi_risk_coverage()
    generate_fig5_cross_domain_performance()
    print("All 5 figures generated successfully in", OUTPUT_DIR)
