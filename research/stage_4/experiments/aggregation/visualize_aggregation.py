#!/usr/bin/env python3
"""
Research Pipeline - Stage 05: Aggregation Visualization Suite

Generates four figure sets from the per-image uncertainty CSV produced by
compute_uncertainty_csv.py:

    Figure A — QC panels (6-subplot per sample):
        (1) Ultrasound image
        (2) Predicted mask overlay
        (3) Raw entropy heatmap (fixed scale 0 → ln 2)
        (4) Foreground region
        (5) Morphological boundary band
        (6) Threshold-selected pixels

    Figure B — Summary statistics table (all 7 metrics)

    Figure C — Spearman correlation heatmap (7 metrics × {DSC, NSD})

    Figure D — Scatter plots (7 × 2 grid, with regression line, ρ, p, N)

IMPORTANT — EXPLORATORY DIAGNOSTICS ONLY:
    All outputs are labelled accordingly. These figures must not be used
    for method selection, threshold calibration, or scientific claims on
    the test set.

Usage:
    python visualize_aggregation.py \\
        --dataset BUSI --split test --seed 666 \\
        [--output-dir /path/to/outputs] [--n-qc-samples 6]
"""

import os
import sys
import argparse
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "research", "stage_4", "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "shared"))

from uncertainty.aggregation import compute_boundary_band_mask  # noqa: E402

LN2 = float(np.log(2))
METRICS = ["U_mean", "U_positive", "U_fg", "U_boundary",
           "U_threshold", "U_std_fg", "U_ratio"]
METRIC_LABELS = {
    "U_mean":      r"$U_{\mathrm{mean}}$",
    "U_positive":  r"$U_{\mathrm{positive}}$",
    "U_fg":        r"$U_{\mathrm{fg}}$",
    "U_boundary":  r"$U_{\mathrm{boundary}}$",
    "U_threshold": r"$U_{\mathrm{threshold}}$",
    "U_std_fg":    r"$U_{\mathrm{std\_fg}}$",
    "U_ratio":     r"$U_{\mathrm{ratio}}$",
}
EXPLORATORY_NOTE = (
    "EXPLORATORY DIAGNOSTIC — BUSI Test Set. "
    "Not used for method selection, threshold calibration, or scientific claims."
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Uncertainty aggregation visualization suite.")
    p.add_argument("--dataset", default="BUSI")
    p.add_argument("--split", default="test")
    p.add_argument("--seed", type=int, default=666)
    p.add_argument("--prompt-design", default="original")
    p.add_argument("--output-dir", default="",
                   help="Root output directory (or MEDCLIPSEG_OUTPUT env var).")
    p.add_argument("--n-qc-samples", type=int, default=6,
                   help="Number of QC panel samples to generate.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def resolve_root(args: argparse.Namespace) -> str:
    root = args.output_dir or os.environ.get(
        "MEDCLIPSEG_OUTPUT",
        os.path.join(REPO_ROOT, "..", "data", "outputs"),
    )
    return os.path.abspath(root)


def raw_data_root() -> str:
    return os.path.abspath(os.path.join(REPO_ROOT, "..", "data", "raw"))


def csv_path(output_root: str, dataset: str, split: str, seed: int) -> str:
    return os.path.join(
        output_root, dataset, "aggregation_results",
        f"seed{seed}", f"uncertainty_scores_{dataset}_{split}_seed{seed}.csv"
    )


def unc_dir(output_root: str, dataset: str, seed: int, prompt: str) -> str:
    tag = f"MedCLIPSeg_unimedclip_ViT-B-16_Prompt-{prompt}"
    return os.path.join(output_root, dataset, "unc_results", f"seed{seed}", tag)


def mask_dir(output_root: str, dataset: str, seed: int, prompt: str) -> str:
    tag = f"MedCLIPSeg_unimedclip_ViT-B-16_Prompt-{prompt}"
    return os.path.join(output_root, dataset, "seg_results", f"seed{seed}", tag)


def us_img_dir(dataset: str, split: str) -> str:
    split_folder = "Test_Folder" if split == "test" else "Val_Folder"
    return os.path.join(raw_data_root(), dataset, split_folder, "img")


def fig_dir(output_root: str, dataset: str, seed: int) -> str:
    d = os.path.join(output_root, dataset, "aggregation_results",
                     f"seed{seed}", "figures")
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Figure A — QC Panel
# ---------------------------------------------------------------------------

def _add_exploratory_stamp(fig: plt.Figure) -> None:
    fig.text(
        0.5, 0.01, EXPLORATORY_NOTE,
        ha="center", va="bottom", fontsize=7,
        color="firebrick", style="italic",
        transform=fig.transFigure,
    )


def _touches_border(fg_mask: np.ndarray) -> bool:
    """Return True when any foreground pixel lies on the image boundary."""
    return bool(
        fg_mask[0].any() or fg_mask[-1].any()
        or fg_mask[:, 0].any() or fg_mask[:, -1].any()
    )


def select_qc_samples(
    df: pd.DataFrame,
    pred_mask_dir: Optional[str] = None,
    n: int = 6,
) -> pd.DataFrame:
    """
    Choose a diverse but representative set of QC samples:
        • best DSC image
        • worst DSC image
        • at least one confirmed border-lesion case (if present)
        • remaining slots filled by DSC-spread sampling
    """
    df_sorted = df.sort_values("DSC", ascending=False).reset_index(drop=True)
    chosen_ids = []

    # Always include best and worst
    chosen_ids.append(df_sorted.iloc[0]["image_id"])
    chosen_ids.append(df_sorted.iloc[-1]["image_id"])

    # Find verified border-lesion case
    if pred_mask_dir and os.path.exists(pred_mask_dir):
        for img_id in df["image_id"]:
            mask_path = os.path.join(pred_mask_dir, img_id)
            if os.path.exists(mask_path):
                m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if m is not None and _touches_border(m > 0):
                    if img_id not in chosen_ids:
                        chosen_ids.append(img_id)
                        break

    # If still not found, fallback to proxy
    if len(chosen_ids) < 3 and "count_fg" in df.columns and "count_boundary" in df.columns:
        df2 = df[df["count_fg"] > 0].copy()
        df2["border_ratio"] = df2["count_boundary"] / df2["count_fg"]
        border_cands = df2.sort_values("border_ratio", ascending=False)
        for _, row in border_cands.iterrows():
            if row["image_id"] not in chosen_ids:
                chosen_ids.append(row["image_id"])
                break

    # Fill remaining slots with evenly spaced DSC ranks
    n_remaining = n - len(chosen_ids)
    if n_remaining > 0:
        step = max(1, len(df_sorted) // (n_remaining + 1))
        for i in range(1, n_remaining + 1):
            img = df_sorted.iloc[min(i * step, len(df_sorted) - 1)]["image_id"]
            if img not in chosen_ids:
                chosen_ids.append(img)

    return df[df["image_id"].isin(chosen_ids)].reset_index(drop=True)


def make_qc_panel(
    row: pd.Series,
    unc_map_dir: str,
    pred_mask_dir: str,
    us_dir: str,
    out_dir: str,
) -> str:
    """Generate a 6-subplot QC panel for one image. Returns saved path."""
    img_name = row["image_id"]
    base = os.path.splitext(img_name)[0]

    # Load data
    npy_path = os.path.join(unc_map_dir, f"{base}.npy")
    mask_path = os.path.join(pred_mask_dir, img_name)
    us_path = os.path.join(us_dir, img_name)

    unc_map = np.load(npy_path).astype(np.float32)
    pred_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    fg_mask = (pred_mask > 0)

    # Ultrasound — may be a different resolution; display as-is
    us_img_bgr = cv2.imread(us_path)
    if us_img_bgr is not None:
        us_rgb = cv2.cvtColor(us_img_bgr, cv2.COLOR_BGR2RGB)
        # Resize US to 224×224 for spatial alignment in the panel
        us_rgb_224 = cv2.resize(us_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    else:
        us_rgb_224 = np.zeros((224, 224, 3), dtype=np.uint8)

    # Morphological boundary band
    bw = int(row.get("boundary_width", 2))
    b_mask = compute_boundary_band_mask(pred_mask, boundary_width=bw)

    # Threshold-selected mask
    t = float(row.get("threshold_t", 0.1))
    thresh_mask = (unc_map > t)

    touches = _touches_border(fg_mask)
    dsc_str = f"DSC={row['DSC']:.3f}" if not np.isnan(row["DSC"]) else "DSC=N/A"
    nsd_str = f"NSD={row['NSD']:.3f}" if not np.isnan(row["NSD"]) else "NSD=N/A"
    border_tag = " [BORDER LESION]" if touches else ""
    suptitle = f"{img_name}  —  {dsc_str}  |  {nsd_str}{border_tag}"

    fig, axes = plt.subplots(1, 6, figsize=(22, 4))
    fig.suptitle(suptitle, fontsize=10, fontweight="bold", y=1.01)

    # (1) Ultrasound
    axes[0].imshow(us_rgb_224)
    axes[0].set_title("Ultrasound (224×224)", fontsize=8)

    # (2) Predicted mask overlay on US
    overlay = us_rgb_224.copy()
    overlay[fg_mask] = (overlay[fg_mask] * 0.4 + np.array([255, 80, 80]) * 0.6).clip(0, 255).astype(np.uint8)
    axes[1].imshow(overlay)
    axes[1].set_title("Pred mask overlay", fontsize=8)

    # (3) Raw entropy heatmap — fixed scale [0, ln 2]
    im = axes[2].imshow(unc_map, cmap="hot", vmin=0.0, vmax=LN2)
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    axes[2].set_title(f"Entropy map [0, ln2]", fontsize=8)

    # (4) Foreground region
    fg_disp = np.zeros((224, 224, 3), dtype=np.uint8)
    fg_disp[fg_mask] = [70, 130, 180]
    fg_disp[~fg_mask] = [30, 30, 30]
    n_fg = int(np.sum(fg_mask))
    axes[3].imshow(fg_disp)
    axes[3].set_title(f"Foreground  (n={n_fg})", fontsize=8)

    # (5) Boundary band
    b_disp = np.zeros((224, 224, 3), dtype=np.uint8)
    b_disp[fg_mask] = [60, 60, 60]
    b_disp[b_mask] = [255, 165, 0]
    n_b = int(np.sum(b_mask))
    axes[4].imshow(b_disp)
    axes[4].set_title(f"Boundary band  (n={n_b}, w={bw})", fontsize=8)

    # (6) Threshold-selected pixels
    t_disp = np.zeros((224, 224, 3), dtype=np.uint8)
    t_disp[thresh_mask] = [50, 200, 50]
    t_disp[~thresh_mask] = [25, 25, 25]
    n_t = int(np.sum(thresh_mask))
    axes[5].imshow(t_disp)
    axes[5].set_title(f"U > {t:.2f}  (n={n_t})", fontsize=8)

    for ax in axes:
        ax.axis("off")

    _add_exploratory_stamp(fig)
    plt.tight_layout()

    out_subdir = os.path.join(out_dir, "qc_panels")
    os.makedirs(out_subdir, exist_ok=True)
    out_path = os.path.join(out_subdir, f"qc_{base}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Figure B — Summary statistics table
# ---------------------------------------------------------------------------

def make_summary_table(df: pd.DataFrame, out_dir: str) -> str:
    """Save a summary statistics table as a PNG."""
    rows = []
    for m in METRICS:
        col = df[m].dropna()
        valid_flag = m.replace("U_", "valid_").replace("_fg", "_fg").replace(
            "U_mean", "valid_mean").replace("U_positive", "valid_pos").replace(
            "U_fg", "valid_fg").replace("U_boundary", "valid_boundary").replace(
            "U_threshold", "valid_threshold").replace("U_std_fg", "valid_std_fg").replace(
            "U_ratio", "valid_ratio")
        # Remap
        flag_map = {
            "U_mean": "valid_mean", "U_positive": "valid_pos",
            "U_fg": "valid_fg", "U_boundary": "valid_boundary",
            "U_threshold": "valid_threshold", "U_std_fg": "valid_std_fg",
            "U_ratio": "valid_ratio",
        }
        flag_col = flag_map[m]
        n_valid = int(df[flag_col].sum()) if flag_col in df.columns else len(col)
        rows.append({
            "Metric": METRIC_LABELS[m],
            "Valid (N)": n_valid,
            "NaN": int(df[m].isna().sum()),
            "Mean": f"{col.mean():.5f}" if len(col) else "—",
            "Std":  f"{col.std():.5f}"  if len(col) else "—",
            "Median": f"{col.median():.5f}" if len(col) else "—",
            "Min":  f"{col.min():.5f}"  if len(col) else "—",
            "Max":  f"{col.max():.5f}"  if len(col) else "—",
        })

    summary_df = pd.DataFrame(rows)
    print("\n" + "=" * 80)
    print("  SUMMARY STATISTICS — All 7 Uncertainty Metrics")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80 + "\n")

    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.axis("off")
    tbl = ax.table(
        cellText=summary_df.values,
        colLabels=summary_df.columns,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)

    # Header style
    for j in range(len(summary_df.columns)):
        tbl[(0, j)].set_facecolor("#2c3e50")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    # Alternating row colours
    for i in range(1, len(rows) + 1):
        fc = "#ecf0f1" if i % 2 == 0 else "white"
        for j in range(len(summary_df.columns)):
            tbl[(i, j)].set_facecolor(fc)

    fig.suptitle(
        "EXPLORATORY DIAGNOSTIC — BUSI Test Set | 7 Uncertainty Aggregation Metrics",
        fontsize=9, color="firebrick", style="italic",
    )
    plt.tight_layout()
    out_path = os.path.join(out_dir, "summary_table.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Figure C — Spearman correlation heatmap
# ---------------------------------------------------------------------------

def make_spearman_heatmap(df: pd.DataFrame, out_dir: str) -> str:
    """7×2 Spearman ρ heatmap (metrics × {DSC, NSD})."""
    perf_cols = ["DSC", "NSD"]
    rho_data = np.full((len(METRICS), len(perf_cols)), np.nan)
    pval_data = np.full_like(rho_data, np.nan)
    annot = np.empty_like(rho_data, dtype=object)

    for i, m in enumerate(METRICS):
        for j, p in enumerate(perf_cols):
            valid = df[[m, p]].dropna()
            if len(valid) >= 5:
                rho, pval = stats.spearmanr(valid[m], valid[p])
                rho_data[i, j] = rho
                pval_data[i, j] = pval
                stars = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
                annot[i, j] = f"ρ={rho:+.3f}{stars}\np={pval:.2e}\nN={len(valid)}"
            else:
                annot[i, j] = "N/A"

    fig, ax = plt.subplots(figsize=(5, 8))
    im = ax.imshow(rho_data, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Spearman ρ", fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(perf_cols)))
    ax.set_xticklabels(perf_cols, fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(METRICS)))
    ax.set_yticklabels([METRIC_LABELS[m] for m in METRICS], fontsize=10)

    for i in range(len(METRICS)):
        for j in range(len(perf_cols)):
            ax.text(j, i, annot[i, j], ha="center", va="center",
                    fontsize=7.5, color="black")

    ax.set_title("Spearman Correlation: Uncertainty Metrics vs Segmentation Quality\n"
                 + EXPLORATORY_NOTE, fontsize=8, color="firebrick", style="italic",
                 pad=12, wrap=True)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "spearman_heatmap.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Figure D — Scatter grid
# ---------------------------------------------------------------------------

def make_scatter_grid(df: pd.DataFrame, out_dir: str) -> str:
    """2-row (DSC top, NSD bottom) × 7-column scatter grid with regression."""
    perf_cols = ["DSC", "NSD"]
    nrows, ncols = len(perf_cols), len(METRICS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(22, 8), sharey=False)

    colors = ["#2980b9", "#8e44ad", "#27ae60", "#e67e22",
              "#c0392b", "#16a085", "#d35400"]

    for j, m in enumerate(METRICS):
        for i, p in enumerate(perf_cols):
            ax = axes[i, j]
            valid = df[[m, p]].dropna()
            x = valid[p].values
            y = valid[m].values
            n = len(x)

            ax.scatter(x, y, alpha=0.65, s=28, color=colors[j],
                       edgecolors="k", linewidths=0.4)

            if n >= 5:
                rho, pval = stats.spearmanr(y, x)
                # OLS regression line
                slope, intercept, *_ = stats.linregress(x, y)
                xfit = np.linspace(x.min(), x.max(), 100)
                ax.plot(xfit, slope * xfit + intercept, color="#c0392b",
                        linewidth=1.5, alpha=0.9)
                stars = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
                ax.set_title(
                    f"ρ={rho:+.3f}{stars}  p={pval:.1e}  N={n}",
                    fontsize=7.5,
                )
            else:
                ax.set_title("insufficient data", fontsize=7.5)

            if i == nrows - 1:
                ax.set_xlabel(p, fontsize=9, fontweight="bold")
            if j == 0:
                ax.set_ylabel(METRIC_LABELS[m], fontsize=9)
            else:
                ax.set_ylabel(METRIC_LABELS[m], fontsize=8)
            ax.tick_params(labelsize=7)

    fig.suptitle(
        "Scatter: Uncertainty Metrics vs Segmentation Quality (DSC / NSD)\n" + EXPLORATORY_NOTE,
        fontsize=9, color="firebrick", style="italic",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = os.path.join(out_dir, "scatter_grid.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = get_args()
    output_root = resolve_root(args)

    print("=" * 70)
    print("  Visualization Suite (Stage 05)")
    print("  EXPLORATORY DIAGNOSTICS ONLY")
    print("=" * 70)

    # Load CSV
    in_csv = csv_path(output_root, args.dataset, args.split, args.seed)
    if not os.path.exists(in_csv):
        print(f"[ERROR] CSV not found — run Stage 04 first:\n  {in_csv}",
              file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(in_csv)
    print(f"  Loaded CSV: {in_csv}  ({len(df)} rows)")

    out_figs = fig_dir(output_root, args.dataset, args.seed)
    unc_d = unc_dir(output_root, args.dataset, args.seed, args.prompt_design)
    msk_d = mask_dir(output_root, args.dataset, args.seed, args.prompt_design)
    us_d = us_img_dir(args.dataset, args.split)

    print(f"  Figures output: {out_figs}")
    print(f"  Uncertainty maps: {unc_d}")
    print(f"  Predicted masks : {msk_d}")
    print(f"  US images       : {us_d}")
    print("-" * 70)

    # --- Figure A: QC panels ---
    print("\n[Figure A] Generating QC panels...")
    qc_subset = select_qc_samples(df, pred_mask_dir=msk_d, n=args.n_qc_samples)
    qc_paths = []
    for _, row in qc_subset.iterrows():
        try:
            p = make_qc_panel(row, unc_d, msk_d, us_d, out_figs)
            print(f"  Saved: {p}")
            qc_paths.append(p)
        except Exception as exc:
            print(f"  [WARN] QC panel failed for {row['image_id']}: {exc}")

    # --- Figure B: Summary table ---
    print("\n[Figure B] Generating summary statistics table...")
    b_path = make_summary_table(df, out_figs)
    print(f"  Saved: {b_path}")

    # --- Figure C: Spearman heatmap ---
    print("\n[Figure C] Generating Spearman correlation heatmap...")
    c_path = make_spearman_heatmap(df, out_figs)
    print(f"  Saved: {c_path}")

    # --- Figure D: Scatter grid ---
    print("\n[Figure D] Generating scatter plot grid...")
    d_path = make_scatter_grid(df, out_figs)
    print(f"  Saved: {d_path}")

    print("\n" + "=" * 70)
    print("  Stage 05 complete.")
    print(f"  All figures written under:\n    {out_figs}")
    print("=" * 70)


if __name__ == "__main__":
    main()

