#!/usr/bin/env python3
"""
Research Pipeline - Stage 06: BUSI Validation Inference & Uncertainty Extraction

Executes frozen MedCLIPSeg probabilistic inference on the BUSI validation set
(78 images) without fine-tuning, training, or modifying model weights.

Outputs generated under separate validation directory:
    <output-root>/BUSI/validation/seed<seed>/
        ├── seg_results/
        │   ├── MedCLIPSeg_unimedclip_ViT-B-16_Prompt-<prompt>/  (*.png masks)
        │   └── val_Prompt-<prompt>.csv                          (Name, DSC, NSD)
        └── unc_results/
            └── MedCLIPSeg_unimedclip_ViT-B-16_Prompt-<prompt>/  (*.npy, *.png)

Features:
- Preserves raw float32 predictive entropy maps (.npy) in natural units [0, ln 2].
- Saves binary predicted segmentation masks (uint8, {0, 255}).
- Computes official baseline DSC and NSD evaluation metrics using SurfaceDice.
- Strictly protects BUSI test outputs (zero overwrite).
- Performs comprehensive integrity and bounds verification.
"""

import os
import sys
import random
import argparse
from collections import OrderedDict
import numpy as np
import pandas as pd
import cv2
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SHARED_DIR = os.path.join(REPO_ROOT, "shared", "medclipseg")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SHARED_DIR)

from utils.main_utils import load_cfg_from_cfg_file, read_text, normalize  # noqa: E402
from trainers import build_medclipseg_unimedclip  # noqa: E402
from datasets.dataloader import DatasetSegmentation, ValGenerator  # noqa: E402
from utils.SurfaceDice import (  # noqa: E402
    compute_surface_distances,
    compute_surface_dice_at_tolerance,
    compute_dice_coefficient,
)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run frozen MedCLIPSeg validation inference and extract raw uncertainty maps."
    )
    parser.add_argument("--dataset", type=str, default="BUSI", help="Dataset name.")
    parser.add_argument("--seed", type=int, default=666, help="Evaluation seed.")
    parser.add_argument("--prompt-design", type=str, default="original", help="Prompt variant tag.")
    parser.add_argument("--batch-size", type=int, default=8, help="DataLoader batch size.")
    parser.add_argument("--num-samples", type=int, default=30, help="Number of Monte Carlo samples.")
    parser.add_argument("--device", type=str, default="", help="Device: mps, cuda, or cpu (empty=auto).")
    parser.add_argument("--config-file", type=str, default="", help="Path to config YAML.")
    parser.add_argument("--output-dir", type=str, default="", help="Base outputs directory.")
    parser.add_argument("--data-root", type=str, default="", help="Raw data root directory.")
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    if requested in ["mps", "cuda", "cpu"]:
        return requested
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main() -> None:
    args = get_args()
    set_random_seed(args.seed)

    # 1. Paths resolution
    data_root = args.data_root or os.environ.get(
        "MEDCLIPSEG_DATA", os.path.abspath(os.path.join(REPO_ROOT, "..", "data", "raw"))
    )
    base_out = args.output_dir or os.environ.get(
        "MEDCLIPSEG_OUTPUT", os.path.abspath(os.path.join(REPO_ROOT, "..", "data", "outputs"))
    )
    cfg_file = args.config_file or os.path.join(REPO_ROOT, "shared", "configs", f"{args.dataset}.yaml")

    print("=" * 75)
    print("  ELEC 6661 — BUSI Validation Inference Pipeline (Frozen Baseline)")
    print("=" * 75)
    print(f"  Dataset        : {args.dataset}")
    print(f"  Split          : Validation")
    print(f"  Seed           : {args.seed}")
    print(f"  Prompt Design  : {args.prompt_design}")
    print(f"  MC Samples     : {args.num_samples}")
    print(f"  Data Root      : {data_root}")
    print(f"  Outputs Root   : {base_out}")

    # Output directories for validation (strictly separated from test)
    val_out_root = os.path.join(base_out, args.dataset, "validation", f"seed{args.seed}")
    tag = f"MedCLIPSeg_unimedclip_ViT-B-16_Prompt-{args.prompt_design}"
    val_seg_dir = os.path.join(val_out_root, "seg_results", tag)
    val_unc_dir = os.path.join(val_out_root, "unc_results", tag)
    val_csv_path = os.path.join(val_out_root, "seg_results", f"val_Prompt-{args.prompt_design}.csv")

    os.makedirs(val_seg_dir, exist_ok=True)
    os.makedirs(val_unc_dir, exist_ok=True)

    print(f"  Val Seg Output : {val_seg_dir}")
    print(f"  Val Unc Output : {val_unc_dir}")
    print(f"  Val Eval CSV   : {val_csv_path}")
    print("-" * 75)

    # 2. Checkpoint path resolution
    results_name = "MedCLIPSeg_unimedclip_ViT-B-16"
    ckpt_path = os.path.join(
        base_out, args.dataset, "trained_models", f"seed{args.seed}", f"{results_name}_latest.pth"
    )
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Required checkpoint not found: {ckpt_path}")
    print(f"  Pretrained Ckpt: {ckpt_path}")

    # 3. Validation and prompts data verification
    val_img_dir = os.path.join(data_root, args.dataset, "Val_Folder", "img")
    val_lbl_dir = os.path.join(data_root, args.dataset, "Val_Folder", "label")
    prompts_dir = os.path.join(data_root, args.dataset, "Prompts_Folder")

    prompt_candidates = [
        os.path.join(prompts_dir, f"Val_text_{args.prompt_design}.xlsx"),
        os.path.join(prompts_dir, "Val_text.xlsx"),
    ]
    val_prompt_file = None
    for cand in prompt_candidates:
        if os.path.exists(cand):
            val_prompt_file = cand
            break

    if not val_prompt_file:
        raise FileNotFoundError(f"No validation prompt file found in {prompts_dir}")
    print(f"  Val Prompts    : {val_prompt_file}")

    val_records = read_text(val_prompt_file)
    n_samples = len(val_records)
    print(f"  Total Val Imgs : {n_samples}")
    if n_samples != 78:
        print(f"  [WARNING] Expected 78 validation images, found {n_samples}")

    # 4. Device and Model setup
    device_str = resolve_device(args.device)
    device = torch.device(device_str)
    print(f"  Execution Dev  : {device_str}")

    cfg = load_cfg_from_cfg_file(cfg_file)
    cfg.MODEL.DEVICE = device_str
    cfg.TEST.NUM_SAMPLES = args.num_samples

    print("\n[Stage 1/3] Building frozen MedCLIPSeg model...")
    model = build_medclipseg_unimedclip(cfg)

    # Disable all gradients explicitly to guarantee no fine-tuning
    for param in model.parameters():
        param.requires_grad = False

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    print("  Model weights successfully loaded into memory (eval mode).")

    # 5. Dataset & DataLoader setup
    val_tf = ValGenerator(output_size=[cfg.DATASET.SIZE, cfg.DATASET.SIZE])
    val_dataset = DatasetSegmentation(
        dataset_path=os.path.join(data_root, args.dataset, "Val_Folder"),
        task_name=args.dataset,
        row_text=val_records,
        joint_transform=val_tf,
        image_size=cfg.DATASET.SIZE,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # 6. Inference Loop
    print("\n[Stage 2/3] Running Monte Carlo probabilistic inference...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Inference"):
            imgs = batch["image"].to(device)
            texts = batch["text_prompt"]
            mask_names = batch["mask_name"]

            # Forward pass: returns seg_samples [N, B, 1, H, W]
            seg_samples = model(image=imgs, text=texts, num_samples=args.num_samples)
            seg_samples = torch.sigmoid(seg_samples)

            # Predictive mean: [B, 1, H, W]
            seg_logits = seg_samples.mean(dim=0)

            # Predictive entropy in nats: [B, 1, H, W]
            seg_unc = -(
                seg_logits * torch.log(seg_logits + 1e-8)
                + (1 - seg_logits) * torch.log(1 - seg_logits + 1e-8)
            )

            # Binary mask thresholded at 0.5
            mask_preds = (seg_logits > 0.5)

            for i in range(len(mask_names)):
                m_name = mask_names[i]
                stem = os.path.splitext(m_name)[0]

                # (a) Save binary segmentation mask: uint8 {0, 255}
                pred_mask = mask_preds[i].cpu().numpy().astype(np.uint8)
                binary_pred = (pred_mask > 0).astype(np.uint8) * 255
                cv2.imwrite(os.path.join(val_seg_dir, m_name), binary_pred)

                # (b) Save RAW float32 uncertainty map (224, 224)
                u_map = seg_unc[i].cpu().numpy().astype(np.float32)
                np.save(os.path.join(val_unc_dir, f"{stem}.npy"), u_map)

                # (c) Save colormapped normalized PNG visualization
                u_norm = normalize(u_map)
                colormap = plt.get_cmap("nipy_spectral")
                u_color = (colormap(u_norm)[:, :, :3] * 255).astype(np.uint8)
                u_bgr = cv2.cvtColor(u_color, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(val_unc_dir, m_name), u_bgr)

    # 7. Compute Official Evaluation Metrics (DSC & NSD)
    print("\n[Stage 3/3] Computing official validation DSC and NSD metrics...")
    seg_metrics = OrderedDict(Name=[], DSC=[], NSD=[])

    for row in val_records:
        gt_name = row["Ground Truth"]
        seg_name = row["Image"]
        seg_metrics["Name"].append(seg_name)

        gt_mask = cv2.imread(os.path.join(val_lbl_dir, gt_name), cv2.IMREAD_GRAYSCALE)
        pred_mask_file = os.path.join(val_seg_dir, seg_name)
        seg_mask = cv2.imread(pred_mask_file, cv2.IMREAD_GRAYSCALE)

        if gt_mask is None or seg_mask is None:
            raise FileNotFoundError(f"Missing mask for {seg_name}")

        # Resize predicted mask to native ground-truth resolution
        seg_mask = cv2.resize(
            seg_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST
        )
        gt_data = np.uint8(gt_mask >= 127)
        seg_data = np.uint8(seg_mask >= 127)

        # Baseline rule: empty prediction and empty ground truth -> DSC=1.0, NSD=1.0
        if np.max(gt_data) == 0 and np.max(seg_data) == 0:
            dsc_val = 1.0
            nsd_val = 1.0
        else:
            gt_labels = np.unique(gt_data)
            seg_labels = np.unique(seg_data)
            gt_labels = gt_labels[gt_labels != 0]
            seg_labels = seg_labels[seg_labels != 0]
            labels = np.union1d(gt_labels, seg_labels)

            dsc_arr = []
            nsd_arr = []
            for lab in labels:
                if np.sum(gt_data == lab) == 0 and np.sum(seg_data == lab) == 0:
                    d_i, n_i = 1.0, 1.0
                elif np.sum(gt_data == lab) == 0 and np.sum(seg_data == lab) > 0:
                    d_i, n_i = 0.0, 0.0
                else:
                    i_gt, i_seg = (gt_data == lab), (seg_data == lab)
                    d_i = compute_dice_coefficient(i_gt, i_seg)
                    sur_dist = compute_surface_distances(
                        i_gt[..., None], i_seg[..., None], [1, 1, 1]
                    )
                    n_i = compute_surface_dice_at_tolerance(sur_dist, 2)
                dsc_arr.append(d_i)
                nsd_arr.append(n_i)

            dsc_val = float(np.mean(dsc_arr)) if dsc_arr else 0.0
            nsd_val = float(np.mean(nsd_arr)) if nsd_arr else 0.0

        seg_metrics["DSC"].append(round(dsc_val, 4))
        seg_metrics["NSD"].append(round(nsd_val, 4))

    val_df = pd.DataFrame(seg_metrics)
    val_df.to_csv(val_csv_path, index=False)
    print(f"  Validation CSV saved to: {val_csv_path}")
    print(f"  Mean DSC : {val_df['DSC'].mean():.4f}")
    print(f"  Mean NSD : {val_df['NSD'].mean():.4f}")

    # 8. Comprehensive Post-Inference Integrity Verification
    print("\n" + "=" * 75)
    print("  INTEGRITY & BOUNDS VERIFICATION")
    print("=" * 75)

    saved_masks = [f for f in os.listdir(val_seg_dir) if f.endswith(".png")]
    saved_npys = [f for f in os.listdir(val_unc_dir) if f.endswith(".npy")]
    saved_pngs = [f for f in os.listdir(val_unc_dir) if f.endswith(".png")]

    assert len(saved_masks) == n_samples, f"Mask count mismatch: {len(saved_masks)} != {n_samples}"
    assert len(saved_npys) == n_samples, f"NPY count mismatch: {len(saved_npys)} != {n_samples}"
    assert len(saved_pngs) == n_samples, f"PNG unc mismatch: {len(saved_pngs)} != {n_samples}"
    assert len(val_df) == n_samples, f"CSV rows mismatch: {len(val_df)} != {n_samples}"
    print(f"  [PASS] File count verified: exactly {n_samples} samples across all directories.")

    ln2 = float(np.log(2.0))
    n_empty_pred = 0
    n_empty_gt = 0
    max_entropy_seen = 0.0
    min_entropy_seen = 1.0

    for idx, row in val_df.iterrows():
        img_name = row["Name"]
        stem = os.path.splitext(img_name)[0]

        # Check mask
        m = cv2.imread(os.path.join(val_seg_dir, img_name), cv2.IMREAD_GRAYSCALE)
        assert m.shape == (224, 224), f"Bad mask shape: {m.shape}"
        assert set(np.unique(m)).issubset({0, 255}), f"Mask non-binary: {np.unique(m)}"
        if (m == 0).all():
            n_empty_pred += 1

        # Check GT
        gt_raw = cv2.imread(os.path.join(val_lbl_dir, img_name), cv2.IMREAD_GRAYSCALE)
        if (gt_raw < 127).all():
            n_empty_gt += 1

        # Check npy
        u = np.load(os.path.join(val_unc_dir, f"{stem}.npy"))
        assert u.shape == (224, 224), f"Bad npy shape: {u.shape}"
        assert u.dtype == np.float32, f"Bad npy dtype: {u.dtype}"
        assert np.all(np.isfinite(u)), "Non-finite values found in uncertainty map!"
        assert np.all(u >= -1e-6), f"Negative entropy found: {u.min()}"
        assert np.all(u <= ln2 + 1e-4), f"Entropy exceeds ln(2): {u.max()} > {ln2}"

        max_entropy_seen = max(max_entropy_seen, float(u.max()))
        min_entropy_seen = min(min_entropy_seen, float(u.min()))

    print(f"  [PASS] All masks are binary {0, 255} with shape (224, 224).")
    print(f"  [PASS] All .npy maps are float32 with shape (224, 224).")
    print(f"  [PASS] Entropy values strictly bounded in [0, ln(2)]: range [{min_entropy_seen:.5f}, {max_entropy_seen:.5f}] (max ln 2 = {ln2:.5f}).")
    print(f"  [INFO] Empty predicted masks count : {n_empty_pred} / {n_samples}")
    print(f"  [INFO] Empty ground-truth masks    : {n_empty_gt} / {n_samples}")
    print("=" * 75)
    print("  Validation inference and verification completed successfully!")
    print("=" * 75)


if __name__ == "__main__":
    main()
