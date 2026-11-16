from pathlib import Path
import sys
import csv
import numpy as np
import nibabel as nib
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
import evaluate_stage2_anatomy_gate as ev

OUT = ROOT / "evaluation" / "stage2_anatomy_gate" / "representative_exports"
OUT.mkdir(parents=True, exist_ok=True)

SELECTED = [
    "LiTS_Batch1_LiTS_Batch1_0010_0000",
    "LiTS_Batch2_LiTS_Batch2_0057_0000",
    "KiTS19_KiTS19_0046_0000",
    "KiTS19_KiTS19_0151_0000",
]

def save_nii(data, reference, path, dtype):
    header = reference.header.copy()
    header.set_data_dtype(dtype)
    nib.save(nib.Nifti1Image(data.astype(dtype), reference.affine, header), str(path))

print("Device:", ev.DEVICE)
print("Loading general Stage-2 checkpoint:", ev.STAGE2_CHECKPOINT)

model = ev.build_stage2_model()
ev.load_checkpoint(model, ev.STAGE2_CHECKPOINT)
model = model.to(ev.DEVICE).eval()

manifest = ev.load_manifest()
entries = {ev.case_key(Path(x["image"])): x for x in manifest}

rows = []

for index, key in enumerate(SELECTED, 1):
    print(f"\n[{index}/{len(SELECTED)}] {key}")

    if key not in entries:
        raise RuntimeError(f"Case not found in validation manifest: {key}")

    entry = entries[key]
    image_path = Path(entry["image"])
    label_path = Path(entry["label"])
    gate_path = ev.STAGE1_MASK_DIR / f"{key}_liver_kidney_gate.nii.gz"

    if not gate_path.exists():
        raise FileNotFoundError(f"Cached anatomy gate missing: {gate_path}")

    image_nii = nib.as_closest_canonical(nib.load(str(image_path)))
    label_nii = nib.as_closest_canonical(nib.load(str(label_path)))
    gate_nii = nib.as_closest_canonical(nib.load(str(gate_path)))

    image = image_nii.get_fdata(dtype=np.float32)
    gt = np.rint(label_nii.get_fdata()).astype(np.uint8)
    gate = np.rint(gate_nii.get_fdata()).astype(np.uint8) > 0

    if image.shape != gt.shape or image.shape != gate.shape:
        raise RuntimeError(
            f"Shape mismatch for {key}: image={image.shape}, "
            f"GT={gt.shape}, gate={gate.shape}"
        )

    spacing = tuple(float(x) for x in image_nii.header.get_zooms()[:3])
    expanded_gate = ev.dilate_physical(gate, spacing, 15.0)

    print("Running Stage-2 inference...")
    p2 = ev.predict_stage2_probability(model, image)[2]

    baseline = p2 >= 0.65
    gated = baseline & expanded_gate
    gt_mask = gt == 2

    base_metrics = ev.binary_metrics(baseline, gt_mask)
    gated_metrics = ev.binary_metrics(gated, gt_mask)

    case_dir = OUT / key
    case_dir.mkdir(parents=True, exist_ok=True)

    save_nii(baseline, image_nii, case_dir / "baseline_prediction.nii.gz", np.uint8)
    save_nii(gated, image_nii, case_dir / "gated_prediction.nii.gz", np.uint8)
    save_nii(gt_mask, image_nii, case_dir / "ground_truth.nii.gz", np.uint8)
    save_nii(p2, image_nii, case_dir / "pathology_probability.nii.gz", np.float32)
    save_nii(expanded_gate, image_nii, case_dir / "expanded_anatomy_gate.nii.gz", np.uint8)

    # Select axial slice containing the largest amount of GT pathology.
    slice_counts = gt_mask.sum(axis=(0, 1))
    z = int(np.argmax(slice_counts))

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    panels = [
        ("CT", None),
        ("Ground Truth", gt_mask),
        ("Baseline Prediction", baseline),
        ("Anatomy-Gated Prediction", gated),
    ]

    for ax, (title, mask) in zip(axes, panels):
        ax.imshow(np.rot90(image[:, :, z]), cmap="gray")
        if mask is not None:
            ax.imshow(
                np.ma.masked_where(~np.rot90(mask[:, :, z]), np.rot90(mask[:, :, z])),
                cmap="autumn",
                alpha=0.65,
                interpolation="nearest",
            )
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle(
        f"{key} | Slice {z} | "
        f"Baseline Dice {base_metrics['dice']:.3f} | "
        f"Gated Dice {gated_metrics['dice']:.3f}"
    )
    fig.tight_layout()
    fig.savefig(case_dir / "comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    rows.append({
        "case": key,
        "slice": z,
        "baseline_dice": base_metrics["dice"],
        "gated_dice": gated_metrics["dice"],
        "baseline_precision": base_metrics["precision"],
        "gated_precision": gated_metrics["precision"],
        "baseline_recall": base_metrics["recall"],
        "gated_recall": gated_metrics["recall"],
        "baseline_tp": base_metrics["tp"],
        "gated_tp": gated_metrics["tp"],
        "baseline_fp": base_metrics["fp"],
        "gated_fp": gated_metrics["fp"],
        "baseline_fn": base_metrics["fn"],
        "gated_fn": gated_metrics["fn"],
    })

    print(
        f"Baseline Dice={base_metrics['dice']:.4f} | "
        f"Gated Dice={gated_metrics['dice']:.4f}"
    )
    print("Saved:", case_dir)

csv_path = OUT / "representative_case_metrics.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("\nAll representative exports completed.")
print("Output:", OUT)
print("Metrics:", csv_path)
