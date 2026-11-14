import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path.cwd().resolve()
manifest_path = root / "datasets/stage2/Task07_Pancreas/task07_split_manifest.json"
results_path = root / "evaluation/task07_stage3_exports/stage3_evaluation_results.json"
out_dir = root / "stage3/ground_truth_validation_v1/qc_overlays"

if not manifest_path.is_file() or not results_path.is_file():
    raise SystemExit("Missing manifest or evaluation results.")

if out_dir.exists():
    raise SystemExit(f"Output directory already exists: {out_dir}")

manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
results = json.loads(results_path.read_text(encoding="utf-8"))

val_key = next(
    (k for k, v in manifest.items()
     if isinstance(v, list) and "validation" in k.lower()),
    None
)
if val_key is None:
    raise SystemExit("Could not locate validation cases in manifest.")

def case_number(text):
    found = re.findall(r"Task07_Pancreas_(\d+)", str(text))
    return int(found[-1]) if found else None

entries = {
    case_number(e["case_id"]): e
    for e in manifest[val_key]
}
predictions = {
    case_number(r["case"]): r
    for r in results.get("case_results", [])
}
selected = [198, 157, 318, 391]

missing = [
    n for n in selected
    if n not in entries or n not in predictions
]
if missing:
    raise SystemExit(f"Selected cases missing from validation/results: {missing}")

out_dir.mkdir(parents=True)

for number in selected:
    entry = entries[number]
    result = predictions[number]

    image_path = root / entry["image"]
    label_path = root / entry["label"]
    pred_path = Path(result["gated_mask_path"])
    if not pred_path.is_absolute():
        pred_path = root / pred_path

    for p in (image_path, label_path, pred_path):
        if not p.is_file():
            raise SystemExit(f"Required file missing: {p}")

    image_nii = nib.load(str(image_path))
    label_nii = nib.load(str(label_path))
    pred_nii = nib.load(str(pred_path))

    image = np.asarray(image_nii.dataobj, dtype=np.float32)
    gt = np.asanyarray(label_nii.dataobj) == 2
    pred = np.asanyarray(pred_nii.dataobj) > 0

    if image.shape != gt.shape or image.shape != pred.shape:
        raise SystemExit(f"Shape mismatch for {entry['case_id']}")
    if not np.allclose(image_nii.affine, label_nii.affine, atol=1e-3):
        raise SystemExit(f"Image/label affine mismatch: {entry['case_id']}")
    if not np.allclose(image_nii.affine, pred_nii.affine, atol=1e-3):
        raise SystemExit(f"Image/prediction affine mismatch: {entry['case_id']}")

    if gt.any():
        center = np.rint(np.argwhere(gt).mean(axis=0)).astype(int)
    else:
        center = np.asarray(image.shape) // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, ax in enumerate(axes):
        index = int(np.clip(center[axis], 0, image.shape[axis] - 1))
        ct_slice = np.rot90(np.take(image, index, axis=axis))
        gt_slice = np.rot90(np.take(gt, index, axis=axis))
        pred_slice = np.rot90(np.take(pred, index, axis=axis))

        lo, hi = np.percentile(ct_slice, [1, 99])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            lo, hi = float(ct_slice.min()), float(ct_slice.max() + 1e-6)

        ax.imshow(ct_slice, cmap="gray", vmin=lo, vmax=hi)
        if gt_slice.any():
            ax.contour(
                gt_slice.astype(float), levels=[0.5],
                colors=["lime"], linewidths=1.4
            )
        if pred_slice.any():
            ax.contour(
                pred_slice.astype(float), levels=[0.5],
                colors=["red"], linewidths=1.2
            )
        ax.set_title(f"Array axis {axis}, slice {index}")
        ax.axis("off")

    dice = float(result["gated"]["dice"])
    gt_ml = float(result["gt_pathology_voxels"]) * float(
        result["stage3"]["voxel_volume_mm3"]
    ) / 1000.0
    pred_ml = float(result["stage3"]["predicted_volume_ml"])

    fig.suptitle(
        f'{entry["case_id"]} | Dice={dice:.3f} | '
        f'GT={gt_ml:.2f} mL | Predicted={pred_ml:.2f} mL'
    )
    fig.tight_layout()
    output = out_dir / f'Task07_Pancreas_{number:03d}_qc.png'
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")

print(f"\nAll overlays saved to: {out_dir}")
print("Green = ground-truth lesion; red = predicted lesion.")
print("Slices pass through the ground-truth lesion centroid.")