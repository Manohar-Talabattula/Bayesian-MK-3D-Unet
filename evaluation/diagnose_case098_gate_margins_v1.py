import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

import run_task07_stage3_export_v1 as stage3

ROOT = stage3.ROOT

IMAGE_PATH = ROOT / "datasets/stage2/Task07_Pancreas/Task07_Pancreas_098_0000.nii.gz"
GT_PATH = ROOT / "datasets/stage2/Task07_Pancreas/Task07_Pancreas_098.nii.gz"
GATE_PATH = ROOT / "evaluation/diagnostics_case098/case098_fresh_stage1_gate_v1.nii.gz"

OUT_DIR = ROOT / "evaluation/diagnostics_case098_stage2_prob_v1"

if OUT_DIR.exists():
    raise FileExistsError(
        f"Output directory already exists. Preserving it: {OUT_DIR}"
    )

for path in [IMAGE_PATH, GT_PATH, GATE_PATH, stage3.STAGE2_CHECKPOINT]:
    if not path.is_file():
        raise FileNotFoundError(path)

OUT_DIR.mkdir(parents=True)

print("Loading Stage 2 model...")
model = stage3.build_stage2_model()
stage3.load_checkpoint(model, stage3.STAGE2_CHECKPOINT)
model = model.to(stage3.DEVICE).eval()

print("Loading case 098...")
image_nii = nib.load(str(IMAGE_PATH))
gt_nii = nib.load(str(GT_PATH))
gate_nii = nib.load(str(GATE_PATH))

image = image_nii.get_fdata().astype(np.float32)
gt = gt_nii.get_fdata().astype(np.uint8)
gate = gate_nii.get_fdata() > 0

if not (image.shape == gt.shape == gate.shape):
    raise RuntimeError(
        f"Grid mismatch: image={image.shape}, GT={gt.shape}, gate={gate.shape}"
    )

print("Image shape:", image.shape)
print("Image range:", float(image.min()), float(image.max()))
print("Gate voxels:", int(gate.sum()))

print("\nRunning Stage 2 probability inference...")
p2_zyx = stage3.predict_stage2_probability(model, np.transpose(image, (2, 1, 0)))[2]
p2 = np.transpose(p2_zyx, (2, 1, 0)).astype(np.float32)

if p2.shape != image.shape:
    raise RuntimeError(f"Probability shape mismatch: {p2.shape}")

prob_path = OUT_DIR / "case098_stage2_lesion_probability.nii.gz"
prob_nii = nib.Nifti1Image(
    p2,
    affine=image_nii.affine,
    header=image_nii.header.copy()
)
prob_nii.set_data_dtype(np.float32)
nib.save(prob_nii, str(prob_path))

print("Probability map saved:", prob_path)
print("Probability range:", float(p2.min()), float(p2.max()))

gt_lesion = gt == 2
threshold = 0.80
baseline = p2 >= threshold

results = {
    "case": "Task07_Pancreas_098",
    "threshold": threshold,
    "probability_map": str(prob_path),
    "baseline": stage3.binary_metrics(baseline, gt_lesion),
    "gate_margin_results": {}
}

spacing = tuple(float(x) for x in image_nii.header.get_zooms()[:3])

print("\n" + "=" * 78)
print("CASE 098 — GATE MARGIN COMPARISON")
print("=" * 78)
print(f"{'Margin':>9} | {'Dice':>8} | {'Precision':>10} | {'Recall':>8} | {'Pred voxels':>12}")

b = results["baseline"]
print(
    f"{'Baseline':>9} | {b['dice']:>8.4f} | "
    f"{b['precision']:>10.4f} | {b['recall']:>8.4f} | "
    f"{b['pred_voxels']:>12}"
)

for margin in [15, 25, 30, 40]:
    region = stage3.dilate_physical(gate, spacing, float(margin))
    prediction = baseline & region
    metrics = stage3.binary_metrics(prediction, gt_lesion)

    results["gate_margin_results"][str(margin)] = {
        "gate_voxels": int(region.sum()),
        "metrics": metrics
    }

    print(
        f"{margin:>6} mm | {metrics['dice']:>8.4f} | "
        f"{metrics['precision']:>10.4f} | {metrics['recall']:>8.4f} | "
        f"{metrics['pred_voxels']:>12}"
    )

report_path = OUT_DIR / "case098_gate_margin_comparison.json"
with open(report_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\nDiagnostic completed.")
print("Probability map:", prob_path)
print("Comparison report:", report_path)
