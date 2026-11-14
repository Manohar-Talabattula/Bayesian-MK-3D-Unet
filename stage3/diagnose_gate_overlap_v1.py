import json
import re
import statistics
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
import csv

ROOT = Path.cwd().resolve()
MANIFEST_PATH = ROOT / "datasets/stage2/Task07_Pancreas/task07_split_manifest.json"
RESULTS_PATH = ROOT / "evaluation/task07_stage3_exports/stage3_evaluation_results.json"
GATE_DIR = ROOT / "evaluation/task07_full_volume_v2/stage1_pancreas_masks"
OUT_DIR = ROOT / "stage3/ground_truth_validation_v1"
CSV_PATH = OUT_DIR / "gate_overlap_validation.csv"
SUMMARY_PATH = OUT_DIR / "gate_overlap_summary.json"
MARGIN_MM = 15.0


def stop(msg):
    raise SystemExit("ERROR: " + msg)


def case_num(text):
    found = re.findall(r"Task07_Pancreas_(\d+)", str(text))
    return int(found[-1]) if found else None


for path in (MANIFEST_PATH, RESULTS_PATH, GATE_DIR):
    if not path.exists():
        stop(f"Missing required input: {path}")

if CSV_PATH.exists() or SUMMARY_PATH.exists():
    stop("An output file already exists. Nothing was overwritten.")

manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))

val_keys = [
    k for k, v in manifest.items()
    if isinstance(v, list) and "validation" in k.lower()
]
if len(val_keys) != 1:
    stop(f"Cannot identify validation list in manifest: {list(manifest.keys())}")

validation = manifest[val_keys[0]]
predictions = {
    case_num(r.get("case", "")): r
    for r in results.get("case_results", [])
    if case_num(r.get("case", "")) is not None
}

if len(validation) != 56:
    stop(f"Expected 56 validation cases, found {len(validation)}")
if len(predictions) < 56:
    stop(f"Expected 56 prediction records, found {len(predictions)}")

rows = []

for i, entry in enumerate(validation, start=1):
    cid = entry["case_id"]
    n = case_num(cid)
    result = predictions.get(n)
    if result is None:
        stop(f"No prediction record for {cid}")

    label_path = (ROOT / entry["label"]).resolve()
    pred_path = Path(result["gated_mask_path"])
    if not pred_path.is_absolute():
        pred_path = ROOT / pred_path

    gates = list(GATE_DIR.glob(
        f"*Task07_Pancreas_{n:03d}_0000_pancreas_gate.nii.gz"
    ))
    if len(gates) != 1:
        stop(f"Expected exactly one cached gate for {cid}; found {len(gates)}")

    for p in (label_path, pred_path, gates[0]):
        if not p.is_file():
            stop(f"Required file missing: {p}")

    label_img = nib.load(str(label_path))
    pred_img = nib.load(str(pred_path))
    gate_img = nib.load(str(gates[0]))

    if not (
        label_img.shape == pred_img.shape == gate_img.shape
    ):
        stop(
            f"Shape mismatch for {cid}: label={label_img.shape}, "
            f"prediction={pred_img.shape}, gate={gate_img.shape}"
        )

    if not (
        np.allclose(label_img.affine, pred_img.affine, atol=1e-3)
        and np.allclose(label_img.affine, gate_img.affine, atol=1e-3)
    ):
        stop(f"Affine mismatch for {cid}; refusing voxelwise comparison.")

    gt = np.asanyarray(label_img.dataobj) == 2
    pred = np.asanyarray(pred_img.dataobj) > 0
    gate = np.rint(
        np.asanyarray(gate_img.dataobj)
    ).astype(np.uint8) > 0

    spacing = tuple(float(x) for x in label_img.header.get_zooms()[:3])
    distance_from_gate = ndimage.distance_transform_edt(
        ~gate, sampling=spacing
    )
    expanded = gate | (distance_from_gate <= MARGIN_MM)

    gt_vox = int(gt.sum())
    pred_vox = int(pred.sum())
    gate_vox = int(gate.sum())
    expanded_vox = int(expanded.sum())

    gt_in_gate = int(np.logical_and(gt, gate).sum())
    gt_in_expanded = int(np.logical_and(gt, expanded).sum())
    pred_in_gate = int(np.logical_and(pred, gate).sum())
    pred_in_expanded = int(np.logical_and(pred, expanded).sum())
    gt_pred_overlap = int(np.logical_and(gt, pred).sum())

    raw_coverage = 100.0 * gt_in_gate / gt_vox if gt_vox else 0.0
    expanded_coverage = (
        100.0 * gt_in_expanded / gt_vox if gt_vox else 0.0
    )
    pred_in_expanded_pct = (
        100.0 * pred_in_expanded / pred_vox if pred_vox else 0.0
    )
    predicted_dice = (
        2.0 * gt_pred_overlap / (gt_vox + pred_vox)
        if gt_vox + pred_vox else 1.0
    )
    voxel_volume_ml = (
        abs(float(np.linalg.det(label_img.affine[:3, :3]))) / 1000.0
    )

    rows.append({
        "case_id": cid,
        "gt_lesion_voxels": gt_vox,
        "gt_volume_ml": round(gt_vox * voxel_volume_ml, 6),
        "original_gate_voxels": gate_vox,
        "original_gate_volume_ml": round(gate_vox * voxel_volume_ml, 6),
        "expanded_gate_voxels": expanded_vox,
        "expanded_gate_volume_ml": round(expanded_vox * voxel_volume_ml, 6),
        "gt_voxels_inside_original_gate": gt_in_gate,
        "gt_coverage_original_gate_pct": round(raw_coverage, 2),
        "gt_voxels_inside_expanded_gate": gt_in_expanded,
        "gt_coverage_expanded_gate_pct": round(expanded_coverage, 2),
        "gt_voxels_excluded_by_expanded_gate": gt_vox - gt_in_expanded,
        "predicted_voxels": pred_vox,
        "pred_voxels_inside_original_gate": pred_in_gate,
        "pred_voxels_inside_expanded_gate": pred_in_expanded,
        "pred_fraction_inside_expanded_gate_pct": round(
            pred_in_expanded_pct, 2
        ),
        "predicted_voxels_outside_expanded_gate": pred_vox - pred_in_expanded,
        "gated_dice_recomputed": round(predicted_dice, 6),
        "gated_dice_reported": round(float(result["gated"]["dice"]), 6),
    })

    print(
        f"[{i}/56] {cid}: GT coverage raw={raw_coverage:.1f}%, "
        f"expanded={expanded_coverage:.1f}%, "
        f"prediction={pred_vox:,} voxels, Dice={predicted_dice:.3f}"
    )

with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

raw = [r["gt_coverage_original_gate_pct"] for r in rows]
expanded_cov = [r["gt_coverage_expanded_gate_pct"] for r in rows]
excluded = [r for r in rows if r["gt_coverage_expanded_gate_pct"] < 99.99]
low_cov = [r for r in rows if r["gt_coverage_expanded_gate_pct"] < 95]
outside = [r for r in rows if r["predicted_voxels_outside_expanded_gate"] > 0]
dice_diffs = [
    abs(r["gated_dice_recomputed"] - r["gated_dice_reported"])
    for r in rows
]

summary = {
    "cases": len(rows),
    "gate_margin_mm": MARGIN_MM,
    "median_ground_truth_coverage_original_gate_pct": statistics.median(raw),
    "median_ground_truth_coverage_expanded_gate_pct": statistics.median(expanded_cov),
    "cases_with_expanded_gate_coverage_below_95_pct": len(low_cov),
    "cases_with_any_gt_voxels_excluded_by_expanded_gate": len(excluded),
    "cases_with_prediction_outside_expanded_gate": len(outside),
    "maximum_recomputed_vs_reported_gated_dice_difference": max(dice_diffs),
    "cases_lowest_expanded_gate_coverage": sorted(
        [
            {
                "case_id": r["case_id"],
                "coverage_pct": r["gt_coverage_expanded_gate_pct"],
                "excluded_gt_voxels": r["gt_voxels_excluded_by_expanded_gate"],
            }
            for r in rows
        ],
        key=lambda x: x["coverage_pct"]
    )[:10],
    "csv_path": str(CSV_PATH),
}

SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("\n" + "=" * 64)
print("PANCREAS GATE OVERLAP SUMMARY")
print("=" * 64)
print(f"Cases: {summary['cases']}")
print(
    "Median GT coverage in original gate: "
    f"{summary['median_ground_truth_coverage_original_gate_pct']:.2f}%"
)
print(
    "Median GT coverage in expanded 15 mm gate: "
    f"{summary['median_ground_truth_coverage_expanded_gate_pct']:.2f}%"
)
print(
    "Cases where expanded gate covers <95% of GT lesion: "
    f"{len(low_cov)}/56"
)
print(
    "Cases with any GT lesion excluded by expanded gate: "
    f"{len(excluded)}/56"
)
print(
    "Cases with predicted voxels outside expanded gate: "
    f"{len(outside)}/56"
)
print(
    "Max Dice discrepancy vs evaluator: "
    f"{max(dice_diffs):.8f}"
)
print("\nLowest gate coverage cases:")
for case in summary["cases_lowest_expanded_gate_coverage"]:
    print(
        f'  {case["case_id"]}: {case["coverage_pct"]:.1f}% coverage; '
        f'{case["excluded_gt_voxels"]} GT voxels excluded'
    )
print(f"\nCSV: {CSV_PATH}")
print(f"Summary: {SUMMARY_PATH}")