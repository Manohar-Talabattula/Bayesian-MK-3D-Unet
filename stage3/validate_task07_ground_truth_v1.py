import csv
import itertools
import json
import re
import statistics
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, distance
from scipy.spatial.qhull import QhullError

ROOT = Path.cwd().resolve()
MANIFEST = ROOT / "datasets/stage2/Task07_Pancreas/task07_split_manifest.json"
RESULTS = ROOT / "evaluation/task07_stage3_exports/stage3_evaluation_results.json"
OUT = ROOT / "stage3/ground_truth_validation_v1"
LESION_LABEL = 2
MIN_COMPONENT_VOXELS = 10


def fail(message):
    raise SystemExit("ERROR: " + message)


def resolve_path(value):
    p = Path(value)
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def case_number(value):
    matches = re.findall(r"Task07_Pancreas_(\d+)", str(value))
    return int(matches[-1]) if matches else None


def max_voxelized_diameter_mm(coords, affine):
    """Maximum distance between voxel-cell corners in one component."""
    if len(coords) == 0:
        return 0.0

    offsets = np.asarray(
        list(itertools.product((-0.5, 0.5), repeat=3)),
        dtype=np.float64,
    )
    corners = (coords[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    corners = np.unique(corners, axis=0)
    world = nib.affines.apply_affine(affine, corners)

    try:
        hull_indices = ConvexHull(world).vertices
        points = world[hull_indices]
    except QhullError:
        # Voxel-cell corners should be 3D; this is a defensive fallback.
        points = world

    maximum_squared = 0.0
    block_size = 512

    # Chunked exact pairwise calculation to limit memory use.
    for i in range(0, len(points), block_size):
        a = points[i:i + block_size]
        for j in range(i, len(points), block_size):
            b = points[j:j + block_size]
            d = distance.cdist(a, b, metric="euclidean")
            if d.size:
                maximum_squared = max(
                    maximum_squared, float(np.max(d) ** 2)
                )

    return float(np.sqrt(maximum_squared))


def quantify_ground_truth(label_img):
    label_data = np.asanyarray(label_img.dataobj)
    if label_data.ndim != 3:
        fail(f"Expected a 3D label image, found shape {label_data.shape}")

    affine = np.asarray(label_img.affine, dtype=np.float64)
    voxel_volume_mm3 = float(abs(np.linalg.det(affine[:3, :3])))
    if not np.isfinite(voxel_volume_mm3) or voxel_volume_mm3 <= 0:
        fail("Invalid label-image affine or voxel volume.")

    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    lesion = np.isclose(label_data, LESION_LABEL)
    voxel_count = int(lesion.sum())
    volume_ml = voxel_count * voxel_volume_mm3 / 1000.0

    if voxel_count == 0:
        return {
            "lesion_label": LESION_LABEL,
            "lesion_voxels": 0,
            "volume_ml": 0.0,
            "voxel_volume_mm3": voxel_volume_mm3,
            "spacing_mm": spacing.tolist(),
            "components_raw": 0,
            "components_at_least_10_voxels": 0,
            "largest_component": None,
        }

    component_map, component_count = ndimage.label(
        lesion, structure=np.ones((3, 3, 3), dtype=np.uint8)
    )
    sizes = np.bincount(component_map.ravel())
    sizes[0] = 0
    component_ids = np.flatnonzero(sizes)
    component_ids = sorted(
        component_ids, key=lambda cid: int(sizes[cid]), reverse=True
    )

    largest_id = int(component_ids[0])
    coords = np.argwhere(component_map == largest_id)
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    centroid_voxel = coords.mean(axis=0)
    centroid_world = nib.affines.apply_affine(
        affine, centroid_voxel.reshape(1, 3)
    )[0]

    axis_dimensions = ((hi - lo + 1) * spacing).tolist()

    edge_bounds = [
        (float(lo[i]) - 0.5, float(hi[i]) + 0.5)
        for i in range(3)
    ]
    bbox_corners = np.asarray(
        list(itertools.product(*edge_bounds)), dtype=np.float64
    )
    bbox_world = nib.affines.apply_affine(affine, bbox_corners)
    ras_min = bbox_world.min(axis=0)
    ras_max = bbox_world.max(axis=0)

    return {
        "lesion_label": LESION_LABEL,
        "lesion_voxels": voxel_count,
        "volume_ml": float(volume_ml),
        "voxel_volume_mm3": voxel_volume_mm3,
        "spacing_mm": [float(x) for x in spacing],
        "components_raw": int(component_count),
        "components_at_least_10_voxels": int(
            sum(int(sizes[cid]) >= MIN_COMPONENT_VOXELS
                for cid in component_ids)
        ),
        "largest_component": {
            "voxel_count": int(sizes[largest_id]),
            "volume_ml": float(
                int(sizes[largest_id]) * voxel_volume_mm3 / 1000.0
            ),
            "dimensions_grid_axes_mm": [float(x) for x in axis_dimensions],
            "centroid_ras_mm": [float(x) for x in centroid_world],
            "bounding_box_ras_min_mm": [float(x) for x in ras_min],
            "bounding_box_ras_max_mm": [float(x) for x in ras_max],
            "bounding_box_ras_dimensions_mm": [
                float(x) for x in ras_max - ras_min
            ],
            "max_3d_diameter_mm": max_voxelized_diameter_mm(
                coords, affine
            ),
            "diameter_definition": (
                "Maximum Euclidean distance between voxel-cell corners "
                "of the largest connected component."
            ),
        },
    }


if not MANIFEST.is_file():
    fail(f"Split manifest not found: {MANIFEST}")
if not RESULTS.is_file():
    fail(f"Stage 3 evaluation results not found: {RESULTS}")
if OUT.exists():
    fail(f"Output directory already exists; nothing overwritten: {OUT}")

manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
results = json.loads(RESULTS.read_text(encoding="utf-8"))

val_keys = [
    key for key, value in manifest.items()
    if isinstance(value, list)
    and ("validation" in key.lower() or key.lower() in {
        "val_cases", "valid_cases", "val"
    })
]
if len(val_keys) != 1:
    fail(
        "Could not uniquely identify the validation list. "
        f"Manifest keys are: {list(manifest.keys())}"
    )

validation = manifest[val_keys[0]]
if len(validation) != 56:
    fail(f"Expected 56 validation cases, found {len(validation)}")

pred_cases = results.get("case_results", [])
pred_by_num = {}
for item in pred_cases:
    number = case_number(item.get("case", ""))
    if number is not None:
        pred_by_num[number] = item

if len(pred_by_num) < 56:
    fail(
        f"Expected prediction results for 56 cases; matched "
        f"{len(pred_by_num)} unique case IDs."
    )

rows = []
reports = []
consistency_diffs = []

for index, entry in enumerate(validation, start=1):
    image_path = resolve_path(entry["image"])
    label_path = resolve_path(entry["label"])
    case_id = entry.get("case_id") or image_path.name.replace(
        "_0000.nii.gz", ""
    )
    number = case_number(case_id)

    if number is None:
        fail(f"Cannot extract case number from {case_id}")
    if not image_path.is_file() or not label_path.is_file():
        fail(f"Missing image or label for {case_id}")

    image = nib.load(str(image_path))
    label = nib.load(str(label_path))

    if image.shape != label.shape:
        fail(
            f"Image/label shapes differ for {case_id}: "
            f"{image.shape} vs {label.shape}"
        )
    if not np.allclose(image.affine, label.affine, atol=1e-3):
        fail(
            f"Image/label affines differ for {case_id}; "
            "refusing to compare measurements on different grids."
        )

    label_values = np.unique(np.asanyarray(label.dataobj))
    gt = quantify_ground_truth(label)

    prediction = pred_by_num.get(number)
    if prediction is None:
        fail(f"No matching prediction results for {case_id}")

    pred_stage3 = prediction["stage3"]
    predicted_ml = float(pred_stage3["predicted_volume_ml"])
    direct_gt_ml = float(gt["volume_ml"])
    old_gt_voxels = int(prediction["gt_pathology_voxels"])
    old_voxel_volume = float(pred_stage3["voxel_volume_mm3"])
    old_gt_ml = old_gt_voxels * old_voxel_volume / 1000.0
    consistency_diff = abs(old_gt_ml - direct_gt_ml)
    consistency_diffs.append(consistency_diff)

    absolute_error_ml = abs(predicted_ml - direct_gt_ml)
    error_pct = (
        absolute_error_ml / direct_gt_ml * 100.0
        if direct_gt_ml > 0 else None
    )
    volume_ratio = (
        predicted_ml / direct_gt_ml if direct_gt_ml > 0 else None
    )

    largest = gt["largest_component"] or {}
    row = {
        "case_id": case_id,
        "ground_truth_voxels": gt["lesion_voxels"],
        "ground_truth_volume_ml": round(direct_gt_ml, 6),
        "previous_evaluation_gt_volume_ml": round(old_gt_ml, 6),
        "gt_volume_consistency_difference_ml": round(
            consistency_diff, 6
        ),
        "predicted_gated_volume_ml": round(predicted_ml, 6),
        "absolute_volume_error_ml": round(absolute_error_ml, 6),
        "absolute_volume_error_pct": (
            round(error_pct, 2) if error_pct is not None else ""
        ),
        "predicted_to_gt_volume_ratio": (
            round(volume_ratio, 5) if volume_ratio is not None else ""
        ),
        "gt_components_raw": gt["components_raw"],
        "gt_components_at_least_10_voxels":
            gt["components_at_least_10_voxels"],
        "gt_largest_component_volume_ml":
            largest.get("volume_ml", ""),
        "gt_largest_component_max_3d_diameter_mm":
            largest.get("max_3d_diameter_mm", ""),
        "baseline_dice": prediction["baseline"]["dice"],
        "gated_dice": prediction["gated"]["dice"],
    }
    rows.append(row)
    reports.append({
        "case_id": case_id,
        "image_path": str(image_path),
        "label_path": str(label_path),
        "label_values": [float(v) for v in label_values],
        "ground_truth_quantification": gt,
        "previous_evaluation_gt_volume_ml": float(old_gt_ml),
        "predicted_gated_volume_ml": predicted_ml,
        "absolute_volume_error_ml": absolute_error_ml,
        "absolute_volume_error_pct": error_pct,
        "baseline_dice": prediction["baseline"]["dice"],
        "gated_dice": prediction["gated"]["dice"],
    })

    print(
        f"[{index}/{len(validation)}] {case_id}: "
        f"GT={direct_gt_ml:.3f} mL, pred={predicted_ml:.3f} mL, "
        f"max diameter={largest.get('max_3d_diameter_mm', 0):.2f} mm"
    )

OUT.mkdir(parents=True)
report_dir = OUT / "case_reports"
report_dir.mkdir()

for report in reports:
    path = report_dir / f'{report["case_id"]}_gt_report.json'
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")

csv_path = OUT / "ground_truth_volume_comparison.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

valid = [r for r in rows if r["ground_truth_volume_ml"] > 0]
errors = [r["absolute_volume_error_pct"] for r in valid]
abs_errors = [r["absolute_volume_error_ml"] for r in valid]

summary = {
    "validation_cases": len(rows),
    "lesion_label": LESION_LABEL,
    "median_absolute_percentage_volume_error": (
        float(statistics.median(errors)) if errors else None
    ),
    "median_absolute_volume_error_ml": (
        float(statistics.median(abs_errors)) if abs_errors else None
    ),
    "cases_within_20_percent_volume_error": sum(
        e <= 20 for e in errors
    ),
    "cases_within_50_percent_volume_error": sum(
        e <= 50 for e in errors
    ),
    "max_ground_truth_volume_consistency_difference_ml": max(
        consistency_diffs, default=0.0
    ),
    "mean_ground_truth_volume_consistency_difference_ml": (
        float(statistics.mean(consistency_diffs))
        if consistency_diffs else None
    ),
    "csv_path": str(csv_path),
    "case_reports_directory": str(report_dir),
    "diameter_definition": (
        "Maximum Euclidean distance between voxel-cell corners "
        "of the largest 26-connected ground-truth lesion component."
    ),
}

summary_path = OUT / "validation_summary.json"
summary_path.write_text(
    json.dumps(summary, indent=2), encoding="utf-8"
)

print("\n" + "=" * 60)
print("GROUND-TRUTH STAGE 3 VALIDATION")
print("=" * 60)
print(f"Cases: {len(rows)}")
print(
    "Median absolute percentage volume error: "
    f"{summary['median_absolute_percentage_volume_error']:.2f}%"
)
print(
    "Median absolute volume error: "
    f"{summary['median_absolute_volume_error_ml']:.3f} mL"
)
print(
    "Cases within 20% / 50% volume error: "
    f"{summary['cases_within_20_percent_volume_error']}/{len(valid)} / "
    f"{summary['cases_within_50_percent_volume_error']}/{len(valid)}"
)
print(
    "Largest old-vs-direct GT volume discrepancy: "
    f"{summary['max_ground_truth_volume_consistency_difference_ml']:.6f} mL"
)
print(f"CSV: {csv_path}")
print(f"Reports: {report_dir}")
print(f"Summary: {summary_path}")