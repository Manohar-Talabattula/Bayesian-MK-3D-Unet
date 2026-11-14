import csv
import itertools
import json
import re
import statistics
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError, distance

ROOT = Path.cwd().resolve()
MANIFEST = ROOT / "datasets/stage2/Task07_Pancreas/task07_split_manifest.json"
RESULTS = ROOT / "evaluation/task07_stage3_exports/stage3_evaluation_results.json"
GT_REPORTS = ROOT / "stage3/ground_truth_validation_v1/case_reports"
OUT = ROOT / "stage3/ground_truth_validation_v1"
CSV_PATH = OUT / "prediction_vs_ground_truth_geometry.csv"
SUMMARY_PATH = OUT / "prediction_geometry_summary.json"
LESION_LABEL = 2


def stop(message):
    raise SystemExit("ERROR: " + message)


def get_case_number(text):
    matches = re.findall(r"Task07_Pancreas_(\d+)", str(text))
    return int(matches[-1]) if matches else None


def max_diameter_mm(coords, affine):
    if len(coords) == 0:
        return None

    offsets = np.asarray(
        list(itertools.product((-0.5, 0.5), repeat=3)),
        dtype=np.float64,
    )
    voxel_corners = (coords[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    voxel_corners = np.unique(voxel_corners, axis=0)
    world = nib.affines.apply_affine(affine, voxel_corners)

    try:
        points = world[ConvexHull(world).vertices]
    except QhullError:
        points = world

    maximum = 0.0
    block = 512
    for i in range(0, len(points), block):
        a = points[i:i + block]
        for j in range(i, len(points), block):
            b = points[j:j + block]
            d = distance.cdist(a, b, metric="euclidean")
            if d.size:
                maximum = max(maximum, float(d.max()))

    return float(maximum)


def measure_largest_component(mask, affine):
    mask = np.asarray(mask, dtype=bool)
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    voxel_volume_mm3 = float(abs(np.linalg.det(affine[:3, :3])))

    if voxel_volume_mm3 <= 0:
        stop("Invalid affine: voxel volume must be positive.")

    labels, n_components = ndimage.label(
        mask, structure=np.ones((3, 3, 3), dtype=np.uint8)
    )
    sizes = np.bincount(labels.ravel())
    if len(sizes):
        sizes[0] = 0

    if n_components == 0 or int(mask.sum()) == 0:
        return {
            "voxel_count": 0,
            "volume_ml": 0.0,
            "components_raw": int(n_components),
            "largest_component_volume_ml": 0.0,
            "max_3d_diameter_mm": None,
            "dimensions_grid_axes_mm": None,
            "bounding_box_ras_dimensions_mm": None,
            "centroid_ras_mm": None,
        }

    largest_id = int(np.argmax(sizes))
    coords = np.argwhere(labels == largest_id)
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)

    centroid = nib.affines.apply_affine(
        affine, coords.mean(axis=0).reshape(1, 3)
    )[0]

    bounds = [
        (float(lo[i]) - 0.5, float(hi[i]) + 0.5)
        for i in range(3)
    ]
    corners = np.asarray(
        list(itertools.product(*bounds)), dtype=np.float64
    )
    world_corners = nib.affines.apply_affine(affine, corners)
    ras_dimensions = world_corners.max(axis=0) - world_corners.min(axis=0)

    return {
        "voxel_count": int(mask.sum()),
        "volume_ml": float(mask.sum() * voxel_volume_mm3 / 1000.0),
        "components_raw": int(n_components),
        "largest_component_voxel_count": int(sizes[largest_id]),
        "largest_component_volume_ml": float(
            int(sizes[largest_id]) * voxel_volume_mm3 / 1000.0
        ),
        "max_3d_diameter_mm": max_diameter_mm(coords, affine),
        "dimensions_grid_axes_mm": [
            float((hi[i] - lo[i] + 1) * spacing[i])
            for i in range(3)
        ],
        "bounding_box_ras_dimensions_mm": [
            float(x) for x in ras_dimensions
        ],
        "centroid_ras_mm": [float(x) for x in centroid],
    }


if not MANIFEST.is_file() or not RESULTS.is_file():
    stop("The validation manifest or Stage 3 evaluation JSON is missing.")
if not GT_REPORTS.is_dir():
    stop(f"Ground-truth reports not found: {GT_REPORTS}")
if CSV_PATH.exists() or SUMMARY_PATH.exists():
    stop("An output file already exists. Nothing was overwritten.")

manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
evaluation = json.loads(RESULTS.read_text(encoding="utf-8"))

validation_keys = [
    k for k, v in manifest.items()
    if isinstance(v, list) and "validation" in k.lower()
]
if len(validation_keys) != 1:
    stop(f"Cannot uniquely identify validation split: {list(manifest.keys())}")

validation = manifest[validation_keys[0]]
predictions = {}
for item in evaluation.get("case_results", []):
    num = get_case_number(item.get("case", ""))
    if num is not None:
        predictions[num] = item

if len(validation) != 56 or len(predictions) < 56:
    stop(
        f"Expected 56 validation cases and predictions; got "
        f"{len(validation)} validation entries and {len(predictions)} predictions."
    )

rows = []
volume_errors = []
diameter_errors = []
diameter_pct_errors = []
centroid_errors = []
dice_differences = []

for index, entry in enumerate(validation, start=1):
    case_id = entry["case_id"]
    num = get_case_number(case_id)
    pred_result = predictions.get(num)
    if pred_result is None:
        stop(f"No prediction entry found for {case_id}")

    label_path = ROOT / entry["label"]
    pred_path = Path(pred_result["gated_mask_path"])
    if not pred_path.is_absolute():
        pred_path = ROOT / pred_path

    report_path = GT_REPORTS / f"{case_id}_gt_report.json"
    for path in (label_path, pred_path, report_path):
        if not path.is_file():
            stop(f"Required file not found: {path}")

    gt_img = nib.load(str(label_path))
    pred_img = nib.load(str(pred_path))

    if gt_img.shape != pred_img.shape:
        stop(
            f"Shape mismatch for {case_id}: "
            f"ground truth {gt_img.shape}, prediction {pred_img.shape}"
        )
    if not np.allclose(gt_img.affine, pred_img.affine, atol=1e-3):
        stop(f"Affine mismatch for {case_id}; cannot compare coordinates.")

    gt = np.isclose(np.asanyarray(gt_img.dataobj), LESION_LABEL)
    pred = np.asanyarray(pred_img.dataobj) > 0

    gt_volume_ml = float(
        gt.sum() * abs(np.linalg.det(gt_img.affine[:3, :3])) / 1000.0
    )
    geom = measure_largest_component(pred, pred_img.affine)
    pred_volume_ml = geom["volume_ml"]
    volume_error_ml = abs(pred_volume_ml - gt_volume_ml)
    volume_error_pct = (
        volume_error_ml / gt_volume_ml * 100.0 if gt_volume_ml > 0 else None
    )

    intersection = int(np.logical_and(gt, pred).sum())
    denom = int(gt.sum()) + int(pred.sum())
    computed_dice = 2.0 * intersection / denom if denom else 1.0
    reported_dice = float(pred_result["gated"]["dice"])
    dice_diff = abs(computed_dice - reported_dice)
    dice_differences.append(dice_diff)

    gt_report = json.loads(report_path.read_text(encoding="utf-8"))
    gt_quant = gt_report["ground_truth_quantification"]
    gt_largest = gt_quant.get("largest_component") or {}
    gt_diameter = gt_largest.get("max_3d_diameter_mm")
    pred_diameter = geom["max_3d_diameter_mm"]

    diameter_error = (
        abs(float(pred_diameter) - float(gt_diameter))
        if pred_diameter is not None and gt_diameter is not None
        else None
    )
    diameter_error_pct = (
        diameter_error / float(gt_diameter) * 100.0
        if diameter_error is not None and float(gt_diameter) > 0
        else None
    )

    gt_centroid = gt_largest.get("centroid_ras_mm")
    pred_centroid = geom["centroid_ras_mm"]
    centroid_distance = (
        float(np.linalg.norm(np.asarray(gt_centroid) - np.asarray(pred_centroid)))
        if gt_centroid is not None and pred_centroid is not None
        else None
    )

    volume_errors.append(volume_error_ml)
    if volume_error_pct is not None:
        # Store percentages separately for the cohort summary.
        pass
    if diameter_error is not None:
        diameter_errors.append(diameter_error)
    if diameter_error_pct is not None:
        diameter_pct_errors.append(diameter_error_pct)
    if centroid_distance is not None:
        centroid_errors.append(centroid_distance)

    rows.append({
        "case_id": case_id,
        "gt_volume_ml": round(gt_volume_ml, 6),
        "predicted_gated_volume_ml": round(pred_volume_ml, 6),
        "absolute_volume_error_ml": round(volume_error_ml, 6),
        "absolute_volume_error_pct": (
            round(volume_error_pct, 2) if volume_error_pct is not None else ""
        ),
        "gt_largest_component_diameter_mm": gt_diameter,
        "predicted_largest_component_diameter_mm": pred_diameter,
        "absolute_diameter_error_mm": (
            round(diameter_error, 3) if diameter_error is not None else ""
        ),
        "absolute_diameter_error_pct": (
            round(diameter_error_pct, 2)
            if diameter_error_pct is not None else ""
        ),
        "gt_centroid_ras_mm": json.dumps(gt_centroid),
        "predicted_centroid_ras_mm": json.dumps(pred_centroid),
        "largest_component_centroid_distance_mm": (
            round(centroid_distance, 3)
            if centroid_distance is not None else ""
        ),
        "gt_largest_component_bbox_ras_dimensions_mm": json.dumps(
            gt_largest.get("bounding_box_ras_dimensions_mm")
        ),
        "pred_largest_component_bbox_ras_dimensions_mm": json.dumps(
            geom["bounding_box_ras_dimensions_mm"]
        ),
        "gt_component_count": gt_quant.get("connected_components_raw"),
        "pred_component_count": geom["components_raw"],
        "dice_recomputed": round(computed_dice, 6),
        "dice_reported": round(reported_dice, 6),
        "dice_difference": round(dice_diff, 8),
    })

    print(
        f"[{index}/56] {case_id}: Dice={computed_dice:.3f}, "
        f"volume error={volume_error_pct:.1f}%, "
        f"diameter error="
        f"{diameter_error:.2f} mm" if diameter_error is not None
        else f"[{index}/56] {case_id}: Dice={computed_dice:.3f}, "
             f"volume error={volume_error_pct:.1f}%, diameter unavailable"
    )

with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

volume_pct_errors = [
    r["absolute_volume_error_pct"] for r in rows
    if r["absolute_volume_error_pct"] != ""
]
summary = {
    "validation_cases": len(rows),
    "median_absolute_volume_error_ml": float(statistics.median(volume_errors)),
    "median_absolute_percentage_volume_error": float(
        statistics.median(volume_pct_errors)
    ),
    "cases_within_20_percent_volume_error": sum(
        float(x) <= 20 for x in volume_pct_errors
    ),
    "cases_within_50_percent_volume_error": sum(
        float(x) <= 50 for x in volume_pct_errors
    ),
    "median_absolute_largest_component_diameter_error_mm": (
        float(statistics.median(diameter_errors)) if diameter_errors else None
    ),
    "median_absolute_largest_component_diameter_error_pct": (
        float(statistics.median(diameter_pct_errors))
        if diameter_pct_errors else None
    ),
    "median_largest_component_centroid_distance_mm": (
        float(statistics.median(centroid_errors)) if centroid_errors else None
    ),
    "maximum_difference_between_recomputed_and_reported_dice": (
        float(max(dice_differences)) if dice_differences else None
    ),
    "csv_path": str(CSV_PATH),
}

SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

print("\n" + "=" * 60)
print("PREDICTION VS GROUND-TRUTH GEOMETRY SUMMARY")
print("=" * 60)
print(f"Cases: {summary['validation_cases']}")
print(
    "Median absolute volume error: "
    f"{summary['median_absolute_volume_error_ml']:.3f} mL"
)
print(
    "Median absolute percentage volume error: "
    f"{summary['median_absolute_percentage_volume_error']:.2f}%"
)
print(
    "Cases within 20% / 50% volume error: "
    f"{summary['cases_within_20_percent_volume_error']}/56 / "
    f"{summary['cases_within_50_percent_volume_error']}/56"
)
print(
    "Median absolute max-diameter error: "
    f"{summary['median_absolute_largest_component_diameter_error_mm']:.2f} mm"
)
print(
    "Median largest-component centroid distance: "
    f"{summary['median_largest_component_centroid_distance_mm']:.2f} mm"
)
print(
    "Maximum recomputed-vs-reported Dice difference: "
    f"{summary['maximum_difference_between_recomputed_and_reported_dice']:.8f}"
)
print(f"\nCSV: {CSV_PATH}")
print(f"Summary JSON: {SUMMARY_PATH}")