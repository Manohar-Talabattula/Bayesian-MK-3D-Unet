import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage

import evaluate_stage2_full_volume as evaluator


PROJECT_ROOT = Path(__file__).resolve().parent

THRESHOLD = 0.80

OUTPUT_DIR = (
    PROJECT_ROOT
    / "datasets"
    / "stage2"
    / "evaluation"
)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "stage2_component_diagnostic.csv"
)


def build_prediction(probability_map, threshold):
    """
    Convert the probability map into the final Stage-2 prediction.

    Class 0 = background
    Class 1 = healthy/non-lesional
    Class 2 = pathology
    """

    prediction = np.argmax(
        probability_map,
        axis=0,
    ).astype(np.uint8)

    pathology_probability = probability_map[2]

    non_pathology_prediction = np.argmax(
        probability_map[:2],
        axis=0,
    ).astype(np.uint8)

    low_confidence_pathology = (
        (prediction == 2)
        &
        (pathology_probability < threshold)
    )

    prediction[low_confidence_pathology] = (
        non_pathology_prediction[
            low_confidence_pathology
        ]
    )

    return prediction


def analyze_components(predicted_pathology, label):
    """
    Analyze predicted pathology connected components.

    26-connectivity is used for 3D connected components.
    """

    structure = np.ones(
        (3, 3, 3),
        dtype=np.uint8,
    )

    component_labels, num_components = ndimage.label(
        predicted_pathology,
        structure=structure,
    )

    if num_components == 0:
        return {
            "predicted_pathology_voxels": 0,
            "num_components": 0,
            "largest_component_voxels": 0,
            "components_le_10_voxels": 0,
            "components_le_50_voxels": 0,
            "components_le_100_voxels": 0,
            "components_le_500_voxels": 0,
            "components_le_1000_voxels": 0,
            "voxels_in_components_le_10": 0,
            "voxels_in_components_le_50": 0,
            "voxels_in_components_le_100": 0,
            "voxels_in_components_le_500": 0,
            "voxels_in_components_le_1000": 0,
            "background_fp_voxels": 0,
            "healthy_fp_voxels": 0,
            "true_positive_voxels": 0,
        }

    component_sizes = np.bincount(
        component_labels.ravel()
    )[1:]

    component_ids = np.arange(
        1,
        num_components + 1,
    )

    sorted_sizes = np.sort(
        component_sizes
    )[::-1]

    largest_component_voxels = int(
        sorted_sizes[0]
    )

    # ---------------------------------------------------------
    # Component-size summaries
    # ---------------------------------------------------------

    le_10 = component_sizes <= 10
    le_50 = component_sizes <= 50
    le_100 = component_sizes <= 100
    le_500 = component_sizes <= 500
    le_1000 = component_sizes <= 1000

    stats = {
        "predicted_pathology_voxels": int(
            np.sum(predicted_pathology)
        ),

        "num_components": int(
            num_components
        ),

        "largest_component_voxels":
            largest_component_voxels,

        "components_le_10_voxels":
            int(np.sum(le_10)),

        "components_le_50_voxels":
            int(np.sum(le_50)),

        "components_le_100_voxels":
            int(np.sum(le_100)),

        "components_le_500_voxels":
            int(np.sum(le_500)),

        "components_le_1000_voxels":
            int(np.sum(le_1000)),

        "voxels_in_components_le_10":
            int(np.sum(component_sizes[le_10])),

        "voxels_in_components_le_50":
            int(np.sum(component_sizes[le_50])),

        "voxels_in_components_le_100":
            int(np.sum(component_sizes[le_100])),

        "voxels_in_components_le_500":
            int(np.sum(component_sizes[le_500])),

        "voxels_in_components_le_1000":
            int(np.sum(component_sizes[le_1000])),
    }

    # ---------------------------------------------------------
    # Detailed pathology breakdown
    # ---------------------------------------------------------

    predicted_mask = component_labels > 0

    gt_pathology = label == 2

    fp_background = np.logical_and(
        predicted_mask,
        label == 0,
    )

    fp_healthy = np.logical_and(
        predicted_mask,
        label == 1,
    )

    true_positive = np.logical_and(
        predicted_mask,
        label == 2,
    )

    stats["background_fp_voxels"] = int(
        np.sum(fp_background)
    )

    stats["healthy_fp_voxels"] = int(
        np.sum(fp_healthy)
    )

    stats["true_positive_voxels"] = int(
        np.sum(true_positive)
    )

    return stats


def analyze_case(model, case, device):
    image_path = evaluator.resolve_path(
        case["image"]
    )

    label_path = evaluator.resolve_path(
        case["label"]
    )

    image_nii = nib.load(
        str(image_path)
    )

    label_nii = nib.load(
        str(label_path)
    )

    image = image_nii.get_fdata(
        dtype=np.float32
    )

    label = label_nii.get_fdata(
        dtype=np.float32
    ).astype(np.uint8)

    if image.shape != label.shape:
        raise RuntimeError(
            f"Shape mismatch for {case['case_id']}: "
            f"{image.shape} vs {label.shape}"
        )

    # ---------------------------------------------------------
    # Full-volume inference ONCE
    # ---------------------------------------------------------

    probability_map = (
        evaluator.sliding_window_predict(
            model,
            image,
            device,
        )
    )

    prediction = build_prediction(
        probability_map,
        THRESHOLD,
    )

    predicted_pathology = (
        prediction == 2
    )

    stats = analyze_components(
        predicted_pathology,
        label,
    )

    gt_pathology_voxels = int(
        np.sum(label == 2)
    )

    stats["case_id"] = case["case_id"]
    stats["dataset"] = case.get(
        "dataset",
        "unknown",
    )
    stats["gt_pathology_voxels"] = (
        gt_pathology_voxels
    )

    if gt_pathology_voxels > 0:
        stats["pred_gt_ratio"] = (
            stats["predicted_pathology_voxels"]
            / gt_pathology_voxels
        )
    else:
        stats["pred_gt_ratio"] = 0.0

    return stats


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Evaluate first N validation cases. 0 = all 69.",
    )

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required."
        )

    device = torch.device("cuda")

    print("=" * 72)
    print("STAGE 2 PATHOLOGY COMPONENT DIAGNOSTIC")
    print("=" * 72)
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )
    print(
        f"Threshold: {THRESHOLD}"
    )
    print(
        f"Patch size: {evaluator.PATCH_SIZE}"
    )
    print(
        f"Overlap: {evaluator.OVERLAP}"
    )
    print()

    cases = (
        evaluator.find_validation_cases()
    )

    print(
        f"Validation cases found: {len(cases)}"
    )

    if args.limit > 0:
        cases = cases[:args.limit]

        print(
            f"LIMIT enabled: "
            f"evaluating {len(cases)} cases"
        )

    model = evaluator.load_model(
        device
    )

    all_results = []

    for index, case in enumerate(
        cases,
        start=1,
    ):

        print("-" * 72)

        print(
            f"[{index}/{len(cases)}] "
            f"{case.get('dataset', 'unknown')} / "
            f"{case['case_id']}"
        )

        result = analyze_case(
            model,
            case,
            device,
        )

        all_results.append(
            result
        )

        print(
            f"GT pathology: "
            f"{result['gt_pathology_voxels']:,}"
        )

        print(
            f"Predicted pathology: "
            f"{result['predicted_pathology_voxels']:,}"
        )

        print(
            f"Pred/GT ratio: "
            f"{result['pred_gt_ratio']:.3f}x"
        )

        print(
            f"Components: "
            f"{result['num_components']:,}"
        )

        print(
            f"Largest component: "
            f"{result['largest_component_voxels']:,} voxels"
        )

        print(
            f"TP: {result['true_positive_voxels']:,} | "
            f"FP healthy: {result['healthy_fp_voxels']:,} | "
            f"FP background: {result['background_fp_voxels']:,}"
        )

        print(
            f"Small components <=100 voxels: "
            f"{result['components_le_100_voxels']:,}"
        )

    # ---------------------------------------------------------
    # Aggregate statistics
    # ---------------------------------------------------------

    total_gt = sum(
        r["gt_pathology_voxels"]
        for r in all_results
    )

    total_pred = sum(
        r["predicted_pathology_voxels"]
        for r in all_results
    )

    total_tp = sum(
        r["true_positive_voxels"]
        for r in all_results
    )

    total_fp_healthy = sum(
        r["healthy_fp_voxels"]
        for r in all_results
    )

    total_fp_background = sum(
        r["background_fp_voxels"]
        for r in all_results
    )

    total_components = sum(
        r["num_components"]
        for r in all_results
    )

    print()
    print("=" * 72)
    print("COMPONENT DIAGNOSTIC SUMMARY")
    print("=" * 72)

    print(
        f"Cases evaluated: "
        f"{len(all_results)}"
    )

    print(
        f"GT pathology voxels: "
        f"{total_gt:,}"
    )

    print(
        f"Predicted pathology voxels: "
        f"{total_pred:,}"
    )

    if total_gt > 0:
        print(
            f"Overall Pred/GT: "
            f"{total_pred / total_gt:.3f}x"
        )

    print(
        f"True-positive voxels: "
        f"{total_tp:,}"
    )

    print(
        f"False-positive healthy voxels: "
        f"{total_fp_healthy:,}"
    )

    print(
        f"False-positive background voxels: "
        f"{total_fp_background:,}"
    )

    print(
        f"Total predicted components: "
        f"{total_components:,}"
    )

    print()

    print(
        "The component-size statistics above are "
        "diagnostic only; no component filtering "
        "has been applied."
    )

    # ---------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------

    csv_fields = [
        "case_id",
        "dataset",
        "gt_pathology_voxels",
        "predicted_pathology_voxels",
        "pred_gt_ratio",
        "num_components",
        "largest_component_voxels",
        "components_le_10_voxels",
        "components_le_50_voxels",
        "components_le_100_voxels",
        "components_le_500_voxels",
        "components_le_1000_voxels",
        "voxels_in_components_le_10",
        "voxels_in_components_le_50",
        "voxels_in_components_le_100",
        "voxels_in_components_le_500",
        "voxels_in_components_le_1000",
        "true_positive_voxels",
        "healthy_fp_voxels",
        "background_fp_voxels",
    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=csv_fields,
        )

        writer.writeheader()

        for result in all_results:
            writer.writerow(
                {
                    field: result[field]
                    for field in csv_fields
                }
            )

    print()
    print(
        f"CSV: {OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()