import os
import json
import csv
import math
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from dynamic_network_architectures.architectures.unet import PlainConvUNet


# ============================================================
# STAGE 2 FULL-VOLUME VALIDATION
# ============================================================

PROJECT_ROOT = Path(r"C:\Users\HPCLAB\Documents\MK_3D_UNET_PROJECT")

SPLIT_MANIFEST = PROJECT_ROOT / "datasets" / "stage2" / "split" / "stage2_split_manifest.json"
BEST_CHECKPOINT = PROJECT_ROOT / "datasets" / "stage2" / "models" / "stage2_best_balanced.pth"

OUTPUT_DIR = PROJECT_ROOT / "datasets" / "stage2" / "evaluation"
OUTPUT_JSON = OUTPUT_DIR / "stage2_full_volume_results.json"
OUTPUT_CSV = OUTPUT_DIR / "stage2_full_volume_case_metrics.csv"

PATCH_SIZE = (32, 96, 96)
OVERLAP = 0.50
INFER_BATCH_SIZE = 2
NUM_CLASSES = 3

THRESHOLDS = [
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
]

# Preprocessed Stage-2 images are already normalized to [0, 1].
# Do NOT perform HU normalization again here.


def build_model():
    return PlainConvUNet(
        input_channels=1,
        n_stages=6,
        features_per_stage=[32, 64, 128, 256, 320, 320],
        conv_op=torch.nn.Conv3d,
        kernel_sizes=[
            [1, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
        ],
        strides=[
            [1, 1, 1],
            [1, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
        ],
        n_conv_per_stage=[2, 2, 2, 2, 2, 2],
        num_classes=NUM_CLASSES,
        n_conv_per_stage_decoder=[2, 2, 2, 2, 2],
        conv_bias=True,
        norm_op=torch.nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=torch.nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=True,
    )


def load_model(device):
    print("Building Stage-2 model...")
    model = build_model()

    print(f"Loading checkpoint:\n{BEST_CHECKPOINT}")
    checkpoint = torch.load(BEST_CHECKPOINT, map_location="cpu", weights_only=False)

    # stage2_best_balanced.pth is produced by train_stage2.py and stores
    # the PyTorch model under "model_state_dict".
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "network_weights" in checkpoint:
        state_dict = checkpoint["network_weights"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    print("Checkpoint loaded successfully.")
    return model


def find_validation_cases():
    with open(SPLIT_MANIFEST, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    cases = []

    def add_case(item, dataset="unknown"):
        if not isinstance(item, dict):
            return False

        image = item.get("image") or item.get("image_path")
        label = item.get("label") or item.get("label_path")

        if isinstance(image, str) and isinstance(label, str):
            cases.append({
                "image": image,
                "label": label,
                "dataset": item.get("dataset", dataset),
                "case_id": item.get("case_id", Path(image).stem),
            })
            return True

        return False

    def collect_under_validation(obj, dataset="unknown"):
        """
        Collect cases only from the validation branch of the manifest.
        Supports:
          val: [...]
          val: {dataset_name: [...]}
          validation: [...]
          validation: {dataset_name: [...]}
        """
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    if not add_case(item, dataset):
                        collect_under_validation(item, dataset)
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    if isinstance(item[0], str) and isinstance(item[1], str):
                        cases.append({
                            "image": item[0],
                            "label": item[1],
                            "dataset": dataset,
                            "case_id": Path(item[0]).stem,
                        })
                elif isinstance(item, dict):
                    collect_under_validation(item, dataset)

        elif isinstance(obj, dict):
            if add_case(obj, dataset):
                return

            for key, value in obj.items():
                collect_under_validation(value, str(key))

    # Prefer explicit validation/val branches.
    if isinstance(manifest, dict):
        for key, value in manifest.items():
            key_lower = str(key).lower()
            if key_lower in {
                "val",
                "validation",
                "validation_cases",
                "val_cases",
                "validation_set",
            }:
                collect_under_validation(value)

    # Some split manifests use a list of records with a split field.
    if not cases:
        def collect_split_records(obj):
            if isinstance(obj, list):
                for item in obj:
                    collect_split_records(item)
            elif isinstance(obj, dict):
                split = str(
                    obj.get("split", obj.get("subset", ""))
                ).lower()

                if split in {"val", "validation"}:
                    add_case(obj, obj.get("dataset", "unknown"))

                for value in obj.values():
                    if isinstance(value, (dict, list)):
                        collect_split_records(value)

        collect_split_records(manifest)

    # Deduplicate.
    unique = {}
    for c in cases:
        unique[str(c["image"])] = c

    cases = list(unique.values())

    if len(cases) != 69:
        raise RuntimeError(
            f"Expected exactly 69 validation cases from the split manifest, "
            f"but found {len(cases)}.\n"
            f"Manifest: {SPLIT_MANIFEST}\n"
            "No evaluation was performed. This protects against accidentally "
            "evaluating the training cases."
        )

    return cases

def resolve_path(path_string):
    p = Path(path_string)

    if p.exists():
        return p

    # Manifest may contain paths relative to the project root.
    candidate = PROJECT_ROOT / p
    if candidate.exists():
        return candidate

    # Try converting forward/backslashes cleanly.
    candidate = PROJECT_ROOT / str(path_string).replace("/", os.sep)
    if candidate.exists():
        return candidate

    raise FileNotFoundError(f"Cannot resolve path: {path_string}")


def make_starts(size, patch, overlap):
    if size <= patch:
        return [0]

    step = max(1, int(round(patch * (1.0 - overlap))))
    starts = list(range(0, size - patch + 1, step))

    last = size - patch
    if starts[-1] != last:
        starts.append(last)

    return starts


def gaussian_importance_map(shape, sigma_scale=1.0 / 8.0):
    """
    Smooth center-weighted importance map used to blend overlapping
    sliding-window predictions.
    """
    axes = []
    for size in shape:
        center = (size - 1) / 2.0
        sigma = max(size * sigma_scale, 1.0)
        x = np.arange(size, dtype=np.float32)
        g = np.exp(-0.5 * ((x - center) / sigma) ** 2)
        axes.append(g)

    weight = (
        axes[0][:, None, None]
        * axes[1][None, :, None]
        * axes[2][None, None, :]
    )

    weight = weight / np.max(weight)
    weight = np.maximum(weight, 1e-4)
    return weight.astype(np.float32)


def sliding_window_predict(
    model,
    image,
    device,
):
    """
    image: numpy array [Z, Y, X], already normalized to [0,1].

    Returns:
        Probability map with shape [3, Z, Y, X].
        Class 0 = background.
        Class 1 = healthy/non-lesional tissue.
        Class 2 = pathology.
    """

    image = image.astype(np.float32, copy=False)

    original_shape = image.shape
    pad_width = []

    for size, patch in zip(original_shape, PATCH_SIZE):
        pad_width.append(
            (0, max(0, patch - size))
        )

    if any(right > 0 for _, right in pad_width):
        padded = np.pad(
            image,
            pad_width,
            mode="constant",
            constant_values=0.0,
        )
    else:
        padded = image

    padded_shape = padded.shape

    z_starts = make_starts(
        padded_shape[0],
        PATCH_SIZE[0],
        OVERLAP,
    )

    y_starts = make_starts(
        padded_shape[1],
        PATCH_SIZE[1],
        OVERLAP,
    )

    x_starts = make_starts(
        padded_shape[2],
        PATCH_SIZE[2],
        OVERLAP,
    )

    locations = [
        (z, y, x)
        for z in z_starts
        for y in y_starts
        for x in x_starts
    ]

    importance = gaussian_importance_map(PATCH_SIZE)

    probability_sum = np.zeros(
        (NUM_CLASSES,) + padded_shape,
        dtype=np.float32,
    )

    weight_sum = np.zeros(
        padded_shape,
        dtype=np.float32,
    )

    with torch.inference_mode():

        for start in range(
            0,
            len(locations),
            INFER_BATCH_SIZE,
        ):

            batch_locations = locations[
                start:start + INFER_BATCH_SIZE
            ]

            patches = np.stack([
                padded[
                    z:z + PATCH_SIZE[0],
                    y:y + PATCH_SIZE[1],
                    x:x + PATCH_SIZE[2],
                ]
                for z, y, x in batch_locations
            ])

            tensor = (
                torch.from_numpy(patches)
                .unsqueeze(1)
                .to(device, non_blocking=True)
            )

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):

                output = model(tensor)

                # Deep-supervision model:
                # highest-resolution output is first.
                if isinstance(output, (list, tuple)):
                    output = output[0]

                probabilities = torch.softmax(
                    output,
                    dim=1,
                )

            probabilities = (
                probabilities
                .float()
                .cpu()
                .numpy()
            )

            for b, (z, y, x) in enumerate(
                batch_locations
            ):

                probability_sum[
                    :,
                    z:z + PATCH_SIZE[0],
                    y:y + PATCH_SIZE[1],
                    x:x + PATCH_SIZE[2],
                ] += (
                    probabilities[b]
                    * importance
                )

                weight_sum[
                    z:z + PATCH_SIZE[0],
                    y:y + PATCH_SIZE[1],
                    x:x + PATCH_SIZE[2],
                ] += importance

    # Convert accumulated probabilities into
    # Gaussian-blended average probabilities.
    probability_sum /= np.maximum(
        weight_sum[None],
        1e-8,
    )

    # Original prediction:
    # whichever of the 3 classes has the highest probability.
    prediction = np.argmax(
        probability_sum,
        axis=0,
    ).astype(np.uint8)

    # ---------------------------------------------------------
    # Pathology confidence threshold
    # ---------------------------------------------------------
    #
    # Only modify voxels where pathology (Class 2)
    # would otherwise win the argmax.
    #
    # If pathology probability is below the threshold,
    # choose between background and healthy instead.
    #
    # This specifically targets the background leakage
    # discovered in the previous evaluation.
    # ---------------------------------------------------------
    probability_sum = probability_sum / (
        weight_sum[None, ...] + 1e-8
    )

    # ---------------------------------------------------------
    # Remove padding.
    # Keep the class dimension:
    #   [classes, Z, Y, X]
    # ---------------------------------------------------------

    probability_sum = probability_sum[
        :,
        :original_shape[0],
        :original_shape[1],
        :original_shape[2],
    ]

    return probability_sum


def binary_metrics(pred, target):
    pred = pred.astype(bool)
    target = target.astype(bool)

    tp = np.logical_and(pred, target).sum(dtype=np.int64)
    fp = np.logical_and(pred, ~target).sum(dtype=np.int64)
    fn = np.logical_and(~pred, target).sum(dtype=np.int64)
    tn = np.logical_and(~pred, ~target).sum(dtype=np.int64)

    dice_den = 2 * tp + fp + fn
    iou_den = tp + fp + fn

    dice = (2 * tp / dice_den) if dice_den else 1.0
    iou = (tp / iou_den) if iou_den else 1.0
    precision = (tp / (tp + fp)) if (tp + fp) else 1.0
    recall = (tp / (tp + fn)) if (tp + fn) else 1.0
    specificity = (tn / (tn + fp)) if (tn + fp) else 1.0

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def evaluate_case(model, case, device):
    image_path = resolve_path(case["image"])
    label_path = resolve_path(case["label"])

    image_nii = nib.load(str(image_path))
    label_nii = nib.load(str(label_path))

    image = image_nii.get_fdata(dtype=np.float32)
    label = label_nii.get_fdata(dtype=np.float32).astype(np.uint8)

    if image.shape != label.shape:
        raise RuntimeError(
            f"Shape mismatch for {case['case_id']}: "
            f"{image.shape} vs {label.shape}"
        )

    image_min = float(np.min(image))
    image_max = float(np.max(image))

    if image_min < -0.01 or image_max > 1.01:
        raise RuntimeError(
            f"Image {case['case_id']} is not normalized to [0,1]. "
            f"Range={image_min:.4f}..{image_max:.4f}"
        )

    # ---------------------------------------------------------
    # ONE full-volume neural-network inference
    # ---------------------------------------------------------

    probability_map = sliding_window_predict(
        model,
        image,
        device,
    )

    # probability_map shape:
    # [3, Z, Y, X]

    if probability_map.shape[0] != NUM_CLASSES:
        raise RuntimeError(
            f"Unexpected probability map shape: "
            f"{probability_map.shape}"
        )

    # ---------------------------------------------------------
    # Base case information
    # ---------------------------------------------------------

    base_result = {
        "case_id": case["case_id"],
        "dataset": case.get("dataset", "unknown"),
        "shape": list(image.shape),
        "spacing_mm": [
            float(x)
            for x in image_nii.header.get_zooms()[:3]
        ],
        "ground_truth_pathology_voxels": int(
            np.sum(label == 2)
        ),
    }

    threshold_results = {}

    # ---------------------------------------------------------
    # Evaluate all thresholds using the SAME probability map
    # ---------------------------------------------------------

    for threshold in THRESHOLDS:

        # Start with normal 3-class argmax.
        prediction = np.argmax(
            probability_map,
            axis=0,
        ).astype(np.uint8)

        pathology_probability = probability_map[2]

        # If pathology wins argmax but does not have enough
        # confidence, choose between background and healthy.
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

        # -----------------------------------------------------
        # Class metrics
        # -----------------------------------------------------

        class_metrics = {}

        for class_id, name in [
            (0, "background"),
            (1, "healthy_non_lesional"),
            (2, "pathology"),
        ]:
            m = binary_metrics(
                prediction == class_id,
                label == class_id,
            )

            class_metrics[name] = m

        pathology = class_metrics["pathology"]
        healthy = class_metrics["healthy_non_lesional"]

        result = {
            "background_dice":
                class_metrics["background"]["dice"],

            "healthy_dice":
                healthy["dice"],

            "pathology_dice":
                pathology["dice"],

            "foreground_dice":
                (
                    healthy["dice"]
                    + pathology["dice"]
                ) / 2.0,

            "pathology_iou":
                pathology["iou"],

            "pathology_precision":
                pathology["precision"],

            "pathology_recall":
                pathology["recall"],

            "pathology_specificity":
                pathology["specificity"],
        }

        # -----------------------------------------------------
        # Pathology voxel counts
        # -----------------------------------------------------

        ground_truth_pathology = (
            label == 2
        )

        predicted_pathology = (
            prediction == 2
        )

        result[
            "ground_truth_pathology_voxels"
        ] = int(
            np.sum(ground_truth_pathology)
        )

        result[
            "predicted_pathology_voxels"
        ] = int(
            np.sum(predicted_pathology)
        )

        # -----------------------------------------------------
        # False-positive breakdown
        # -----------------------------------------------------

        false_positive_healthy = np.logical_and(
            predicted_pathology,
            label == 1,
        )

        false_positive_background = np.logical_and(
            predicted_pathology,
            label == 0,
        )

        true_positive_pathology = np.logical_and(
            predicted_pathology,
            label == 2,
        )

        result[
            "pathology_true_positive_voxels"
        ] = int(
            np.sum(true_positive_pathology)
        )

        result[
            "pathology_false_positive_healthy_voxels"
        ] = int(
            np.sum(false_positive_healthy)
        )

        result[
            "pathology_false_positive_background_voxels"
        ] = int(
            np.sum(false_positive_background)
        )

        # Sanity check
        predicted_breakdown = (
            result[
                "pathology_true_positive_voxels"
            ]
            + result[
                "pathology_false_positive_healthy_voxels"
            ]
            + result[
                "pathology_false_positive_background_voxels"
            ]
        )

        if (
            predicted_breakdown
            != result["predicted_pathology_voxels"]
        ):
            raise RuntimeError(
                f"Pathology voxel breakdown mismatch "
                f"for {case['case_id']} at threshold "
                f"{threshold}: "
                f"predicted="
                f"{result['predicted_pathology_voxels']}, "
                f"breakdown={predicted_breakdown}"
            )

        threshold_results[
            threshold
        ] = result

    base_result["thresholds"] = threshold_results

    return base_result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Evaluate only the first N validation cases. 0 = all 69.",
    )

    args = parser.parse_args()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this full-volume evaluation."
        )

    device = torch.device("cuda")

    print("=" * 72)
    print("STAGE 2 FULL-VOLUME THRESHOLD SWEEP")
    print("=" * 72)

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

    print(
        f"Patch size: {PATCH_SIZE}"
    )

    print(
        f"Overlap: {OVERLAP}"
    )

    print(
        f"Inference batch size: {INFER_BATCH_SIZE}"
    )

    print(
        f"Thresholds: {THRESHOLDS}"
    )

    print()

    cases = find_validation_cases()

    print(
        f"Validation cases found: {len(cases)}"
    )

    if args.limit > 0:
        cases = cases[:args.limit]

        print(
            f"LIMIT enabled: evaluating "
            f"{len(cases)} cases"
        )

    model = load_model(device)

    all_results = []

    # ---------------------------------------------------------
    # One inference pass per case
    # ---------------------------------------------------------

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

        result = evaluate_case(
            model,
            case,
            device,
        )

        all_results.append(result)

        # Show pathology Dice for each threshold.
        threshold_text = " | ".join(
            [
                (
                    f"{threshold:.2f}: "
                    f"D={result['thresholds'][threshold]['pathology_dice']:.3f}"
                )
                for threshold in THRESHOLDS
            ]
        )

        print(
            f"Pathology Dice by threshold: "
            f"{threshold_text}"
        )

    # ---------------------------------------------------------
    # Aggregate results
    # ---------------------------------------------------------

    print()
    print("=" * 72)
    print("THRESHOLD SWEEP RESULTS")
    print("=" * 72)

    sweep_summary = []

    for threshold in THRESHOLDS:

        threshold_results = [
            result["thresholds"][threshold]
            for result in all_results
        ]

        def mean(key):
            return float(
                np.mean(
                    [
                        r[key]
                        for r in threshold_results
                    ]
                )
            )

        gt_voxels = sum(
            r["ground_truth_pathology_voxels"]
            for r in threshold_results
        )

        predicted_voxels = sum(
            r["predicted_pathology_voxels"]
            for r in threshold_results
        )

        true_positive = sum(
            r["pathology_true_positive_voxels"]
            for r in threshold_results
        )

        false_positive_healthy = sum(
            r[
                "pathology_false_positive_healthy_voxels"
            ]
            for r in threshold_results
        )

        false_positive_background = sum(
            r[
                "pathology_false_positive_background_voxels"
            ]
            for r in threshold_results
        )

        volume_ratio = (
            predicted_voxels / gt_voxels
            if gt_voxels > 0
            else 0.0
        )

        summary_row = {
            "threshold": threshold,
            "num_cases": len(threshold_results),

            "mean_background_dice":
                mean("background_dice"),

            "mean_healthy_dice":
                mean("healthy_dice"),

            "mean_pathology_dice":
                mean("pathology_dice"),

            "mean_foreground_dice":
                mean("foreground_dice"),

            "mean_pathology_iou":
                mean("pathology_iou"),

            "mean_pathology_precision":
                mean("pathology_precision"),

            "mean_pathology_recall":
                mean("pathology_recall"),

            "mean_pathology_specificity":
                mean("pathology_specificity"),

            "ground_truth_pathology_voxels":
                gt_voxels,

            "predicted_pathology_voxels":
                predicted_voxels,

            "prediction_gt_volume_ratio":
                volume_ratio,

            "true_positive_voxels":
                true_positive,

            "false_positive_healthy_voxels":
                false_positive_healthy,

            "false_positive_background_voxels":
                false_positive_background,
        }

        sweep_summary.append(
            summary_row
        )

        print(
            f"Threshold {threshold:.2f} | "
            f"Dice={summary_row['mean_pathology_dice']:.5f} | "
            f"IoU={summary_row['mean_pathology_iou']:.5f} | "
            f"Precision={summary_row['mean_pathology_precision']:.5f} | "
            f"Recall={summary_row['mean_pathology_recall']:.5f} | "
            f"Pred/GT={summary_row['prediction_gt_volume_ratio']:.3f}x"
        )

    # ---------------------------------------------------------
    # Best threshold
    # ---------------------------------------------------------

    best = max(
        sweep_summary,
        key=lambda x: x["mean_pathology_dice"],
    )

    print()
    print("-" * 72)
    print(
        "BEST THRESHOLD BY MEAN PATHOLOGY DICE"
    )
    print("-" * 72)

    print(
        f"Threshold:  {best['threshold']:.2f}"
    )

    print(
        f"Dice:       "
        f"{best['mean_pathology_dice']:.5f}"
    )

    print(
        f"IoU:        "
        f"{best['mean_pathology_iou']:.5f}"
    )

    print(
        f"Precision:  "
        f"{best['mean_pathology_precision']:.5f}"
    )

    print(
        f"Recall:     "
        f"{best['mean_pathology_recall']:.5f}"
    )

    print(
        f"Pred/GT:    "
        f"{best['prediction_gt_volume_ratio']:.3f}x"
    )

    # ---------------------------------------------------------
    # Save sweep JSON
    # ---------------------------------------------------------

    sweep_json = (
        OUTPUT_DIR
        / "stage2_balanced_threshold_sweep_results.json"
    )

    sweep_output = {
        "evaluation_type":
            "full_volume_threshold_sweep",

        "checkpoint":
            str(BEST_CHECKPOINT),

        "num_cases":
            len(all_results),

        "patch_size":
            list(PATCH_SIZE),

        "overlap":
            OVERLAP,

        "inference_batch_size":
            INFER_BATCH_SIZE,

        "thresholds":
            THRESHOLDS,

        "best_threshold":
            best,

        "summary":
            sweep_summary,

        "cases":
            all_results,
    }

    with open(
        sweep_json,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            sweep_output,
            f,
            indent=2,
        )

    # ---------------------------------------------------------
    # Save threshold summary CSV
    # ---------------------------------------------------------

    sweep_csv = (
        OUTPUT_DIR
        / "stage2_balanced_threshold_sweep_summary.csv"
    )

    csv_fields = [
        "threshold",
        "num_cases",
        "mean_background_dice",
        "mean_healthy_dice",
        "mean_pathology_dice",
        "mean_foreground_dice",
        "mean_pathology_iou",
        "mean_pathology_precision",
        "mean_pathology_recall",
        "mean_pathology_specificity",
        "ground_truth_pathology_voxels",
        "predicted_pathology_voxels",
        "prediction_gt_volume_ratio",
        "true_positive_voxels",
        "false_positive_healthy_voxels",
        "false_positive_background_voxels",
    ]

    with open(
        sweep_csv,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=csv_fields,
        )

        writer.writeheader()

        for row in sweep_summary:
            writer.writerow(row)

    print()
    print("=" * 72)
    print("THRESHOLD SWEEP FINISHED")
    print("=" * 72)

    print(
        f"JSON: {sweep_json}"
    )

    print(
        f"CSV : {sweep_csv}"
    )


if __name__ == "__main__":
    main()

