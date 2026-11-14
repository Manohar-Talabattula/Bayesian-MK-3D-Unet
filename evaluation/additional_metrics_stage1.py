from pathlib import Path
import sys
import numpy as np
import torch
import nibabel as nib

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.bayesian_mk_unet3d import BayesianMKUNet3D


# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT = (
    ROOT
    / "training"
    / "checkpoints_exp5_samples4"
    / "stage1_baseline_epoch71.pth"
)

VAL_IMAGES = ROOT / "datasets" / "stage1_preprocessed" / "val" / "images"
VAL_LABELS = ROOT / "datasets" / "stage1_preprocessed" / "val" / "labels"

OUTPUT_DIR = ROOT / "evaluation" / "additional_metrics"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 13

CLASS_NAMES = [
    "Background",
    "Spleen",
    "Right Kidney",
    "Left Kidney",
    "Gallbladder",
    "Esophagus",
    "Liver",
    "Stomach",
    "Aorta",
    "IVC",
    "Pancreas",
    "Right Adrenal",
    "Left Adrenal",
]


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("STAGE-1 ADDITIONAL SEGMENTATION METRICS")
print("=" * 70)
print(f"Device: {device}")

if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading model...")

model = BayesianMKUNet3D(
    in_channels=1,
    num_classes=NUM_CLASSES,
    base_channels=16,
    dropout=0.1,
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
    weights_only=False,
)

if "model_state" in checkpoint:
    model.load_state_dict(checkpoint["model_state"])
elif "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
elif "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"])
else:
    model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()

print("Model loaded successfully.")


# ============================================================
# METRIC STORAGE
# ============================================================

tp_total = np.zeros(NUM_CLASSES, dtype=np.float64)
fp_total = np.zeros(NUM_CLASSES, dtype=np.float64)
fn_total = np.zeros(NUM_CLASSES, dtype=np.float64)
tn_total = np.zeros(NUM_CLASSES, dtype=np.float64)

case_count = 0


# ============================================================
# PROCESS VALIDATION VOLUMES
# ============================================================

image_files = sorted(VAL_IMAGES.glob("*.nii.gz"))

print(f"\nValidation volumes: {len(image_files)}")
print("Processing...\n")


for index, image_path in enumerate(image_files, start=1):

    label_path = VAL_LABELS / image_path.name

    if not label_path.exists():
        print(f"[WARNING] Missing label: {image_path.name}")
        continue

    print(
        f"[{index}/{len(image_files)}] "
        f"{image_path.stem.replace('.nii', '')}"
    )

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image_nii = nib.load(str(image_path))
    image = image_nii.get_fdata(dtype=np.float32)

    # --------------------------------------------------------
    # Load ground truth
    # --------------------------------------------------------

    label_nii = nib.load(str(label_path))
    label = label_nii.get_fdata(dtype=np.float32).astype(np.int16)

    # --------------------------------------------------------
    # Tensor
    # --------------------------------------------------------

    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    image_tensor = image_tensor.to(device)

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    with torch.no_grad():

        output = model(image_tensor)

        if isinstance(output, dict):
            output = output["main"]
        elif isinstance(output, tuple):
            output = output[0]

        prediction = torch.argmax(output, dim=1)

    prediction = prediction.squeeze(0).cpu().numpy().astype(np.int16)

    # --------------------------------------------------------
    # Shape verification
    # --------------------------------------------------------

    if prediction.shape != label.shape:
        print(
            f"  [WARNING] Shape mismatch: "
            f"prediction={prediction.shape}, "
            f"label={label.shape}"
        )
        continue

    # --------------------------------------------------------
    # Confusion counts per class
    # --------------------------------------------------------

    for class_id in range(NUM_CLASSES):

        pred_class = prediction == class_id
        true_class = label == class_id

        tp_total[class_id] += np.logical_and(
            pred_class,
            true_class
        ).sum()

        fp_total[class_id] += np.logical_and(
            pred_class,
            ~true_class
        ).sum()

        fn_total[class_id] += np.logical_and(
            ~pred_class,
            true_class
        ).sum()

        tn_total[class_id] += np.logical_and(
            ~pred_class,
            ~true_class
        ).sum()

    case_count += 1


# ============================================================
# CALCULATE METRICS
# ============================================================

precision = np.divide(
    tp_total,
    tp_total + fp_total,
    out=np.zeros_like(tp_total),
    where=(tp_total + fp_total) > 0,
)

recall = np.divide(
    tp_total,
    tp_total + fn_total,
    out=np.zeros_like(tp_total),
    where=(tp_total + fn_total) > 0,
)

specificity = np.divide(
    tn_total,
    tn_total + fp_total,
    out=np.zeros_like(tn_total),
    where=(tn_total + fp_total) > 0,
)

f1 = np.divide(
    2 * precision * recall,
    precision + recall,
    out=np.zeros_like(precision),
    where=(precision + recall) > 0,
)


# ============================================================
# SAVE RESULTS
# ============================================================

output_file = OUTPUT_DIR / "additional_metrics_summary.txt"

with open(output_file, "w", encoding="utf-8") as f:

    f.write("STAGE-1 ADDITIONAL SEGMENTATION METRICS\n")
    f.write("=" * 70 + "\n\n")

    f.write(f"Validation cases: {case_count}\n")
    f.write(f"Checkpoint: {CHECKPOINT}\n\n")

    f.write(
        f"{'Class':<22}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'Specificity':>14}"
        f"{'F1':>12}\n"
    )

    f.write("-" * 72 + "\n")

    for class_id in range(1, NUM_CLASSES):

        f.write(
            f"{CLASS_NAMES[class_id]:<22}"
            f"{precision[class_id]:>12.4f}"
            f"{recall[class_id]:>12.4f}"
            f"{specificity[class_id]:>14.4f}"
            f"{f1[class_id]:>12.4f}\n"
        )

    foreground_precision = precision[1:].mean()
    foreground_recall = recall[1:].mean()
    foreground_specificity = specificity[1:].mean()
    foreground_f1 = f1[1:].mean()

    f.write("\n" + "-" * 72 + "\n")

    f.write(
        f"{'Mean Foreground':<22}"
        f"{foreground_precision:>12.4f}"
        f"{foreground_recall:>12.4f}"
        f"{foreground_specificity:>14.4f}"
        f"{foreground_f1:>12.4f}\n"
    )


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("ADDITIONAL METRICS")
print("=" * 70)

print(
    f"{'Class':<22}"
    f"{'Precision':>12}"
    f"{'Recall':>12}"
    f"{'Specificity':>14}"
    f"{'F1':>12}"
)

print("-" * 72)

for class_id in range(1, NUM_CLASSES):

    print(
        f"{CLASS_NAMES[class_id]:<22}"
        f"{precision[class_id]:>12.4f}"
        f"{recall[class_id]:>12.4f}"
        f"{specificity[class_id]:>14.4f}"
        f"{f1[class_id]:>12.4f}"
    )

foreground_precision = precision[1:].mean()
foreground_recall = recall[1:].mean()
foreground_specificity = specificity[1:].mean()
foreground_f1 = f1[1:].mean()

print("-" * 72)

print(
    f"{'Mean Foreground':<22}"
    f"{foreground_precision:>12.4f}"
    f"{foreground_recall:>12.4f}"
    f"{foreground_specificity:>14.4f}"
    f"{foreground_f1:>12.4f}"
)

print("\nCases successfully evaluated:", case_count)

print("\nResults saved to:")
print(output_file)

print("\n" + "=" * 70)
print("ADDITIONAL METRIC ANALYSIS COMPLETED")
print("=" * 70)