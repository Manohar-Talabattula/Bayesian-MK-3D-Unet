import sys
from pathlib import Path

import torch
from monai.inferers import sliding_window_inference

# --------------------------------------------------
# Make project root importable
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.stage1_dataloader import build_stage1_dataloader
from models.bayesian_mk_unet3d import BayesianMKUNet3D


# ==================================================
# Configuration
# ==================================================

DATA_ROOT = ROOT / "datasets" / "stage1_preprocessed"
CHECKPOINT = ROOT / "training" / "checkpoints_exp5_samples4" / "stage1_baseline_epoch71.pth"

NUM_CLASSES = 13
PATCH_SIZE = (96, 96, 64)

ORGAN_NAMES = {
    1: "Spleen",
    2: "Right Kidney",
    3: "Left Kidney",
    4: "Gallbladder",
    5: "Esophagus",
    6: "Liver",
    7: "Stomach",
    8: "Aorta",
    9: "Inferior Vena Cava",
    10: "Pancreas",
    11: "Right Adrenal Gland",
    12: "Left Adrenal Gland",
}


# ==================================================
# Device
# ==================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("STAGE-1 ORGAN-WISE EVALUATION")
print("=" * 70)

print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ==================================================
# Checkpoint
# ==================================================

if not CHECKPOINT.exists():
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT}"
    )

print("\nLoading checkpoint:")
print(CHECKPOINT)


# ==================================================
# Validation DataLoader
# ==================================================

print("\nBuilding validation DataLoader...")

val_loader = build_stage1_dataloader(
    data_root=DATA_ROOT,
    split="val",
    batch_size=1,
    num_workers=0
)

print("Validation DataLoader ready.")


# ==================================================
# Model
# ==================================================

print("\nBuilding Bayesian MK-3D-UNet...")

model = BayesianMKUNet3D(
    in_channels=1,
    num_classes=NUM_CLASSES,
    base_channels=16,
    dropout=0.1
).to(device)


# ==================================================
# Load best model
# ==================================================

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state"]
)

model.eval()

print(
    "\nBest checkpoint epoch:",
    checkpoint.get("epoch")
)

print(
    "Best validation Dice:",
    checkpoint.get("best_val_dice")
)


# ==================================================
# Per-organ statistics
# ==================================================

dice_sum = {
    class_id: 0.0
    for class_id in ORGAN_NAMES
}

iou_sum = {
    class_id: 0.0
    for class_id in ORGAN_NAMES
}

count = {
    class_id: 0
    for class_id in ORGAN_NAMES
}

prediction_count = {
    class_id: 0
    for class_id in ORGAN_NAMES
}

label_count = {
    class_id: 0
    for class_id in ORGAN_NAMES
}


# ==================================================
# Evaluation
# ==================================================

print("\nRunning validation evaluation...")

with torch.no_grad():

    for batch_index, batch in enumerate(val_loader, start=1):

        images = batch["image"].to(
            device,
            non_blocking=True
        )

        labels = batch["label"].to(
            device,
            non_blocking=True
        )

        labels = labels.long()

        if labels.ndim == 5:
            labels = labels.squeeze(1)

        logits = sliding_window_inference(
            inputs=images,
            roi_size=PATCH_SIZE,
            sw_batch_size=1,
            predictor=lambda x: model(x)["main"],
            overlap=0.25
        )

        predictions = torch.argmax(
            logits,
            dim=1
        )

        for class_id in ORGAN_NAMES:

            prediction_mask = (
                predictions == class_id
            )

            label_mask = (
                labels == class_id
            )

            intersection = (
                prediction_mask & label_mask
            ).sum().float()

            prediction_pixels = (
                prediction_mask.sum().item()
            )

            label_pixels = (
                label_mask.sum().item()
            )

            union = (
                prediction_mask | label_mask
            ).sum().float()

            denominator = (
                prediction_pixels + label_pixels
            )

            prediction_count[class_id] += prediction_pixels
            label_count[class_id] += label_pixels

            if denominator == 0:
                continue

            dice = (
                2.0 * intersection.item()
                / (denominator + 1e-8)
            )

            if union.item() == 0:
                iou = 1.0
            else:
                iou = (
                    intersection.item()
                    / (union.item() + 1e-8)
                )

            dice_sum[class_id] += dice
            iou_sum[class_id] += iou
            count[class_id] += 1

        print(
            f"Validated volume {batch_index}/"
            f"{len(val_loader)}",
            end="\r"
        )


# ==================================================
# Results
# ==================================================

print("\n\n" + "=" * 70)
print("ORGAN-WISE RESULTS")
print("=" * 70)

print(
    f"{'Organ':25s}"
    f"{'Dice':>10s}"
    f"{'IoU':>10s}"
    f"{'Predicted':>15s}"
    f"{'Ground Truth':>15s}"
)

print("-" * 70)

valid_dice = []
valid_iou = []

for class_id, organ_name in ORGAN_NAMES.items():

    if count[class_id] > 0:

        mean_dice = (
            dice_sum[class_id]
            / count[class_id]
        )

        mean_iou = (
            iou_sum[class_id]
            / count[class_id]
        )

        valid_dice.append(mean_dice)
        valid_iou.append(mean_iou)

        print(
            f"{organ_name:25s}"
            f"{mean_dice:10.4f}"
            f"{mean_iou:10.4f}"
            f"{prediction_count[class_id]:15,d}"
            f"{label_count[class_id]:15,d}"
        )

    else:

        print(
            f"{organ_name:25s}"
            f"{'N/A':>10s}"
            f"{'N/A':>10s}"
            f"{prediction_count[class_id]:15,d}"
            f"{label_count[class_id]:15,d}"
        )


# ==================================================
# Mean foreground metrics
# ==================================================

print("-" * 70)

if valid_dice:

    print(
        f"{'Mean Foreground Dice':25s}"
        f"{sum(valid_dice) / len(valid_dice):10.4f}"
    )

if valid_iou:

    print(
        f"{'Mean Foreground IoU':25s}"
        f"{sum(valid_iou) / len(valid_iou):10.4f}"
    )

print("=" * 70)
print("Evaluation complete.")
