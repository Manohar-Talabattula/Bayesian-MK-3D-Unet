import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceCELoss
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
CHECKPOINT_DIR = ROOT / "training" / "checkpoints"

NUM_CLASSES = 13

PATCH_SIZE = (96, 96, 64)

BATCH_SIZE = 1
NUM_SAMPLES = 2

EPOCHS = 20

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

DEEP_SUPERVISION_WEIGHT = 0.2

VAL_EVERY = 1

NUM_WORKERS = 0

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==================================================
# Device
# ==================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 65)
print("BAYESIAN MK-3D-UNET — STAGE 1 TRAINING")
print("=" * 65)

print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "GPU memory:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / (1024 ** 3),
            2
        ),
        "GB"
    )


# ==================================================
# DataLoaders
# ==================================================

print("\nBuilding training DataLoader...")

train_loader = build_stage1_dataloader(
    data_root=DATA_ROOT,
    split="train",
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS
)

print("Training DataLoader ready.")

print("\nBuilding validation DataLoader...")

val_loader = build_stage1_dataloader(
    data_root=DATA_ROOT,
    split="val",
    batch_size=1,
    num_workers=NUM_WORKERS
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

parameter_count = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print(
    "Trainable parameters:",
    f"{parameter_count:,}"
)


# ==================================================
# Loss
# ==================================================

criterion = DiceCELoss(
    to_onehot_y=True,
    softmax=True,
    squared_pred=True,
    jaccard=False
)


# ==================================================
# Optimizer
# ==================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ==================================================
# Learning-rate scheduler
# ==================================================

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)


# ==================================================
# AMP
# ==================================================

use_amp = device.type == "cuda"

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=use_amp
)


# ==================================================
# Dice / IoU calculation
# ==================================================

def calculate_metrics(logits, labels):

    predictions = torch.argmax(
        logits,
        dim=1
    )

    labels = labels.long()

    if labels.ndim == 5:
        labels = labels.squeeze(1)

    dice_scores = []
    iou_scores = []

    for class_id in range(1, NUM_CLASSES):

        prediction_mask = (
            predictions == class_id
        )

        label_mask = (
            labels == class_id
        )

        intersection = (
            prediction_mask & label_mask
        ).sum().float()

        prediction_sum = (
            prediction_mask.sum().float()
        )

        label_sum = (
            label_mask.sum().float()
        )

        union = (
            prediction_mask | label_mask
        ).sum().float()

        denominator = (
            prediction_sum + label_sum
        )

        # If the organ does not appear in either
        # prediction or ground truth, skip it.
        if denominator == 0:
            continue

        dice = (
            2.0 * intersection
            / (denominator + 1e-8)
        )

        if union == 0:
            iou = torch.tensor(
                1.0,
                device=device
            )
        else:
            iou = (
                intersection
                / (union + 1e-8)
            )

        dice_scores.append(dice)
        iou_scores.append(iou)

    if not dice_scores:
        return 0.0, 0.0

    mean_dice = torch.stack(
        dice_scores
    ).mean().item()

    mean_iou = torch.stack(
        iou_scores
    ).mean().item()

    return mean_dice, mean_iou


# ==================================================
# Resume support
# ==================================================

latest_checkpoint = CHECKPOINT_DIR / "latest.pth"

start_epoch = 1
best_val_dice = 0.0

if latest_checkpoint.exists():

    print(
        "\nCheckpoint found:",
        latest_checkpoint
    )

    checkpoint = torch.load(
        latest_checkpoint,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state"]
    )

    scheduler.load_state_dict(
        checkpoint["scheduler_state"]
    )

    if "scaler_state" in checkpoint:
        scaler.load_state_dict(
            checkpoint["scaler_state"]
        )

    start_epoch = (
        checkpoint["epoch"] + 1
    )

    best_val_dice = checkpoint.get(
        "best_val_dice",
        0.0
    )

    print(
        "Resuming from epoch:",
        start_epoch
    )

    print(
        "Previous best Dice:",
        best_val_dice
    )


# ==================================================
# Training
# ==================================================

for epoch in range(
    start_epoch,
    EPOCHS + 1
):

    model.train()

    running_loss = 0.0
    batch_count = 0

    print("\n" + "=" * 65)
    print(
        f"Epoch {epoch}/{EPOCHS}"
    )
    print("=" * 65)

    for batch in train_loader:

        images = batch["image"].to(
            device,
            non_blocking=True
        )

        labels = batch["label"].to(
            device,
            non_blocking=True
        )

        labels = labels.long()

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp
        ):

            outputs = model(images)

            main_logits = outputs["main"]

            main_loss = criterion(
                main_logits,
                labels
            )

            # Deep supervision
            aux2 = F.interpolate(
                outputs["aux2"],
                size=main_logits.shape[2:],
                mode="trilinear",
                align_corners=False
            )

            aux3 = F.interpolate(
                outputs["aux3"],
                size=main_logits.shape[2:],
                mode="trilinear",
                align_corners=False
            )

            aux4 = F.interpolate(
                outputs["aux4"],
                size=main_logits.shape[2:],
                mode="trilinear",
                align_corners=False
            )

            aux_loss = (
                criterion(aux2, labels)
                + criterion(aux3, labels)
                + criterion(aux4, labels)
            ) / 3.0

            loss = (
                main_loss
                + DEEP_SUPERVISION_WEIGHT * aux_loss
            )

        scaler.scale(loss).backward()

        scaler.step(
            optimizer
        )

        scaler.update()

        running_loss += loss.item()
        batch_count += 1

        if batch_count % 10 == 0:

            print(
                f"Batch {batch_count} | "
                f"Loss: {loss.item():.4f}"
            )

    epoch_loss = (
        running_loss / max(batch_count, 1)
    )

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"\nEpoch {epoch} training loss: "
        f"{epoch_loss:.4f}"
    )

    print(
        f"Learning rate: {current_lr:.8f}"
    )


    # ==================================================
    # Validation
    # ==================================================

    if epoch % VAL_EVERY == 0:

        model.eval()

        val_loss = 0.0
        val_dice = 0.0
        val_iou = 0.0
        val_count = 0

        print("\nRunning validation...")

        with torch.no_grad():

            for batch in val_loader:

                images = batch["image"].to(
                    device,
                    non_blocking=True
                )

                labels = batch["label"].to(
                    device,
                    non_blocking=True
                )

                labels = labels.long()

                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=use_amp
                ):

                    logits = sliding_window_inference(
                        inputs=images,
                        roi_size=PATCH_SIZE,
                        sw_batch_size=1,
                        predictor=lambda x: model(x)["main"],
                        overlap=0.25
                    )

                    loss = criterion(
                        logits,
                        labels
                    )

                dice, iou = calculate_metrics(
                    logits,
                    labels
                )

                val_loss += loss.item()
                val_dice += dice
                val_iou += iou

                val_count += 1

        val_loss /= max(val_count, 1)
        val_dice /= max(val_count, 1)
        val_iou /= max(val_count, 1)

        print(
            f"Validation loss: {val_loss:.4f}"
        )

        print(
            f"Validation Dice: {val_dice:.4f}"
        )

        print(
            f"Validation IoU: {val_iou:.4f}"
        )


    # ==================================================
    # Best checkpoint
    # ==================================================

    if val_dice > best_val_dice:

        best_val_dice = val_dice

    best_checkpoint = (
        CHECKPOINT_DIR / "best.pth"
    )

    temp_best_checkpoint = (
        CHECKPOINT_DIR / "best_tmp.pth"
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_val_dice": best_val_dice
        },
        temp_best_checkpoint
    )

    if best_checkpoint.exists():
        best_checkpoint.unlink()

    temp_best_checkpoint.replace(best_checkpoint)

    print(
        "New best model saved:",
        best_checkpoint
    )

    # ==================================================
    # Latest checkpoint
    # ==================================================

    temp_checkpoint = CHECKPOINT_DIR / "latest_tmp.pth"

    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_val_dice": best_val_dice
        },
        temp_checkpoint
    )

    if latest_checkpoint.exists():
        latest_checkpoint.unlink()

    temp_checkpoint.replace(latest_checkpoint)

    print(
        "Latest checkpoint saved."
    )

    scheduler.step()


# ==================================================
# Finished
# ==================================================

print("\n" + "=" * 65)
print("STAGE-1 TRAINING COMPLETED")
print("=" * 65)

print(
    "Best validation Dice:",
    f"{best_val_dice:.4f}"
)