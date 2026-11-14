from pathlib import Path
import torch

from models.bayesian_mk_unet3d import BayesianMKUNet3D


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT = (
    ROOT
    / "training"
    / "checkpoints_exp5_samples4"
    / "stage1_baseline_epoch71.pth"
)

OUTPUT_DIR = ROOT / "evaluation" / "model_export"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "stage1_bayesian_mk_unet3d_epoch71.pth"


# ---------------------------------------------------------
# Model configuration
# ---------------------------------------------------------
IN_CHANNELS = 1
NUM_CLASSES = 13
BASE_CHANNELS = 16
DROPOUT = 0.1


# ---------------------------------------------------------
# Device
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 70)
print("STAGE-1 MODEL EXPORT")
print("=" * 70)

print("Checkpoint:")
print(CHECKPOINT)

print("Output:")
print(OUTPUT_FILE)

print("Device:", device)


# ---------------------------------------------------------
# Load checkpoint
# ---------------------------------------------------------
if not CHECKPOINT.exists():
    raise FileNotFoundError(
        f"Checkpoint not found:\n{CHECKPOINT}"
    )

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device,
    weights_only=False
)

print("\nCheckpoint epoch:", checkpoint.get("epoch"))
print(
    "Best validation Dice:",
    checkpoint.get("best_val_dice")
)


# ---------------------------------------------------------
# Build model
# ---------------------------------------------------------
model = BayesianMKUNet3D(
    in_channels=IN_CHANNELS,
    num_classes=NUM_CLASSES,
    base_channels=BASE_CHANNELS,
    dropout=DROPOUT,
).to(device)


# ---------------------------------------------------------
# Load model weights
# ---------------------------------------------------------
if "model_state" not in checkpoint:
    raise KeyError(
        "Checkpoint does not contain 'model_state'."
    )

model.load_state_dict(checkpoint["model_state"])
model.eval()


# ---------------------------------------------------------
# Count parameters
# ---------------------------------------------------------
total_params = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_params = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print("Total parameters:", total_params)
print("Trainable parameters:", trainable_params)


# ---------------------------------------------------------
# Save inference-ready artifact
# ---------------------------------------------------------
export = {
    "model_state": model.state_dict(),
    "model_config": {
        "architecture": "BayesianMKUNet3D",
        "in_channels": IN_CHANNELS,
        "num_classes": NUM_CLASSES,
        "base_channels": BASE_CHANNELS,
        "dropout": DROPOUT,
    },
    "training_info": {
        "checkpoint_epoch": checkpoint.get("epoch"),
        "best_val_dice": checkpoint.get("best_val_dice"),
        "source_checkpoint": str(CHECKPOINT),
    },
    "stage1_info": {
        "target_classes": 13,
        "target_spacing": [1.5, 1.5, 3.0],
        "patch_size": [96, 96, 64],
    },
}

torch.save(export, OUTPUT_FILE)


# ---------------------------------------------------------
# Verify saved artifact
# ---------------------------------------------------------
if not OUTPUT_FILE.exists():
    raise RuntimeError(
        "Export file was not created."
    )

saved = torch.load(
    OUTPUT_FILE,
    map_location="cpu",
    weights_only=False
)

print("\nExport verification:")
print("File exists:", OUTPUT_FILE.exists())
print("Export keys:", list(saved.keys()))
print(
    "Export epoch:",
    saved["training_info"]["checkpoint_epoch"]
)
print(
    "Export best Dice:",
    saved["training_info"]["best_val_dice"]
)

print("\n" + "=" * 70)
print("MODEL EXPORT COMPLETED SUCCESSFULLY")
print("=" * 70)
print("Saved to:")
print(OUTPUT_FILE)