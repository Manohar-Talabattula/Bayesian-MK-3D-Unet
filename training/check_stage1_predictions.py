from pathlib import Path
import sys
import torch
import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.bayesian_mk_unet3d import BayesianMKUNet3D

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "datasets" / "stage1_preprocessed"
CHECKPOINT = ROOT / "training" / "checkpoints" / "best.pth"

NUM_CLASSES = 13
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dice_score(pred, target):
    inter = np.logical_and(pred, target).sum()
    total = pred.sum() + target.sum()

    if total == 0:
        return None

    return (2.0 * inter) / total


print("=" * 70)
print("STAGE-1 PREDICTION DIAGNOSTIC")
print("=" * 70)

print("Device:", DEVICE)
print("Checkpoint:", CHECKPOINT)

model = BayesianMKUNet3D(
    in_channels=1,
    num_classes=NUM_CLASSES,
).to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False,
)

model.load_state_dict(checkpoint["model_state"])
model.eval()

print("Checkpoint epoch:", checkpoint["epoch"])
print("Stored best Dice:", checkpoint["best_val_dice"])

image_dir = DATA_ROOT / "val" / "images"
label_dir = DATA_ROOT / "val" / "labels"

image_files = sorted(image_dir.glob("*.nii.gz"))
label_files = sorted(label_dir.glob("*.nii.gz"))

print("Validation volumes:", len(image_files))

total_dice = []
class_prediction_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
class_label_counts = np.zeros(NUM_CLASSES, dtype=np.int64)

for index, (image_file, label_file) in enumerate(
    zip(image_files[:5], label_files[:5]), 1
):
    image = nib.load(str(image_file)).get_fdata(dtype=np.float32)
    label = nib.load(str(label_file)).get_fdata(dtype=np.float32)

    image_tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image_tensor)

        if isinstance(output, dict):
            output = output["main"]

        elif isinstance(output, (tuple, list)):
            output = output[0]

        prediction = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    print()
    print("-" * 70)
    print(f"Volume {index}: {image_file.name}")
    print("Image shape:", image.shape)
    print("Label shape:", label.shape)
    print("Prediction shape:", prediction.shape)

    unique_pred, counts_pred = np.unique(prediction, return_counts=True)
    unique_label, counts_label = np.unique(label.astype(np.int64), return_counts=True)

    print("\nPredicted classes:")
    for cls, count in zip(unique_pred, counts_pred):
        print(f"  Class {cls:2d}: {count:,} voxels")

    print("\nGround-truth classes:")
    for cls, count in zip(unique_label, counts_label):
        print(f"  Class {cls:2d}: {count:,} voxels")

    print("\nPer-class Dice:")

    volume_dice = []

    for class_id in range(1, NUM_CLASSES):
        pred_mask = prediction == class_id
        label_mask = label == class_id

        score = dice_score(pred_mask, label_mask)

        if score is not None:
            volume_dice.append(score)
            total_dice.append(score)

        class_prediction_counts[class_id] += pred_mask.sum()
        class_label_counts[class_id] += label_mask.sum()

        pred_voxels = pred_mask.sum()
        label_voxels = label_mask.sum()

        if score is None:
            print(
                f"  Class {class_id:2d}: "
                f"pred={pred_voxels:,}, "
                f"label={label_voxels:,}, "
                f"Dice=N/A"
            )
        else:
            print(
                f"  Class {class_id:2d}: "
                f"pred={pred_voxels:,}, "
                f"label={label_voxels:,}, "
                f"Dice={score:.4f}"
            )

    if volume_dice:
        print(f"\nMean foreground Dice: {np.mean(volume_dice):.4f}")

print()
print("=" * 70)
print("OVERALL PREDICTION SUMMARY")
print("=" * 70)

print("\nPrediction voxel totals across checked volumes:")

for class_id in range(1, NUM_CLASSES):
    print(
        f"Class {class_id:2d}: "
        f"predicted={class_prediction_counts[class_id]:,} | "
        f"label={class_label_counts[class_id]:,}"
    )

if total_dice:
    print(f"\nMean Dice across checked volumes: {np.mean(total_dice):.4f}")

print()
print("Diagnostic complete.")