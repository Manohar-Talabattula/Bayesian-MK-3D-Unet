from pathlib import Path
import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

TRAIN_LABEL_DIR = ROOT / "datasets" / "stage1_preprocessed" / "train" / "labels"
VAL_LABEL_DIR = ROOT / "datasets" / "stage1_preprocessed" / "val" / "labels"

CLASS_NAMES = {
    0: "background",
    1: "spleen",
    2: "right_kidney",
    3: "left_kidney",
    4: "gallbladder",
    5: "esophagus",
    6: "liver",
    7: "stomach",
    8: "aorta",
    9: "inferior_vena_cava",
    10: "pancreas",
    11: "right_adrenal_gland",
    12: "left_adrenal_gland",
}

NUM_CLASSES = 13


def calculate_distribution(label_dir):
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    files = sorted(label_dir.glob("*.nii.gz"))

    for i, label_file in enumerate(files, 1):
        label = nib.load(str(label_file)).get_fdata(dtype=np.float32)
        label = label.astype(np.int64)

        values, voxel_counts = np.unique(label, return_counts=True)

        for value, count in zip(values, voxel_counts):
            if 0 <= value < NUM_CLASSES:
                counts[value] += count

        if i % 100 == 0 or i == len(files):
            print(f"  Processed {i}/{len(files)}")

    return counts, len(files)


print("=" * 70)
print("STAGE-1 CLASS DISTRIBUTION")
print("=" * 70)

print("\nTRAINING SET")
print("-" * 70)

train_counts, train_files = calculate_distribution(TRAIN_LABEL_DIR)

print(f"\nTraining volumes: {train_files}")

train_total = train_counts.sum()

for class_id in range(NUM_CLASSES):
    percentage = 100.0 * train_counts[class_id] / train_total

    print(
        f"Class {class_id:2d} "
        f"{CLASS_NAMES[class_id]:25s} : "
        f"{train_counts[class_id]:15,} voxels "
        f"({percentage:7.3f}%)"
    )


print("\n\nVALIDATION SET")
print("-" * 70)

val_counts, val_files = calculate_distribution(VAL_LABEL_DIR)

print(f"\nValidation volumes: {val_files}")

val_total = val_counts.sum()

for class_id in range(NUM_CLASSES):
    percentage = 100.0 * val_counts[class_id] / val_total

    print(
        f"Class {class_id:2d} "
        f"{CLASS_NAMES[class_id]:25s} : "
        f"{val_counts[class_id]:15,} voxels "
        f"({percentage:7.3f}%)"
    )


print("\n" + "=" * 70)
print("FOREGROUND CLASS COMPARISON")
print("=" * 70)

foreground_train = train_counts[1:]

print("\nTraining foreground distribution:")

for class_id in range(1, NUM_CLASSES):
    print(
        f"Class {class_id:2d} "
        f"{CLASS_NAMES[class_id]:25s} : "
        f"{train_counts[class_id]:15,}"
    )

print("\nLargest foreground class:",
      CLASS_NAMES[np.argmax(foreground_train) + 1])

print("Smallest foreground class:",
      CLASS_NAMES[np.argmin(foreground_train) + 1])

print("\nDiagnostic complete.")