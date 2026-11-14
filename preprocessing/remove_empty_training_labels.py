from pathlib import Path
import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

LABEL_DIR = ROOT / "datasets" / "stage1_preprocessed" / "train" / "labels"
IMAGE_DIR = ROOT / "datasets" / "stage1_preprocessed" / "train" / "images"

removed = 0
checked = 0

print("=" * 70)
print("REMOVING EMPTY STAGE-1 TRAINING CASES")
print("=" * 70)

for label_path in sorted(LABEL_DIR.glob("*.nii.gz")):
    checked += 1

    label = nib.load(str(label_path)).get_fdata()

    if not np.any(label > 0):
        image_path = IMAGE_DIR / label_path.name

        print(f"Removing empty case: {label_path.name}")

        label_path.unlink()

        if image_path.exists():
            image_path.unlink()

        removed += 1

print()
print("-" * 70)
print(f"Cases checked : {checked}")
print(f"Cases removed : {removed}")
print(f"Cases remaining: {checked - removed}")
print("-" * 70)

print("Empty-case cleanup completed.")
print("=" * 70)