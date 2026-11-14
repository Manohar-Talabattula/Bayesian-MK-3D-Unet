from pathlib import Path
import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LABEL_DIR = ROOT / "datasets" / "stage1_preprocessed" / "train" / "labels"

files = sorted(LABEL_DIR.glob("*.nii.gz"))

print("=" * 70)
print("STAGE-1 TRAINING LABEL FOREGROUND CHECK")
print("=" * 70)

print(f"Label directory : {LABEL_DIR}")
print(f"Total labels    : {len(files)}")
print()

empty = 0
non_empty = 0

foreground_counts = []

for i, path in enumerate(files, 1):
    data = nib.load(str(path)).get_fdata()

    foreground = np.count_nonzero(data > 0)
    unique = np.unique(data)

    if foreground == 0:
        empty += 1
        print(f"[EMPTY] {path.name} | IDs: {unique.tolist()}")
    else:
        non_empty += 1

    foreground_counts.append(foreground)

    if i <= 10:
        print(
            f"{path.name:35s} "
            f"foreground voxels: {foreground:,} "
            f"IDs: {unique.astype(int).tolist()}"
        )

print()
print("-" * 70)
print(f"Non-empty labels : {non_empty}")
print(f"Empty labels     : {empty}")
print(f"Minimum FG voxels: {min(foreground_counts):,}")
print(f"Maximum FG voxels: {max(foreground_counts):,}")
print(f"Mean FG voxels   : {np.mean(foreground_counts):,.0f}")
print("-" * 70)

if empty == 0:
    print("RESULT: All training labels contain foreground.")
else:
    print("RESULT: Some training labels contain NO foreground.")

print("=" * 70)