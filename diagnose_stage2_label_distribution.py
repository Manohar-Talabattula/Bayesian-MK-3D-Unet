import json
import random
from pathlib import Path

import nibabel as nib
import numpy as np


MANIFEST = Path(
    r".\datasets\stage2\split\stage2_split_manifest.json"
)

with open(MANIFEST, "r", encoding="utf-8") as f:
    manifest = json.load(f)

cases = manifest["train"]

rng = random.Random(42)
sample = rng.sample(cases, min(50, len(cases)))

print("=" * 70)
print("STAGE 2 TRAINING LABEL DISTRIBUTION")
print("=" * 70)

print(f"Training cases: {len(cases)}")
print(f"Sampled cases:  {len(sample)}")

totals = np.zeros(3, dtype=np.int64)

pathology_cases = 0
pathology_voxels = []
organ_voxels = []
bg_voxels = []

print()
print("Per-case pathology statistics:")
print("-" * 70)

for i, item in enumerate(sample):

    label_path = Path(item["label"])

    label = nib.load(label_path).get_fdata(dtype=np.float32)

    label_uint8 = label.astype(np.uint8)

    counts = np.bincount(
        label_uint8.ravel(),
        minlength=3
    )

    counts = counts[:3]

    totals += counts.astype(np.int64)

    bg = int(counts[0])
    healthy = int(counts[1])
    pathology = int(counts[2])

    pathology_voxels.append(pathology)
    organ_voxels.append(healthy)
    bg_voxels.append(bg)

    if pathology > 0:
        pathology_cases += 1

    print(
        f"{i + 1:02d}. "
        f"{item['dataset']}/{label_path.name} | "
        f"BG={bg:>10} "
        f"Healthy={healthy:>10} "
        f"Path={pathology:>10}"
    )


total = int(totals.sum())

print()
print("=" * 70)
print("AGGREGATE")
print("=" * 70)

print(
    f"Background voxels: {int(totals[0]):>12} "
    f"({totals[0] / total * 100:.4f}%)"
)

print(
    f"Healthy voxels:   {int(totals[1]):>12} "
    f"({totals[1] / total * 100:.4f}%)"
)

print(
    f"Pathology voxels: {int(totals[2]):>12} "
    f"({totals[2] / total * 100:.4f}%)"
)

print()

print(
    f"Cases with pathology: "
    f"{pathology_cases} / {len(sample)}"
)

print()

print("Pathology voxels per positive case:")

if pathology_voxels:
    print(f"  min:    {int(np.min(pathology_voxels))}")
    print(f"  median: {int(np.median(pathology_voxels))}")
    print(f"  mean:   {float(np.mean(pathology_voxels)):.2f}")
    print(f"  max:    {int(np.max(pathology_voxels))}")

print("=" * 70)
