import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from datasets.stage1_dataloader import build_stage1_dataloader

DATA_ROOT = "datasets/stage1_preprocessed"

loader = build_stage1_dataloader(
    data_root=DATA_ROOT,
    split="train",
    batch_size=1,
    num_workers=0,
    cache_rate=0.0,
    samples_per_volume=2,
)

organ_names = [
    "Background",
    "Spleen",
    "Right Kidney",
    "Left Kidney",
    "Gallbladder",
    "Esophagus",
    "Liver",
    "Stomach",
    "Aorta",
    "Inferior Vena Cava",
    "Pancreas",
    "Right Adrenal",
    "Left Adrenal",
]

counts = torch.zeros(13, dtype=torch.long)
patch_count = 0
TARGET_PATCHES = 200

for batch in loader:
    labels = batch["label"]

    for label in labels:
        patch_count += 1
        present = torch.unique(label).long()

        for class_id in present:
            if 0 <= class_id < 13:
                counts[class_id] += 1

        if patch_count >= TARGET_PATCHES:
            break

    if patch_count >= TARGET_PATCHES:
        break

print("=" * 70)
print("TRAINING PATCH ORGAN-PRESENCE DIAGNOSTIC")
print("=" * 70)
print(f"Patches examined: {patch_count}")
print()

for class_id, name in enumerate(organ_names):
    percentage = counts[class_id].item() / patch_count * 100
    print(
        f"{class_id:2d} {name:<22} "
        f"{counts[class_id].item():3d}/{patch_count} "
        f"({percentage:6.2f}%)"
    )

print("=" * 70)
