import sys
sys.path.insert(0, ".")

import torch
from datasets.stage1_dataloader import build_stage1_dataloader

loader = build_stage1_dataloader(
    "datasets/stage1_preprocessed",
    "train",
    batch_size=1,
    num_workers=0,
    cache_rate=0.0
)

class_names = [
    "background",
    "spleen",
    "right_kidney",
    "left_kidney",
    "gallbladder",
    "esophagus",
    "liver",
    "stomach",
    "aorta",
    "inferior_vena_cava",
    "pancreas",
    "right_adrenal_gland",
    "left_adrenal_gland"
]

presence = torch.zeros(13, dtype=torch.long)

print("Checking first 100 sampled training patches...")

for i, batch in enumerate(loader):
    label = batch["label"]

    for class_id in torch.unique(label).tolist():
        presence[int(class_id)] += 1

    if i + 1 >= 100:
        break

print("\nClass presence in 100 sampled patches:")
print("-" * 50)

for i, name in enumerate(class_names):
    print(f"{i:2d}  {name:22s} : {int(presence[i]):3d}/100")
