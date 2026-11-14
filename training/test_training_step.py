import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from monai.losses import DiceCELoss

from datasets.stage1_dataloader import build_stage1_dataloader
from models.bayesian_mk_unet3d import BayesianMKUNet3D


# --------------------------------------------------
# Paths
# --------------------------------------------------

DATA_ROOT = ROOT / "datasets" / "stage1_preprocessed"


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# --------------------------------------------------
# Build DataLoader
# --------------------------------------------------

print("\nBuilding training DataLoader...")

train_loader = build_stage1_dataloader(
    data_root=DATA_ROOT,
    split="train",
    batch_size=1,
    num_workers=0
)

print("Training DataLoader ready.")


# --------------------------------------------------
# Build model
# --------------------------------------------------

print("\nBuilding Bayesian MK-3D-UNet...")

model = BayesianMKUNet3D(
    in_channels=1,
    num_classes=13,
    base_channels=16,
    dropout=0.1
)

model = model.to(device)

print("Model ready.")


# --------------------------------------------------
# Loss
# --------------------------------------------------

loss_function = DiceCELoss(
    to_onehot_y=True,
    softmax=True
)


# --------------------------------------------------
# Optimizer
# --------------------------------------------------

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-5
)


# --------------------------------------------------
# Get one real batch
# --------------------------------------------------

print("\nLoading one real training batch...")

batch = next(iter(train_loader))

images = batch["image"].to(device)
labels = batch["label"].to(device)

print("Input shape:", tuple(images.shape))
print("Label shape:", tuple(labels.shape))

print(
    "Input range:",
    float(images.min()),
    "to",
    float(images.max())
)

print(
    "Label IDs:",
    torch.unique(labels).detach().cpu().tolist()
)


# --------------------------------------------------
# Forward pass
# --------------------------------------------------

print("\nRunning forward pass...")

model.train()

optimizer.zero_grad(set_to_none=True)

outputs = model(images)

logits = outputs["main"]

print("Output shape:", tuple(logits.shape))


# --------------------------------------------------
# Loss
# --------------------------------------------------

print("\nCalculating loss...")

loss = loss_function(logits, labels)

print("Loss:", float(loss.item()))


# --------------------------------------------------
# Backward pass
# --------------------------------------------------

print("\nRunning backward pass...")

loss.backward()

print("Backward pass successful.")


# --------------------------------------------------
# Optimizer update
# --------------------------------------------------

print("\nRunning optimizer update...")

optimizer.step()

print("Optimizer update successful.")


# --------------------------------------------------
# Final checks
# --------------------------------------------------

print("\nChecking gradients...")

gradient_found = False

for name, parameter in model.named_parameters():

    if parameter.grad is not None:

        gradient_found = True

        print(
            "Example gradient:",
            name,
            "mean =",
            float(parameter.grad.abs().mean())
        )

        break


if not gradient_found:
    raise RuntimeError("No gradients were found.")


print("\n" + "=" * 55)
print("TRAINING PIPELINE SMOKE TEST SUCCESSFUL")
print("=" * 55)

print("\nVerified:")
print("1. Real DataLoader works")
print("2. Real CT batch reaches the model")
print("3. Model produces 13-class output")
print("4. Dice + Cross Entropy loss works")
print("5. Backward propagation works")
print("6. Gradients are produced")
print("7. Optimizer update works")