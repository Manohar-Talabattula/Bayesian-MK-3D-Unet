import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Project root
# ---------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.bayesian_mk_unet3d import BayesianMKUNet3D


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
DATA_ROOT = ROOT / "datasets" / "stage1_preprocessed"

IMAGE_DIR = DATA_ROOT / "val" / "images"
LABEL_DIR = DATA_ROOT / "val" / "labels"

CHECKPOINT = ROOT / "training" / "checkpoints_exp5_samples4" / "stage1_baseline_epoch71.pth"

OUTPUT_DIR = ROOT / "evaluation" / "visual_results"

NUM_CLASSES = 13
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------
def load_model():

    print("Loading Bayesian MK-3D-UNet...")

    model = BayesianMKUNet3D(
        in_channels=1,
        num_classes=NUM_CLASSES,
        base_channels=16,
        dropout=0.1,
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=False,
    )

    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Device: {DEVICE}")

    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    return model


# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
@torch.no_grad()
def predict_volume(model, image):

    # image:
    # (H, W, D)

    image_tensor = torch.from_numpy(
        image.astype(np.float32)
    )

    # (1, 1, H, W, D)
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)

    image_tensor = image_tensor.to(DEVICE)

    output = model(image_tensor)

    # Main segmentation output
    if isinstance(output, dict):
        output = output["main"]
    elif isinstance(output, tuple):
        output = output[0]

    prediction = torch.argmax(
        output,
        dim=1
    )

    prediction = prediction.squeeze(0).cpu().numpy()

    return prediction


# ---------------------------------------------------------
# Select useful axial slice
# ---------------------------------------------------------
def choose_slice(label):

    foreground = label > 0

    if not np.any(foreground):
        return label.shape[2] // 2

    coords = np.where(foreground)

    z_min = coords[2].min()
    z_max = coords[2].max()

    return (z_min + z_max) // 2


# ---------------------------------------------------------
# Save visualization
# ---------------------------------------------------------
def save_visualization(
    image,
    label,
    prediction,
    case_name,
    slice_index,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    ct_slice = image[:, :, slice_index]
    gt_slice = label[:, :, slice_index]
    pred_slice = prediction[:, :, slice_index]

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(20, 5)
    )

    # -----------------------------
    # CT
    # -----------------------------
    axes[0].imshow(
        ct_slice.T,
        cmap="gray",
        origin="lower"
    )

    axes[0].set_title("CT Image")
    axes[0].axis("off")

    # -----------------------------
    # Ground truth
    # -----------------------------
    axes[1].imshow(
        gt_slice.T,
        cmap="nipy_spectral",
        origin="lower",
        vmin=0,
        vmax=NUM_CLASSES - 1
    )

    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    # -----------------------------
    # Prediction
    # -----------------------------
    axes[2].imshow(
        pred_slice.T,
        cmap="nipy_spectral",
        origin="lower",
        vmin=0,
        vmax=NUM_CLASSES - 1
    )

    axes[2].set_title("Prediction")
    axes[2].axis("off")

    # -----------------------------
    # Overlay
    # -----------------------------
    axes[3].imshow(
        ct_slice.T,
        cmap="gray",
        origin="lower"
    )

    axes[3].imshow(
        pred_slice.T,
        cmap="nipy_spectral",
        origin="lower",
        alpha=0.45,
        vmin=0,
        vmax=NUM_CLASSES - 1
    )

    axes[3].set_title("Prediction Overlay")
    axes[3].axis("off")

    plt.suptitle(
        f"{case_name} | Slice {slice_index}"
    )

    plt.tight_layout()

    output_file = OUTPUT_DIR / f"{case_name}_slice_{slice_index}.png"

    plt.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {output_file}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():

    print("=" * 60)
    print("STAGE-1 VISUAL EVALUATION")
    print("=" * 60)

    print(f"Images: {IMAGE_DIR}")
    print(f"Labels: {LABEL_DIR}")

    model = load_model()

    image_files = sorted(
        IMAGE_DIR.glob("*.nii.gz")
    )

    label_files = sorted(
        LABEL_DIR.glob("*.nii.gz")
    )

    if len(image_files) == 0:
        raise RuntimeError(
            f"No validation images found in {IMAGE_DIR}"
        )

    print(f"Validation images found: {len(image_files)}")
    print(f"Validation labels found: {len(label_files)}")

    # -----------------------------------------------------
    # Visualize first 5 validation cases
    # -----------------------------------------------------
    for image_path in image_files[:5]:

        case_name = image_path.name.replace(
            ".nii.gz",
            ""
        )

        label_path = LABEL_DIR / image_path.name

        if not label_path.exists():

            print(
                f"Skipping {case_name}: "
                f"label not found."
            )

            continue

        print()
        print("-" * 60)
        print(f"Processing: {case_name}")

        image_nii = nib.load(str(image_path))
        label_nii = nib.load(str(label_path))

        image = image_nii.get_fdata(
            dtype=np.float32
        )

        label = label_nii.get_fdata(
            dtype=np.float32
        ).astype(np.int16)

        print(f"Image shape: {image.shape}")
        print(f"Label shape: {label.shape}")

        print(
            f"Image range: "
            f"{image.min():.3f} - {image.max():.3f}"
        )

        print(
            f"Ground-truth IDs: "
            f"{np.unique(label).astype(int).tolist()}"
        )

        # ---------------------------------------------
        # Prediction
        # ---------------------------------------------
        print("Running model prediction...")

        prediction = predict_volume(
            model,
            image
        )

        print(
            f"Prediction IDs: "
            f"{np.unique(prediction).tolist()}"
        )

        # ---------------------------------------------
        # Choose slice
        # ---------------------------------------------
        slice_index = choose_slice(label)

        print(
            f"Selected slice: {slice_index}"
        )

        # ---------------------------------------------
        # Save result
        # ---------------------------------------------
        save_visualization(
            image=image,
            label=label,
            prediction=prediction,
            case_name=case_name,
            slice_index=slice_index,
        )

    print()
    print("=" * 60)
    print("VISUAL EVALUATION COMPLETED")
    print("=" * 60)

    print(
        f"Results saved to:\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()