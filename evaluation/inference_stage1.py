from pathlib import Path
import sys

import numpy as np
import nibabel as nib
import torch
from monai.inferers import sliding_window_inference

# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# IMPORT MODEL
# ============================================================

from models.bayesian_mk_unet3d import BayesianMKUNet3D


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHECKPOINT = (
    ROOT
    / "training"
    / "checkpoints_exp5_samples4"
    / "stage1_baseline_epoch71.pth"
)

INPUT_DIR = (
    ROOT
    / "datasets"
    / "stage1_preprocessed"
    / "val"
    / "images"
)

OUTPUT_DIR = (
    ROOT
    / "evaluation"
    / "inference_results"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Same patch size used during Stage-1 training
ROI_SIZE = (96, 96, 64)

# Process one window at a time to control GPU memory
SW_BATCH_SIZE = 1

# 25% overlap between neighboring windows
OVERLAP = 0.25

NUM_CLASSES = 13


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("STAGE-1 FULL-VOLUME INFERENCE")
print("=" * 60)

print(f"Checkpoint: {CHECKPOINT}")
print(f"Device: {DEVICE}")

if DEVICE.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


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


# Our training checkpoints store the model under "model_state"
if "model_state" in checkpoint:
    model.load_state_dict(checkpoint["model_state"])

elif "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])

elif "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"])

else:
    model.load_state_dict(checkpoint)


model = model.to(DEVICE)
model.eval()

print("Model loaded successfully.")


# ============================================================
# HELPER: EXTRACT MAIN OUTPUT
# ============================================================

def get_main_output(output):
    """
    BayesianMKUNet3D returns a dictionary containing
    the main segmentation output and auxiliary outputs.
    """

    if isinstance(output, dict):
        return output["main"]

    if isinstance(output, tuple):
        return output[0]

    return output


# ============================================================
# FULL-VOLUME INFERENCE
# ============================================================

def predict_volume(image):
    """
    Run sliding-window inference over the complete 3D CT volume.

    Input:
        image: numpy array [H, W, D]

    Returns:
        prediction: numpy array [H, W, D]
                    containing class IDs 0-12
    """

    image_tensor = torch.from_numpy(
        image.astype(np.float32)
    )

    # [H, W, D]
    # -> [1, 1, H, W, D]
    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)

    image_tensor = image_tensor.to(DEVICE)

    with torch.no_grad():

        def predictor(x):
            model_output = model(x)

            if isinstance(model_output, dict):
                return model_output["main"]

            if isinstance(model_output, tuple):
                return model_output[0]

            return model_output

        output = sliding_window_inference(
            inputs=image_tensor,
            roi_size=ROI_SIZE,
            sw_batch_size=SW_BATCH_SIZE,
            predictor=predictor,
            overlap=OVERLAP,
            mode="gaussian",
        )

        prediction = torch.argmax(
            output,
            dim=1,
        )

    prediction = prediction.squeeze(0).cpu().numpy().astype(np.uint8)

    return prediction


# ============================================================
# FIND VALIDATION IMAGES
# ============================================================

image_files = sorted(
    INPUT_DIR.glob("*.nii.gz")
)

print(f"Validation images found: {len(image_files)}")

if len(image_files) == 0:
    raise RuntimeError(
        f"No NIfTI images found in:\n{INPUT_DIR}"
    )


# ============================================================
# PROCESS ALL VALIDATION VOLUMES
# ============================================================

successful_cases = 0
failed_cases = []

total_cases = len(image_files)

print()
print("=" * 60)
print("STARTING FULL-VALIDATION INFERENCE")
print("=" * 60)
print(f"Total validation volumes: {total_cases}")
print()


for case_index, image_path in enumerate(image_files, start=1):

    case_name = image_path.name.replace(".nii.gz", "")

    print()
    print("-" * 60)
    print(f"[{case_index}/{total_cases}] Processing: {case_name}")
    print("-" * 60)

    try:

        # ========================================================
        # LOAD IMAGE
        # ========================================================

        image_nii = nib.load(str(image_path))

        image = image_nii.get_fdata(
            dtype=np.float32
        )

        print(f"Image shape: {image.shape}")
        print(
            f"Image spacing: "
            f"{image_nii.header.get_zooms()[:3]}"
        )

        print(
            f"Image range: "
            f"{image.min():.3f} - {image.max():.3f}"
        )

        # ========================================================
        # PREDICT
        # ========================================================

        print(
            "Running sliding-window full-volume inference..."
        )

        prediction = predict_volume(image)

        print("Inference completed.")

        print(
            f"Prediction shape: {prediction.shape}"
        )

        print(
            f"Prediction IDs: "
            f"{np.unique(prediction).tolist()}"
        )

        # ========================================================
        # VERIFY OUTPUT
        # ========================================================

        if prediction.shape != image.shape:
            raise RuntimeError(
                "Prediction shape does not match input shape!"
            )

        unique_ids = np.unique(prediction)

        invalid_ids = unique_ids[
            (unique_ids < 0) |
            (unique_ids >= NUM_CLASSES)
        ]

        if len(invalid_ids) > 0:
            raise RuntimeError(
                f"Invalid prediction IDs found: {invalid_ids}"
            )

        # ========================================================
        # SAVE NIFTI
        # ========================================================

        output_name = (
            image_path.name.replace(
                ".nii.gz",
                "_prediction.nii.gz"
            )
        )

        output_path = OUTPUT_DIR / output_name

        # Preserve the affine and header from the preprocessed CT.
        prediction_nii = nib.Nifti1Image(
            prediction,
            affine=image_nii.affine,
            header=image_nii.header.copy(),
        )

        # Ensure label data is stored as uint8.
        prediction_nii.set_data_dtype(np.uint8)

        nib.save(
            prediction_nii,
            str(output_path)
        )

        print()
        print("Saved prediction:")
        print(output_path)

        # ========================================================
        # VERIFY SAVED FILE
        # ========================================================

        saved_nii = nib.load(str(output_path))

        saved_prediction = saved_nii.get_fdata()

        print()
        print("Saved-file verification:")
        print(f"Shape: {saved_prediction.shape}")
        print(
            f"Spacing: "
            f"{saved_nii.header.get_zooms()[:3]}"
        )
        print(
            f"IDs: "
            f"{np.unique(saved_prediction).astype(int).tolist()}"
        )

        successful_cases += 1

        print()
        print(
            f"Case completed successfully "
            f"({successful_cases}/{total_cases})"
        )

    except Exception as e:

        failed_cases.append(
            (case_name, str(e))
        )

        print()
        print("!!! CASE FAILED !!!")
        print(f"Case: {case_name}")
        print(f"Error: {e}")

        # Continue with the next validation case.
        continue


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 60)
print("FULL-VALIDATION INFERENCE COMPLETED")
print("=" * 60)

print(f"Total cases:      {total_cases}")
print(f"Successful cases: {successful_cases}")
print(f"Failed cases:     {len(failed_cases)}")

if failed_cases:

    print()
    print("Failed cases:")

    for case_name, error_message in failed_cases:
        print(f"- {case_name}: {error_message}")

else:

    print()
    print("All validation volumes processed successfully.")

print("=" * 60)