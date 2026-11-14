from pathlib import Path
import sys

import nibabel as nib
import numpy as np
import torch

from monai.inferers import sliding_window_inference


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ============================================================
# MODEL
# ============================================================

from models.bayesian_mk_unet3d import BayesianMKUNet3D


# ============================================================
# CONFIGURATION
# ============================================================

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
    / "uncertainty_results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

NUM_CLASSES = 13

# Bayesian stochastic forward passes per patch.
MC_SAMPLES = 8

# Same patch size used during Stage-1 training.
ROI_SIZE = (
    96,
    96,
    64,
)

# Keep this at 1 because each MC prediction is expensive.
SW_BATCH_SIZE = 1

# Same overlap used by deterministic inference.
OVERLAP = 0.25


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("STAGE-1 BAYESIAN MC-DROPOUT UNCERTAINTY")
print("=" * 60)

print(f"Checkpoint: {CHECKPOINT}")
print(f"Device: {DEVICE}")

if DEVICE.type == "cuda":
    print(
        f"GPU: "
        f"{torch.cuda.get_device_name(0)}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

model = BayesianMKUNet3D(
    in_channels=1,
    num_classes=NUM_CLASSES,
    base_channels=16,
    dropout=0.1,
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
)

if "model_state" in checkpoint:
    model.load_state_dict(
        checkpoint["model_state"]
    )

elif "model_state_dict" in checkpoint:
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

elif "state_dict" in checkpoint:
    model.load_state_dict(
        checkpoint["state_dict"]
    )

else:
    model.load_state_dict(
        checkpoint
    )


model = model.to(DEVICE)

print("Model loaded successfully.")


# ============================================================
# MC DROPOUT
# ============================================================

model.eval()

if hasattr(
    model,
    "enable_mc_dropout"
):

    model.enable_mc_dropout()

    print(
        "MC Dropout enabled."
    )

else:

    raise RuntimeError(
        "Model does not contain "
        "enable_mc_dropout()."
    )


# ============================================================
# SLIDING-WINDOW MC PREDICTOR
# ============================================================

def mc_predictor(x):

    """
    x:
        [B, 1, X, Y, Z]

    Returns concatenated:

        mean probability
        uncertainty

    as:

        [B, 26, X, Y, Z]

    First 13 channels:
        mean probability

    Last 13 channels:
        voxel-wise variance
    """

    mean_prediction, uncertainty = (
        model.predict_with_uncertainty(
            x,
            n_passes=MC_SAMPLES,
        )
    )

    return torch.cat(
        [
            mean_prediction,
            uncertainty,
        ],
        dim=1,
    )


# ============================================================
# PROCESS ONE VOLUME
# ============================================================

def process_volume(
    image_path,
    case_number,
    total_cases,
):

    case_name = image_path.name.replace(
        ".nii.gz",
        "",
    )

    print()
    print("-" * 60)
    print(
        f"[{case_number}/{total_cases}] "
        f"Processing: {case_name}"
    )
    print("-" * 60)

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    image_nii = nib.load(
        str(image_path)
    )

    image = image_nii.get_fdata(
        dtype=np.float32
    )

    print(
        f"Image shape: {image.shape}"
    )

    print(
        f"Image spacing: "
        f"{image_nii.header.get_zooms()[:3]}"
    )

    print(
        f"Image range: "
        f"{image.min():.3f} - "
        f"{image.max():.3f}"
    )

    # --------------------------------------------------------
    # TENSOR
    # --------------------------------------------------------

    image_tensor = torch.from_numpy(
        image
    ).unsqueeze(0).unsqueeze(0)

    image_tensor = image_tensor.to(
        DEVICE,
        non_blocking=True,
    )

    # --------------------------------------------------------
    # SLIDING-WINDOW MC INFERENCE
    # --------------------------------------------------------

    print()
    print(
        f"Running sliding-window MC inference "
        f"({MC_SAMPLES} stochastic passes)..."
    )

    with torch.no_grad():

        combined_output = (
            sliding_window_inference(
                inputs=image_tensor,
                roi_size=ROI_SIZE,
                sw_batch_size=SW_BATCH_SIZE,
                predictor=mc_predictor,
                overlap=OVERLAP,
                mode="gaussian",
            )
        )

    # --------------------------------------------------------
    # SPLIT PROBABILITY / UNCERTAINTY
    # --------------------------------------------------------

    mean_probability = (
        combined_output[
            :,
            :NUM_CLASSES,
        ]
    )

    uncertainty = (
        combined_output[
            :,
            NUM_CLASSES:,
        ]
    )

    # --------------------------------------------------------
    # SEGMENTATION
    # --------------------------------------------------------

    prediction = torch.argmax(
        mean_probability,
        dim=1,
    )

    prediction = (
        prediction
        .squeeze(0)
        .cpu()
        .numpy()
        .astype(np.uint8)
    )

    # --------------------------------------------------------
    # UNCERTAINTY
    # --------------------------------------------------------

    uncertainty = (
        uncertainty
        .squeeze(0)
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    # [C, X, Y, Z]
    #
    # Convert class-wise uncertainty into one
    # voxel-wise uncertainty map.

    voxel_uncertainty = np.mean(
        uncertainty,
        axis=0,
    )

    # --------------------------------------------------------
    # VERIFY
    # --------------------------------------------------------

    if prediction.shape != image.shape:

        raise RuntimeError(
            "Prediction shape does not "
            "match input shape: "
            f"{prediction.shape} vs "
            f"{image.shape}"
        )

    if voxel_uncertainty.shape != image.shape:

        raise RuntimeError(
            "Uncertainty shape does not "
            "match input shape: "
            f"{voxel_uncertainty.shape} vs "
            f"{image.shape}"
        )

    unique_ids = np.unique(
        prediction
    )

    invalid_ids = unique_ids[
        (unique_ids < 0)
        |
        (unique_ids >= NUM_CLASSES)
    ]

    if len(invalid_ids) > 0:

        raise RuntimeError(
            f"Invalid prediction IDs: "
            f"{invalid_ids}"
        )

    print()
    print(
        f"Prediction IDs: "
        f"{unique_ids.tolist()}"
    )

    print(
        f"Uncertainty range: "
        f"{voxel_uncertainty.min():.8f} - "
        f"{voxel_uncertainty.max():.8f}"
    )

    # --------------------------------------------------------
    # NORMALIZE UNCERTAINTY
    # --------------------------------------------------------

    unc_min = float(
        voxel_uncertainty.min()
    )

    unc_max = float(
        voxel_uncertainty.max()
    )

    if unc_max > unc_min:

        uncertainty_normalized = (
            voxel_uncertainty - unc_min
        ) / (
            unc_max - unc_min
        )

    else:

        uncertainty_normalized = (
            np.zeros_like(
                voxel_uncertainty,
                dtype=np.float32,
            )
        )

    # --------------------------------------------------------
    # OUTPUT PATHS
    # --------------------------------------------------------

    prediction_path = (
        OUTPUT_DIR
        / f"{case_name}_mc_prediction.nii.gz"
    )

    uncertainty_path = (
        OUTPUT_DIR
        / f"{case_name}_uncertainty.nii.gz"
    )

    normalized_path = (
        OUTPUT_DIR
        / f"{case_name}_uncertainty_normalized.nii.gz"
    )

    # --------------------------------------------------------
    # SAVE PREDICTION
    # --------------------------------------------------------

    prediction_nii = nib.Nifti1Image(
        prediction,
        affine=image_nii.affine,
        header=image_nii.header.copy(),
    )

    prediction_nii.set_data_dtype(
        np.uint8
    )

    nib.save(
        prediction_nii,
        str(prediction_path),
    )

    # --------------------------------------------------------
    # SAVE RAW UNCERTAINTY
    # --------------------------------------------------------

    uncertainty_nii = nib.Nifti1Image(
        voxel_uncertainty,
        affine=image_nii.affine,
        header=image_nii.header.copy(),
    )

    uncertainty_nii.set_data_dtype(
        np.float32
    )

    nib.save(
        uncertainty_nii,
        str(uncertainty_path),
    )

    # --------------------------------------------------------
    # SAVE NORMALIZED UNCERTAINTY
    # --------------------------------------------------------

    normalized_nii = nib.Nifti1Image(
        uncertainty_normalized.astype(
            np.float32
        ),
        affine=image_nii.affine,
        header=image_nii.header.copy(),
    )

    normalized_nii.set_data_dtype(
        np.float32
    )

    nib.save(
        normalized_nii,
        str(normalized_path),
    )

    # --------------------------------------------------------
    # VERIFY SAVED FILES
    # --------------------------------------------------------

    saved_prediction = nib.load(
        str(prediction_path)
    )

    saved_uncertainty = nib.load(
        str(uncertainty_path)
    )

    if (
        saved_prediction.shape
        != image.shape
    ):

        raise RuntimeError(
            "Saved prediction shape "
            "verification failed."
        )

    if (
        saved_uncertainty.shape
        != image.shape
    ):

        raise RuntimeError(
            "Saved uncertainty shape "
            "verification failed."
        )

    print()
    print(
        f"Saved prediction: "
        f"{prediction_path}"
    )

    print(
        f"Saved uncertainty: "
        f"{uncertainty_path}"
    )

    print(
        f"Saved normalized uncertainty: "
        f"{normalized_path}"
    )

    print()
    print(
        "Case completed successfully."
    )


# ============================================================
# FIND VALIDATION IMAGES
# ============================================================

image_files = sorted(
    INPUT_DIR.glob("*.nii.gz")
)

print()
print(
    f"Validation images found: "
    f"{len(image_files)}"
)

if len(image_files) == 0:

    raise RuntimeError(
        f"No NIfTI images found in:\n"
        f"{INPUT_DIR}"
    )


# ============================================================
# PROCESS ALL CASES
# ============================================================

successful_cases = 0
failed_cases = []

total_cases = len(
    image_files
)

for case_number, image_path in enumerate(
    image_files,
    start=1,
):

    try:

        process_volume(
            image_path,
            case_number,
            total_cases,
        )

        successful_cases += 1

    except Exception as e:

        case_name = (
            image_path.name.replace(
                ".nii.gz",
                "",
            )
        )

        failed_cases.append(
            (
                case_name,
                str(e),
            )
        )

        print()
        print(
            "!!! CASE FAILED !!!"
        )

        print(
            f"Case: {case_name}"
        )

        print(
            f"Error: {e}"
        )

        print(
            "Continuing to next case..."
        )

        continue


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 60)
print(
    "FULL-VALIDATION MC UNCERTAINTY "
    "COMPLETED"
)
print("=" * 60)

print(
    f"Total cases:      {total_cases}"
)

print(
    f"Successful cases: {successful_cases}"
)

print(
    f"Failed cases:     {len(failed_cases)}"
)

if failed_cases:

    print()
    print("Failed cases:")

    for case_name, error in failed_cases:

        print(
            f"- {case_name}: {error}"
        )

else:

    print()
    print(
        "All validation volumes processed "
        "successfully."
    )

print("=" * 60)