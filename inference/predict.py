
from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from evaluation import run_task07_stage3_export_v1 as stage3
from models.bayesian_mk_unet3d import BayesianMKUNet3D


ROOT = Path(__file__).resolve().parents[1]

STAGE1_CHECKPOINT = (
    ROOT
    / "training"
    / "checkpoints_exp5_samples4"
    / "best.pth"
)

STAGE2_CHECKPOINT = (
    ROOT
    / "datasets"
    / "stage2"
    / "models"
    / "task07_finetune_v2"
    / "latest_task07_finetune.pth"
)

DEFAULT_THRESHOLD = 0.80
DEFAULT_GATE_MARGIN_MM = 15.0


def load_models():
    """Build and load the two existing trained models."""

    if not STAGE1_CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Stage 1 checkpoint not found: {STAGE1_CHECKPOINT}"
        )

    if not STAGE2_CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Stage 2 checkpoint not found: {STAGE2_CHECKPOINT}"
        )

    device = stage3.DEVICE

    # ---------------------------------------------------------
    # Stage 1: Multi-organ segmentation model
    # ---------------------------------------------------------

    stage1_model = BayesianMKUNet3D(
        in_channels=1,
        num_classes=13,
        base_channels=16,
        dropout=0.1,
    ).to(device)

    stage3.load_checkpoint(
        stage1_model,
        STAGE1_CHECKPOINT,
    )

    stage1_model.eval()

    # ---------------------------------------------------------
    # Stage 2: Pancreatic pathology segmentation model
    # ---------------------------------------------------------

    stage2_model = stage3.build_stage2_model().to(device)

    stage3.load_checkpoint(
        stage2_model,
        STAGE2_CHECKPOINT,
    )

    stage2_model.eval()

    return stage1_model, stage2_model


def predict_case(
    image_path: str | Path,
    stage1_model,
    stage2_model,
    threshold: float = DEFAULT_THRESHOLD,
    gate_margin_mm: float = DEFAULT_GATE_MARGIN_MM,
):
    """
    Run the complete Stage 1 -> Stage 2 -> Stage 3 cascade.

    Input:
        Original 3D CT scan (.nii or .nii.gz)

    Output:
        Predicted lesion mask
        Lesion probability map
        Pancreas gate
        Quantitative measurements
        JSON report
    """

    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Input CT scan not found: {image_path}"
        )

    if not (
        image_path.name.endswith(".nii")
        or image_path.name.endswith(".nii.gz")
    ):
        raise ValueError(
            "Input must be a .nii or .nii.gz file."
        )

    if not 0.0 < threshold < 1.0:
        raise ValueError(
            "Threshold must be between 0 and 1."
        )

    if gate_margin_mm < 0:
        raise ValueError(
            "Gate margin cannot be negative."
        )

    # ---------------------------------------------------------
    # Load original CT scan in canonical RAS orientation
    # ---------------------------------------------------------

    source_nii = nib.as_closest_canonical(
        nib.load(str(image_path))
    )

    if len(source_nii.shape) != 3:
        raise ValueError(
            f"Expected a 3D CT volume; received shape "
            f"{source_nii.shape}"
        )

    # ---------------------------------------------------------
    # Stage 2 input preparation
    # ---------------------------------------------------------

    # Match the original Stage 3 evaluation pipeline exactly.
    # Do not perform additional resampling or normalization.

    stage2_nii = source_nii

    stage2_raw = stage2_nii.get_fdata(
        dtype=np.float32
    )

    if not np.isfinite(stage2_raw).all():
        raise ValueError(
            "The input CT scan contains invalid intensity values."
        )

    # Original Stage 3 evaluation uses Z,Y,X ordering.
    stage2_model_image = np.transpose(
        stage2_raw,
        (2, 1, 0),
    ).copy()

    # ---------------------------------------------------------
    # Stage 1: Pancreas localization
    # ---------------------------------------------------------

    stage1_transform = stage3.get_stage1_preprocessor()

    stage1_image, stage1_affine = (
        stage3.preprocess_stage1_case(
            image_path,
            stage1_transform,
        )
    )

    stage1_prediction = stage3.predict_stage1_organs(
        stage1_model,
        stage1_image,
    )

    # Class 10 = pancreas.
    stage1_pancreas = (
        stage1_prediction == 10
    )

    # Map Stage 1 pancreas gate to the Stage 2 CT grid.
    stage2_pancreas_gate = (
        stage3.map_stage1_gate_to_stage2(
            stage1_pancreas,
            stage1_affine,
            stage2_nii,
        )
    )

    # ---------------------------------------------------------
    # Stage 2: Lesion probability prediction
    # ---------------------------------------------------------

    stage2_probabilities_zyx = (
        stage3.predict_stage2_probability(
            stage2_model,
            stage2_model_image,
        )
    )

    # Convert probabilities from Z,Y,X to X,Y,Z.
    stage2_probabilities_xyz = np.transpose(
        stage2_probabilities_zyx,
        (0, 3, 2, 1),
    ).copy()

    # Class 2 = pancreatic lesion/pathology.
    lesion_probability = (
        stage2_probabilities_xyz[2]
    )

    if lesion_probability.shape != stage2_nii.shape[:3]:
        raise RuntimeError(
            "Stage 2 output shape does not match the "
            "reference CT grid: "
            f"{lesion_probability.shape} vs "
            f"{stage2_nii.shape[:3]}"
        )

    # ---------------------------------------------------------
    # Stage 3: Physical pancreas gate and thresholding
    # ---------------------------------------------------------

    spacing_xyz = np.linalg.norm(
        stage2_nii.affine[:3, :3],
        axis=0,
    )

    expanded_gate = stage3.dilate_physical(
        stage2_pancreas_gate,
        spacing_xyz,
        gate_margin_mm,
    )

    # Threshold-only baseline prediction.
    baseline_mask = (
        lesion_probability >= threshold
    )

    # Final prediction: probability threshold + pancreas gate.
    final_mask = (
        baseline_mask & expanded_gate
    )

    # ---------------------------------------------------------
    # Quantification
    # ---------------------------------------------------------

    measurements = stage3.quantify_lesion_mask(
        final_mask,
        stage2_nii.affine,
    )

    baseline_measurements = (
        stage3.quantify_lesion_mask(
            baseline_mask,
            stage2_nii.affine,
        )
    )

    # ---------------------------------------------------------
    # Prediction report
    # ---------------------------------------------------------

    report = {
        "project": "Bayesian MK-3D-UNet",
        "task": "Research pancreatic lesion segmentation",
        "input_file": image_path.name,
        "device": str(stage3.DEVICE),

        "stage1": {
            "checkpoint": str(
                STAGE1_CHECKPOINT.relative_to(ROOT)
            ),
            "predicted_pancreas_voxels": int(
                stage1_pancreas.sum()
            ),
            "mapped_pancreas_voxels": int(
                stage2_pancreas_gate.sum()
            ),
        },

        "stage2": {
            "checkpoint": str(
                STAGE2_CHECKPOINT.relative_to(ROOT)
            ),
            "lesion_class": 2,
            "probability_threshold": float(threshold),
        },

        "stage3": {
            "gate_margin_mm": float(gate_margin_mm),

            "baseline_thresholded_volume_ml": (
                baseline_measurements[
                    "predicted_volume_ml"
                ]
            ),

            "final_gated_volume_ml": (
                measurements[
                    "predicted_volume_ml"
                ]
            ),
        },

        "final_lesion_measurements": measurements,

        "warning": (
            "Research output only. Predicted regions and "
            "connected components are not clinically verified "
            "lesions. This software is not intended for "
            "diagnosis or treatment."
        ),
    }

    # ---------------------------------------------------------
    # Create output NIfTI files
    # ---------------------------------------------------------

    final_nii = nib.Nifti1Image(
        final_mask.astype(np.uint8),
        affine=stage2_nii.affine,
        header=stage2_nii.header.copy(),
    )

    final_nii.set_data_dtype(np.uint8)

    probability_nii = nib.Nifti1Image(
        lesion_probability.astype(np.float32),
        affine=stage2_nii.affine,
        header=stage2_nii.header.copy(),
    )

    probability_nii.set_data_dtype(np.float32)

    # ---------------------------------------------------------
    # Return all prediction outputs
    # ---------------------------------------------------------

    return {
        "report": report,

        "mask_nii": final_nii,

        "probability_nii": probability_nii,

        "final_mask": final_mask,

        "lesion_probability": lesion_probability,

        "pancreas_gate": stage2_pancreas_gate,

        "expanded_gate": expanded_gate,
    }


def report_to_json(report: dict) -> bytes:
    """Convert prediction report into downloadable JSON."""

    return json.dumps(
        report,
        indent=2,
        allow_nan=False,
    ).encode("utf-8")
from pathlib import Path
import json
import nibabel as nib
import numpy as np
import torch

from evaluation import evaluate_stage2_anatomy_gate as litskits


LITS_KITS_STAGE1_CHECKPOINT = litskits.STAGE1_CHECKPOINT
LITS_KITS_STAGE2_CHECKPOINT = litskits.STAGE2_CHECKPOINT

LITS_KITS_DEFAULT_THRESHOLD = 0.65
LITS_KITS_DEFAULT_GATE_MARGIN_MM = 15.0


def load_lits_kits_models():
    """Load the existing Stage 1 and general LiTS/KiTS Stage 2 models."""

    for checkpoint_path in (
        LITS_KITS_STAGE1_CHECKPOINT,
        LITS_KITS_STAGE2_CHECKPOINT,
    ):
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

    device = litskits.DEVICE

    stage1_model = BayesianMKUNet3D(
        in_channels=1,
        num_classes=13,
        base_channels=16,
        dropout=0.1,
    ).to(device)

    litskits.load_checkpoint(
        stage1_model,
        LITS_KITS_STAGE1_CHECKPOINT,
    )
    stage1_model.eval()

    stage2_model = litskits.build_stage2_model().to(device)

    litskits.load_checkpoint(
        stage2_model,
        LITS_KITS_STAGE2_CHECKPOINT,
    )
    stage2_model.eval()

    return stage1_model, stage2_model


def predict_lits_kits_case(
    image_path: str | Path,
    stage1_model,
    stage2_model,
    threshold: float = LITS_KITS_DEFAULT_THRESHOLD,
    gate_margin_mm: float = LITS_KITS_DEFAULT_GATE_MARGIN_MM,
):
    """Run the evaluated liver/kidney anatomy-gated pathology pipeline."""

    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Input CT scan not found: {image_path}")

    if not (
        image_path.name.endswith(".nii")
        or image_path.name.endswith(".nii.gz")
    ):
        raise ValueError("Input must be a .nii or .nii.gz file.")

    if not 0.0 < threshold < 1.0:
        raise ValueError("Threshold must be between 0 and 1.")

    if gate_margin_mm < 0:
        raise ValueError("Gate margin cannot be negative.")

    # Match the LiTS/KiTS evaluation: canonical XYZ reference grid.
    source_nii = nib.as_closest_canonical(
        nib.load(str(image_path))
    )

    if len(source_nii.shape) != 3:
        raise ValueError(
            f"Expected a 3D CT volume; received {source_nii.shape}"
        )

    stage2_image = source_nii.get_fdata(dtype=np.float32)

    if not np.isfinite(stage2_image).all():
        raise ValueError("Input CT contains invalid intensity values.")

    # Stage 1 preprocessing and inference: same helper functions as evaluation.
    transform = litskits.get_stage1_preprocessor()

    stage1_image, stage1_affine = (
        litskits.preprocess_stage1_case(image_path, transform)
    )

    stage1_prediction = litskits.predict_stage1_organs(
        stage1_model,
        stage1_image,
    )

    # Classes 2, 3, 6 = right kidney, left kidney, liver.
    stage1_anatomy_gate = np.isin(
        stage1_prediction,
        [2, 3, 6],
    )

    anatomy_gate = litskits.map_stage1_gate_to_stage2(
        stage1_anatomy_gate,
        stage1_affine,
        source_nii,
    )

    # Stage 2 inference: preserve the evaluation script's XYZ ordering.
    probabilities = litskits.predict_stage2_probability(
        stage2_model,
        stage2_image,
    )

    pathology_probability = probabilities[2]

    if pathology_probability.shape != source_nii.shape[:3]:
        raise RuntimeError(
            "Stage 2 output shape does not match the CT grid: "
            f"{pathology_probability.shape} vs {source_nii.shape[:3]}"
        )

    spacing = tuple(
        float(x) for x in source_nii.header.get_zooms()[:3]
    )

    expanded_gate = litskits.dilate_physical(
        anatomy_gate,
        spacing,
        gate_margin_mm,
    )

    baseline_mask = pathology_probability >= threshold
    final_mask = baseline_mask & expanded_gate

    measurements = stage3.quantify_lesion_mask(
        final_mask,
        source_nii.affine,
    )

    baseline_measurements = stage3.quantify_lesion_mask(
        baseline_mask,
        source_nii.affine,
    )

    report = {
        "project": "Bayesian MK-3D-UNet",
        "task": "Research liver/kidney-region pathology segmentation",
        "input_file": image_path.name,
        "device": str(litskits.DEVICE),
        "stage1": {
            "checkpoint": str(
                LITS_KITS_STAGE1_CHECKPOINT.relative_to(ROOT)
            ),
            "anatomy_classes": [2, 3, 6],
            "mapped_anatomy_gate_voxels": int(anatomy_gate.sum()),
        },
        "stage2": {
            "checkpoint": str(
                LITS_KITS_STAGE2_CHECKPOINT.relative_to(ROOT)
            ),
            "pathology_class": 2,
            "probability_threshold": float(threshold),
        },
        "anatomy_gate": {
            "margin_mm": float(gate_margin_mm),
            "expanded_gate_voxels": int(expanded_gate.sum()),
        },
        "baseline_thresholded_volume_ml": (
            baseline_measurements["predicted_volume_ml"]
        ),
        "final_pathology_measurements": measurements,
        "warning": (
            "Research output only. Predicted regions are not clinically "
            "verified lesions. This software is not intended for "
            "diagnosis or treatment."
        ),
    }

    mask_nii = nib.Nifti1Image(
        final_mask.astype(np.uint8),
        affine=source_nii.affine,
        header=source_nii.header.copy(),
    )
    mask_nii.set_data_dtype(np.uint8)

    probability_nii = nib.Nifti1Image(
        pathology_probability.astype(np.float32),
        affine=source_nii.affine,
        header=source_nii.header.copy(),
    )
    probability_nii.set_data_dtype(np.float32)

    return {
        "report": report,
        "mask_nii": mask_nii,
        "probability_nii": probability_nii,
        "final_mask": final_mask,
        "lesion_probability": pathology_probability,
        "anatomy_gate": anatomy_gate,
        "expanded_gate": expanded_gate,
    }
