"""
Stage 2 dataset preparation for LiTS and KiTS.

Stage 2 classes:
    0 = background
    1 = healthy/non-lesional organ tissue
    2 = abnormal/pathological tissue

LiTS source labels:
    0 = background
    1 = liver
    2 = tumor

KiTS source labels:
    0 = background
    1 = kidney
    2 = tumor
    3 = cyst

KiTS label 3 (cyst) is merged into Stage-2 class 2
because Stage-2 class 2 represents abnormal/pathological tissue.

The original source datasets are never modified.
Only the preprocessed output labels are converted.
"""

from pathlib import Path
import argparse
import json

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to


# ============================================================
# PREPROCESSING SETTINGS
# ============================================================

TARGET_SPACING = np.array(
    [2.5, 1.5, 1.5],
    dtype=np.float32
)

CLIP_LOW = -1000.0
CLIP_HIGH = 1000.0


# ============================================================
# NIFTI LOADING
# ============================================================

def load_nifti(path):
    """Load a NIfTI file."""
    return nib.load(str(path))


# ============================================================
# RESAMPLING
# ============================================================

def resample_case(image, label):
    """
    Convert image and label to RAS orientation and resample
    them to the target voxel spacing.

    Image:
        linear interpolation

    Label:
        nearest-neighbor interpolation
    """

    # Convert both to canonical RAS orientation
    image = nib.as_closest_canonical(image)
    label = nib.as_closest_canonical(label)

    # Original image shape and spacing
    shape = np.asarray(
        image.shape[:3],
        dtype=np.int64
    )

    old_spacing = np.asarray(
        image.header.get_zooms()[:3],
        dtype=np.float64
    )

    # Calculate new shape
    new_shape = np.maximum(
        np.round(
            shape * old_spacing / TARGET_SPACING
        ).astype(int),
        1
    )

    # Build target affine
    target_affine = image.affine.copy()

    for i in range(3):
        direction = target_affine[:3, i]

        norm = np.linalg.norm(direction)

        if norm > 0:
            target_affine[:3, i] = (
                direction / norm * TARGET_SPACING[i]
            )

    target = (
        tuple(new_shape.tolist()),
        target_affine
    )

    # Image: linear interpolation
    image_r = resample_from_to(
        image,
        target,
        order=1
    )

    # Label: nearest-neighbor interpolation
    label_r = resample_from_to(
        label,
        target,
        order=0
    )

    return image_r, label_r


# ============================================================
# CT NORMALIZATION
# ============================================================

def normalize_ct(data):
    """
    Clip CT intensity to [-1000, 1000] HU
    and normalize to [0, 1].
    """

    data = np.asarray(
        data,
        dtype=np.float32
    )

    data = np.clip(
        data,
        CLIP_LOW,
        CLIP_HIGH
    )

    return (
        (data - CLIP_LOW)
        / (CLIP_HIGH - CLIP_LOW)
    )


# ============================================================
# LABEL CONVERSION
# ============================================================

def convert_label(data, dataset_name):
    """
    Convert source dataset labels into the fixed Stage-2 labels.

    Stage 2:
        0 = background
        1 = healthy/non-lesional organ tissue
        2 = abnormal/pathological tissue

    LiTS:
        0 -> 0
        1 -> 1
        2 -> 2

    KiTS:
        0 -> 0
        1 -> 1
        2 -> 2
        3 -> 2
    """

    data = np.rint(
        np.asarray(data)
    ).astype(np.int16)

    unique_labels = np.unique(data)

    dataset_lower = dataset_name.lower()

    # --------------------------------------------------------
    # Expected source labels
    # --------------------------------------------------------

    if "lits" in dataset_lower:

        expected_labels = {0, 1, 2}

        unexpected = set(
            unique_labels.tolist()
        ) - expected_labels

        if unexpected:
            raise ValueError(
                f"{dataset_name}: unexpected LiTS labels found: "
                f"{sorted(unexpected)}"
            )

        out = np.zeros_like(
            data,
            dtype=np.uint8
        )

        # LiTS:
        # 0 = background
        # 1 = liver
        # 2 = tumor
        out[data == 1] = 1
        out[data == 2] = 2

        return out

    # --------------------------------------------------------
    # KiTS
    # --------------------------------------------------------

    elif "kits" in dataset_lower:

        expected_labels = {0, 1, 2, 3}

        unexpected = set(
            unique_labels.tolist()
        ) - expected_labels

        if unexpected:
            raise ValueError(
                f"{dataset_name}: unexpected KiTS labels found: "
                f"{sorted(unexpected)}"
            )

        out = np.zeros_like(
            data,
            dtype=np.uint8
        )

        # KiTS:
        # 0 = background
        # 1 = kidney
        # 2 = tumor
        # 3 = cyst
        #
        # Stage 2:
        # 0 = background
        # 1 = healthy/non-lesional organ tissue
        # 2 = abnormal/pathological tissue
        #
        # Therefore:
        # tumor (2) -> pathology (2)
        # cyst  (3) -> pathology (2)

        out[data == 1] = 1
        out[data == 2] = 2
        out[data == 3] = 2

        return out

    else:

        raise ValueError(
            "Unknown dataset. "
            "dataset-name must contain either "
            "'LiTS' or 'KiTS'."
        )


# ============================================================
# PROCESS ONE CASE
# ============================================================

def process_case(
    image_path,
    label_path,
    out_dir,
    case_id,
    dataset_name
):
    """
    Process one CT image and its segmentation.
    """

    print(
        f"\nProcessing {case_id}"
    )

    print(
        f"  Image: {image_path.name}"
    )

    print(
        f"  Label: {label_path.name}"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    image = load_nifti(
        image_path
    )

    label = load_nifti(
        label_path
    )

    # --------------------------------------------------------
    # Verify original dimensions
    # --------------------------------------------------------

    if image.shape[:3] != label.shape[:3]:

        raise ValueError(
            f"{case_id}: image/label shape mismatch:\n"
            f"Image: {image.shape}\n"
            f"Label: {label.shape}"
        )

    # --------------------------------------------------------
    # Read original labels BEFORE resampling
    # --------------------------------------------------------

    original_label_data = label.get_fdata(
        dtype=np.float32
    )

    original_labels = np.unique(
        np.rint(
            original_label_data
        ).astype(np.int16)
    )

    print(
        f"  Original labels: "
        f"{original_labels.tolist()}"
    )

    # --------------------------------------------------------
    # Resample
    # --------------------------------------------------------

    image_r, label_r = resample_case(
        image,
        label
    )

    # --------------------------------------------------------
    # Image normalization
    # --------------------------------------------------------

    image_np = normalize_ct(
        image_r.get_fdata(
            dtype=np.float32
        )
    )

    # --------------------------------------------------------
    # Label conversion
    # --------------------------------------------------------

    label_np = convert_label(
        label_r.get_fdata(
            dtype=np.float32
        ),
        dataset_name
    )

    # --------------------------------------------------------
    # Verify converted labels
    # --------------------------------------------------------

    output_labels = np.unique(
        label_np
    )

    allowed_stage2_labels = {
        0,
        1,
        2
    }

    unexpected_output = set(
        output_labels.tolist()
    ) - allowed_stage2_labels

    if unexpected_output:

        raise ValueError(
            f"{case_id}: unexpected Stage-2 labels "
            f"after conversion: "
            f"{sorted(unexpected_output)}"
        )

    print(
        f"  Stage-2 labels: "
        f"{output_labels.tolist()}"
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Output filenames
    # --------------------------------------------------------

    img_out = (
        out_dir
        / f"{case_id}_0000.nii.gz"
    )

    lab_out = (
        out_dir
        / f"{case_id}.nii.gz"
    )

    # --------------------------------------------------------
    # Save image
    # --------------------------------------------------------

    nib.save(
        nib.Nifti1Image(
            image_np.astype(
                np.float32
            ),
            image_r.affine
        ),
        str(img_out)
    )

    # --------------------------------------------------------
    # Save label
    # --------------------------------------------------------

    nib.save(
        nib.Nifti1Image(
            label_np,
            label_r.affine
        ),
        str(lab_out)
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    background_voxels = int(
        np.count_nonzero(
            label_np == 0
        )
    )

    healthy_voxels = int(
        np.count_nonzero(
            label_np == 1
        )
    )

    pathology_voxels = int(
        np.count_nonzero(
            label_np == 2
        )
    )

    # --------------------------------------------------------
    # Return record
    # --------------------------------------------------------

    return {
        "case_id": case_id,
        "image": str(img_out),
        "label": str(lab_out),
        "shape": list(
            image_np.shape
        ),
        "background_voxels": background_voxels,
        "healthy_voxels": healthy_voxels,
        "pathology_voxels": pathology_voxels,
        "positive_voxels": int(
            np.count_nonzero(label_np)
        )
    }


# ============================================================
# FIND AND PAIR FILES
# ============================================================

def find_files(
    image_dir,
    label_dir,
    dataset_name
):
    """
    Find and pair image/label files according
    to the dataset naming convention.
    """

    dataset_lower = dataset_name.lower()

    # ========================================================
    # LiTS
    # ========================================================

    if "lits" in dataset_lower:

        image_files = sorted(
            [
                p
                for p in image_dir.rglob(
                    "volume-*.nii"
                )
                if p.is_file()
            ],
            key=lambda p: int(
                p.stem.split("-")[1]
            )
        )

        label_files = sorted(
            [
                p
                for p in label_dir.rglob(
                    "segmentation-*.nii"
                )
                if p.is_file()
            ],
            key=lambda p: int(
                p.stem.split("-")[1]
            )
        )

        image_ids = [
            int(
                p.stem.split("-")[1]
            )
            for p in image_files
        ]

        label_ids = [
            int(
                p.stem.split("-")[1]
            )
            for p in label_files
        ]

    # ========================================================
    # KiTS
    # ========================================================

    elif "kits" in dataset_lower:

        # KiTS:
        # images/case_00000.nii.gz
        # segmentations/case_00000.nii.gz

        image_files = sorted(
            [
                p
                for p in image_dir.rglob(
                    "case_*.nii.gz"
                )
                if p.is_file()
            ],
            key=lambda p: int(
                p.name.split("_")[1].split(".")[0]
            )
        )

        label_files = sorted(
            [
                p
                for p in label_dir.rglob(
                    "case_*.nii.gz"
                )
                if p.is_file()
            ],
            key=lambda p: int(
                p.name.split("_")[1].split(".")[0]
            )
        )

        image_ids = [
            int(
                p.name.split("_")[1].split(".")[0]
            )
            for p in image_files
        ]

        label_ids = [
            int(
                p.name.split("_")[1].split(".")[0]
            )
            for p in label_files
        ]

    else:

        raise RuntimeError(
            "Unknown dataset. "
            "dataset-name must contain "
            "'LiTS' or 'KiTS'."
        )

    # ========================================================
    # Basic checks
    # ========================================================

    if not image_files:

        raise RuntimeError(
            f"No image files found in:\n"
            f"{image_dir}"
        )

    if not label_files:

        raise RuntimeError(
            f"No label files found in:\n"
            f"{label_dir}"
        )

    if image_ids != label_ids:

        raise RuntimeError(
            f"Image/label case mismatch "
            f"for {dataset_name}.\n\n"
            f"Image IDs:\n"
            f"{image_ids}\n\n"
            f"Label IDs:\n"
            f"{label_ids}"
        )

    if len(image_files) != len(label_files):

        raise RuntimeError(
            f"Found {len(image_files)} images "
            f"and {len(label_files)} labels."
        )

    return image_files, label_files


# ============================================================
# MAIN
# ============================================================

def main():

    ap = argparse.ArgumentParser(
        description=(
            "Prepare LiTS or KiTS data for "
            "Stage-2 pathology segmentation."
        )
    )

    ap.add_argument(
        "--image-dir",
        required=True
    )

    ap.add_argument(
        "--label-dir",
        required=True
    )

    ap.add_argument(
        "--output-dir",
        required=True
    )

    ap.add_argument(
        "--dataset-name",
        required=True
    )

    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "Process only the first N cases. "
            "0 = process all cases."
        )
    )

    args = ap.parse_args()

    image_dir = Path(
        args.image_dir
    )

    label_dir = Path(
        args.label_dir
    )

    output_dir = Path(
        args.output_dir
    )

    # ========================================================
    # Validate directories
    # ========================================================

    if not image_dir.exists():

        raise FileNotFoundError(
            f"Image directory does not exist:\n"
            f"{image_dir}"
        )

    if not label_dir.exists():

        raise FileNotFoundError(
            f"Label directory does not exist:\n"
            f"{label_dir}"
        )

    # ========================================================
    # Find files
    # ========================================================

    image_files, label_files = find_files(
        image_dir,
        label_dir,
        args.dataset_name
    )

    print("=" * 70)
    print(
        f"STAGE 2 PREPROCESSING: "
        f"{args.dataset_name}"
    )
    print("=" * 70)

    print(
        f"\nImages found: "
        f"{len(image_files)}"
    )

    print(
        f"Labels found: "
        f"{len(label_files)}"
    )

    print(
        f"Target spacing: "
        f"{TARGET_SPACING.tolist()}"
    )

    print(
        f"Output directory:\n"
        f"{output_dir}"
    )

    # ========================================================
    # Apply limit
    # ========================================================

    if args.limit > 0:

        image_files = image_files[
            :args.limit
        ]

        label_files = label_files[
            :args.limit
        ]

        print(
            f"\nLIMIT MODE: "
            f"processing {args.limit} case(s)"
        )

    # ========================================================
    # Process cases
    # ========================================================

    records = []

    for i, (
        image_path,
        label_path
    ) in enumerate(
        zip(
            image_files,
            label_files
        )
    ):

        case_id = (
            f"{args.dataset_name}_{i:04d}"
        )

        record = process_case(
            image_path,
            label_path,
            output_dir,
            case_id,
            args.dataset_name
        )

        records.append(
            record
        )

    # ========================================================
    # Manifest
    # ========================================================

    meta = {

        "dataset": args.dataset_name,

        "num_processed": len(
            records
        ),

        "target_spacing": (
            TARGET_SPACING.tolist()
        ),

        "ct_clip_hu": [
            CLIP_LOW,
            CLIP_HIGH
        ],

        "stage2_classes": {

            "0": "background",

            "1": (
                "healthy_non_lesional_"
                "organ_tissue"
            ),

            "2": (
                "abnormal_pathological_"
                "tissue"
            )
        },

        "label_mapping": (
            {
                "LiTS": {
                    "0": 0,
                    "1": 1,
                    "2": 2
                },

                "KiTS": {
                    "0": 0,
                    "1": 1,
                    "2": 2,
                    "3": 2
                }
            }
        ),

        "records": records
    }

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    manifest_path = (
        output_dir
        / "dataset_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(
            meta,
            indent=2
        ),
        encoding="utf-8"
    )

    # ========================================================
    # Final summary
    # ========================================================

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print("=" * 70)

    print(
        f"Cases processed: "
        f"{len(records)}"
    )

    print(
        f"Manifest:\n"
        f"{manifest_path}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()