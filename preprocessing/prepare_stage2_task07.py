from pathlib import Path
import argparse
import json

import nibabel as nib
import numpy as np

from preprocessing.prepare_stage2_dataset import (
    TARGET_SPACING,
    normalize_ct,
    resample_case,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SOURCE = (
    Path(r"C:\Users\HPCLAB\Downloads\Stage2_Clean")
    / "MSD_Task07_Pancreas"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT / "datasets" / "stage2" / "Task07_Pancreas"
)


def process_case(image_path, label_path, output_dir):
    case_suffix = image_path.name.removeprefix(
        "pancreas_"
    ).removesuffix(".nii.gz")

    case_id = f"Task07_Pancreas_{case_suffix}"

    print(f"\nProcessing: {case_id}")
    print(f"  Image: {image_path.name}")
    print(f"  Label: {label_path.name}")

    image = nib.load(str(image_path))
    label = nib.load(str(label_path))

    if len(image.shape) != 3 or len(label.shape) != 3:
        raise ValueError(
            f"{case_id}: expected 3D CT and label volumes."
        )

    if image.shape != label.shape:
        raise ValueError(
            f"{case_id}: image/label shape mismatch: "
            f"{image.shape} vs {label.shape}"
        )

    if not np.allclose(
        image.affine, label.affine, atol=1e-4
    ):
        raise ValueError(
            f"{case_id}: source image and label geometry differ."
        )

    # Validate the native mask before resampling.
    source_label = label.get_fdata(dtype=np.float32)

    if not np.isfinite(source_label).all():
        raise ValueError(
            f"{case_id}: mask contains non-finite values."
        )

    source_values = set(
        np.unique(np.rint(source_label).astype(np.int16)).tolist()
    )

    if not source_values.issubset({0, 1, 2}):
        raise ValueError(
            f"{case_id}: unexpected source labels "
            f"{sorted(source_values)}"
        )

    del source_label

    # Reuse the established project resampling implementation.
    image_r, label_r = resample_case(image, label)

    if image_r.shape != label_r.shape:
        raise ValueError(
            f"{case_id}: resampled image/mask shapes differ."
        )

    if not np.allclose(
        image_r.affine, label_r.affine, atol=1e-4
    ):
        raise ValueError(
            f"{case_id}: resampled image/mask geometry differs."
        )

    # Match the existing Stage-2 CT normalization.
    image_np = normalize_ct(
        image_r.get_fdata(dtype=np.float32)
    ).astype(np.float32)

    label_values = label_r.get_fdata(dtype=np.float32)
    label_np = np.rint(label_values).astype(np.int16)
    del label_values

    output_values = set(np.unique(label_np).tolist())

    if not output_values.issubset({0, 1, 2}):
        raise ValueError(
            f"{case_id}: unexpected resampled labels "
            f"{sorted(output_values)}"
        )

    if not np.isfinite(image_np).all():
        raise ValueError(
            f"{case_id}: normalized CT contains non-finite values."
        )

    if image_np.min() < 0 or image_np.max() > 1:
        raise ValueError(
            f"{case_id}: normalized CT is outside [0, 1]."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    image_out = output_dir / f"{case_id}_0000.nii.gz"
    label_out = output_dir / f"{case_id}.nii.gz"

    nib.save(
        nib.Nifti1Image(image_np, image_r.affine),
        str(image_out),
    )

    nib.save(
        nib.Nifti1Image(label_np, label_r.affine),
        str(label_out),
    )

    spacing = np.asarray(
        nib.load(str(image_out)).header.get_zooms()[:3]
    )

    record = {
        "case_id": case_id,
        "dataset": "Task07_Pancreas",
        "source_image": str(image_path.resolve()),
        "source_label": str(label_path.resolve()),
        "image": str(image_out.resolve()),
        "label": str(label_out.resolve()),
        "shape": list(image_np.shape),
        "spacing": spacing.tolist(),
        "image_range": [
            float(image_np.min()),
            float(image_np.max()),
        ],
        "source_labels": sorted(source_values),
        "output_labels": sorted(output_values),
        "background_voxels": int(np.count_nonzero(label_np == 0)),
        "pancreas_voxels": int(np.count_nonzero(label_np == 1)),
        "pathology_voxels": int(np.count_nonzero(label_np == 2)),
    }

    print(f"  Output shape: {record['shape']}")
    print(f"  Output spacing: {record['spacing']}")
    print(f"  Output labels: {record['output_labels']}")
    print(f"  Pathology voxels: {record['pathology_voxels']:,}")
    print(f"  Saved CT: {image_out.name}")
    print(f"  Saved label: {label_out.name}")

    return record


def main():
    parser = argparse.ArgumentParser(
        description="Prepare MSD Task07 Pancreas for Stage 2."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process only the first N cases; 0 processes all.",
    )
    args = parser.parse_args()

    source_dir = args.source_dir
    images_dir = source_dir / "imagesTr"
    labels_dir = source_dir / "labelsTr"

    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(
            f"Expected imagesTr and labelsTr under {source_dir}"
        )

    images = sorted(images_dir.glob("*.nii.gz"))
    label_map = {
        path.name: path
        for path in labels_dir.glob("*.nii.gz")
    }

    if not images:
        raise RuntimeError(f"No CT images found in {images_dir}")

    missing = [
        image.name
        for image in images
        if image.name not in label_map
    ]

    if missing:
        raise RuntimeError(
            f"{len(missing)} images have no matching labels. "
            f"Examples: {missing[:10]}"
        )

    if len(images) != len(label_map):
        raise RuntimeError(
            f"Image count {len(images)} differs from "
            f"label count {len(label_map)}."
        )

    if args.limit < 0:
        raise ValueError("--limit cannot be negative.")

    if args.limit:
        images = images[:args.limit]

    print("=" * 70)
    print("STAGE-2 TASK07 PANCREAS PREPROCESSING")
    print("=" * 70)
    print(f"Available source pairs: {len(label_map)}")
    print(f"Cases selected: {len(images)}")
    print(f"Target spacing: {TARGET_SPACING.tolist()}")
    print("CT clip: [-1000, 1000] HU")
    print("CT normalization: [0, 1]")
    print("Label mapping: 0->0, 1->1, 2->2")
    print(f"Output directory: {args.output_dir}")

    records = []

    for index, image_path in enumerate(images, start=1):
        print(f"\nCase {index}/{len(images)}")

        record = process_case(
            image_path,
            label_map[image_path.name],
            args.output_dir,
        )
        records.append(record)

    manifest = {
        "dataset": "MSD Task07 Pancreas",
        "source_case_count": len(label_map),
        "processed_case_count": len(records),
        "target_spacing": TARGET_SPACING.tolist(),
        "ct_clip_hu": [-1000, 1000],
        "ct_normalization": "[0, 1]",
        "label_mapping": {
            "0": "background",
            "1": "pancreas",
            "2": "pancreatic mass/pathology",
        },
        "records": records,
    }

    manifest_path = (
        args.output_dir / "task07_preprocessing_manifest.json"
    )

    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("PREPROCESSING COMPLETE")
    print(f"Cases processed: {len(records)}")
    print(f"Manifest: {manifest_path}")
    print("Original source files were not modified.")
    print("=" * 70)


if __name__ == "__main__":
    main()
