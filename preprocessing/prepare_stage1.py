from pathlib import Path
import csv
import shutil

import nibabel as nib
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "stage1_dataset.yaml"
OUTPUT_ROOT = PROJECT_ROOT / "datasets" / "stage1"


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def remap_label(label_path, mapping):
    """
    Convert dataset-specific label IDs into the unified Stage-1 IDs.
    All labels not present in the mapping become background (0).
    """
    nii = nib.load(str(label_path))
    data = np.asarray(nii.dataobj)

    output = np.zeros(data.shape, dtype=np.uint8)

    for source_id, target_id in mapping.items():
        output[data == int(source_id)] = int(target_id)

    return nib.Nifti1Image(output, nii.affine, nii.header)


def copy_image(image_path, output_path):
    """
    Preserve the original CT image without modifying voxel values.
    """
    shutil.copy2(image_path, output_path)


def process_standard_dataset(name, dataset_config, split):
    image_dir = Path(dataset_config["image_dir"])
    label_dir = Path(dataset_config["label_dir"])

    mapping = {
        int(k): int(v)
        for k, v in dataset_config["label_mapping"].items()
    }

    image_files = sorted(image_dir.glob("*.nii.gz"))

    if not image_files:
        print(f"[WARNING] No images found for {name}: {image_dir}")
        return 0

    processed = 0

    output_images = OUTPUT_ROOT / split / "images"
    output_labels = OUTPUT_ROOT / split / "labels"

    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    for image_path in image_files:
        label_name = image_path.name.replace("img", "label", 1) if image_path.name.startswith("img") else image_path.name
        label_path = label_dir / label_name

        if not label_path.exists():
            print(f"[WARNING] Missing label for {image_path.name}")
            continue

        output_name = f"{name}_{image_path.name}"

        output_image = output_images / output_name
        output_label = output_labels / output_name

        copy_image(image_path, output_image)

        remapped = remap_label(label_path, mapping)
        nib.save(remapped, output_label)

        processed += 1

        if processed % 10 == 0:
            print(f"[{name}] Processed: {processed}")

    print(f"[{name}] Completed: {processed} cases")
    return processed


def load_totalsegmentator_metadata(metadata_path):
    """
    Read TotalSegmentator metadata and return image_id -> split.
    """
    metadata = {}

    with open(metadata_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")

        for row in reader:
            image_id = row["image_id"].strip()
            split = row["split"].strip().lower()

            metadata[image_id] = split

    return metadata


def process_totalsegmentator(dataset_config):
    image_dir = Path(dataset_config["image_dir"])
    label_dir = Path(dataset_config["label_dir"])
    metadata_path = Path(dataset_config["metadata"])

    mapping = {
        int(k): int(v)
        for k, v in dataset_config["label_mapping"].items()
    }

    metadata = load_totalsegmentator_metadata(metadata_path)

    image_files = sorted(image_dir.glob("*.nii.gz"))

    if not image_files:
        print(f"[WARNING] No TotalSegmentator images found: {image_dir}")
        return 0

    processed = 0

    for image_path in image_files:
        case_id = image_path.stem.replace(".nii", "")

        if case_id not in metadata:
            print(f"[WARNING] No metadata entry for {case_id}")
            continue

        dataset_split = metadata[case_id]

        if dataset_split == "train":
            output_split = "train"
        elif dataset_split == "val":
            output_split = "val"
        else:
            print(f"[INFO] Skipping TotalSegmentator test case: {case_id}")
            continue

        label_path = label_dir / image_path.name

        if not label_path.exists():
            print(f"[WARNING] Missing label for {image_path.name}")
            continue

        output_images = OUTPUT_ROOT / output_split / "images"
        output_labels = OUTPUT_ROOT / output_split / "labels"

        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)

        output_name = f"totalsegmentator_{image_path.name}"

        output_image = output_images / output_name
        output_label = output_labels / output_name

        copy_image(image_path, output_image)

        remapped = remap_label(label_path, mapping)
        nib.save(remapped, output_label)

        processed += 1

        if processed % 50 == 0:
            print(f"[TotalSegmentator] Processed: {processed}")

    print(f"[TotalSegmentator] Completed: {processed} cases")
    return processed


def main():
    print("=" * 70)
    print("Stage-1 Dataset Preparation")
    print("Unified 12-Organ Segmentation")
    print("=" * 70)

    config = load_config()

    total_processed = 0

    # BTCV
    total_processed += process_standard_dataset(
        "btcv",
        config["datasets"]["btcv"],
        "train",
    )

    # AMOS22 training
    total_processed += process_standard_dataset(
        "amos22",
        config["datasets"]["amos22"],
        "train",
    )

    # AMOS22 validation
    total_processed += process_standard_dataset(
        "amos22",
        config["datasets"]["amos22_valid"],
        "val",
    )

    # TotalSegmentator
    total_processed += process_totalsegmentator(
        config["datasets"]["totalsegmentator"]
    )

    print("=" * 70)
    print(f"TOTAL PROCESSED CASES: {total_processed}")
    print("=" * 70)


if __name__ == "__main__":
    main()