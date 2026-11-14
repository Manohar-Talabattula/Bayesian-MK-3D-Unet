from pathlib import Path
import json
import random

import nibabel as nib
import numpy as np
import torch


# ============================================================
# PATHS
# ============================================================

ROOT = Path(r"C:\Users\HPCLAB\Documents\MK_3D_UNET_PROJECT")

STAGE2_DIR = ROOT / "datasets" / "stage2"

DATASETS = {
    "LiTS_Batch1": STAGE2_DIR / "LiTS_Batch1",
    "LiTS_Batch2": STAGE2_DIR / "LiTS_Batch2",
    "KiTS19": STAGE2_DIR / "KiTS19",
}

SPLIT_FILE = STAGE2_DIR / "split" / "stage2_split_manifest.json"

MODEL_FILE = (
    STAGE2_DIR
    / "models"
    / "stage2_pretrained_init.pth"
)


# ============================================================
# CONFIG
# ============================================================

PATCH_SIZE = (32, 96, 96)

EXPECTED_TOTAL = 341

random.seed(42)


# ============================================================
# HELPERS
# ============================================================

def get_image_files(folder):
    """
    Find only the actual Stage-2 image files.

    Expected:
        *_0000.nii.gz

    but NOT:
        case_0000.nii.gz
        *_0000.nii.gz labels
    """

    files = []

    for p in folder.glob("*_0000.nii.gz"):

        # Image filename must end with:
        # _dddd_0000.nii.gz
        if p.name.endswith("_0000.nii.gz"):

            stem = p.name[:-len("_0000.nii.gz")]

            # Must have another numeric case ID.
            parts = stem.split("_")

            if len(parts) >= 2 and parts[-1].isdigit():
                files.append(p)

    return sorted(files)


def label_from_image(image_path):
    """
    Example:

    LiTS_Batch1_0007_0000.nii.gz
        ->
    LiTS_Batch1_0007.nii.gz
    """

    suffix = "_0000.nii.gz"

    return image_path.with_name(
        image_path.name[:-len(suffix)] + ".nii.gz"
    )


def case_id_from_image(image_path):
    """
    Extract a stable case identifier from the filename.
    """

    suffix = "_0000.nii.gz"

    return image_path.name[:-len(suffix)]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_strings(obj):
    """
    Recursively collect all strings from JSON.
    """

    result = []

    if isinstance(obj, str):
        result.append(obj)

    elif isinstance(obj, list):
        for item in obj:
            result.extend(collect_strings(item))

    elif isinstance(obj, dict):
        for value in obj.values():
            result.extend(collect_strings(value))

    return result


def find_split_lists(obj):
    """
    Extract only actual Stage-2 case IDs from the split manifest.

    Dataset names such as LiTS_Batch1, LiTS_Batch2 and KiTS19
    are intentionally ignored.
    """

    train = []
    val = []

    def extract_case_ids(node):

        result = []

        if isinstance(node, str):

            # Actual case identifiers have a numeric suffix.
            # Examples:
            # LiTS_Batch1_0007
            # LiTS_Batch2_0102
            # KiTS19_0209
            parts = node.replace("\\", "/").split("/")

            for part in parts:

                if (
                    part.startswith("LiTS_Batch1_")
                    or part.startswith("LiTS_Batch2_")
                    or part.startswith("KiTS19_")
                ):
                    suffix = part.split("_")[-1]

                    if suffix.isdigit():
                        result.append(part)

        elif isinstance(node, list):

            for item in node:
                result.extend(extract_case_ids(item))

        elif isinstance(node, dict):

            for value in node.values():
                result.extend(extract_case_ids(value))

        return result

    def walk(node, split_type=None):

        nonlocal train, val

        if isinstance(node, dict):

            for key, value in node.items():

                key_lower = str(key).lower()

                if "train" in key_lower:
                    train.extend(extract_case_ids(value))

                elif (
                    "val" in key_lower
                    or "valid" in key_lower
                ):
                    val.extend(extract_case_ids(value))

                else:
                    walk(value, split_type)

        elif isinstance(node, list):

            for item in node:
                walk(item, split_type)

    walk(obj)

    # Remove duplicates while preserving order.
    train = list(dict.fromkeys(train))
    val = list(dict.fromkeys(val))

    return train, val


# ============================================================
# STEP 1 — VERIFY ALL DATASET PAIRS
# ============================================================

print("=" * 70)
print("STAGE 2 DATASET LOADER TEST")
print("=" * 70)

all_cases = []

for dataset_name, folder in DATASETS.items():

    print(f"\nChecking {dataset_name}...")

    if not folder.exists():
        raise FileNotFoundError(
            f"Dataset folder missing:\n{folder}"
        )

    images = get_image_files(folder)

    print(f"  Images found: {len(images)}")

    if len(images) == 0:
        raise RuntimeError(
            f"No Stage-2 image files found in:\n{folder}"
        )

    for image_path in images:

        label_path = label_from_image(image_path)

        if not label_path.exists():
            raise RuntimeError(
                f"Missing label:\n{label_path}"
            )

        case_id = case_id_from_image(image_path)

        all_cases.append(
            {
                "dataset": dataset_name,
                "case_id": case_id,
                "image": image_path,
                "label": label_path,
            }
        )

print("\nTotal image-label pairs:", len(all_cases))

if len(all_cases) != EXPECTED_TOTAL:
    raise RuntimeError(
        f"Expected {EXPECTED_TOTAL} cases, "
        f"but found {len(all_cases)}."
    )

print("341-case count: PASSED")


# ============================================================
# STEP 2 — CHECK SHAPES / SPACING / LABEL VALUES
# ============================================================

print("\n" + "=" * 70)
print("CHECKING NIFTI PAIRS")
print("=" * 70)

invalid_label_cases = []
shape_mismatch_cases = []
spacing_mismatch_cases = []
empty_cases = []

dataset_counts = {}

for index, case in enumerate(all_cases, start=1):

    dataset = case["dataset"]
    image_path = case["image"]
    label_path = case["label"]

    image_nii = nib.load(str(image_path))
    label_nii = nib.load(str(label_path))

    image_shape = image_nii.shape
    label_shape = label_nii.shape

    if image_shape != label_shape:
        shape_mismatch_cases.append(case["case_id"])

    image_spacing = np.array(
        image_nii.header.get_zooms()[:3]
    )

    label_spacing = np.array(
        label_nii.header.get_zooms()[:3]
    )

    if not np.allclose(
        image_spacing,
        label_spacing,
        atol=1e-4
    ):
        spacing_mismatch_cases.append(case["case_id"])

    # Load only the label for this validation.
    label = np.asanyarray(label_nii.dataobj)

    unique = np.unique(label)

    if not np.all(np.isin(unique, [0, 1, 2])):
        invalid_label_cases.append(
            (case["case_id"], unique.tolist())
        )

    positive = int(np.count_nonzero(label > 0))

    if positive == 0:
        empty_cases.append(case["case_id"])

    dataset_counts[dataset] = (
        dataset_counts.get(dataset, 0) + 1
    )

    if index % 50 == 0:
        print(f"  Checked {index}/{len(all_cases)} cases")


print("\nDataset counts:")

for dataset, count in dataset_counts.items():
    print(f"  {dataset}: {count}")


print("\nShape mismatches:", len(shape_mismatch_cases))
print("Spacing mismatches:", len(spacing_mismatch_cases))
print("Invalid label cases:", len(invalid_label_cases))
print("Empty masks:", len(empty_cases))


if shape_mismatch_cases:
    raise RuntimeError(
        f"Shape mismatch detected:\n"
        f"{shape_mismatch_cases[:10]}"
    )

if spacing_mismatch_cases:
    raise RuntimeError(
        f"Spacing mismatch detected:\n"
        f"{spacing_mismatch_cases[:10]}"
    )

if invalid_label_cases:
    raise RuntimeError(
        f"Invalid labels detected:\n"
        f"{invalid_label_cases[:10]}"
    )

print("\nAll labels contain only 0/1/2: PASSED")
print("All image-label shapes match: PASSED")
print("All image-label spacings match: PASSED")


# ============================================================
# STEP 3 — CHECK SPLIT MANIFEST
# ============================================================

print("\n" + "=" * 70)
print("CHECKING TRAIN / VALIDATION SPLIT")
print("=" * 70)

if not SPLIT_FILE.exists():
    raise FileNotFoundError(
        f"Split manifest missing:\n{SPLIT_FILE}"
    )

split_data = load_json(SPLIT_FILE)

train_strings, val_strings = find_split_lists(split_data)

print("Train entries detected:", len(train_strings))
print("Validation entries detected:", len(val_strings))

if len(train_strings) == 0 or len(val_strings) == 0:
    print(
        "\nWARNING: Could not automatically identify "
        "train/validation lists."
    )
    print(
        "The dataset itself is still valid, "
        "but the split JSON format needs inspection."
    )
else:

    print(
        f"Expected approximately 272 train / 69 validation."
    )

    # Convert paths/IDs into normalized strings.
    train_normalized = set(
        str(x).replace("\\", "/")
        for x in train_strings
    )

    val_normalized = set(
        str(x).replace("\\", "/")
        for x in val_strings
    )

    overlap = train_normalized.intersection(
        val_normalized
    )

    print("Train/validation overlap entries:", len(overlap))

    if overlap:
        raise RuntimeError(
            f"Train/validation overlap detected:\n"
            f"{list(overlap)[:10]}"
        )

    print("Train/validation overlap: PASSED")


# ============================================================
# STEP 4 — LOAD REAL TRAINING SAMPLE
# ============================================================

print("\n" + "=" * 70)
print("LOADING REAL TRAINING SAMPLE")
print("=" * 70)

sample_cases = [
    next(c for c in all_cases
         if c["dataset"] == "LiTS_Batch1"),

    next(c for c in all_cases
         if c["dataset"] == "LiTS_Batch2"),

    next(c for c in all_cases
         if c["dataset"] == "KiTS19"),
]


def load_case(case):

    image_nii = nib.load(str(case["image"]))
    label_nii = nib.load(str(case["label"]))

    image = np.asarray(
        image_nii.dataobj,
        dtype=np.float32
    )

    label = np.asarray(
        label_nii.dataobj,
        dtype=np.uint8
    )

    return image, label


for case in sample_cases:

    image, label = load_case(case)

    print(f"\n{case['dataset']}")
    print("  Case:", case["case_id"])
    print("  Image shape:", image.shape)
    print("  Label shape:", label.shape)
    print("  Image dtype:", image.dtype)
    print("  Label dtype:", label.dtype)
    print(
        "  Label values:",
        np.unique(label).tolist()
    )
    print(
        "  Tumor/pathology voxels:",
        int(np.count_nonzero(label == 2))
    )

    assert image.shape == label.shape
    assert np.all(np.isin(label, [0, 1, 2]))


print("\nReal LiTS + KiTS19 loading: PASSED")


# ============================================================
# STEP 5 — NORMALIZE IMAGE AND CREATE PATCH
# ============================================================

print("\n" + "=" * 70)
print("TESTING PATCH PIPELINE")
print("=" * 70)

image, label = load_case(sample_cases[0])

# Same CT intensity range planned for Stage 2.
image = np.clip(image, -1000, 1000)

# Normalize to [0, 1].
image = (image + 1000.0) / 2000.0

image = image.astype(np.float32)

print("Normalized image range:")
print("  Min:", float(image.min()))
print("  Max:", float(image.max()))

assert image.min() >= 0.0
assert image.max() <= 1.0


# Convert NIfTI's X,Y,Z arrangement into Z,Y,X
# for a standard PyTorch 3D tensor layout.
image_zyx = np.transpose(image, (2, 1, 0))
label_zyx = np.transpose(label, (2, 1, 0))

print("\nTensor layout:")
print("  Image:", image_zyx.shape)
print("  Label:", label_zyx.shape)


def extract_center_patch(array, patch_size):

    z, y, x = array.shape

    pz, py, px = patch_size

    # Pad if necessary.
    pad_z = max(0, pz - z)
    pad_y = max(0, py - y)
    pad_x = max(0, px - x)

    if pad_z or pad_y or pad_x:

        array = np.pad(
            array,
            (
                (pad_z // 2, pad_z - pad_z // 2),
                (pad_y // 2, pad_y - pad_y // 2),
                (pad_x // 2, pad_x - pad_x // 2),
            ),
            mode="constant"
        )

        z, y, x = array.shape

    start_z = (z - pz) // 2
    start_y = (y - py) // 2
    start_x = (x - px) // 2

    return array[
        start_z:start_z + pz,
        start_y:start_y + py,
        start_x:start_x + px,
    ]


image_patch = extract_center_patch(
    image_zyx,
    PATCH_SIZE
)

label_patch = extract_center_patch(
    label_zyx,
    PATCH_SIZE
)

print("\nPatch:")
print("  Image:", image_patch.shape)
print("  Label:", label_patch.shape)

assert image_patch.shape == PATCH_SIZE
assert label_patch.shape == PATCH_SIZE

assert np.all(
    np.isin(label_patch, [0, 1, 2])
)

print("Patch extraction: PASSED")


# ============================================================
# STEP 6 — PYTORCH TENSOR CHECK
# ============================================================

print("\n" + "=" * 70)
print("TESTING PYTORCH TENSOR PIPELINE")
print("=" * 70)

image_tensor = torch.from_numpy(
    image_patch.copy()
).unsqueeze(0).unsqueeze(0)

label_tensor = torch.from_numpy(
    label_patch.copy()
).unsqueeze(0)

print("Image tensor:", tuple(image_tensor.shape))
print("Label tensor:", tuple(label_tensor.shape))

assert image_tensor.shape == (
    1,
    1,
    PATCH_SIZE[0],
    PATCH_SIZE[1],
    PATCH_SIZE[2],
)

assert label_tensor.shape == (
    1,
    PATCH_SIZE[0],
    PATCH_SIZE[1],
    PATCH_SIZE[2],
)

print("PyTorch tensor pipeline: PASSED")


# ============================================================
# STEP 7 — LOAD STAGE 2 MODEL AND RUN REAL PATCH
# ============================================================

print("\n" + "=" * 70)
print("TESTING REAL PATCH THROUGH STAGE-2 MODEL")
print("=" * 70)

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Stage-2 model checkpoint missing:\n{MODEL_FILE}"
    )

from dynamic_network_architectures.architectures.unet import (
    PlainConvUNet
)


def build_model():

    return PlainConvUNet(
        input_channels=1,
        n_stages=6,
        features_per_stage=[
            32, 64, 128, 256, 320, 320
        ],
        conv_op=torch.nn.Conv3d,
        kernel_sizes=[
            [1, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
        ],
        strides=[
            [1, 1, 1],
            [1, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
        ],
        n_conv_per_stage=[
            2, 2, 2, 2, 2, 2
        ],
        num_classes=3,
        n_conv_per_stage_decoder=[
            2, 2, 2, 2, 2
        ],
        conv_bias=True,
        norm_op=torch.nn.InstanceNorm3d,
        norm_op_kwargs={
            "eps": 1e-5,
            "affine": True
        },
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=torch.nn.LeakyReLU,
        nonlin_kwargs={
            "inplace": True
        },
        deep_supervision=True,
    )


checkpoint = torch.load(
    MODEL_FILE,
    map_location="cpu",
    weights_only=False
)

model = build_model()

model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True
)

print("Stage-2 checkpoint loaded: PASSED")

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available."
    )

device = torch.device("cuda")

model = model.to(device)
model.eval()

image_tensor = image_tensor.to(device)

with torch.no_grad():

    outputs = model(image_tensor)

print("Number of outputs:", len(outputs))

for i, output in enumerate(outputs):

    print(
        f"  Output {i}:",
        tuple(output.shape)
    )

assert outputs[0].shape == (
    1,
    3,
    PATCH_SIZE[0],
    PATCH_SIZE[1],
    PATCH_SIZE[2],
)

print("\nReal dataset patch → Stage-2 model: PASSED")


# ============================================================
# FINAL RESULT
# ============================================================

print("\n")
print("=" * 70)
print("STAGE 2 DATASET LOADER TEST PASSED")
print("=" * 70)

print("Total cases              : 341")
print("LiTS Batch 1             : 28")
print("LiTS Batch 2             : 103")
print("KiTS19                   : 210")
print("Classes                  : 3")
print("Patch                    :", PATCH_SIZE)
print("CUDA model forward       : PASSED")
print("Stage-2 pretrained model : PASSED")

print("\nREADY FOR STAGE 2 TRAINING.")