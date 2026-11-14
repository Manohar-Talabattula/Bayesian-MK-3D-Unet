import json
import random
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn

from dynamic_network_architectures.architectures.unet import PlainConvUNet


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

CHECKPOINT = (
    PROJECT_ROOT
    / "datasets"
    / "stage2"
    / "models"
    / "stage2_best.pth"
)

SPLIT_MANIFEST = (
    PROJECT_ROOT
    / "datasets"
    / "stage2"
    / "split"
    / "stage2_split_manifest.json"
)

PATCH_SIZE = (32, 96, 96)
NUM_CASES = 10
PATCHES_PER_CASE = 3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# MODEL
# ============================================================

def build_model():
    model = PlainConvUNet(
        input_channels=1,
        n_stages=6,
        features_per_stage=[32, 64, 128, 256, 320, 320],
        conv_op=nn.Conv3d,
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
        n_conv_per_stage=[2, 2, 2, 2, 2, 2],
        num_classes=3,
        n_conv_per_stage_decoder=[2, 2, 2, 2, 2],
        conv_bias=True,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=True,
    )

    return model


# ============================================================
# CHECKPOINT
# ============================================================

def load_model():
    print("Building Stage-2 model...")

    model = build_model()

    print("Loading checkpoint:")
    print(CHECKPOINT)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
    )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "network_weights" in checkpoint:
        state_dict = checkpoint["network_weights"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)

    model = model.to(DEVICE)
    model.eval()

    print("Checkpoint loaded successfully.")
    print()

    return model


# ============================================================
# SPLIT MANIFEST
# ============================================================

def get_validation_cases():
    with open(SPLIT_MANIFEST, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    validation = manifest["validation"]

    if not isinstance(validation, list):
        raise RuntimeError(
            f"Expected validation to be a list, "
            f"got {type(validation).__name__}"
        )

    cases = []

    for item in validation:

        if not isinstance(item, dict):
            continue

        dataset = item["dataset"]
        image_path = Path(item["image"])
        label_path = Path(item["label"])

        # The manifest's case_id field is only the dataset name,
        # so derive the actual case ID from the image filename.
        filename = image_path.name

        if filename.endswith("_0000.nii.gz"):
            case_id = filename[:-len("_0000.nii.gz")]
        else:
            case_id = image_path.stem

        cases.append(
            {
                "dataset": dataset,
                "case_id": case_id,
                "image_path": image_path,
                "label_path": label_path,
            }
        )

    if len(cases) != 69:
        raise RuntimeError(
            f"Expected 69 validation cases, "
            f"found {len(cases)}"
        )

    return cases


# ============================================================
# FIND FILES
# ============================================================

def find_case_files(dataset, case_id, image_path=None, label_path=None):

    # Prefer the exact paths stored in the split manifest.
    if image_path is not None and label_path is not None:

        image_path = Path(image_path)
        label_path = Path(label_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Manifest image does not exist:\n{image_path}"
            )

        if not label_path.exists():
            raise FileNotFoundError(
                f"Manifest label does not exist:\n{label_path}"
            )

        return image_path, label_path

    # Fallback for manually supplied cases.
    base = (
        PROJECT_ROOT
        / "datasets"
        / "stage2"
        / dataset
    )

    image_candidates = list(
        base.glob(f"{case_id}_0000.nii.gz")
    )

    label_candidates = list(
        base.glob(f"{case_id}.nii.gz")
    )

    if not image_candidates:
        raise FileNotFoundError(
            f"Image not found for {dataset}/{case_id}"
        )

    if not label_candidates:
        raise FileNotFoundError(
            f"Label not found for {dataset}/{case_id}"
        )

    return image_candidates[0], label_candidates[0]

# ============================================================
# PATHOLOGY-CENTERED PATCH
# ============================================================

def extract_pathology_patch(image, label):

    pathology = np.argwhere(label == 2)

    if len(pathology) == 0:
        return None

    center = pathology[
        random.randint(0, len(pathology) - 1)
    ]

    zc, yc, xc = center

    pd, ph, pw = PATCH_SIZE

    shape = image.shape

    z0 = max(0, zc - pd // 2)
    y0 = max(0, yc - ph // 2)
    x0 = max(0, xc - pw // 2)

    z0 = min(z0, max(0, shape[0] - pd))
    y0 = min(y0, max(0, shape[1] - ph))
    x0 = min(x0, max(0, shape[2] - pw))

    z1 = min(shape[0], z0 + pd)
    y1 = min(shape[1], y0 + ph)
    x1 = min(shape[2], x0 + pw)

    image_patch = image[z0:z1, y0:y1, x0:x1]
    label_patch = label[z0:z1, y0:y1, x0:x1]

    # Pad if necessary
    pad_z = pd - image_patch.shape[0]
    pad_y = ph - image_patch.shape[1]
    pad_x = pw - image_patch.shape[2]

    if pad_z > 0 or pad_y > 0 or pad_x > 0:

        image_patch = np.pad(
            image_patch,
            (
                (0, max(0, pad_z)),
                (0, max(0, pad_y)),
                (0, max(0, pad_x)),
            ),
            mode="constant",
            constant_values=0,
        )

        label_patch = np.pad(
            label_patch,
            (
                (0, max(0, pad_z)),
                (0, max(0, pad_y)),
                (0, max(0, pad_x)),
            ),
            mode="constant",
            constant_values=0,
        )

    return image_patch, label_patch


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("STAGE 2 PATHOLOGY PATCH DIAGNOSTIC")
    print("=" * 72)

    print("Device:", DEVICE)
    print("Patch:", PATCH_SIZE)
    print()

    model = load_model()

    validation_cases = get_validation_cases()

    random.seed(42)

    # Shuffle so we don't systematically inspect only one dataset
    random.shuffle(validation_cases)

    tested = 0
    total_patches = 0

    total_gt_pathology = 0
    total_pred_pathology = 0
    total_correct_pathology = 0

    pathology_patch_count = 0

    print("Searching validation cases for pathology-containing patches...")
    print()

    for item in validation_cases:

        if tested >= NUM_CASES:
            break

        dataset = item["dataset"]
        case_id = item["case_id"]

        image_path, label_path = find_case_files(
            dataset,
            case_id,
            item["image_path"],
            item["label_path"],
        )

        image = nib.load(image_path).get_fdata(
            dtype=np.float32
        )

        label = nib.load(label_path).get_fdata(
            dtype=np.float32
        )

        label = label.astype(np.int64)

        pathology_voxels = int(
            np.sum(label == 2)
        )

        if pathology_voxels == 0:
            continue

        print("-" * 72)
        print(
            f"Case: {dataset} / {case_id}"
        )
        print(
            f"Full volume: {image.shape}"
        )
        print(
            f"Ground-truth pathology voxels: "
            f"{pathology_voxels:,}"
        )

        for patch_number in range(
            PATCHES_PER_CASE
        ):

            result = extract_pathology_patch(
                image,
                label,
            )

            if result is None:
                continue

            image_patch, label_patch = result

            gt_pathology = int(
                np.sum(label_patch == 2)
            )

            if gt_pathology == 0:
                print(
                    "WARNING: extracted patch has no pathology"
                )
                continue

            # Preprocessed images are already [0,1]
            image_patch = np.clip(
                image_patch,
                0.0,
                1.0,
            )

            x = torch.from_numpy(
                image_patch
            ).float()

            x = x.unsqueeze(0).unsqueeze(0)
            x = x.to(DEVICE)

            with torch.no_grad():

                outputs = model(x)

                # Highest-resolution output
                logits = outputs[0]

                probabilities = torch.softmax(
                    logits,
                    dim=1,
                )

                prediction = torch.argmax(
                    probabilities,
                    dim=1,
                )

            prediction = (
                prediction[0]
                .cpu()
                .numpy()
            )

            probabilities = (
                probabilities[0]
                .cpu()
                .numpy()
            )

            pred_pathology = int(
                np.sum(prediction == 2)
            )

            correct_pathology = int(
                np.sum(
                    (prediction == 2)
                    & (label_patch == 2)
                )
            )

            pathology_probability = (
                probabilities[2]
            )

            gt_probability_region = (
                pathology_probability[
                    label_patch == 2
                ]
            )

            mean_pathology_probability = float(
                gt_probability_region.mean()
            )

            max_pathology_probability = float(
                gt_probability_region.max()
            )

            total_gt_pathology += gt_pathology
            total_pred_pathology += pred_pathology
            total_correct_pathology += correct_pathology

            total_patches += 1
            pathology_patch_count += 1

            print(
                f"  Patch {patch_number + 1}: "
                f"GT pathology={gt_pathology:,} | "
                f"Pred pathology={pred_pathology:,} | "
                f"Correct={correct_pathology:,}"
            )

            print(
                f"    Pathology probability on GT voxels: "
                f"mean={mean_pathology_probability:.4f} | "
                f"max={max_pathology_probability:.4f}"
            )

        tested += 1

    print()
    print("=" * 72)
    print("PATHOLOGY PATCH DIAGNOSTIC SUMMARY")
    print("=" * 72)

    print("Cases tested:", tested)
    print("Patches tested:", total_patches)

    print(
        f"Total GT pathology voxels: "
        f"{total_gt_pathology:,}"
    )

    print(
        f"Total predicted pathology voxels: "
        f"{total_pred_pathology:,}"
    )

    print(
        f"Total correctly predicted pathology voxels: "
        f"{total_correct_pathology:,}"
    )

    if total_gt_pathology > 0:

        recall = (
            total_correct_pathology
            / total_gt_pathology
        )

        print(
            f"Patch pathology recall: "
            f"{recall:.6f}"
        )

    print("=" * 72)


if __name__ == "__main__":
    main()