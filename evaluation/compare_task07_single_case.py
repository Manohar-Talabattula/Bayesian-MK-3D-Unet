
from pathlib import Path
import json

import nibabel as nib
import numpy as np
import torch

from dynamic_network_architectures.architectures.unet import PlainConvUNet
from monai.inferers import sliding_window_inference


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path.cwd()
CASE_NAME = "Task07_Pancreas_157"

CHECKPOINT = (
    ROOT / "datasets" / "stage2" / "models"
    / "task07_finetune_v2" / "latest_task07_finetune.pth"
)

MANIFEST = (
    ROOT / "datasets" / "stage2" / "Task07_Pancreas"
    / "task07_split_manifest.json"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATCH = (32, 96, 96)
THRESHOLD = 0.80
OVERLAPS = (0.25, 0.50, 0.75)


# ============================================================
# MODEL
# ============================================================

def build_model():
    return PlainConvUNet(
        input_channels=1,
        n_stages=6,
        features_per_stage=[32, 64, 128, 256, 320, 320],
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
        n_conv_per_stage=[2, 2, 2, 2, 2, 2],
        num_classes=3,
        n_conv_per_stage_decoder=[2, 2, 2, 2, 2],
        conv_bias=True,
        norm_op=torch.nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=torch.nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
    )


def main_output(output):
    if isinstance(output, dict):
        return output["main"]
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def dice_score(pred, truth):
    pred = np.asarray(pred, dtype=bool)
    truth = np.asarray(truth, dtype=bool)

    denominator = int(pred.sum()) + int(truth.sum())
    if denominator == 0:
        return 1.0

    intersection = int(np.logical_and(pred, truth).sum())
    return 2.0 * intersection / denominator


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


# ============================================================
# LOAD ONE VALIDATION CASE
# ============================================================

def main():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    entries = manifest.get("validation_cases", [])

    entry = next(
        (item for item in entries if CASE_NAME in str(item["image"])),
        None,
    )

    if entry is None:
        raise RuntimeError(
            f"{CASE_NAME} was not found in validation_cases."
        )

    image_path = resolve_path(entry["image"])
    label_path = resolve_path(entry["label"])

    print("Loading case:", CASE_NAME)
    print("Device:", DEVICE)

    # Match the Stage 2 training loader: load the original NIfTI
    # array, clip image intensities to [0, 1], then transpose to Z,Y,X.
    image_nii = nib.load(str(image_path))
    label_nii = nib.load(str(label_path))

    image = np.transpose(
        np.clip(image_nii.get_fdata(dtype=np.float32), 0.0, 1.0),
        (2, 1, 0),
    ).copy()

    label = np.transpose(
        np.rint(label_nii.get_fdata()).astype(np.uint8),
        (2, 1, 0),
    ).copy()

    if image.shape != label.shape:
        raise RuntimeError(
            f"Image/label shape mismatch: {image.shape} vs {label.shape}"
        )

    # ========================================================
    # REPRODUCE THE LESION-CENTRED VALIDATION PATCH
    # ========================================================

    pz, py, px = PATCH
    z, y, x = label.shape

    pad_z = max(0, pz - z)
    pad_y = max(0, py - y)
    pad_x = max(0, px - x)

    if pad_z or pad_y or pad_x:
        pad_width = (
            (pad_z // 2, pad_z - pad_z // 2),
            (pad_y // 2, pad_y - pad_y // 2),
            (pad_x // 2, pad_x - pad_x // 2),
        )

        image = np.pad(image, pad_width, mode="constant")
        label = np.pad(label, pad_width, mode="constant")

    lesion_voxels = np.argwhere(label == 2)

    if len(lesion_voxels) == 0:
        raise RuntimeError("No class-2 pathology voxels found in this case.")

    z, y, x = label.shape

    center_z, center_y, center_x = np.median(
        lesion_voxels, axis=0
    ).astype(int)

    sz = int(np.clip(center_z - pz // 2, 0, z - pz))
    sy = int(np.clip(center_y - py // 2, 0, y - py))
    sx = int(np.clip(center_x - px // 2, 0, x - px))

    patch_image = image[
        sz:sz + pz,
        sy:sy + py,
        sx:sx + px,
    ].copy()

    patch_label = label[
        sz:sz + pz,
        sy:sy + py,
        sx:sx + px,
    ].copy()

    truth = patch_label == 2

    print("Patch origin (Z,Y,X):", (sz, sy, sx))
    print("Ground-truth lesion voxels:", int(truth.sum()))

    # ========================================================
    # LOAD EXISTING CHECKPOINT — NO TRAINING
    # ========================================================

    model = build_model().to(DEVICE)

    checkpoint = torch.load(
        str(CHECKPOINT),
        map_location=DEVICE,
        weights_only=False,
    )

    if isinstance(checkpoint, dict):
        for key in (
            "model_state",
            "model_state_dict",
            "network_weights",
            "state_dict",
        ):
            if key in checkpoint:
                checkpoint = checkpoint[key]
                break

    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    # ========================================================
    # PATCH AND FULL-VOLUME COMPARISON
    # ========================================================

    with torch.inference_mode():
        patch_tensor = torch.from_numpy(
            patch_image
        ).float()[None, None].to(DEVICE)

        patch_probability = torch.softmax(
            main_output(model(patch_tensor)),
            dim=1,
        )[0, 2].cpu().numpy()

        patch_prediction = patch_probability >= THRESHOLD

        print("\n=== SLIDING-WINDOW OVERLAP COMPARISON ===")
        print("Case:", CASE_NAME)
        print("Threshold:", THRESHOLD)
        print(
            "Patch inference:",
            f"predicted voxels={int(patch_prediction.sum())}",
            f"| Dice={dice_score(patch_prediction, truth):.4f}",
        )

        full_tensor = torch.from_numpy(
            image
        ).float()[None, None].to(DEVICE)

        for overlap in OVERLAPS:
            print(f"\nRunning full-volume inference at overlap={overlap:.2f}...")

            logits = sliding_window_inference(
                inputs=full_tensor,
                roi_size=PATCH,
                sw_batch_size=2,
                predictor=lambda tensor: main_output(model(tensor)),
                overlap=overlap,
                mode="gaussian",
            )

            full_probability = torch.softmax(
                logits, dim=1
            )[0, 2].cpu().numpy()

            crop_probability = full_probability[
                sz:sz + pz,
                sy:sy + py,
                sx:sx + px,
            ]

            full_prediction = crop_probability >= THRESHOLD
	    
            intersection = int(np.logical_and(full_prediction, truth).sum())
            false_positive = int(np.logical_and(full_prediction, ~truth).sum())
            false_negative = int(np.logical_and(~full_prediction, truth).sum())

            print(
                f"  TP={intersection} | FP={false_positive} | "
                f"FN={false_negative} | GT={int(truth.sum())}"
            )
            print(
                f"  Probability min={crop_probability.min():.6f} | "
                f"max={crop_probability.max():.6f} | "
                f"mean={crop_probability.mean():.6f}"
            )


            score = dice_score(full_prediction, truth)
            probability_difference = float(
                np.mean(np.abs(patch_probability - crop_probability))
            )

            print(
                f"Overlap={overlap:.2f} | "
                f"predicted voxels={int(full_prediction.sum())} | "
                f"Dice={score:.4f} | "
                f"mean probability difference={probability_difference:.6f}"
            )

            del logits, full_probability, crop_probability

            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

            del full_prediction

        del full_tensor, patch_tensor

    print("\nTest complete. No model training was performed.")
    print("The original checkpoint and evaluator were not modified.")


if __name__ == "__main__":
    main()
