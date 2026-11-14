from pathlib import Path
import yaml
import numpy as np
import nibabel as nib

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Spacingd,
    ScaleIntensityRanged,
    EnsureTyped,
)


ROOT = Path(__file__).resolve().parents[1]

with open(ROOT / "configs" / "stage1_dataset.yaml") as f:
    cfg = yaml.safe_load(f)

prep = cfg["preprocessing"]

SPACING = (
    prep["target_spacing"]["x"],
    prep["target_spacing"]["y"],
    prep["target_spacing"]["z"],
)

CLIP_MIN = prep["intensity"]["clip_min"]
CLIP_MAX = prep["intensity"]["clip_max"]

SRC = ROOT / "datasets" / "stage1"
DST = ROOT / "datasets" / "stage1_preprocessed"


def process_split(split):
    image_dir = SRC / split / "images"
    label_dir = SRC / split / "labels"

    out_image_dir = DST / split / "images"
    out_label_dir = DST / split / "labels"

    out_image_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(image_dir.glob("*.nii.gz"))

    transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(
            keys=["image", "label"],
            pixdim=SPACING,
            mode=("bilinear", "nearest"),
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=CLIP_MIN,
            a_max=CLIP_MAX,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        EnsureTyped(keys=["image", "label"]),
    ])

    for i, image_path in enumerate(images, 1):
        label_path = label_dir / image_path.name

        sample = transforms({
            "image": str(image_path),
            "label": str(label_path),
        })

        image = sample["image"].numpy()[0]
        label = sample["label"].numpy()[0]

        image_nii = nib.Nifti1Image(
            image.astype(np.float32),
            np.diag([SPACING[0], SPACING[1], SPACING[2], 1.0])
        )

        label_nii = nib.Nifti1Image(
            label.astype(np.int16),
            np.diag([SPACING[0], SPACING[1], SPACING[2], 1.0])
        )

        nib.save(
            image_nii,
            out_image_dir / image_path.name
        )

        nib.save(
            label_nii,
            out_label_dir / label_path.name
        )

        print(f"{split}: {i}/{len(images)} - {image_path.name}")


process_split("train")
process_split("val")

print("\nMONAI preprocessing completed successfully.")