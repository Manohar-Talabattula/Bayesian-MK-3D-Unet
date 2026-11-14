from pathlib import Path
import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "stage1_preprocessed"

expected = {
    "train": 1312,
    "val": 157
}

for split, expected_count in expected.items():

    image_dir = DATASET / split / "images"
    label_dir = DATASET / split / "labels"

    images = sorted(image_dir.glob("*.nii.gz"))
    labels = sorted(label_dir.glob("*.nii.gz"))

    print(f"\n{split.upper()}")
    print("-" * 40)

    print("Images:", len(images))
    print("Labels:", len(labels))

    if len(images) != expected_count:
        print(f"WARNING: expected {expected_count} images")

    if len(labels) != expected_count:
        print(f"WARNING: expected {expected_count} labels")

    image_names = {x.name for x in images}
    label_names = {x.name for x in labels}

    missing_labels = image_names - label_names
    missing_images = label_names - image_names

    print("Missing labels:", len(missing_labels))
    print("Missing images:", len(missing_images))

    if images:
        image = nib.load(images[0])
        label = nib.load(labels[0])

        image_data = image.get_fdata()
        label_data = label.get_fdata()

        print("\nSample:", images[0].name)
        print("Image shape:", image_data.shape)
        print("Label shape:", label_data.shape)
        print("Image spacing:", image.header.get_zooms()[:3])
        print("Label spacing:", label.header.get_zooms()[:3])
        print("Image dtype:", image_data.dtype)
        print("Label dtype:", label_data.dtype)

        label_ids = np.unique(label_data)
        print("Label IDs:", label_ids.astype(int).tolist())

        print("Image min:", float(image_data.min()))
        print("Image max:", float(image_data.max()))

        if image_data.shape != label_data.shape:
            print("ERROR: image and label shapes do not match!")

        if not np.all(np.isin(label_ids, np.arange(13))):
            print("ERROR: unexpected label ID found!")

print("\nVerification completed.")