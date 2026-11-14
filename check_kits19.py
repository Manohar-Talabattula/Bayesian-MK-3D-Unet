from pathlib import Path
import nibabel as nib
import numpy as np

root = Path(r"C:\Users\HPCLAB\Downloads\KiTS19")
img_dir = root / "images"
seg_dir = root / "segmentations"

imgs = sorted(img_dir.glob("*.nii.gz"))
segs = sorted(seg_dir.glob("*.nii.gz"))

print("=" * 60)
print("KiTS19 COMPLETE DATASET INTEGRITY CHECK")
print("=" * 60)

print("\nFILE COUNTS")
print("-" * 60)
print(f"Images:         {len(imgs)}")
print(f"Segmentations:  {len(segs)}")
print(f"Total:          {len(imgs) + len(segs)}")

# Extract case IDs
img_ids = {
    p.name.replace("case_", "").replace(".nii.gz", "")
    for p in imgs
}

seg_ids = {
    p.name.replace("case_", "").replace(".nii.gz", "")
    for p in segs
}

print("\nCASE ID MATCH")
print("-" * 60)

missing_segs = sorted(img_ids - seg_ids)
missing_imgs = sorted(seg_ids - img_ids)

print("Missing segmentations:", missing_segs)
print("Missing images:       ", missing_imgs)

# Read all images
print("\nREADING ALL CT IMAGES")
print("-" * 60)

bad_images = []
valid_images = 0

for i, p in enumerate(imgs, 1):
    try:
        nii = nib.load(str(p))
        data = nii.get_fdata(dtype=np.float32)

        if data.size == 0:
            raise ValueError("Empty image")

        valid_images += 1

        if i % 25 == 0 or i == len(imgs):
            print(f"  {i}/{len(imgs)} images checked")

    except Exception as e:
        bad_images.append((p.name, str(e)))

# Read all segmentations
print("\nREADING ALL SEGMENTATIONS")
print("-" * 60)

bad_segs = []
valid_segs = 0

for i, p in enumerate(segs, 1):
    try:
        nii = nib.load(str(p))
        data = nii.get_fdata(dtype=np.float32)

        if data.size == 0:
            raise ValueError("Empty segmentation")

        valid_segs += 1

        if i % 25 == 0 or i == len(segs):
            print(f"  {i}/{len(segs)} segmentations checked")

    except Exception as e:
        bad_segs.append((p.name, str(e)))

# Check image/segmentation compatibility
print("\nCHECKING IMAGE ↔ SEGMENTATION PAIRS")
print("-" * 60)

pair_errors = []

common_ids = sorted(img_ids & seg_ids)

for case_id in common_ids:
    img_path = img_dir / f"case_{case_id}.nii.gz"
    seg_path = seg_dir / f"case_{case_id}.nii.gz"

    try:
        img = nib.load(str(img_path))
        seg = nib.load(str(seg_path))

        if img.shape != seg.shape:
            pair_errors.append(
                f"case_{case_id}: shape mismatch "
                f"{img.shape} vs {seg.shape}"
            )

    except Exception as e:
        pair_errors.append(
            f"case_{case_id}: {e}"
        )

# Check labels
print("\nCHECKING SEGMENTATION LABELS")
print("-" * 60)

label_errors = []
all_labels = set()

for p in segs:
    try:
        data = nib.load(str(p)).get_fdata()

        unique = np.unique(data)

        all_labels.update(unique.tolist())

        invalid = unique[~np.isin(unique, [0, 1, 2])]

        if len(invalid) > 0:
            label_errors.append(
                f"{p.name}: invalid labels {invalid.tolist()}"
            )

    except Exception:
        pass

# Final report
print("\n" + "=" * 60)
print("FINAL RESULT")
print("=" * 60)

print(f"Images found:             {len(imgs)}")
print(f"Segmentations found:      {len(segs)}")
print(f"Readable images:          {valid_images}")
print(f"Readable segmentations:   {valid_segs}")
print(f"Bad images:               {len(bad_images)}")
print(f"Bad segmentations:        {len(bad_segs)}")
print(f"Pair errors:              {len(pair_errors)}")
print(f"Label errors:             {len(label_errors)}")
print(f"All labels found:         {sorted(all_labels)}")

if bad_images:
    print("\nBAD IMAGES:")
    for x in bad_images:
        print(x)

if bad_segs:
    print("\nBAD SEGMENTATIONS:")
    for x in bad_segs:
        print(x)

if pair_errors:
    print("\nPAIR ERRORS:")
    for x in pair_errors:
        print(x)

if label_errors:
    print("\nLABEL ERRORS:")
    for x in label_errors:
        print(x)

print("\n" + "=" * 60)

if (
    len(imgs) == 210
    and len(segs) == 210
    and not bad_images
    and not bad_segs
    and not missing_imgs
    and not missing_segs
    and not pair_errors
    and not label_errors
):
    print("STATUS: KiTS19 DATASET PASSED INTEGRITY CHECK")
else:
    print("STATUS: PROBLEMS FOUND — DO NOT PREPROCESS YET")

print("=" * 60)