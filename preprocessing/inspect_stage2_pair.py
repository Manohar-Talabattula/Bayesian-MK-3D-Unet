from pathlib import Path
import argparse
import json
import nibabel as nib
import numpy as np

def inspect_pair(image_path, label_path):
    img = nib.load(str(image_path))
    lab = nib.load(str(label_path))
    img_shape = tuple(img.shape)
    lab_shape = tuple(lab.shape)
    img_spacing = tuple(float(x) for x in img.header.get_zooms()[:3])
    lab_spacing = tuple(float(x) for x in lab.header.get_zooms()[:3])
    labels = np.unique(np.asanyarray(lab.dataobj))
    return {
        "image": str(image_path),
        "label": str(label_path),
        "image_shape": img_shape,
        "label_shape": lab_shape,
        "image_spacing": img_spacing,
        "label_spacing": lab_spacing,
        "shape_match": img_shape == lab_shape,
        "spacing_match": np.allclose(img_spacing, lab_spacing, atol=1e-4),
        "unique_labels": [int(x) for x in labels.tolist()],
        "label_counts": {
            str(int(x)): int(np.count_nonzero(np.asanyarray(lab.dataobj) == x))
            for x in labels
        },
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    result = inspect_pair(Path(args.image), Path(args.label))
    print(json.dumps(result, indent=2))

    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
