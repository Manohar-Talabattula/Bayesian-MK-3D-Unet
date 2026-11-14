import json
import random
from pathlib import Path

ROOT = Path(r"C:\Users\HPCLAB\Documents\MK_3D_UNET_PROJECT")
STAGE2 = ROOT / "datasets" / "stage2"
OUT = STAGE2 / "split" / "stage2_split_manifest.json"

random.seed(42)

datasets = [
    ("LiTS_Batch1", STAGE2 / "LiTS_Batch1", 28),
    ("LiTS_Batch2", STAGE2 / "LiTS_Batch2", 103),
    ("KiTS19", STAGE2 / "KiTS19", 210),
]

all_cases = []

for dataset_name, folder, expected_count in datasets:
    image_files = sorted(folder.glob("*_????_0000.nii.gz"))

    print(f"{dataset_name}: found {len(image_files)} images")

    if len(image_files) != expected_count:
        raise RuntimeError(
            f"{dataset_name}: expected {expected_count}, found {len(image_files)}"
        )

    for image_path in image_files:
        label_name = image_path.name[:-len("_0000.nii.gz")] + ".nii.gz"
        label_path = folder / label_name

        if not label_path.exists():
            raise RuntimeError(
                f"Missing label for {image_path.name}: {label_name}"
            )

        case_id = image_path.name[:-len("_0000_0000.nii.gz")]

        all_cases.append({
            "case_id": case_id,
            "dataset": dataset_name,
            "image": str(image_path),
            "label": str(label_path)
        })

train = []
validation = []

for dataset_name, _, _ in datasets:
    cases = [
        c for c in all_cases
        if c["dataset"] == dataset_name
    ]

    random.shuffle(cases)

    validation_count = round(len(cases) * 0.20)

    validation.extend(cases[:validation_count])
    train.extend(cases[validation_count:])

random.shuffle(train)
random.shuffle(validation)

manifest = {
    "stage": 2,
    "task": "general_abnormal_pathological_tissue_segmentation",
    "classes": {
        "0": "background",
        "1": "healthy_non_lesional_tissue",
        "2": "abnormal_pathological_tissue"
    },
    "seed": 42,
    "split_strategy": "stratified_by_dataset_then_case_level",
    "total_cases": len(all_cases),
    "train_cases": len(train),
    "validation_cases": len(validation),
    "train": train,
    "validation": validation
}

OUT.parent.mkdir(parents=True, exist_ok=True)

OUT.write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8"
)

print()
print("=" * 60)
print("STAGE 2 SPLIT CREATED")
print("=" * 60)
print(f"Total cases:      {len(all_cases)}")
print(f"Training cases:   {len(train)}")
print(f"Validation cases: {len(validation)}")
print()

for dataset_name, _, _ in datasets:
    train_count = sum(
        c["dataset"] == dataset_name
        for c in train
    )

    val_count = sum(
        c["dataset"] == dataset_name
        for c in validation
    )

    print(
        f"{dataset_name}: "
        f"train={train_count}, "
        f"val={val_count}"
    )

print()
print(f"Manifest: {OUT}")
print("=" * 60)
