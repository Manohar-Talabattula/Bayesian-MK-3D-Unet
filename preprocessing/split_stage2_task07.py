import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "datasets" / "stage2" / "Task07_Pancreas"
OUTPUT_FILE = DATA_DIR / "task07_split_manifest.json"

SEED = 42
TRAIN_RATIO = 0.80
EXPECTED_CASES = 281


def main():
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {DATA_DIR}")

    image_files = sorted(DATA_DIR.glob("*_0000.nii.gz"))

    if not image_files:
        raise RuntimeError(f"No preprocessed CT images found in {DATA_DIR}")

    cases = []

    for image_path in image_files:
        case_id = image_path.name.removesuffix("_0000.nii.gz")
        label_path = DATA_DIR / f"{case_id}.nii.gz"

        if not label_path.is_file():
            raise FileNotFoundError(
                f"Missing mask for {case_id}: {label_path}"
            )

        cases.append({
            "case_id": case_id,
            "image": image_path.relative_to(PROJECT_ROOT).as_posix(),
            "label": label_path.relative_to(PROJECT_ROOT).as_posix(),
        })

    if len(cases) != EXPECTED_CASES:
        raise RuntimeError(
            f"Expected {EXPECTED_CASES} cases, found {len(cases)}. "
            "Check the processed dataset before creating the split."
        )

    # Shuffle case IDs deterministically; each patient stays in one split.
    rng = random.Random(SEED)
    rng.shuffle(cases)

    train_count = round(len(cases) * TRAIN_RATIO)
    train_cases = cases[:train_count]
    val_cases = cases[train_count:]

    train_ids = {case["case_id"] for case in train_cases}
    val_ids = {case["case_id"] for case in val_cases}

    if train_ids & val_ids:
        raise RuntimeError("Data leakage detected: a case appears in both splits.")

    if train_ids | val_ids != {case["case_id"] for case in cases}:
        raise RuntimeError("Split does not cover every case exactly once.")

    manifest = {
        "dataset": "MSD Task07 Pancreas",
        "seed": SEED,
        "train_ratio": TRAIN_RATIO,
        "split_method": "Deterministic case-level random split",
        "total_cases": len(cases),
        "train_count": len(train_cases),
        "validation_count": len(val_cases),
        "train_cases": train_cases,
        "validation_cases": val_cases,
    }

    OUTPUT_FILE.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print("=" * 65)
    print("TASK07 SPLIT CREATED")
    print("=" * 65)
    print(f"Total cases:      {len(cases)}")
    print(f"Training cases:   {len(train_cases)}")
    print(f"Validation cases: {len(val_cases)}")
    print(f"Random seed:      {SEED}")
    print(f"Manifest:         {OUTPUT_FILE}")
    print("Case overlap:     0")
    print("Result:           SPLIT PASSED")


if __name__ == "__main__":
    main()
