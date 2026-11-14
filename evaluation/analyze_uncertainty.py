from pathlib import Path
import numpy as np
import nibabel as nib

ROOT = Path(__file__).resolve().parents[1]
UNCERTAINTY_DIR = ROOT / "evaluation" / "uncertainty_results"

FILES = sorted(UNCERTAINTY_DIR.glob("*_uncertainty_normalized.nii.gz"))

if not FILES:
    raise FileNotFoundError(
        f"No normalized uncertainty files found in: {UNCERTAINTY_DIR}"
    )

print("=" * 70)
print("NORMALIZED UNCERTAINTY ANALYSIS")
print("=" * 70)
print(f"Files found: {len(FILES)}")
print()

case_means = []
case_medians = []
case_max = []
case_p95 = []
case_p99 = []

thresholds = [0.10, 0.25, 0.50, 0.75, 0.90]

threshold_percentages = {
    threshold: []
    for threshold in thresholds
}

for index, path in enumerate(FILES, start=1):

    nii = nib.load(str(path))
    data = nii.get_fdata(dtype=np.float32)

    # Remove invalid values
    data = data[np.isfinite(data)]

    if data.size == 0:
        print(f"[WARNING] Empty uncertainty file: {path.name}")
        continue

    # Clamp to expected normalized range
    data = np.clip(data, 0.0, 1.0)

    case_means.append(float(np.mean(data)))
    case_medians.append(float(np.median(data)))
    case_max.append(float(np.max(data)))
    case_p95.append(float(np.percentile(data, 95)))
    case_p99.append(float(np.percentile(data, 99)))

    for threshold in thresholds:
        percentage = float(np.mean(data >= threshold) * 100.0)
        threshold_percentages[threshold].append(percentage)

    if index <= 5:
        print(
            f"{path.name}: "
            f"mean={np.mean(data):.6f}, "
            f"median={np.median(data):.6f}, "
            f"max={np.max(data):.6f}, "
            f"p95={np.percentile(data, 95):.6f}, "
            f"p99={np.percentile(data, 99):.6f}"
        )

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"Cases successfully analyzed : {len(case_means)}")

print(f"Mean case uncertainty       : {np.mean(case_means):.6f}")
print(f"Median case uncertainty     : {np.median(case_means):.6f}")
print(f"Mean case maximum           : {np.mean(case_max):.6f}")
print(f"Maximum observed            : {np.max(case_max):.6f}")
print(f"Mean 95th percentile        : {np.mean(case_p95):.6f}")
print(f"Mean 99th percentile        : {np.mean(case_p99):.6f}")

print()
print("=" * 70)
print("VOXELS ABOVE NORMALIZED UNCERTAINTY THRESHOLDS")
print("=" * 70)

print(f"{'Threshold':<15}{'Mean voxel percentage':>25}")
print("-" * 40)

for threshold in thresholds:
    values = threshold_percentages[threshold]

    if values:
        print(
            f"{threshold:<15.2f}"
            f"{np.mean(values):>24.6f}%"
        )

print()
print("=" * 70)
print("INTERPRETATION GUIDE")
print("=" * 70)

print("Normalized uncertainty is scaled approximately from 0 to 1.")
print("Higher values indicate regions where the MC-Dropout predictions")
print("show greater variability across stochastic forward passes.")
print()
print("Thresholds are reported for analysis only.")
print("They do not change the model predictions or checkpoint.")
print("=" * 70)