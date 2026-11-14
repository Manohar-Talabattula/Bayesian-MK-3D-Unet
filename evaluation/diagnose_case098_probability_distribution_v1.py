import nibabel as nib
import numpy as np
from pathlib import Path

root = Path.cwd()
prob_path = root / "evaluation/diagnostics_case098_stage2_prob_v1/case098_stage2_lesion_probability.nii.gz"
gt_path = root / "datasets/stage2/Task07_Pancreas/Task07_Pancreas_098.nii.gz"

p = nib.load(str(prob_path)).get_fdata().astype(np.float32)
gt = nib.load(str(gt_path)).get_fdata().astype(np.uint8)

lesion = gt == 2
pancreas = gt == 1
background = gt == 0

def show_stats(name, values):
    if values.size == 0:
        print(f"\n{name}: EMPTY")
        return
    q = np.percentile(values, [0, 25, 50, 75, 90, 95, 99, 99.9, 100])
    print(f"\n{name} — voxels: {values.size:,}")
    print("Min, 25%, 50%, 75%, 90%, 95%, 99%, 99.9%, Max:")
    print(np.array2string(q, precision=6))

print("Probability shape:", p.shape)
print("GT shape:", gt.shape)

show_stats("LESION probability", p[lesion])
show_stats("PANCREAS probability", p[pancreas])
show_stats("BACKGROUND probability", p[background])

print("\n" + "=" * 75)
print("THRESHOLD SWEEP — UNGATED STAGE 2")
print("=" * 75)
print(f"{'Threshold':>10} | {'Dice':>8} | {'Precision':>10} | {'Recall':>8} | {'Pred voxels':>12}")

for threshold in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]:
    pred = p >= threshold
    tp = int(np.logical_and(pred, lesion).sum())
    fp = int(np.logical_and(pred, ~lesion).sum())
    fn = int(np.logical_and(~pred, lesion).sum())

    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0

    print(f"{threshold:>10.2f} | {dice:>8.4f} | {precision:>10.4f} | {recall:>8.4f} | {int(pred.sum()):>12,}")

flat_idx = int(np.argmax(p))
coord = np.unravel_index(flat_idx, p.shape)

print("\nHighest-probability voxel:")
print("Coordinate:", coord)
print("Probability:", float(p[coord]))
print("Ground-truth class at coordinate:", int(gt[coord]))

print("\nLesion voxels with probability >= thresholds:")
for threshold in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 0.90]:
    count = int((p[lesion] >= threshold).sum())
    print(f"{threshold:.2f}: {count:,} / {int(lesion.sum()):,}")

