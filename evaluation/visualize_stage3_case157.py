from pathlib import Path
import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

ROOT = Path(__file__).resolve().parents[1]

GT_PATH = ROOT / "datasets/stage2/Task07_Pancreas/Task07_Pancreas_157.nii.gz"
PRED_PATH = ROOT / "evaluation/task07_stage3_exports_overlap075/masks/Task07_Pancreas_Task07_Pancreas_157_0000_lesion_mask.nii.gz"

OUT_DIR = ROOT / "evaluation/visual_results_stage3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading masks...", flush=True)

gt_nii = nib.load(str(GT_PATH))
pred_nii = nib.load(str(PRED_PATH))

gt = np.asarray(gt_nii.dataobj)
pred = np.asarray(pred_nii.dataobj)

# Ground truth labels: 1 = pancreas, 2 = lesion
pancreas = gt == 1
gt_lesion = gt == 2
pred_lesion = pred > 0

# Downsample for manageable 3D rendering
step = 2
pancreas = pancreas[::step, ::step, ::step]
gt_lesion = gt_lesion[::step, ::step, ::step]
pred_lesion = pred_lesion[::step, ::step, ::step]

print("Downsampled shape:", pancreas.shape, flush=True)
print("Pancreas voxels:", int(pancreas.sum()), flush=True)
print("GT lesion voxels:", int(gt_lesion.sum()), flush=True)
print("Predicted lesion voxels:", int(pred_lesion.sum()), flush=True)

def draw_voxels(ax, mask, color, title):
    if not mask.any():
        ax.text2D(0.25, 0.5, "No voxels to display", transform=ax.transAxes)
        ax.set_title(title)
        return

    ax.voxels(
        mask,
        facecolors=color,
        edgecolor="none",
        shade=True
    )
    ax.set_title(title, pad=12)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=25, azim=45)
    ax.set_box_aspect(mask.shape)

fig = plt.figure(figsize=(17, 6))

ax1 = fig.add_subplot(131, projection="3d")
draw_voxels(ax1, pancreas, "lightblue", "Ground Truth Pancreas")

ax2 = fig.add_subplot(132, projection="3d")
draw_voxels(ax2, gt_lesion, "green", "Ground Truth Lesion")

ax3 = fig.add_subplot(133, projection="3d")
draw_voxels(ax3, pred_lesion, "red", "Stage 3 Predicted Lesion")

fig.suptitle(
    "Bayesian MK-3D-UNet | Stage 3 | Case 157",
    fontsize=15
)
plt.tight_layout()

output = OUT_DIR / "case157_stage3_3d_visualization.png"
print("Rendering 3D figure...", flush=True)
plt.savefig(output, dpi=160, bbox_inches="tight")
plt.close(fig)

print("\nSaved:", output, flush=True)
print("3D visualization completed.", flush=True)
