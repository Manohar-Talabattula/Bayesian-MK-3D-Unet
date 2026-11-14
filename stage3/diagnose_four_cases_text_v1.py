import json
import re
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

root = Path.cwd()
manifest = json.loads(
    (root / "datasets/stage2/Task07_Pancreas/task07_split_manifest.json")
    .read_text(encoding="utf-8")
)
results = json.loads(
    (root / "evaluation/task07_stage3_exports/stage3_evaluation_results.json")
    .read_text(encoding="utf-8")
)

val_key = next(
    k for k, v in manifest.items()
    if isinstance(v, list) and "validation" in k.lower()
)
entries = {
    int(re.findall(r"Task07_Pancreas_(\d+)", e["case_id"])[-1]): e
    for e in manifest[val_key]
}
predictions = {
    int(re.findall(r"Task07_Pancreas_(\d+)", r["case"])[-1]): r
    for r in results["case_results"]
}

def components(mask, affine):
    lab, n = ndimage.label(
        mask, structure=np.ones((3, 3, 3), dtype=np.uint8)
    )
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    ids = sorted(range(1, n + 1), key=lambda i: int(sizes[i]), reverse=True)
    info = []
    for cid in ids[:5]:
        coords = np.argwhere(lab == cid)
        if not len(coords):
            continue
        center_voxel = coords.mean(axis=0)
        center_ras = nib.affines.apply_affine(
            affine, center_voxel.reshape(1, 3)
        )[0]
        info.append({
            "id": cid,
            "voxels": int(sizes[cid]),
            "center": center_ras,
            "overlap_gt": int(np.logical_and(lab == cid, gt_mask).sum()),
        })
    return lab, n, info

for number in [198, 157, 318, 391]:
    e = entries[number]
    r = predictions[number]
    gt_img = nib.load(str(root / e["label"]))
    pred_path = Path(r["gated_mask_path"])
    if not pred_path.is_absolute():
        pred_path = root / pred_path
    pred_img = nib.load(str(pred_path))

    gt_mask = np.asanyarray(gt_img.dataobj) == 2
    pred_mask = np.asanyarray(pred_img.dataobj) > 0
    if gt_mask.shape != pred_mask.shape:
        raise RuntimeError(f"Shape mismatch in case {number}")
    if not np.allclose(gt_img.affine, pred_img.affine, atol=1e-3):
        raise RuntimeError(f"Affine mismatch in case {number}")

    voxel_mm3 = abs(float(np.linalg.det(gt_img.affine[:3, :3])))
    gt_vox = int(gt_mask.sum())
    pred_vox = int(pred_mask.sum())
    inter = int(np.logical_and(gt_mask, pred_mask).sum())
    fp = int(np.logical_and(pred_mask, ~gt_mask).sum())
    fn = int(np.logical_and(gt_mask, ~pred_mask).sum())
    dice = 2 * inter / (gt_vox + pred_vox) if gt_vox + pred_vox else 1.0

    gt_lab, gt_n, gt_comps = components(gt_mask, gt_img.affine)
    pred_lab, pred_n, pred_comps = components(pred_mask, pred_img.affine)

    print("\n" + "=" * 68)
    print(f"CASE {number}: {e['case_id']}")
    print(f"Dice={dice:.4f}; GT={gt_vox*voxel_mm3/1000:.3f} mL; "
          f"Pred={pred_vox*voxel_mm3/1000:.3f} mL")
    print(f"Overlap voxels={inter}; false-positive voxels={fp}; "
          f"missed GT voxels={fn}")
    print(f"Connected components: GT={gt_n}, prediction={pred_n}")

    if gt_comps:
        gt_center = gt_comps[0]["center"]
        print("Largest GT component:")
        print(f"  voxels={gt_comps[0]['voxels']}; "
              f"centroid RAS mm={np.round(gt_center, 1).tolist()}")
    else:
        gt_center = None
        print("GT has no lesion component.")

    if not pred_comps:
        print("Prediction is empty.")
    else:
        print("Largest predicted components (up to 5):")
        for pc in pred_comps:
            dist = (
                float(np.linalg.norm(pc["center"] - gt_center))
                if gt_center is not None else float("nan")
            )
            purity = 100 * pc["overlap_gt"] / pc["voxels"]
            gt_coverage = 100 * pc["overlap_gt"] / gt_vox if gt_vox else 0
            print(
                f"  component={pc['id']}, voxels={pc['voxels']}, "
                f"centroid distance to largest GT={dist:.1f} mm, "
                f"component voxels on GT={purity:.1f}%, "
                f"GT lesion captured={gt_coverage:.1f}%"
            )