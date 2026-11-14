from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

# Allow direct/module execution to import project modules.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Spacingd, EnsureTyped
from scipy.ndimage import distance_transform_edt
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from models.bayesian_mk_unet3d import BayesianMKUNet3D


STAGE1_CHECKPOINT = ROOT / "training" / "checkpoints_exp5_samples4" / "best.pth"
STAGE2_CHECKPOINT = ROOT / "datasets" / "stage2" / "models" / "task07_finetune_v2" / "latest_task07_finetune.pth"
MANIFEST_PATH = ROOT / "datasets" / "stage2" / "Task07_Pancreas" / "task07_split_manifest.json"
CACHE_DIR = ROOT / "evaluation" / "task07_full_volume_v2"
STAGE1_MASK_DIR = CACHE_DIR / "stage1_pancreas_masks"
RESULTS_PATH = CACHE_DIR / "task07_pancreas_full_volume_v2_epoch25_results.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STAGE1_SPACING = (1.5, 1.5, 3.0)
STAGE1_ROI = (96, 96, 64)
STAGE1_SW_BATCH = 1
STAGE1_OVERLAP = 0.25
STAGE1_NUM_CLASSES = 13
STAGE2_ROI = (32, 96, 96)
STAGE2_SW_BATCH = 2
STAGE2_OVERLAP = 0.50
STAGE2_NUM_CLASSES = 3
DEFAULT_THRESHOLD = 0.80
DEFAULT_GATE_MARGIN_MM = 15.0


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> None:
    checkpoint = torch.load(str(checkpoint_path), map_location=DEVICE, weights_only=False)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state = checkpoint["model_state"]
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "network_weights" in checkpoint:
        state = checkpoint["network_weights"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint
    model.load_state_dict(state, strict=True)


def get_main_output(output):
    if isinstance(output, dict):
        return output["main"]
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def load_manifest() -> list[dict]:
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    validation = manifest.get("validation_cases")
    if not isinstance(validation, list) or not validation:
        raise RuntimeError(
            f"Expected a non-empty 'validation_cases' list in {MANIFEST_PATH}"
        )

    for entry in validation:
        for field in ("image", "label"):
            if field not in entry:
                raise RuntimeError(
                    f"Validation entry is missing '{field}': {entry}"
                )

            case_path = Path(entry[field])
            if not case_path.is_absolute():
                case_path = ROOT / case_path

            case_path = case_path.resolve()
            if not case_path.is_file():
                raise FileNotFoundError(
                    f"Task07 validation file not found: {case_path}"
                )
            entry[field] = str(case_path)

    return validation


def case_key(image_path: Path) -> str:
    return f"{image_path.parent.name}_{image_path.name.replace('.nii.gz', '')}"


def get_stage1_preprocessor():
    # Stage-2 images already use [-1000,1000] -> [0,1], so do not
    # normalize them a second time. Match Stage-1 spatial resampling.
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Spacingd(keys=["image"], pixdim=STAGE1_SPACING, mode=("bilinear",)),
        EnsureTyped(keys=["image"]),
    ])


def derive_resampled_affine(source_nii: nib.Nifti1Image, target_spacing) -> np.ndarray:
    source = nib.as_closest_canonical(source_nii)
    affine = np.array(source.affine, dtype=np.float64, copy=True)
    for axis in range(3):
        direction = affine[:3, axis]
        norm = np.linalg.norm(direction)
        if norm <= 0:
            raise RuntimeError(f"Invalid affine axis {axis}")
        affine[:3, axis] = direction / norm * float(target_spacing[axis])
    return affine


def preprocess_stage1_case(image_path: Path, transform):
    source_nii = nib.as_closest_canonical(nib.load(str(image_path)))
    result = transform({"image": str(image_path)})
    image_tensor = result["image"]
    stage1_image = image_tensor.cpu().numpy()[0].astype(np.float32)
    stage1_affine = None
    try:
        if hasattr(image_tensor, "affine"):
            stage1_affine = np.asarray(image_tensor.affine.cpu().numpy(), dtype=np.float64)
    except Exception:
        stage1_affine = None
    if stage1_affine is None or stage1_affine.shape != (4, 4):
        stage1_affine = derive_resampled_affine(source_nii, STAGE1_SPACING)
    return stage1_image, stage1_affine


def predict_stage1_organs(model, image: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = sliding_window_inference(
            inputs=tensor,
            roi_size=STAGE1_ROI,
            sw_batch_size=STAGE1_SW_BATCH,
            predictor=lambda x: get_main_output(model(x)),
            overlap=STAGE1_OVERLAP,
            mode="gaussian",
        )
        prediction = torch.argmax(output, dim=1)
    return prediction.squeeze(0).cpu().numpy().astype(np.uint8)


def map_stage1_gate_to_stage2(stage1_gate: np.ndarray, stage1_affine: np.ndarray, stage2_nii: nib.Nifti1Image) -> np.ndarray:
    source_nii = nib.Nifti1Image(stage1_gate.astype(np.uint8), affine=stage1_affine)
    target = (stage2_nii.shape[:3], stage2_nii.affine)
    from nibabel.processing import resample_from_to
    mapped = resample_from_to(source_nii, target, order=0).get_fdata()
    return (mapped > 0.5).astype(bool)


def dilate_physical(binary_mask: np.ndarray, spacing, margin_mm: float) -> np.ndarray:
    if margin_mm <= 0:
        return binary_mask
    dist = distance_transform_edt(~binary_mask, sampling=np.asarray(spacing, dtype=np.float64))
    return binary_mask | (dist <= margin_mm)


def predict_stage2_probability(model, image: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(image.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = sliding_window_inference(
            inputs=tensor,
            roi_size=STAGE2_ROI,
            sw_batch_size=STAGE2_SW_BATCH,
            predictor=lambda x: get_main_output(model(x)),
            overlap=STAGE2_OVERLAP,
            mode="gaussian",
        )
        probabilities = torch.softmax(logits, dim=1)
    return probabilities.squeeze(0).cpu().numpy().astype(np.float32)


def binary_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    tp = int(np.count_nonzero(pred & gt))
    fp = int(np.count_nonzero(pred & ~gt))
    fn = int(np.count_nonzero(~pred & gt))
    pred_voxels = int(pred.sum())
    gt_voxels = int(gt.sum())
    precision = tp / pred_voxels if pred_voxels else 0.0
    recall = tp / gt_voxels if gt_voxels else 0.0
    denom = pred_voxels + gt_voxels
    dice = 2.0 * tp / denom if denom else 1.0
    union = tp + fp + fn
    iou = tp / union if union else 1.0
    ratio = pred_voxels / gt_voxels if gt_voxels else (float("inf") if pred_voxels else 1.0)
    return {"dice": float(dice), "iou": float(iou), "precision": float(precision), "recall": float(recall), "tp": tp, "fp": fp, "fn": fn, "pred_voxels": pred_voxels, "gt_voxels": gt_voxels, "pred_gt_ratio": float(ratio)}


def build_stage2_model():
    return PlainConvUNet(
        input_channels=1,
        n_stages=6,
        features_per_stage=[32, 64, 128, 256, 320, 320],
        conv_op=torch.nn.Conv3d,
        kernel_sizes=[
            [1, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
            [3, 3, 3],
        ],
        strides=[
            [1, 1, 1],
            [1, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
            [2, 2, 2],
        ],
        n_conv_per_stage=[2, 2, 2, 2, 2, 2],
        num_classes=STAGE2_NUM_CLASSES,
        n_conv_per_stage_decoder=[2, 2, 2, 2, 2],
        conv_bias=True,
        norm_op=torch.nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=torch.nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
        deep_supervision=True,
    )


def save_mask(mask: np.ndarray, reference_nii: nib.Nifti1Image, path: Path):
    output = nib.Nifti1Image(mask.astype(np.uint8), affine=reference_nii.affine, header=reference_nii.header.copy())
    output.set_data_dtype(np.uint8)
    nib.save(output, str(path))


def aggregate(tp, fp, fn):
    pred_count = tp + fp
    gt_count = tp + fn
    precision = tp / pred_count if pred_count else 0.0
    recall = tp / gt_count if gt_count else 0.0
    dice = 2.0 * tp / (pred_count + gt_count) if (pred_count + gt_count) else 1.0
    union = tp + fp + fn
    iou = tp / union if union else 1.0
    ratio = pred_count / gt_count if gt_count else (float("inf") if pred_count else 1.0)
    return {"dice": float(dice), "iou": float(iou), "precision": float(precision), "recall": float(recall), "tp": int(tp), "fp": int(fp), "fn": int(fn), "pred_voxels": int(pred_count), "gt_voxels": int(gt_count), "pred_gt_ratio": float(ratio)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_cases", type=int, default=0, help="0 = all validation cases")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--gate_margin_mm", type=float, default=DEFAULT_GATE_MARGIN_MM)
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    STAGE1_MASK_DIR.mkdir(parents=True, exist_ok=True)
    for p in (STAGE1_CHECKPOINT, STAGE2_CHECKPOINT, MANIFEST_PATH):
        if not p.exists():
            raise FileNotFoundError(p)

    print("=" * 72)
    print("STAGE-2 TASK07 PANCREAS FULL-VOLUME EVALUATION")
    print("=" * 72)
    print(f"Device:             {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU:                {torch.cuda.get_device_name(0)}")
    print(f"Stage-1 checkpoint: {STAGE1_CHECKPOINT}")
    print(f"Stage-2 checkpoint: {STAGE2_CHECKPOINT}")
    print(f"Threshold:          {args.threshold:.2f}")
    print(f"Gate margin:        {args.gate_margin_mm:.1f} mm")

    stage1_model = BayesianMKUNet3D(in_channels=1, num_classes=13, base_channels=16, dropout=0.1)
    load_checkpoint(stage1_model, STAGE1_CHECKPOINT)
    stage1_model = stage1_model.to(DEVICE).eval()

    stage2_model = build_stage2_model()
    load_checkpoint(stage2_model, STAGE2_CHECKPOINT)
    stage2_model = stage2_model.to(DEVICE).eval()

    print("Both models loaded successfully.")

    transform = get_stage1_preprocessor()
    validation = load_manifest()
    if args.max_cases > 0:
        validation = validation[:args.max_cases]

    total_tp_base = total_fp_base = total_fn_base = 0
    total_tp_gate = total_fp_gate = total_fn_gate = 0
    case_results = []

    for index, entry in enumerate(validation, start=1):
        image_path = Path(entry["image"])
        label_path = Path(entry["label"])
        key = case_key(image_path)
        print("-" * 72)
        print(f"[{index}/{len(validation)}] {key}")

        stage2_nii = nib.as_closest_canonical(nib.load(str(image_path)))
        gt_nii = nib.as_closest_canonical(nib.load(str(label_path)))
        stage2_image = stage2_nii.get_fdata(dtype=np.float32)
        gt = np.rint(gt_nii.get_fdata()).astype(np.uint8)

        if stage2_image.shape != gt.shape:
            raise RuntimeError(f"Image/label shape mismatch for {key}: {stage2_image.shape} vs {gt.shape}")
        if np.any((gt < 0) | (gt > 2)):
            raise RuntimeError(f"Unexpected labels in {key}: {np.unique(gt).tolist()}")

        mask_path = STAGE1_MASK_DIR / f"{key}_pancreas_gate.nii.gz"
        if mask_path.exists():
            print("Stage-1 pancreas gate: cached")
            gate = np.rint(nib.load(str(mask_path)).get_fdata()).astype(np.uint8) > 0
        else:
            print("Stage-1 anatomy: inference")
            stage1_image, stage1_affine = preprocess_stage1_case(image_path, transform)
            print(f"Stage-1 image: {stage1_image.shape}; range={stage1_image.min():.4f}-{stage1_image.max():.4f}")
            stage1_pred = predict_stage1_organs(stage1_model, stage1_image)
            stage1_gate = (stage1_pred == 10)
            gate = map_stage1_gate_to_stage2(stage1_gate, stage1_affine, stage2_nii)
            save_mask(gate.astype(np.uint8), stage2_nii, mask_path)
            print(f"Mapped pancreas gate voxels: {int(gate.sum()):,}")

        spacing = tuple(float(x) for x in stage2_nii.header.get_zooms()[:3])
        gated_region = dilate_physical(gate, spacing, args.gate_margin_mm)

        print("Stage-2 pathology inference")
        # Match the Z,Y,X array convention used by the Stage-2 training loader.
        stage2_image_zyx = np.transpose(stage2_image, (2, 1, 0)).copy()
        p2_zyx = predict_stage2_probability(
            stage2_model, stage2_image_zyx
        )[2]

        # Return probabilities to the original X,Y,Z grid for metrics and gating.
        p2 = np.transpose(p2_zyx, (2, 1, 0))
        gt_pathology = gt == 2
        baseline_pred = p2 >= args.threshold
        gated_pred = baseline_pred & gated_region

        base = binary_metrics(baseline_pred, gt_pathology)
        gated = binary_metrics(gated_pred, gt_pathology)
        total_tp_base += base["tp"]
        total_fp_base += base["fp"]
        total_fn_base += base["fn"]
        total_tp_gate += gated["tp"]
        total_fp_gate += gated["fp"]
        total_fn_gate += gated["fn"]

        case_results.append({"case": key, "threshold": args.threshold, "gate_margin_mm": args.gate_margin_mm, "gt_pathology_voxels": gated["gt_voxels"], "gate_voxels": int(gated_region.sum()), "baseline": base, "gated": gated})
        print(f"Baseline: Dice={base['dice']:.4f}, Precision={base['precision']:.4f}, Recall={base['recall']:.4f}, Pred/GT={base['pred_gt_ratio']:.3f}")
        print(f"Gated:    Dice={gated['dice']:.4f}, Precision={gated['precision']:.4f}, Recall={gated['recall']:.4f}, Pred/GT={gated['pred_gt_ratio']:.3f}")

    result = {
        "num_cases": len(validation),
        "threshold": args.threshold,
        "gate_margin_mm": args.gate_margin_mm,
        "baseline_micro": aggregate(total_tp_base, total_fp_base, total_fn_base),
        "gated_micro": aggregate(total_tp_gate, total_fp_gate, total_fn_gate),
        "baseline_macro_mean_dice": float(np.mean([x["baseline"]["dice"] for x in case_results])) if case_results else 0.0,
        "gated_macro_mean_dice": float(np.mean([x["gated"]["dice"] for x in case_results])) if case_results else 0.0,
        "case_results": case_results,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    base = result["baseline_micro"]
    gate = result["gated_micro"]
    print("\n" + "=" * 72)
    print("FINAL SUMMARY")
    print("=" * 72)
    print(f"Cases: {len(validation)}")
    print(f"BASELINE  Dice={base['dice']:.4f}  IoU={base['iou']:.4f}  Precision={base['precision']:.4f}  Recall={base['recall']:.4f}  Pred/GT={base['pred_gt_ratio']:.3f}")
    print(f"GATED     Dice={gate['dice']:.4f}  IoU={gate['iou']:.4f}  Precision={gate['precision']:.4f}  Recall={gate['recall']:.4f}  Pred/GT={gate['pred_gt_ratio']:.3f}")
    print(f"Dice change:      {gate['dice'] - base['dice']:+.4f}")
    print(f"Precision change: {gate['precision'] - base['precision']:+.4f}")
    print(f"Recall change:    {gate['recall'] - base['recall']:+.4f}")
    print(f"Pred/GT change:   {gate['pred_gt_ratio'] - base['pred_gt_ratio']:+.3f}")
    print(f"Results saved:    {RESULTS_PATH}")
    print("=" * 72)


if __name__ == "__main__":
    main()
