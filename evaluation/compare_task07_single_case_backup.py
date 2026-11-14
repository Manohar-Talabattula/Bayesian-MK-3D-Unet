from pathlib import Path
import json
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from dynamic_network_architectures.architectures.unet import PlainConvUNet
from monai.inferers import sliding_window_inference

ROOT = Path.cwd()
CASE_NAME = "Task07_Pancreas_157"
CHECKPOINT = ROOT / "datasets/stage2/models/task07_finetune_v2/latest_task07_finetune.pth"
MANIFEST = ROOT / "datasets/stage2/Task07_Pancreas/task07_split_manifest.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PATCH = (32, 96, 96)

def build_model():
    return PlainConvUNet(
        input_channels=1,
        n_stages=6,
        features_per_stage=[32, 64, 128, 256, 320, 320],
        conv_op=torch.nn.Conv3d,
        kernel_sizes=[[1,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3]],
        strides=[[1,1,1],[1,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2]],
        n_conv_per_stage=[2,2,2,2,2,2],
        num_classes=3,
        n_conv_per_stage_decoder=[2,2,2,2,2],
        conv_bias=True,
        norm_op=torch.nn.InstanceNorm3d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=torch.nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
    )

manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
entries = manifest["validation_cases"]
entry = next(
    (e for e in entries if CASE_NAME in str(e["image"])),
    None
)
if entry is None:
    raise RuntimeError(f"{CASE_NAME} was not found in validation_cases")

def resolve(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p

image_path, label_path = resolve(entry["image"]), resolve(entry["label"])
image_nii = nib.load(str(image_path))
label_nii = nib.load(str(label_path))

# Match the Stage-2 training loader: original NIfTI array, clip, then Z,Y,X.
image = np.transpose(
    np.clip(image_nii.get_fdata(dtype=np.float32), 0, 1), (2,1,0)
).copy()
label = np.transpose(
    np.rint(label_nii.get_fdata()).astype(np.uint8), (2,1,0)
).copy()

if image.shape != label.shape:
    raise RuntimeError(f"Image/label shape mismatch: {image.shape} vs {label.shape}")

lesion = np.argwhere(label == 2)
if len(lesion) == 0:
    raise RuntimeError("This case has no class-2 pathology voxels")

# Match the training validation_patch() centre and boundary handling.
pz, py, px = PATCH
z, y, x = label.shape
pad = (
    (max(0,pz-z)//2, max(0,pz-z)-max(0,pz-z)//2),
    (max(0,py-y)//2, max(0,py-y)-max(0,py-y)//2),
    (max(0,px-x)//2, max(0,px-x)-max(0,px-x)//2),
)
if any(a or b for a,b in pad):
    image = np.pad(image, pad, mode="constant")
    label = np.pad(label, pad, mode="constant")
    lesion = np.argwhere(label == 2)
z, y, x = label.shape
cz, cy, cx = np.median(lesion, axis=0).astype(int)
sz = int(np.clip(cz-pz//2, 0, z-pz))
sy = int(np.clip(cy-py//2, 0, y-py))
sx = int(np.clip(cx-px//2, 0, x-px))
patch_image = image[sz:sz+pz, sy:sy+py, sx:sx+px]
patch_label = label[sz:sz+pz, sy:sy+py, sx:sx+px]

model = build_model().to(DEVICE)
ckpt = torch.load(str(CHECKPOINT), map_location=DEVICE, weights_only=False)
if isinstance(ckpt, dict):
    for key in ("model_state", "model_state_dict", "network_weights", "state_dict"):
        if key in ckpt:
            ckpt = ckpt[key]
            break
model.load_state_dict(ckpt, strict=True)
model.eval()

def main_output(output):
    if isinstance(output, dict):
        return output["main"]
    if isinstance(output, (tuple, list)):
        return output[0]
    return output

def dice(pred, truth):
    pred, truth = pred.astype(bool), truth.astype(bool)
    denom = pred.sum() + truth.sum()
    return float(2 * np.logical_and(pred, truth).sum() / denom) if denom else 1.0

with torch.no_grad():
    inp = torch.from_numpy(patch_image.copy()).float()[None, None].to(DEVICE)
    patch_prob = torch.softmax(main_output(model(inp)), dim=1)[0, 2].cpu().numpy()

    # Full-volume sliding-window inference, matching evaluator settings.
    full_inp = torch.from_numpy(image.copy()).float()[None, None].to(DEVICE)
    full_logits = sliding_window_inference(
        full_inp, roi_size=PATCH, sw_batch_size=2,
        predictor=lambda t: main_output(model(t)),
        overlap=0.50, mode="gaussian",
    )
    full_prob = torch.softmax(full_logits, dim=1)[0, 2].cpu().numpy()
    full_patch_prob = full_prob[sz:sz+pz, sy:sy+py, sx:sx+px]

truth = patch_label == 2
threshold = 0.80
patch_pred = patch_prob >= threshold
full_pred = full_patch_prob >= threshold

print("\n=== SINGLE-CASE PATCH VS FULL-VOLUME TEST ===")
print("Case:", CASE_NAME)
print("Device:", DEVICE)
print("Patch origin (Z,Y,X):", (sz, sy, sx))
print("Ground-truth lesion voxels in patch:", int(truth.sum()))
print("Patch inference:  pred voxels =", int(patch_pred.sum()),
      "| Dice =", round(dice(patch_pred, truth), 4))
print("Full-volume crop: pred voxels =", int(full_pred.sum()),
      "| Dice =", round(dice(full_pred, truth), 4))
print("Mean absolute class-2 probability difference:",
      round(float(np.mean(np.abs(patch_prob-full_patch_prob))), 6))
