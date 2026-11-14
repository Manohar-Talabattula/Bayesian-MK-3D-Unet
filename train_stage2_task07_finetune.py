import argparse
import ctypes
import json
import math
import os
import random
import re
import time
from collections import OrderedDict
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from dynamic_network_architectures.architectures.unet import PlainConvUNet


# ============================================================
# PATHS AND EXPERIMENT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
STAGE2_DIR = ROOT / "datasets" / "stage2"

DATASET_DIRS = {
    "LiTS_Batch1": STAGE2_DIR / "LiTS_Batch1",
    "LiTS_Batch2": STAGE2_DIR / "LiTS_Batch2",
    "KiTS19": STAGE2_DIR / "KiTS19",
}

ORIGINAL_SPLIT_FILE = (
    STAGE2_DIR / "split" / "stage2_split_manifest.json"
)

TASK07_DIR = STAGE2_DIR / "Task07_Pancreas"
TASK07_SPLIT_FILE = TASK07_DIR / "task07_split_manifest.json"

V1_CHECKPOINT = (
    STAGE2_DIR / "models" / "stage2_best_balanced.pth"
)

# All experimental outputs are isolated from the original model.
OUTPUT_DIR = (
    STAGE2_DIR / "models" / "task07_finetune_v2"
)

BEST_CHECKPOINT = OUTPUT_DIR / "best_task07_finetune.pth"
LATEST_CHECKPOINT = OUTPUT_DIR / "latest_task07_finetune.pth"
BASELINE_FILE = OUTPUT_DIR / "baseline_metrics.json"
HISTORY_FILE = OUTPUT_DIR / "training_history.jsonl"

IMAGE_SUFFIX = "_0000.nii.gz"

SEED = 42

PATCH_SIZE = (32, 96, 96)
BATCH_SIZE = 1

# Conservative initial fine-tuning settings.
EPOCHS = 25
STEPS_PER_EPOCH = 100
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 1e-5

NUM_WORKERS = 0
AMP_ENABLED = True
PATHOLOGY_PATCH_PROBABILITY = 0.70
NUM_CLASSES = 3

# Cache no more than roughly 45% of currently available RAM.
RAM_CACHE_FRACTION = 0.45

CASE_ID_PATTERN = re.compile(
    r"(?:LiTS_Batch1|LiTS_Batch2|KiTS19)_\d{4}"
)


# ============================================================
# REPRODUCIBILITY AND DEVICE
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    DEVICE = torch.device("cuda")
else:
    raise RuntimeError(
        "CUDA is required. No training has been started."
    )


# ============================================================
# ORIGINAL LITS/KITS DATASET AND SPLIT
# ============================================================

def get_image_files(folder):
    # Match the original Stage-2 trainer's image discovery logic.
    # Mask files like LiTS_Batch1_0000.nii.gz must not be
    # mistaken for CT images.
    files = []

    for p in folder.glob("*_0000.nii.gz"):
        name = p.name

        if not name.endswith(IMAGE_SUFFIX):
            continue

        stem = name[:-len(IMAGE_SUFFIX)]
        parts = stem.split("_")

        if len(parts) >= 2 and parts[-1].isdigit():
            files.append(p)

    return sorted(files)


def case_id_from_image(image_path):
    return image_path.name[:-len(IMAGE_SUFFIX)]


def label_from_image(image_path):
    return image_path.with_name(
        case_id_from_image(image_path) + ".nii.gz"
    )


def collect_original_cases():
    cases = {}

    for dataset_name, folder in DATASET_DIRS.items():
        if not folder.is_dir():
            raise FileNotFoundError(
                f"Original dataset directory not found: {folder}"
            )

        images = get_image_files(folder)

        if not images:
            raise RuntimeError(
                f"No preprocessed CT images found in {folder}"
            )

        for image_path in images:
            label_path = label_from_image(image_path)

            if not label_path.is_file():
                raise FileNotFoundError(
                    f"Missing mask for {image_path.name}: {label_path}"
                )

            case_id = case_id_from_image(image_path)

            if case_id in cases:
                raise RuntimeError(
                    f"Duplicate original case ID: {case_id}"
                )

            cases[case_id] = {
                "case_id": case_id,
                "dataset": dataset_name,
                "image": image_path,
                "label": label_path,
            }

    if len(cases) != 341:
        raise RuntimeError(
            f"Expected 341 original cases; found {len(cases)}."
        )

    return cases


def extract_original_case_ids(node):
    if isinstance(node, str):
        return CASE_ID_PATTERN.findall(node)

    if isinstance(node, list):
        result = []
        for item in node:
            result.extend(extract_original_case_ids(item))
        return result

    if isinstance(node, dict):
        result = []
        for value in node.values():
            result.extend(extract_original_case_ids(value))
        return result

    return []


def find_original_split_cases(obj):
    train_ids = []
    val_ids = []

    def walk(node):
        nonlocal train_ids, val_ids

        if isinstance(node, dict):
            for key, value in node.items():
                key_lower = str(key).lower()

                if "train" in key_lower and "val" not in key_lower:
                    train_ids.extend(
                        extract_original_case_ids(value)
                    )
                elif "val" in key_lower or "valid" in key_lower:
                    val_ids.extend(
                        extract_original_case_ids(value)
                    )
                else:
                    walk(value)

        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(obj)

    return (
        list(dict.fromkeys(train_ids)),
        list(dict.fromkeys(val_ids)),
    )


def load_original_splits(all_cases):
    if not ORIGINAL_SPLIT_FILE.is_file():
        raise FileNotFoundError(
            f"Original split manifest not found: {ORIGINAL_SPLIT_FILE}"
        )

    with ORIGINAL_SPLIT_FILE.open(
        "r", encoding="utf-8"
    ) as file:
        split_json = json.load(file)

    train_ids, val_ids = find_original_split_cases(split_json)

    if len(train_ids) != 272 or len(val_ids) != 69:
        raise RuntimeError(
            "Original split count mismatch. "
            f"Expected 272/69; got {len(train_ids)}/{len(val_ids)}."
        )

    if set(train_ids) & set(val_ids):
        raise RuntimeError(
            "Original split has training/validation overlap."
        )

    if set(train_ids) | set(val_ids) != set(all_cases):
        raise RuntimeError(
            "Original split does not cover exactly the 341 indexed cases."
        )

    return (
        [all_cases[case_id] for case_id in train_ids],
        [all_cases[case_id] for case_id in val_ids],
    )


# ============================================================
# TASK07 DATASET AND SPLIT
# ============================================================

def resolve_manifest_path(value):
    path = Path(value)
    if path.is_absolute():
        return path
    return ROOT / path


def load_task07_splits():
    if not TASK07_SPLIT_FILE.is_file():
        raise FileNotFoundError(
            f"Task07 split manifest not found: {TASK07_SPLIT_FILE}"
        )

    with TASK07_SPLIT_FILE.open(
        "r", encoding="utf-8"
    ) as file:
        manifest = json.load(file)

    train_entries = manifest.get("train_cases", [])
    val_entries = manifest.get("validation_cases", [])

    if len(train_entries) != 225 or len(val_entries) != 56:
        raise RuntimeError(
            "Task07 split count mismatch. "
            f"Expected 225/56; got {len(train_entries)}/{len(val_entries)}."
        )

    if manifest.get("total_cases") != 281:
        raise RuntimeError(
            "Task07 manifest does not report 281 total cases."
        )

    train_ids = [entry["case_id"] for entry in train_entries]
    val_ids = [entry["case_id"] for entry in val_entries]

    if len(set(train_ids)) != len(train_ids):
        raise RuntimeError("Duplicate Task07 training case IDs.")

    if len(set(val_ids)) != len(val_ids):
        raise RuntimeError("Duplicate Task07 validation case IDs.")

    if set(train_ids) & set(val_ids):
        raise RuntimeError("Task07 training/validation overlap detected.")

    def convert_entry(entry):
        image_path = resolve_manifest_path(entry["image"]).resolve()
        label_path = resolve_manifest_path(entry["label"]).resolve()

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Task07 CT image missing: {image_path}"
            )

        if not label_path.is_file():
            raise FileNotFoundError(
                f"Task07 mask missing: {label_path}"
            )

        return {
            "case_id": entry["case_id"],
            "dataset": "Task07_Pancreas",
            "image": image_path,
            "label": label_path,
        }

    train_cases = [convert_entry(x) for x in train_entries]
    val_cases = [convert_entry(x) for x in val_entries]

    return train_cases, val_cases


# ============================================================
# MEMORY-AWARE CACHE CAPACITY
# ============================================================

def get_available_ram_bytes():
    if os.name == "nt":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)

        if ctypes.windll.kernel32.GlobalMemoryStatusEx(
            ctypes.byref(status)
        ):
            return int(status.ullAvailPhys)

    # Conservative fallback if physical memory information is unavailable.
    return 8 * 1024**3


def choose_cache_capacity(cases):
    estimated_case_bytes = []

    for record in cases:
        image_nii = nib.load(str(record["image"]))

        if len(image_nii.shape) != 3:
            raise RuntimeError(
                f"Expected a 3D CT image: {record['image']}"
            )

        voxel_count = math.prod(image_nii.shape)

        # Approximate in-memory image float32 + mask uint8.
        estimated_case_bytes.append(voxel_count * 5)

        del image_nii

    if not estimated_case_bytes:
        raise RuntimeError("Cannot estimate cache size for empty training data.")

    average_bytes = max(
        1, int(sum(estimated_case_bytes) / len(estimated_case_bytes))
    )

    total_estimated_bytes = sum(estimated_case_bytes)
    available_bytes = get_available_ram_bytes()
    budget_bytes = int(available_bytes * RAM_CACHE_FRACTION)

    cache_capacity = max(
        1,
        min(
            len(cases),
            budget_bytes // average_bytes
        )
    )

    print("\nRAM/cache estimate:")
    print(f"  Training cases:             {len(cases)}")
    print(
        f"  Approximate cache for all:  "
        f"{total_estimated_bytes / 1024**3:.2f} GiB"
    )
    print(
        f"  Currently available RAM:    "
        f"{available_bytes / 1024**3:.2f} GiB"
    )
    print(f"  RAM cache budget:           {RAM_CACHE_FRACTION:.0%}")
    print(f"  Maximum cached cases:       {cache_capacity}")

    return cache_capacity


# ============================================================
# CT / MASK LOADING
# ============================================================

class NiftiCase:
    def __init__(self, record):
        self.record = record
        self.case_id = record["case_id"]
        self.image_path = record["image"]
        self.label_path = record["label"]
        self.image = None
        self.label = None

    def load(self):
        if self.image is not None and self.label is not None:
            return

        image_nii = nib.load(str(self.image_path))
        label_nii = nib.load(str(self.label_path))

        image = np.asarray(
            image_nii.dataobj,
            dtype=np.float32
        )

        label = np.asarray(
            label_nii.dataobj,
            dtype=np.uint8
        )

        if image.shape != label.shape:
            raise RuntimeError(
                f"Image/mask shape mismatch for {self.case_id}: "
                f"{image.shape} versus {label.shape}"
            )

        unique_labels = np.unique(label)

        if not np.all(np.isin(unique_labels, [0, 1, 2])):
            raise RuntimeError(
                f"Invalid labels in {self.case_id}: {unique_labels}"
            )

        # Inputs have already been normalized during preprocessing.
        image = np.clip(image, 0.0, 1.0)

        # NIfTI X,Y,Z -> model Z,Y,X.
        image = np.transpose(image, (2, 1, 0))
        label = np.transpose(label, (2, 1, 0))

        self.image = np.ascontiguousarray(image, dtype=np.float32)
        self.label = np.ascontiguousarray(label, dtype=np.uint8)


# ============================================================
# PATCH SAMPLING
# ============================================================

class Stage2PatchDataset(Dataset):
    def __init__(self, cases, steps_per_epoch, cache_capacity):
        self.records = list(cases)
        self.steps_per_epoch = steps_per_epoch
        self.cache_capacity = cache_capacity
        self.case_cache = OrderedDict()

        if not self.records:
            raise RuntimeError("Cannot create an empty training dataset.")

    def __len__(self):
        return self.steps_per_epoch

    def get_case(self, record):
        case_id = record["case_id"]

        if case_id in self.case_cache:
            case = self.case_cache.pop(case_id)
            self.case_cache[case_id] = case
            return case

        case = NiftiCase(record)
        case.load()

        self.case_cache[case_id] = case

        while len(self.case_cache) > self.cache_capacity:
            self.case_cache.popitem(last=False)

        return case

    def random_patch(self, image, label):
        z, y, x = label.shape
        pz, py, px = PATCH_SIZE

        has_pathology = np.any(label == 2)

        target_pathology = (
            has_pathology
            and random.random() < PATHOLOGY_PATCH_PROBABILITY
        )

        if target_pathology:
            coords = np.argwhere(label == 2)
            center_z, center_y, center_x = coords[
                random.randrange(len(coords))
            ]
        else:
            center_z = random.randrange(z)
            center_y = random.randrange(y)
            center_x = random.randrange(x)

        start_z = int(np.clip(center_z - pz // 2, 0, max(0, z - pz)))
        start_y = int(np.clip(center_y - py // 2, 0, max(0, y - py)))
        start_x = int(np.clip(center_x - px // 2, 0, max(0, x - px)))

        pad_z = max(0, pz - z)
        pad_y = max(0, py - y)
        pad_x = max(0, px - x)

        if pad_z or pad_y or pad_x:
            pad_width = (
                (pad_z // 2, pad_z - pad_z // 2),
                (pad_y // 2, pad_y - pad_y // 2),
                (pad_x // 2, pad_x - pad_x // 2),
            )

            image = np.pad(
                image, pad_width, mode="constant", constant_values=0
            )
            label = np.pad(
                label, pad_width, mode="constant", constant_values=0
            )

            z, y, x = label.shape
            start_z = min(start_z, z - pz)
            start_y = min(start_y, y - py)
            start_x = min(start_x, x - px)

        image_patch = image[
            start_z:start_z + pz,
            start_y:start_y + py,
            start_x:start_x + px,
        ]

        label_patch = label[
            start_z:start_z + pz,
            start_y:start_y + py,
            start_x:start_x + px,
        ]

        return image_patch, label_patch

    def __getitem__(self, index):
        record = random.choice(self.records)
        case = self.get_case(record)

        image_patch, label_patch = self.random_patch(
            case.image, case.label
        )

        image_tensor = torch.from_numpy(
            image_patch.copy()
        ).unsqueeze(0)

        label_tensor = torch.from_numpy(
            label_patch.copy()
        ).long()

        return image_tensor, label_tensor


# ============================================================
# MODEL — SAME ARCHITECTURE AS V1
# ============================================================

def build_model():
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
        num_classes=3,
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


# ============================================================
# LOSS — MATCH V1
# ============================================================

class PathologyWeightedDiceCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_weight = torch.tensor(
            [0.25, 1.0, 3.0],
            dtype=torch.float32
        )

    def dice_loss(self, logits, target):
        probabilities = torch.softmax(logits, dim=1)

        target_one_hot = F.one_hot(
            target.long(), num_classes=NUM_CLASSES
        )
        target_one_hot = (
            target_one_hot.permute(0, 4, 1, 2, 3).float()
        )

        dims = (0, 2, 3, 4)

        intersection = (probabilities * target_one_hot).sum(dims)
        denominator = (
            probabilities.sum(dims) + target_one_hot.sum(dims)
        )

        dice = (
            2.0 * intersection + 1e-5
        ) / (
            denominator + 1e-5
        )

        foreground_dice = dice[1:]
        foreground_weights = torch.tensor(
            [1.0, 2.0],
            device=dice.device,
            dtype=dice.dtype
        )

        weighted_foreground_dice = (
            foreground_dice * foreground_weights
        ).sum() / foreground_weights.sum()

        return 1.0 - weighted_foreground_dice

    def forward(self, logits, target):
        ce_weight = self.ce_weight.to(
            device=logits.device,
            dtype=torch.float32
        )

        ce = F.cross_entropy(
            logits, target, weight=ce_weight
        )

        dice = self.dice_loss(logits, target)
        return ce + dice


def deep_supervision_loss(outputs, target, criterion):
    total_loss = 0.0

    weights = [1.0, 0.5, 0.25, 0.125, 0.0625]

    for output, weight in zip(outputs, weights):
        if output.shape[2:] != target.shape[1:]:
            target_scaled = F.interpolate(
                target.unsqueeze(1).float(),
                size=output.shape[2:],
                mode="nearest"
            ).squeeze(1).long()
        else:
            target_scaled = target

        total_loss = total_loss + weight * criterion(
            output, target_scaled
        )

    return total_loss / sum(weights)


# ============================================================
# VALIDATION — SAME CENTERED-PATCH PROTOCOL AS V1
# ============================================================

def validation_patch(image, label):
    z, y, x = label.shape
    pz, py, px = PATCH_SIZE

    pad_z = max(0, pz - z)
    pad_y = max(0, py - y)
    pad_x = max(0, px - x)

    if pad_z or pad_y or pad_x:
        pad_width = (
            (pad_z // 2, pad_z - pad_z // 2),
            (pad_y // 2, pad_y - pad_y // 2),
            (pad_x // 2, pad_x - pad_x // 2),
        )

        image = np.pad(image, pad_width, mode="constant")
        label = np.pad(label, pad_width, mode="constant")
        z, y, x = label.shape

    pathology_voxels = np.argwhere(label == 2)

    if len(pathology_voxels) > 0:
        center_z, center_y, center_x = np.median(
            pathology_voxels, axis=0
        ).astype(int)
    else:
        center_z, center_y, center_x = z // 2, y // 2, x // 2

    start_z = int(np.clip(center_z - pz // 2, 0, z - pz))
    start_y = int(np.clip(center_y - py // 2, 0, y - py))
    start_x = int(np.clip(center_x - px // 2, 0, x - px))

    return (
        image[
            start_z:start_z + pz,
            start_y:start_y + py,
            start_x:start_x + px,
        ],
        label[
            start_z:start_z + pz,
            start_y:start_y + py,
            start_x:start_x + px,
        ],
    )


def dice_per_class(prediction, target):
    scores = []

    for class_id in range(NUM_CLASSES):
        pred = prediction == class_id
        true = target == class_id

        intersection = np.logical_and(pred, true).sum()
        pred_count = pred.sum()
        true_count = true.sum()

        if pred_count == 0 and true_count == 0:
            dice = 1.0
        else:
            dice = (
                2.0 * intersection + 1e-5
            ) / (
                pred_count + true_count + 1e-5
            )

        scores.append(float(dice))

    return scores


@torch.no_grad()
def evaluate_cases(model, cases, cohort_name):
    model.eval()
    all_scores = []

    print(f"\nValidating {cohort_name}: {len(cases)} cases")

    for index, record in enumerate(cases, start=1):
        case = NiftiCase(record)
        case.load()

        image_patch, label_patch = validation_patch(
            case.image, case.label
        )

        image_tensor = torch.from_numpy(
            image_patch.copy()
        ).unsqueeze(0).unsqueeze(0).to(DEVICE)

        with torch.amp.autocast("cuda", enabled=AMP_ENABLED):
            outputs = model(image_tensor)

        logits = outputs[0]

        prediction = torch.argmax(
            logits, dim=1
        ).cpu().numpy()[0]

        all_scores.append(
            dice_per_class(prediction, label_patch)
        )

        del case, image_tensor, outputs, logits, prediction

        if index % 10 == 0 or index == len(cases):
            print(f"  {cohort_name}: {index}/{len(cases)}")

    scores = np.asarray(all_scores, dtype=np.float64)
    mean_scores = scores.mean(axis=0)

    return {
        "case_count": len(cases),
        "dice_background": float(mean_scores[0]),
        "dice_nonlesional_target_tissue": float(mean_scores[1]),
        "dice_pathology": float(mean_scores[2]),
        "dice_foreground_mean": float(mean_scores[1:].mean()),
        "metric_protocol": (
            "One deterministic pathology-centered 3D patch per case; "
            "not a full-volume evaluation."
        ),
    }


def balanced_cohort_score(original_metrics, task07_metrics):
    return (
        original_metrics["dice_pathology"]
        + task07_metrics["dice_pathology"]
    ) / 2.0


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2),
        encoding="utf-8"
    )


# ============================================================
# CHECKPOINT HELPERS
# ============================================================

def make_checkpoint(
    epoch,
    model,
    optimizer,
    scheduler,
    scaler,
    best_score,
    original_metrics,
    task07_metrics,
    train_loss,
):
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_val_dice": original_metrics["dice_pathology"],
        "best_balanced_cohort_pathology_dice": best_score,
        "train_loss": train_loss,
        "val_dice": [
            original_metrics["dice_background"],
            original_metrics["dice_nonlesional_target_tissue"],
            original_metrics["dice_pathology"],
        ],
        "val_foreground_dice": original_metrics["dice_foreground_mean"],
        "val_metrics_original_lits_kits": original_metrics,
        "val_metrics_task07": task07_metrics,
        "source_checkpoint": str(V1_CHECKPOINT),
        "config": {
            "experiment": "Task07 mixed-cohort fine-tuning",
            "patch_size": PATCH_SIZE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "steps_per_epoch": STEPS_PER_EPOCH,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "seed": SEED,
            "training_cases_original": 272,
            "training_cases_task07": 225,
            "validation_cases_original": 69,
            "validation_cases_task07": 56,
        },
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Stage-2 V1 using Task07 and original cohorts."
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate manifests, checkpoint compatibility, and one patch; do not train.",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("STAGE-2 TASK07 MIXED-COHORT FINE-TUNING")
    print("=" * 72)
    print("Device:", torch.cuda.get_device_name(0))
    print("Initial checkpoint:", V1_CHECKPOINT)
    print("Experiment output:", OUTPUT_DIR)

    if not V1_CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Trained V1 checkpoint not found: {V1_CHECKPOINT}"
        )

    # Read and verify the original split without changing it.
    all_original_cases = collect_original_cases()
    original_train, original_val = load_original_splits(
        all_original_cases
    )

    # Read and verify the independent Task07 split.
    task07_train, task07_val = load_task07_splits()

    # Original training cases and Task07 training cases are mixed.
    train_cases = original_train + task07_train

    # Validation cohorts remain separate and held out.
    print("\nVerified data split:")
    print(f"  Original training:       {len(original_train)}")
    print(f"  Task07 training:         {len(task07_train)}")
    print(f"  Combined training:       {len(train_cases)}")
    print(f"  Original validation:     {len(original_val)}")
    print(f"  Task07 validation:       {len(task07_val)}")

    if len(train_cases) != 497:
        raise RuntimeError(
            f"Expected 497 combined training cases; found {len(train_cases)}."
        )

    if set(r["case_id"] for r in train_cases) & set(
        r["case_id"] for r in original_val + task07_val
    ):
        raise RuntimeError("Training/validation overlap detected.")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this experiment.")

    # Estimate a safe RAM-based cache capacity for training volumes.
    cache_capacity = choose_cache_capacity(train_cases)

    train_dataset = Stage2PatchDataset(
        train_cases,
        STEPS_PER_EPOCH,
        cache_capacity,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Load the trained V1 model—not the original pretrained initialization.
    model = build_model()

    v1_state = torch.load(
        V1_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    if "model_state_dict" not in v1_state:
        raise RuntimeError(
            "V1 checkpoint has no model_state_dict."
        )

    model.load_state_dict(
        v1_state["model_state_dict"],
        strict=True,
    )

    model = model.to(DEVICE)

    print(
        f"\nLoaded V1 checkpoint from epoch index "
        f"{v1_state.get('epoch', 'unknown')}."
    )
    print(
        "Recorded V1 pathology Dice:",
        v1_state.get("best_val_dice", "unknown"),
    )

    criterion = PathologyWeightedDiceCELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=AMP_ENABLED,
    )

    # Optional, non-training smoke test.
    if args.check_only:
        image_tensor, label_tensor = train_dataset[0]

        if tuple(image_tensor.shape) != (1, *PATCH_SIZE):
            raise RuntimeError(
                f"Unexpected input patch shape: {tuple(image_tensor.shape)}"
            )

        if tuple(label_tensor.shape) != PATCH_SIZE:
            raise RuntimeError(
                f"Unexpected target patch shape: {tuple(label_tensor.shape)}"
            )

        model.eval()

        with torch.no_grad(), torch.amp.autocast(
            "cuda", enabled=AMP_ENABLED
        ):
            outputs = model(
                image_tensor.unsqueeze(0).to(DEVICE)
            )

        if not outputs:
            raise RuntimeError("Model returned no deep-supervision outputs.")

        if tuple(outputs[0].shape[2:]) != PATCH_SIZE:
            raise RuntimeError(
                "Model output shape is incompatible with the training patch: "
                f"{tuple(outputs[0].shape)}"
            )

        print("\nCHECK-ONLY PASSED")
        print("  Manifests: passed")
        print("  Case files: passed")
        print("  V1 checkpoint architecture: passed")
        print("  Input patch shape:", tuple(image_tensor.shape))
        print("  Target labels:", torch.unique(label_tensor).tolist())
        print("  Output logits shape:", tuple(outputs[0].shape))
        print("  Optimizer steps performed: 0")
        print("  Existing V1 checkpoint changed: no")
        print("  Fine-tuning checkpoints written: no")
        return

    # Ask before resuming, but only from this experiment's latest checkpoint.
    resume_run = False

    if LATEST_CHECKPOINT.is_file():
        answer = input(
            "\nExisting Task07 experiment checkpoint found. "
            "Resume this experiment? [y/N]: "
        ).strip().lower()
        resume_run = answer == "y"

    # Establish or load baseline metrics using the untouched V1 weights.
    if resume_run and BASELINE_FILE.is_file():
        baseline = json.loads(
            BASELINE_FILE.read_text(encoding="utf-8")
        )
    else:
        print("\nEvaluating the original V1 baseline...")
        baseline_original = evaluate_cases(
            model, original_val, "V1 original LiTS/KiTS validation"
        )
        baseline_task07 = evaluate_cases(
            model, task07_val, "V1 Task07 validation"
        )

        baseline = {
            "source_checkpoint": str(V1_CHECKPOINT),
            "source_checkpoint_epoch_index": v1_state.get("epoch"),
            "original_lits_kits": baseline_original,
            "task07_pancreas": baseline_task07,
            "balanced_cohort_pathology_dice": balanced_cohort_score(
                baseline_original, baseline_task07
            ),
        }

        write_json(BASELINE_FILE, baseline)

    print("\nV1 baseline results (pathology-centered patch Dice):")
    print(
        "  Original LiTS/KiTS:",
        f"{baseline['original_lits_kits']['dice_pathology']:.5f}",
    )
    print(
        "  Task07 Pancreas:   ",
        f"{baseline['task07_pancreas']['dice_pathology']:.5f}",
    )
    print(
        "  Balanced mean:     ",
        f"{baseline['balanced_cohort_pathology_dice']:.5f}",
    )

    start_epoch = 0

    if resume_run:
        resume = torch.load(
            LATEST_CHECKPOINT,
            map_location=DEVICE,
            weights_only=False,
        )

        model.load_state_dict(
            resume["model_state_dict"], strict=True
        )
        optimizer.load_state_dict(
            resume["optimizer_state_dict"]
        )
        scheduler.load_state_dict(
            resume["scheduler_state_dict"]
        )
        scaler.load_state_dict(
            resume["scaler_state_dict"]
        )

        start_epoch = int(resume["epoch"]) + 1
        best_score = float(
            resume["best_balanced_cohort_pathology_dice"]
        )

        print(
            f"\nResuming this Task07 experiment at epoch "
            f"{start_epoch + 1}."
        )
    else:
        # Reset only this experiment's files; never touch V1.
        LATEST_CHECKPOINT.unlink(missing_ok=True)
        HISTORY_FILE.write_text("", encoding="utf-8")

        best_score = float(
            baseline["balanced_cohort_pathology_dice"]
        )

        # Save the original V1 weights as a fallback candidate.
        base_checkpoint = make_checkpoint(
            epoch=-1,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_score=best_score,
            original_metrics=baseline["original_lits_kits"],
            task07_metrics=baseline["task07_pancreas"],
            train_loss=float("nan"),
        )

        base_checkpoint["checkpoint_role"] = (
            "Original V1 baseline before Task07 fine-tuning"
        )

        torch.save(base_checkpoint, BEST_CHECKPOINT)

    if start_epoch >= EPOCHS:
        print(
            "\nThe configured epoch limit has already been reached. "
            "Increase EPOCHS in the script to continue training."
        )
        return

    print("\n" + "=" * 72)
    print("STARTING MIXED-COHORT FINE-TUNING")
    print("=" * 72)
    print("Epochs:", EPOCHS)
    print("Steps per epoch:", STEPS_PER_EPOCH)
    print("Patch size:", PATCH_SIZE)
    print("Learning rate:", LEARNING_RATE)
    print("Task07 pathology-centred patch probability:",
          PATHOLOGY_PATCH_PROBABILITY)
    print("Checkpoint directory:", OUTPUT_DIR)

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for step, (image, target) in enumerate(train_loader, start=1):
            image = image.to(DEVICE, non_blocking=True)
            target = target.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                "cuda", enabled=AMP_ENABLED
            ):
                outputs = model(image)
                loss = deep_supervision_loss(
                    outputs, target, criterion
                )

            if not torch.isfinite(loss).item():
                raise FloatingPointError(
                    f"Non-finite loss at epoch {epoch + 1}, step {step}."
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=12.0,
            )

            scaler.step(optimizer)
            scaler.update()

            epoch_loss += float(loss.item())

            if step % 25 == 0:
                print(
                    f"Epoch {epoch + 1}/{EPOCHS} | "
                    f"step {step}/{STEPS_PER_EPOCH} | "
                    f"loss {loss.item():.5f}"
                )

        epoch_loss /= len(train_loader)
        scheduler.step()

        # Validate each held-out cohort independently.
        original_metrics = evaluate_cases(
            model, original_val, "LiTS/KiTS validation"
        )
        task07_metrics = evaluate_cases(
            model, task07_val, "Task07 validation"
        )

        balanced_score = balanced_cohort_score(
            original_metrics, task07_metrics
        )

        improved = balanced_score > best_score

        if improved:
            best_score = balanced_score

        learning_rate = float(
            optimizer.param_groups[0]["lr"]
        )

        elapsed_minutes = (time.time() - start_time) / 60.0

        print("\n" + "-" * 72)
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print(f"Train loss:                    {epoch_loss:.5f}")
        print(
            "LiTS/KiTS pathology Dice:     "
            f"{original_metrics['dice_pathology']:.5f}"
        )
        print(
            "Task07 pathology Dice:        "
            f"{task07_metrics['dice_pathology']:.5f}"
        )
        print(f"Balanced cohort Dice:          {balanced_score:.5f}")
        print(f"Best balanced cohort Dice:     {best_score:.5f}")
        print(f"Learning rate:                 {learning_rate:.8f}")
        print(f"Epoch duration:                {elapsed_minutes:.2f} min")

        state = make_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            best_score=best_score,
            original_metrics=original_metrics,
            task07_metrics=task07_metrics,
            train_loss=epoch_loss,
        )

        state["checkpoint_role"] = "Task07 mixed-cohort fine-tuning"

        torch.save(state, LATEST_CHECKPOINT)

        if improved:
            torch.save(state, BEST_CHECKPOINT)
            print("NEW BEST BALANCED-COHORT CHECKPOINT SAVED")

        history_row = {
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "original_lits_kits": original_metrics,
            "task07_pancreas": task07_metrics,
            "balanced_cohort_pathology_dice": balanced_score,
            "best_balanced_cohort_pathology_dice": best_score,
            "learning_rate": learning_rate,
            "epoch_minutes": elapsed_minutes,
        }

        with HISTORY_FILE.open(
            "a", encoding="utf-8"
        ) as file:
            file.write(json.dumps(history_row) + "\n")

        print("-" * 72)

    print("\n" + "=" * 72)
    print("TASK07 FINE-TUNING FINISHED")
    print("=" * 72)
    print("Best checkpoint:", BEST_CHECKPOINT)
    print("Latest checkpoint:", LATEST_CHECKPOINT)
    print("Baseline metrics:", BASELINE_FILE)
    print("Training history:", HISTORY_FILE)
    print(f"Best balanced-cohort pathology Dice: {best_score:.5f}")
    print("Original V1 checkpoint was not overwritten.")


if __name__ == "__main__":
    main()
