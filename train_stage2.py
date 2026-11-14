from pathlib import Path
import json
import random
import re
import time

import nibabel as nib
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from dynamic_network_architectures.architectures.unet import PlainConvUNet


# ============================================================
# PATHS
# ============================================================

ROOT = Path(
    r"C:\Users\HPCLAB\Documents\MK_3D_UNET_PROJECT"
)

STAGE2_DIR = ROOT / "datasets" / "stage2"

DATASET_DIRS = {
    "LiTS_Batch1": STAGE2_DIR / "LiTS_Batch1",
    "LiTS_Batch2": STAGE2_DIR / "LiTS_Batch2",
    "KiTS19": STAGE2_DIR / "KiTS19",
}

SPLIT_FILE = (
    STAGE2_DIR
    / "split"
    / "stage2_split_manifest.json"
)

INIT_CHECKPOINT = (
    STAGE2_DIR
    / "models"
    / "stage2_pretrained_init.pth"
)

MODEL_DIR = STAGE2_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BEST_CHECKPOINT = (
    MODEL_DIR / "stage2_best_balanced.pth"
)

LATEST_CHECKPOINT = (
    MODEL_DIR / "stage2_latest_balanced.pth"
)


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

SEED = 42

PATCH_SIZE = (
    32,
    96,
    96
)

BATCH_SIZE = 1

EPOCHS = 60

STEPS_PER_EPOCH = 100

VAL_CASE_LIMIT = None

LEARNING_RATE = 1e-4

WEIGHT_DECAY = 1e-5

NUM_WORKERS = 0

AMP_ENABLED = True

# Probability of sampling around class-2 pathology.
PATHOLOGY_PATCH_PROBABILITY = 0.70

# Number of classes.
NUM_CLASSES = 3


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ============================================================
# DEVICE
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is required for Stage-2 training."
    )

DEVICE = torch.device("cuda")

print("=" * 72)
print("STAGE 2 TRAINING")
print("=" * 72)

print("Device:", torch.cuda.get_device_name(0))
print("CUDA:", torch.version.cuda)
print("Patch size:", PATCH_SIZE)
print("Batch size:", BATCH_SIZE)
print("Epochs:", EPOCHS)
print("Steps / epoch:", STEPS_PER_EPOCH)
print("Learning rate:", LEARNING_RATE)


# ============================================================
# CASE UTILITIES
# ============================================================

IMAGE_SUFFIX = "_0000.nii.gz"


def get_image_files(folder):

    files = []

    for p in folder.glob("*_0000.nii.gz"):

        name = p.name

        if not name.endswith(IMAGE_SUFFIX):
            continue

        stem = name[:-len(IMAGE_SUFFIX)]

        parts = stem.split("_")

        if (
            len(parts) >= 2
            and parts[-1].isdigit()
        ):
            files.append(p)

    return sorted(files)


def label_from_image(image_path):

    return image_path.with_name(
        image_path.name[:-len(IMAGE_SUFFIX)]
        + ".nii.gz"
    )


def case_id_from_image(image_path):

    return image_path.name[
        :-len(IMAGE_SUFFIX)
    ]


def collect_all_cases():

    cases = {}

    for dataset_name, folder in DATASET_DIRS.items():

        images = get_image_files(folder)

        if not images:
            raise RuntimeError(
                f"No images found in {folder}"
            )

        for image_path in images:

            label_path = label_from_image(
                image_path
            )

            if not label_path.exists():
                raise RuntimeError(
                    f"Missing label:\n{label_path}"
                )

            case_id = case_id_from_image(
                image_path
            )

            cases[case_id] = {
                "case_id": case_id,
                "dataset": dataset_name,
                "image": image_path,
                "label": label_path,
            }

    return cases


# ============================================================
# SPLIT MANIFEST PARSER
# ============================================================

CASE_PATTERN = re.compile(
    r"(LiTS_Batch1|LiTS_Batch2|KiTS19)_\d{4}"
)


def extract_case_ids(node):

    found = []

    if isinstance(node, str):

        matches = CASE_PATTERN.findall(node)

        if matches:

            # Need complete case ID, not only dataset prefix.
            complete = re.findall(
                r"(?:LiTS_Batch1|LiTS_Batch2|KiTS19)_\d{4}",
                node
            )

            found.extend(complete)

    elif isinstance(node, list):

        for item in node:
            found.extend(
                extract_case_ids(item)
            )

    elif isinstance(node, dict):

        for value in node.values():
            found.extend(
                extract_case_ids(value)
            )

    return found


def find_split_cases(obj):

    train = []
    val = []

    def walk(node):

        nonlocal train, val

        if isinstance(node, dict):

            for key, value in node.items():

                key_lower = str(key).lower()

                if (
                    "train" in key_lower
                    and "val" not in key_lower
                ):

                    train.extend(
                        extract_case_ids(value)
                    )

                elif (
                    "val" in key_lower
                    or "valid" in key_lower
                ):

                    val.extend(
                        extract_case_ids(value)
                    )

                else:
                    walk(value)

        elif isinstance(node, list):

            for item in node:
                walk(item)

    walk(obj)

    train = list(dict.fromkeys(train))
    val = list(dict.fromkeys(val))

    return train, val


# ============================================================
# LOAD CASES
# ============================================================

print("\nLoading dataset index...")

ALL_CASES = collect_all_cases()

print("Total cases indexed:", len(ALL_CASES))

if len(ALL_CASES) != 341:
    raise RuntimeError(
        f"Expected 341 cases, found {len(ALL_CASES)}"
    )


# ============================================================
# LOAD SPLIT
# ============================================================

if not SPLIT_FILE.exists():

    raise FileNotFoundError(
        f"Split manifest not found:\n{SPLIT_FILE}"
    )

with open(
    SPLIT_FILE,
    "r",
    encoding="utf-8"
) as f:

    split_json = json.load(f)


TRAIN_IDS, VAL_IDS = find_split_cases(
    split_json
)

print("\nSplit:")
print("  Train:", len(TRAIN_IDS))
print("  Val  :", len(VAL_IDS))


if len(TRAIN_IDS) != 272:

    raise RuntimeError(
        "Expected exactly 272 training cases, "
        f"but found {len(TRAIN_IDS)}."
    )

if len(VAL_IDS) != 69:

    raise RuntimeError(
        "Expected exactly 69 validation cases, "
        f"but found {len(VAL_IDS)}."
    )


overlap = set(TRAIN_IDS).intersection(
    set(VAL_IDS)
)

if overlap:

    raise RuntimeError(
        f"Train/validation overlap:\n{overlap}"
    )


if (
    set(TRAIN_IDS).union(set(VAL_IDS))
    != set(ALL_CASES.keys())
):

    missing = (
        set(ALL_CASES.keys())
        - set(TRAIN_IDS)
        - set(VAL_IDS)
    )

    extra = (
        set(TRAIN_IDS)
        | set(VAL_IDS)
    ) - set(ALL_CASES.keys())

    raise RuntimeError(
        f"Split does not cover exactly all cases.\n"
        f"Missing: {missing}\n"
        f"Extra: {extra}"
    )


TRAIN_CASES = [
    ALL_CASES[x]
    for x in TRAIN_IDS
]

VAL_CASES = [
    ALL_CASES[x]
    for x in VAL_IDS
]

print("Split verification: PASSED")


# ============================================================
# NIFTI CACHE
# ============================================================

class NiftiCase:

    def __init__(
        self,
        case
    ):

        self.case_id = case["case_id"]

        self.image_path = case["image"]

        self.label_path = case["label"]

        self.image = None

        self.label = None

    def load(self):

        if (
            self.image is not None
            and self.label is not None
        ):
            return

        image_nii = nib.load(
            str(self.image_path)
        )

        label_nii = nib.load(
            str(self.label_path)
        )

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
                f"Shape mismatch: "
                f"{self.case_id}"
            )

        unique = np.unique(label)

        if not np.all(
            np.isin(unique, [0, 1, 2])
        ):

            raise RuntimeError(
                f"Invalid labels in "
                f"{self.case_id}: {unique}"
            )

        # IMPORTANT:
        # The preprocessing pipeline already normalized
        # images to [0, 1].
        #
        # Do NOT apply HU normalization again.
        image = np.clip(
            image,
            0.0,
            1.0
        )

        # NIfTI is stored X,Y,Z.
        # Training uses Z,Y,X.
        image = np.transpose(
            image,
            (2, 1, 0)
        )

        label = np.transpose(
            label,
            (2, 1, 0)
        )

        self.image = np.ascontiguousarray(
            image,
            dtype=np.float32
        )

        self.label = np.ascontiguousarray(
            label,
            dtype=np.uint8
        )


# ============================================================
# DATASET
# ============================================================

class Stage2PatchDataset(Dataset):

    def __init__(
        self,
        cases,
        steps_per_epoch
    ):

        self.cases = [
            NiftiCase(c)
            for c in cases
        ]

        self.steps_per_epoch = (
            steps_per_epoch
        )

    def __len__(self):

        return self.steps_per_epoch

    def _random_patch(
        self,
        image,
        label
    ):

        z, y, x = label.shape

        pz, py, px = PATCH_SIZE

        # Decide whether to target pathology.
        has_pathology = np.any(
            label == 2
        )

        target_pathology = (
            has_pathology
            and random.random()
            < PATHOLOGY_PATCH_PROBABILITY
        )

        if target_pathology:

            coords = np.argwhere(
                label == 2
            )

            center = coords[
                random.randrange(
                    len(coords)
                )
            ]

            cz, cy, cx = center

        else:

            cz = random.randrange(z)
            cy = random.randrange(y)
            cx = random.randrange(x)

        start_z = cz - pz // 2
        start_y = cy - py // 2
        start_x = cx - px // 2

        max_z = max(0, z - pz)
        max_y = max(0, y - py)
        max_x = max(0, x - px)

        start_z = min(
            max(0, start_z),
            max_z
        )

        start_y = min(
            max(0, start_y),
            max_y
        )

        start_x = min(
            max(0, start_x),
            max_x
        )

        # If the volume is smaller than patch size,
        # pad it.
        pad_z = max(
            0,
            pz - z
        )

        pad_y = max(
            0,
            py - y
        )

        pad_x = max(
            0,
            px - x
        )

        if (
            pad_z
            or pad_y
            or pad_x
        ):

            image = np.pad(
                image,
                (
                    (
                        pad_z // 2,
                        pad_z - pad_z // 2
                    ),
                    (
                        pad_y // 2,
                        pad_y - pad_y // 2
                    ),
                    (
                        pad_x // 2,
                        pad_x - pad_x // 2
                    ),
                ),
                mode="constant",
                constant_values=0
            )

            label = np.pad(
                label,
                (
                    (
                        pad_z // 2,
                        pad_z - pad_z // 2
                    ),
                    (
                        pad_y // 2,
                        pad_y - pad_y // 2
                    ),
                    (
                        pad_x // 2,
                        pad_x - pad_x // 2
                    ),
                ),
                mode="constant",
                constant_values=0
            )

            z, y, x = label.shape

            start_z = min(
                start_z,
                z - pz
            )

            start_y = min(
                start_y,
                y - py
            )

            start_x = min(
                start_x,
                x - px
            )

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

        return (
            image_patch,
            label_patch
        )

    def __getitem__(self, index):

        case = random.choice(
            self.cases
        )

        case.load()

        image_patch, label_patch = (
            self._random_patch(
                case.image,
                case.label
            )
        )

        image_tensor = torch.from_numpy(
            image_patch.copy()
        ).unsqueeze(0)

        label_tensor = torch.from_numpy(
            label_patch.copy()
        ).long()

        return (
            image_tensor,
            label_tensor
        )


# ============================================================
# MODEL
# ============================================================

def build_model():

    return PlainConvUNet(

        input_channels=1,

        n_stages=6,

        features_per_stage=[
            32,
            64,
            128,
            256,
            320,
            320
        ],

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

        n_conv_per_stage=[
            2,
            2,
            2,
            2,
            2,
            2
        ],

        num_classes=3,

        n_conv_per_stage_decoder=[
            2,
            2,
            2,
            2,
            2
        ],

        conv_bias=True,

        norm_op=torch.nn.InstanceNorm3d,

        norm_op_kwargs={
            "eps": 1e-5,
            "affine": True
        },

        dropout_op=None,

        dropout_op_kwargs=None,

        nonlin=torch.nn.LeakyReLU,

        nonlin_kwargs={
            "inplace": True
        },

        deep_supervision=True
    )


# ============================================================
# LOAD PRETRAINED INITIALIZATION
# ============================================================

print("\nLoading Stage-2 pretrained initialization...")

model = build_model()

checkpoint = torch.load(
    INIT_CHECKPOINT,
    map_location="cpu",
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True
)

model = model.to(DEVICE)

print("Pretrained initialization loaded.")


# ============================================================
# LOSS
# ============================================================

# ============================================================
# LOSS
# ============================================================

class PathologyWeightedDiceCELoss(nn.Module):

    def __init__(self):
        super().__init__()

        # [background, healthy/non-lesional, pathology]
        self.ce_weight = torch.tensor(
            [0.25, 1.0, 3.0],
            dtype=torch.float32
        )

    def dice_loss(
        self,
        logits,
        target
    ):

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        target_one_hot = F.one_hot(
            target.long(),
            num_classes=NUM_CLASSES
        )

        target_one_hot = (
            target_one_hot
            .permute(0, 4, 1, 2, 3)
            .float()
        )

        dims = (
            0,
            2,
            3,
            4
        )

        intersection = (
            probabilities
            * target_one_hot
        ).sum(dims)

        denominator = (
            probabilities.sum(dims)
            + target_one_hot.sum(dims)
        )

        dice = (
            2.0 * intersection + 1e-5
        ) / (
            denominator + 1e-5
        )

        # Class 1 = healthy/non-lesional
        # Class 2 = pathology
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

    def forward(
        self,
        logits,
        target
    ):

        ce_weight = self.ce_weight.to(
            device=logits.device,
            dtype=torch.float32
        )

        ce = F.cross_entropy(
            logits,
            target,
            weight=ce_weight
        )

        dice = self.dice_loss(
            logits,
            target
        )

        return ce + dice


criterion = PathologyWeightedDiceCELoss()

# ============================================================
# DEEP SUPERVISION LOSS
# ============================================================

def deep_supervision_loss(
    outputs,
    target
):

    total_loss = 0.0

    weights = [
        1.0,
        0.5,
        0.25,
        0.125,
        0.0625
    ]

    for output, weight in zip(
        outputs,
        weights
    ):

        if output.shape[2:] != target.shape[1:]:

            target_scaled = F.interpolate(
                target.unsqueeze(1).float(),
                size=output.shape[2:],
                mode="nearest"
            ).squeeze(1).long()

        else:

            target_scaled = target

        loss = criterion(
            output,
            target_scaled
        )

        total_loss += (
            weight * loss
        )

    return total_loss / sum(weights)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=AMP_ENABLED
)


# ============================================================
# DATASETS / LOADERS
# ============================================================

train_dataset = Stage2PatchDataset(
    TRAIN_CASES,
    STEPS_PER_EPOCH
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True
)


# ============================================================
# VALIDATION UTILITIES
# ============================================================

def validation_patch(
    image,
    label
):

    z, y, x = label.shape

    pz, py, px = PATCH_SIZE

    # --------------------------------------------------------
    # Pad if volume is smaller than the validation patch.
    # --------------------------------------------------------

    pad_z = max(0, pz - z)
    pad_y = max(0, py - y)
    pad_x = max(0, px - x)

    if (
        pad_z
        or pad_y
        or pad_x
    ):

        image = np.pad(
            image,
            (
                (
                    pad_z // 2,
                    pad_z - pad_z // 2
                ),
                (
                    pad_y // 2,
                    pad_y - pad_y // 2
                ),
                (
                    pad_x // 2,
                    pad_x - pad_x // 2
                )
            ),
            mode="constant"
        )

        label = np.pad(
            label,
            (
                (
                    pad_z // 2,
                    pad_z - pad_z // 2
                ),
                (
                    pad_y // 2,
                    pad_y - pad_y // 2
                ),
                (
                    pad_x // 2,
                    pad_x - pad_x // 2
                )
            ),
            mode="constant"
        )

        z, y, x = label.shape

    # --------------------------------------------------------
    # Pathology-aware validation
    #
    # If pathology exists, center the patch on the first
    # pathology voxel found.
    #
    # Otherwise use the geometric center of the volume.
    # --------------------------------------------------------

    pathology_voxels = np.argwhere(
        label == 2
    )

    if len(pathology_voxels) > 0:

        # Deterministic selection:
        # use the median pathology voxel rather than a random
        # voxel, giving a stable validation patch every epoch.
        center_z, center_y, center_x = np.median(
            pathology_voxels,
            axis=0
        ).astype(int)

    else:

        center_z = z // 2
        center_y = y // 2
        center_x = x // 2

    # --------------------------------------------------------
    # Convert center coordinates into patch start coordinates.
    # --------------------------------------------------------

    start_z = int(
        np.clip(
            center_z - pz // 2,
            0,
            z - pz
        )
    )

    start_y = int(
        np.clip(
            center_y - py // 2,
            0,
            y - py
        )
    )

    start_x = int(
        np.clip(
            center_x - px // 2,
            0,
            x - px
        )
    )

    image_patch = image[
        start_z:start_z + pz,
        start_y:start_y + py,
        start_x:start_x + px
    ]

    label_patch = label[
        start_z:start_z + pz,
        start_y:start_y + py,
        start_x:start_x + px
    ]

    return (
        image_patch,
        label_patch
    )


def dice_per_class(
    prediction,
    target
):

    scores = []

    for class_id in range(
        NUM_CLASSES
    ):

        pred = prediction == class_id
        true = target == class_id

        intersection = np.logical_and(
            pred,
            true
        ).sum()

        pred_count = pred.sum()
        true_count = true.sum()

        if (
            pred_count == 0
            and true_count == 0
        ):

            # Perfect agreement when both
            # are absent.
            dice = 1.0

        else:

            dice = (
                2.0 * intersection
                + 1e-5
            ) / (
                pred_count
                + true_count
                + 1e-5
            )

        scores.append(
            float(dice)
        )

    return scores


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate():

    model.eval()

    all_scores = []

    cases_to_validate = VAL_CASES

    if VAL_CASE_LIMIT is not None:

        cases_to_validate = (
            cases_to_validate[
                :VAL_CASE_LIMIT
            ]
        )

    for case_dict in cases_to_validate:

        case = NiftiCase(case_dict)

        case.load()

        image_patch, label_patch = (
            validation_patch(
                case.image,
                case.label
            )
        )

        image_tensor = (
            torch.from_numpy(
                image_patch.copy()
            )
            .unsqueeze(0)
            .unsqueeze(0)
            .to(DEVICE)
        )

        with torch.amp.autocast(
            "cuda",
            enabled=AMP_ENABLED
        ):

            outputs = model(
                image_tensor
            )

        logits = outputs[0]

        prediction = torch.argmax(
            logits,
            dim=1
        )

        prediction = (
            prediction
            .cpu()
            .numpy()[0]
        )

        scores = dice_per_class(
            prediction,
            label_patch
        )

        all_scores.append(
            scores
        )

    scores = np.asarray(
        all_scores,
        dtype=np.float64
    )

    mean_scores = scores.mean(
        axis=0
    )

    foreground_mean = mean_scores[
        1:
    ].mean()

    return (
        mean_scores,
        float(foreground_mean)
    )


# ============================================================
# RESUME
# ============================================================

start_epoch = 0
best_val_dice = -1.0

if LATEST_CHECKPOINT.exists():

    print(
        "\nFound previous checkpoint."
    )

    response = input(
        "Resume previous training? [y/N]: "
    ).strip().lower()

    if response == "y":

        resume = torch.load(
            LATEST_CHECKPOINT,
            map_location=DEVICE,
            weights_only=False
        )

        model.load_state_dict(
            resume["model_state_dict"]
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

        start_epoch = (
            resume["epoch"] + 1
        )

        best_val_dice = resume[
            "best_val_dice"
        ]

        print(
            f"Resuming from epoch "
            f"{start_epoch}"
        )


# ============================================================
# TRAINING LOOP
# ============================================================

print("\n")
print("=" * 72)
print("STARTING STAGE 2 TRAINING")
print("=" * 72)

for epoch in range(
    start_epoch,
    EPOCHS
):

    model.train()

    epoch_loss = 0.0

    start_time = time.time()

    for step, batch in enumerate(
        train_loader
    ):

        image, target = batch

        image = image.to(
            DEVICE,
            non_blocking=True
        )

        target = target.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.amp.autocast(
            "cuda",
            enabled=AMP_ENABLED
        ):

            outputs = model(
                image
            )

            loss = deep_supervision_loss(
                outputs,
                target
            )

        scaler.scale(
            loss
        ).backward()

        scaler.unscale_(
            optimizer
        )

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=12.0
        )

        scaler.step(
            optimizer
        )

        scaler.update()

        epoch_loss += (
            loss.item()
        )

    epoch_loss /= len(
        train_loader
    )

    scheduler.step()

    # Validation.
    val_scores, val_fg_dice = validate()

    elapsed = (
        time.time()
        - start_time
    )

    lr = optimizer.param_groups[0][
        "lr"
    ]

    print("\n" + "-" * 72)

    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )

    print(
        f"Train loss : {epoch_loss:.5f}"
    )

    print(
        f"Val Dice BG       : {val_scores[0]:.5f}"
    )

    print(
        f"Val Dice Healthy  : {val_scores[1]:.5f}"
    )

    print(
        f"Val Dice Pathology: {val_scores[2]:.5f}"
    )

    print(
        f"Val FG Dice       : {val_fg_dice:.5f}"
    )

    print(
        f"Learning rate     : {lr:.7f}"
    )

    print(
        f"Epoch time        : {elapsed / 60:.2f} min"
    )

    # Save latest.
    latest_state = {
        "epoch": epoch,
        "model_state_dict":
            model.state_dict(),
        "optimizer_state_dict":
            optimizer.state_dict(),
        "scheduler_state_dict":
            scheduler.state_dict(),
        "scaler_state_dict":
            scaler.state_dict(),
        "best_val_dice":
            best_val_dice,
        "train_loss":
            epoch_loss,
        "val_dice":
            val_scores.tolist(),
        "val_foreground_dice":
            val_fg_dice,
        "config": {
            "patch_size":
                PATCH_SIZE,
            "batch_size":
                BATCH_SIZE,
            "learning_rate":
                LEARNING_RATE,
            "epochs":
                EPOCHS,
            "steps_per_epoch":
                STEPS_PER_EPOCH
        }
    }

    torch.save(
        latest_state,
        LATEST_CHECKPOINT
    )

    # Save best.
    if val_scores[2] > best_val_dice:

        best_val_dice = val_scores[2]

        latest_state[
            "best_val_dice"
        ] = best_val_dice

        torch.save(
            latest_state,
            BEST_CHECKPOINT
        )

        print(
            f"NEW BEST PATHOLOGY MODEL: "
            f"{best_val_dice:.5f}"
        )

    print("-" * 72)


# ============================================================
# FINAL
# ============================================================

print("\n")
print("=" * 72)
print("STAGE 2 TRAINING FINISHED")
print("=" * 72)

print(
    "Best pathology Dice:",
    f"{best_val_dice:.5f}"
)
print(
    f"NEW BEST PATHOLOGY MODEL: "
    f"{best_val_dice:.5f}"
)

print(
    "Latest checkpoint:",
    LATEST_CHECKPOINT
)
