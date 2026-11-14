from pathlib import Path
import torch

files = [
    Path("datasets/stage2/models/stage2_best_balanced.pth"),
    Path("datasets/stage2/models/task07_finetune_v2/latest_task07_finetune.pth"),
]

for path in files:
    print("\n" + "=" * 70)
    print("FILE:", path)
    print("EXISTS:", path.exists())

    if not path.exists():
        continue

    print("SIZE (MB):", round(path.stat().st_size / (1024 * 1024), 2))

    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict):
        for key in [
            "epoch", "epoch_index", "best_val_dice",
            "best_balanced_cohort_pathology_dice",
            "model_state", "model_state_dict", "state_dict"
        ]:
            if key in ckpt:
                value = ckpt[key]
                if isinstance(value, dict):
                    print(f"{key}: state dictionary ({len(value)} entries)")
                else:
                    print(f"{key}: {value}")
        print("CHECKPOINT KEYS:", list(ckpt.keys()))
    else:
        print("Checkpoint type:", type(ckpt))
