from pathlib import Path
import torch
from dynamic_network_architectures.architectures.unet import PlainConvUNet

ROOT = Path(r"C:\Users\HPCLAB\Documents\MK_3D_UNET_PROJECT")

PRETRAINED = Path(
    r"C:\Users\HPCLAB\Downloads\Stage2_Clean\LesionSegmenter"
    r"\nnUNet_results\Dataset1357_RADFULL"
    r"\nnUNetTrainer__nnUNetPlans__3d_fullres"
    r"\fold_all\checkpoint_final.pth"
)

OUTPUT = ROOT / "datasets" / "stage2" / "models" / "stage2_pretrained_init.pth"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)


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


print("Loading LesionSegmenter checkpoint...")
checkpoint = torch.load(
    PRETRAINED,
    map_location="cpu",
    weights_only=False
)

pretrained = checkpoint["network_weights"]

print("Building 3-class Stage-2 model...")
model = build_model()
current = model.state_dict()

compatible = {
    k: v
    for k, v in pretrained.items()
    if k in current and current[k].shape == v.shape
}

incompatible = [
    k for k in pretrained
    if k not in compatible
]

missing = [
    k for k in current
    if k not in compatible
]

print(f"Compatible tensors: {len(compatible)}")
print(f"New tensors:        {len(incompatible)}")

assert len(compatible) == 282
assert len(incompatible) == 10
assert len(missing) == 10

assert all(
    k.startswith("decoder.seg_layers.")
    for k in incompatible
)

# Transfer the 282 compatible LesionSegmenter tensors.
model.load_state_dict(compatible, strict=False)

# Save the complete 3-class initialization.
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "source_checkpoint": str(PRETRAINED),
        "pretrained_tensor_count": 282,
        "new_tensor_count": 10,
        "new_tensors": incompatible,
        "num_classes": 3,
        "classes": {
            0: "background",
            1: "healthy_non_lesional_tissue",
            2: "abnormal_pathological_tissue",
        },
    },
    OUTPUT,
)

print("\nSaved:")
print(OUTPUT)

# Reload and verify everything.
print("\nReloading checkpoint...")
saved = torch.load(
    OUTPUT,
    map_location="cpu",
    weights_only=False
)

check_model = build_model()
check_model.load_state_dict(
    saved["model_state_dict"],
    strict=True
)

print("All 292 tensors loaded successfully.")

# CUDA forward test
if torch.cuda.is_available():
    print("\nRunning CUDA forward test...")

    device = torch.device("cuda")
    check_model = check_model.to(device)
    check_model.eval()

    with torch.no_grad():
        x = torch.randn(
            1, 1, 32, 96, 96,
            device=device
        )

        output = check_model(x)

    assert isinstance(output, list)
    assert output[0].shape == (1, 3, 32, 96, 96)

    print("CUDA forward: PASSED")
    print("Output:", tuple(output[0].shape))
else:
    print("CUDA unavailable.")

print("\n========================================")
print("STAGE 2 PRETRAINED INITIALIZATION PASSED")
print("========================================")
print("LesionSegmenter tensors transferred : 282 / 292")
print("Fresh segmentation-head tensors    : 10 / 292")
print("Classes                            : 3")
print("Checkpoint                         :", OUTPUT)