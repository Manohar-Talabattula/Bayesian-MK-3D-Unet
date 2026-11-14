import torch
from dynamic_network_architectures.architectures.unet import PlainConvUNet


CHECKPOINT = r"C:\Users\HPCLAB\Downloads\Stage2_Clean\LesionSegmenter\nnUNet_results\Dataset1357_RADFULL\nnUNetTrainer__nnUNetPlans__3d_fullres\fold_all\checkpoint_final.pth"


# ---------------------------------------------------------
# 1. Build our 3-class Stage 2 architecture
# ---------------------------------------------------------

model = PlainConvUNet(
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
        [3, 3, 3]
    ],
    strides=[
        [1, 1, 1],
        [1, 2, 2],
        [2, 2, 2],
        [2, 2, 2],
        [2, 2, 2],
        [2, 2, 2]
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
    deep_supervision=True
)


# ---------------------------------------------------------
# 2. Load pretrained checkpoint
# ---------------------------------------------------------

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False
)

pretrained = checkpoint["network_weights"]

model_state = model.state_dict()


# ---------------------------------------------------------
# 3. Compare weights
# ---------------------------------------------------------

compatible = []
incompatible_shape = []
missing = []

for key, pretrained_tensor in pretrained.items():

    if key not in model_state:
        missing.append(key)

    elif pretrained_tensor.shape == model_state[key].shape:
        compatible.append(key)

    else:
        incompatible_shape.append(
            (
                key,
                tuple(pretrained_tensor.shape),
                tuple(model_state[key].shape)
            )
        )


# ---------------------------------------------------------
# 4. Print results
# ---------------------------------------------------------

print("=" * 70)
print("STAGE 2 PRETRAINED WEIGHT COMPATIBILITY CHECK")
print("=" * 70)

print("\nTotal pretrained tensors:", len(pretrained))
print("Total Stage 2 model tensors:", len(model_state))

print("\nCompatible tensors:", len(compatible))
print("Missing from Stage 2 model:", len(missing))
print("Shape-incompatible tensors:", len(incompatible_shape))


print("\n" + "-" * 70)
print("SHAPE-INCOMPATIBLE TENSORS")
print("-" * 70)

for key, old_shape, new_shape in incompatible_shape:
    print(f"\n{key}")
    print("  Pretrained:", old_shape)
    print("  Stage 2:   ", new_shape)


print("\n" + "-" * 70)
print("MISSING / EXTRA TENSORS")
print("-" * 70)

for key in missing:
    print(key)


print("\n" + "-" * 70)
print("FIRST 30 COMPATIBLE TENSORS")
print("-" * 70)

for key in compatible[:30]:
    print(key)


print("\n" + "=" * 70)
print("CHECK COMPLETE")
print("=" * 70)