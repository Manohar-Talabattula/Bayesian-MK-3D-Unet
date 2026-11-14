import torch
from dynamic_network_architectures.architectures.unet import PlainConvUNet


print("=" * 70)
print("STAGE 2 ARCHITECTURE TEST")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Exact LesionSegmenter 3D full-resolution architecture
model = PlainConvUNet(
    input_channels=1,
    n_stages=6,
    features_per_stage=[32, 64, 128, 256, 320, 320],
    conv_op=torch.nn.Conv3d,
    kernel_sizes=[[1, 3, 3],
                  [3, 3, 3],
                  [3, 3, 3],
                  [3, 3, 3],
                  [3, 3, 3],
                  [3, 3, 3]],
    strides=[[1, 1, 1],
              [1, 2, 2],
              [2, 2, 2],
              [2, 2, 2],
              [2, 2, 2],
              [2, 2, 2]],
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

model = model.to(device)

print("Model created successfully.")

# Use a small test volume first to avoid unnecessary VRAM usage.
x = torch.randn(1, 1, 32, 96, 96, device=device)

print("Input shape:", tuple(x.shape))

with torch.no_grad():
    output = model(x)

print("Output type:", type(output))

if isinstance(output, (list, tuple)):
    print("Number of deep-supervision outputs:", len(output))

    for i, out in enumerate(output):
        print(f"Output {i} shape:", tuple(out.shape))
else:
    print("Output shape:", tuple(output.shape))

print("\nParameters:", sum(p.numel() for p in model.parameters()))

print("\n" + "=" * 70)
print("STAGE 2 ARCHITECTURE TEST PASSED")
print("=" * 70)