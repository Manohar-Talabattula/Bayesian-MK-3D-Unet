import torch

checkpoint_path = r"C:\Users\HPCLAB\Downloads\Stage2_Clean\LesionSegmenter\nnUNet_results\Dataset1357_RADFULL\nnUNetTrainer__nnUNetPlans__3d_fullres\fold_all\checkpoint_final.pth"

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=False
)

weights = checkpoint["network_weights"]

print("=" * 70)
print("LESIONSEGMENTER WEIGHT SHAPES")
print("=" * 70)

print("\nENCODER CONVOLUTION WEIGHTS:")
for key, tensor in weights.items():
    if "encoder" in key and key.endswith(".conv.weight"):
        print(f"{key:70s} {tuple(tensor.shape)}")

print("\nDECODER TRANSCONV WEIGHTS:")
for key, tensor in weights.items():
    if key.startswith("decoder.transpconvs") and key.endswith(".weight"):
        print(f"{key:70s} {tuple(tensor.shape)}")

print("\nSEGMENTATION HEADS:")
for key, tensor in weights.items():
    if key.startswith("decoder.seg_layers") and key.endswith(".weight"):
        print(f"{key:70s} {tuple(tensor.shape)}")

print("\n" + "=" * 70)
print("CHECKPOINT METADATA")
print("=" * 70)

print("\nTRAINER:")
print(checkpoint.get("trainer_name"))

print("\nCURRENT EPOCH:")
print(checkpoint.get("current_epoch"))

print("\nINIT ARGS:")
print(checkpoint.get("init_args"))

print("=" * 70)