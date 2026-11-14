# Stage 2 preparation

Fixed target:
- 0 = background
- 1 = healthy/non-lesional tissue
- 2 = abnormal/pathological tissue

Datasets:
- LiTS
- KiTS19

Pretrained initialization:
- LesionSegmenter nnU-Net v2

Important:
Do not run `prepare_stage2_dataset.py` on the extracted LiTS/KiTS data until
the actual extracted folder structure has been inspected and the image/label
pairing is confirmed. The script deliberately refuses to guess pairings.

Next actions after downloads:
1. Inspect extracted LiTS structure.
2. Inspect extracted KiTS19 structure.
3. Run a one-case compatibility test.
4. Process the full datasets.
5. Build/fine-tune Stage 2.
