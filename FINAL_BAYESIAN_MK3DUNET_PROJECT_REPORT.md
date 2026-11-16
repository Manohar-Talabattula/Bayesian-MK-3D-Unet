# Bayesian MK-3D-UNet
## Consolidated Project Report

**Project:** Bayesian MK-3D-UNet: Multi-Organ and Lesion Segmentation from 3D Abdominal CT  
**Architecture:** Bayesian MK-3D-UNet  
**Framework:** PyTorch and MONAI  
**Environment:** Python 3.12.10, CUDA  
**Project status:** Training, evaluation and inference prototype completed

---

## 1. Project Overview

This project develops a Bayesian MK-3D-UNet-based pipeline for 3D abdominal CT segmentation. It combines multi-organ segmentation, pathology segmentation and a pancreas-specific cascaded inference workflow.

The project is organized into three major stages:

1. Stage 1: Multi-organ segmentation using BTCV, AMOS22 and TotalSegmentator.
2. Stage 2: Liver and kidney pathology segmentation using LiTS and KiTS19.
3. Stage 3: Pancreatic lesion segmentation using Task07 Pancreas and a cascaded pancreas-localization pipeline.

A Streamlit application provides an interface for running the trained models on uploaded NIfTI CT volumes.

This is a research prototype, not a clinically validated medical device.

---

## 2. Stage 1 — Multi-Organ Segmentation

### Datasets
- BTCV
- AMOS22
- TotalSegmentator

### Model
- Bayesian MK-3D-UNet
- 13 output classes
- Parameters: 15,111,462
- Training patch size: 96 × 96 × 64
- Number of Bayesian samples: 2
- Optimizer learning rate: 0.0001
- Loss: DiceCELoss

### Best checkpoint
`training/checkpoints_exp5_samples4/best.pth`

### Recorded result
- Best validation Dice: 0.6144059
- Best checkpoint epoch: 71

The Stage 1 model provides the multi-organ segmentation foundation and is also used for pancreas localization in the cascaded inference pipeline.

---

## 3. Stage 2 — Liver and Kidney Pathology Segmentation

### Datasets
- LiTS (Liver Tumor Segmentation)
- KiTS19 (Kidney Tumor Segmentation)

The local dataset folders include LiTS batches, KiTS19 cases and associated dataset manifests.

### Existing model checkpoints

- `datasets/stage2/models/stage2_best.pth`
- `datasets/stage2/models/stage2_best_balanced.pth`
- `datasets/stage2/models/stage2_best_weighted.pth`
- `datasets/stage2/models/stage2_latest.pth`
- `datasets/stage2/models/stage2_latest_balanced.pth`
- `datasets/stage2/models/stage2_latest_weighted.pth`

The weighted and balanced checkpoints are distinct model variants. Their evaluation results should not be treated as interchangeable.

### Full-volume evaluation

Evaluation configuration:
- Evaluation cases: 69
- Inference: Full-volume sliding window
- Patch size: 32 × 96 × 96
- Overlap: 0.50
- Inference batch size: 2

Dataset distribution:

| Dataset | Evaluated cases |
|---|---:|
| LiTS Batch 1 | 6 |
| LiTS Batch 2 | 21 |
| KiTS19 | 42 |
| **Total** | **69** |

### Weighted checkpoint results

Checkpoint: `stage2_best_weighted.pth`

| Metric | Result |
|---|---:|
| Mean pathology Dice | 0.30118 |
| Mean pathology IoU | 0.20966 |
| Mean pathology precision | 0.22751 |
| Mean pathology recall | 0.66677 |
| Mean pathology specificity | 0.99725 |

### Balanced checkpoint results

The threshold-sweep evaluation reports the following results at threshold 0.80:

| Metric | Result |
|---|---:|
| Mean pathology Dice | 0.33439 |
| Mean pathology IoU | 0.23710 |
| Mean pathology precision | 0.26077 |
| Mean pathology recall | 0.65825 |
| Mean pathology specificity | 0.99784 |

The balanced checkpoint achieved a higher reported mean pathology Dice than the weighted checkpoint in these recorded evaluations. These results come from separate evaluation files and should be compared with awareness of their respective configurations.

Evaluation files:
- `datasets/stage2/evaluation/stage2_full_volume_results.json`
- `datasets/stage2/evaluation/stage2_balanced_threshold_sweep_results.json`
- `datasets/stage2/evaluation/stage2_threshold_sweep_results.json`

**Verification note:** The 69-case evaluation confirms LiTS and KiTS19 coverage. The exact original training split and complete training history have not yet been independently verified for this consolidated report.

### Anatomy-Guided LiTS/KiTS Evaluation

A second evaluation experiment investigated whether Stage 1 anatomical localization could improve the general Stage 2 liver and kidney pathology predictions.

**Configuration**
- Stage 1 checkpoint: `training/checkpoints_exp5_samples4/best.pth`
- General Stage 2 checkpoint: `datasets/stage2/models/stage2_best_balanced.pth`
- Evaluation coverage: 69 cases
- Stage 1 anatomical classes: right kidney, left kidney and liver
- Anatomy-gate expansion margin: 15 mm
- Stage 2 probability threshold: 0.65
- Cached Stage 1 anatomical masks were reused for evaluation.
- No model retraining was performed for this experiment.

**Overall micro-averaged results**

| Metric | Baseline | Anatomy-gated |
|---|---:|---:|
| Dice | 0.5143 | **0.5808** |
| IoU | 0.3461 | **0.4092** |
| Precision | 0.3965 | **0.5691** |
| Recall | **0.7315** | 0.5929 |
| True positives | 796,575 | 645,668 |
| False positives | 1,212,304 | 488,782 |
| False negatives | 292,426 | 443,333 |

Anatomy gating increased micro Dice by approximately 0.0665 and substantially reduced false positives. It also reduced recall, showing that restricting predictions to an anatomical region can suppress some true-positive pathology voxels.

**Dataset-wise micro-averaged results**

| Dataset | Cases | Method | Dice | IoU | Precision | Recall |
|---|---:|---|---:|---:|---:|---:|
| LiTS Batch 1 | 6 | Baseline | 0.4299 | 0.2738 | 0.2881 | 0.8463 |
| LiTS Batch 1 | 6 | Gated | **0.6392** | **0.4697** | **0.5161** | 0.8394 |
| LiTS Batch 2 | 21 | Baseline | 0.4592 | 0.2980 | 0.3234 | 0.7918 |
| LiTS Batch 2 | 21 | Gated | **0.6437** | **0.4746** | **0.5654** | 0.7471 |
| KiTS19 | 42 | Baseline | 0.5390 | 0.3689 | 0.4343 | **0.7102** |
| KiTS19 | 42 | Gated | **0.5581** | **0.3871** | **0.5760** | 0.5413 |

The improvement was more pronounced on the LiTS subsets. KiTS19 showed a smaller Dice improvement and a more substantial recall reduction. These results indicate that the benefit of anatomical gating varies across datasets.

**Representative qualitative comparisons**

Four representative cases were exported. Each comparison image contains the CT slice, ground-truth pathology, baseline prediction and anatomy-gated prediction.

1. LiTS Batch 1 — `LiTS_Batch1_LiTS_Batch1_0010_0000`: Dice 0.4278 → 0.7530.
2. LiTS Batch 2 — `LiTS_Batch2_LiTS_Batch2_0057_0000`: Dice 0.1868 → 0.5348.
3. KiTS19 — `KiTS19_KiTS19_0046_0000`: Dice 0.4027 → 0.7388.
4. KiTS19 failure example — `KiTS19_KiTS19_0151_0000`: Dice 0.8107 → 0.0890.

Comparison images and per-case metrics are stored under:

`evaluation/stage2_anatomy_gate/representative_exports/`

The four exported case metrics were checked against the original 69-case evaluation results and matched to four decimal places.

**Interpretation:** Anatomical gating is a useful false-positive reduction strategy, particularly for the evaluated LiTS cases. However, the KiTS19 failure example demonstrates that gating can remove substantial true-positive regions. It should therefore be treated as an experimental inference refinement, not as a universally superior replacement for the baseline prediction.

---

## 4. Stage 2 Extension — Pancreatic Lesion Fine-Tuning

A separate pancreas-specific fine-tuning experiment was conducted using Task07 Pancreas.

### Checkpoint
`datasets/stage2/models/task07_finetune_v2/latest_task07_finetune.pth`

### Recorded training result
- Epoch: 24
- Best validation Dice: 0.6982574
- Training patch size: 32 × 96 × 96
- Batch size: 1
- Learning rate: 0.00002

This pancreas-specific model is separate from the general LiTS/KiTS Stage 2 checkpoints.

---

## 5. Stage 3 — Cascaded Pancreas Segmentation

Stage 3 combines Stage 1 pancreas localization with Stage 2 pancreas-specific lesion prediction.

### Pipeline

1. Load the abdominal CT volume.
2. Run Stage 1 multi-organ segmentation.
3. Extract the predicted pancreas region.
4. Map the pancreas region to the Stage 2 inference grid.
5. Run pancreas-specific lesion segmentation.
6. Apply pancreas-guided spatial gating.
7. Generate the final lesion mask.
8. Calculate lesion measurements and connected components.

### Evaluation configuration

- Dataset: Task07 Pancreas
- Evaluation cases: 56
- Probability threshold: 0.80
- Sliding-window overlap: 0.50
- Pancreas gate margin: 15 mm

### Cascaded evaluation results

| Metric | Cascaded result |
|---|---:|
| Micro Dice | 0.261418 |
| Micro IoU | 0.150363 |
| Precision | 0.322174 |
| Recall | 0.219942 |
| Mean case Dice | 0.2037 |
| Median case Dice | 0.0623 |
| Cases with nonzero Dice | 33 |
| Cases with zero Dice | 23 |
| Cases with Dice >= 0.50 | 9 |

Voxel counts:
- True positives: 20,007
- False positives: 42,093
- False negatives: 70,958

### Baseline comparison

| Metric | Baseline | Cascaded |
|---|---:|---:|
| Micro Dice | 0.094959 | 0.261418 |
| Precision | 0.053896 | 0.322174 |
| Recall | 0.398802 | 0.219942 |

The pancreas gate substantially reduced false-positive predictions, improving precision and micro Dice, while reducing recall. This demonstrates the precision-recall trade-off introduced by spatial gating.

### Existing outputs

- `evaluation/FINAL_STAGE3_PROJECT_REPORT.md`
- `evaluation/task07_stage3_exports_overlap075/`
- `evaluation/visual_results_stage3/case157_stage3_3d_visualization.png`

The export directory name includes `overlap075`, but the recorded inference configuration used 0.50 overlap. The actual configuration should take precedence over the folder name.

---

## 6. Streamlit Inference Application

Application file: `inference/app.py`  
Prediction module: `inference/predict.py`

The application supports:
- Uploading 3D NIfTI CT scans (`.nii` and `.nii.gz`).
- Running the trained two-stage pancreas segmentation pipeline.
- Configuring the lesion probability threshold.
- Configuring the pancreas gate margin.
- Displaying predicted lesion volume.
- Displaying predicted lesion voxel count.
- Reporting connected components.
- Downloading prediction outputs and JSON measurements.

### Application verification

A real inference test was completed successfully using:

`case_00209.nii.gz`

Recorded test output:
- Device: CUDA
- Predicted lesion volume: 1.5564 mL
- Predicted lesion voxels: 510
- Connected components: 4
- Mapped pancreas voxels: 5,266
- Probability threshold: 0.80
- Gate margin: 15 mm

This confirms that the upload-to-prediction application workflow executes successfully. It does not independently establish segmentation accuracy for that input.

---

## 7. Project Limitations

1. The Stage 2 pathology Dice values indicate that pathology segmentation remains challenging.
2. Stage 3 has a substantial number of zero-Dice cases, indicating missed lesions or unsuccessful predictions on some evaluation cases.
3. Spatial gating improves precision but reduces recall.
4. The reported evaluation results are dataset-specific and should not be generalized without external validation.
5. No clinical validation has been performed.
6. Predicted connected components are image-derived candidates, not clinically verified individual lesions.
7. The exact training split and complete training history of the original LiTS/KiTS experiment require further archival verification.

---

## 8. Conclusion

The project implements a multi-stage 3D abdominal CT segmentation research pipeline using Bayesian MK-3D-UNet.

The completed work includes multi-organ segmentation, LiTS/KiTS pathology model development and evaluation, pancreas-specific fine-tuning, cascaded pancreas lesion evaluation and a working Streamlit inference prototype.

The results demonstrate the feasibility of combining multi-organ localization with pathology-specific segmentation. However, the measured performance also highlights the need for improved lesion recall, further validation and more extensive testing.

**Research-use disclaimer:** This software is intended for research and educational purposes only. Its predictions are not clinically verified and must not be used for diagnosis or treatment.
