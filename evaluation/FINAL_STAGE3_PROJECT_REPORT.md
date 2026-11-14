# Bayesian MK-3D-UNet ? Stage 3 Evaluation Report

## 1. Project overview

This project implements a Bayesian MK-3D-UNet-based 3D medical image segmentation pipeline. Stage 1 performs multi-organ segmentation, Stage 2 fine-tunes the segmentation model for pancreas and lesion classes, and Stage 3 combines pancreas localization with lesion prediction.

## 2. Stage 3 evaluation configuration

- Dataset: Medical Segmentation Decathlon Task07 Pancreas
- Evaluated cases: 56
- Stage 1 checkpoint: `training/checkpoints_exp5_samples4/best.pth`
- Stage 2 checkpoint: `datasets/stage2/models/task07_finetune_v2/latest_task07_finetune.pth`
- Stage 2 sliding-window overlap: 0.50
- Lesion probability threshold: 0.80
- Pancreas gate dilation margin: 15 mm
- Minimum reported connected-component size: 10 voxels

Note: The export folder contains `overlap075` in its name, but the script configuration confirms that the actual Stage 2 overlap used was 0.50.

## 3. Aggregate evaluation metrics

| Metric | Baseline | Gated Stage 3 |
|---|---:|---:|
| Micro Dice | 0.09496 | 0.26142 |
| Precision | 0.05390 | 0.32217 |
| Recall | 0.39880 | 0.21994 |

The pancreas gate substantially reduces false-positive predictions, but also reduces recall. The gated results therefore represent a precision-recall trade-off.

## 4. Case-level results

- Total evaluated cases: 56
- Mean case Dice: 0.2037
- Median case Dice: 0.0623
- Cases with non-zero Dice: 33
- Cases with zero Dice: 23
- Cases with Dice >= 0.50: 9

### Top five cases

| Case | Dice | Precision | Recall |
|---|---:|---:|---:|
| Task07_Pancreas_Task07_Pancreas_155_0000 | 0.8068 | 0.7880 | 0.8266 |
| Task07_Pancreas_Task07_Pancreas_042_0000 | 0.7375 | 0.7916 | 0.6902 |
| Task07_Pancreas_Task07_Pancreas_198_0000 | 0.7072 | 0.5883 | 0.8865 |
| Task07_Pancreas_Task07_Pancreas_187_0000 | 0.7040 | 0.9664 | 0.5537 |
| Task07_Pancreas_Task07_Pancreas_166_0000 | 0.7019 | 0.7761 | 0.6406 |

### Bottom five cases

| Case | Dice | Precision | Recall |
|---|---:|---:|---:|
| Task07_Pancreas_Task07_Pancreas_347_0000 | 0.0000 | 0.0000 | 0.0000 |
| Task07_Pancreas_Task07_Pancreas_258_0000 | 0.0000 | 0.0000 | 0.0000 |
| Task07_Pancreas_Task07_Pancreas_261_0000 | 0.0000 | 0.0000 | 0.0000 |
| Task07_Pancreas_Task07_Pancreas_058_0000 | 0.0000 | 0.0000 | 0.0000 |
| Task07_Pancreas_Task07_Pancreas_294_0000 | 0.0000 | 0.0000 | 0.0000 |

## 5. 3D visualization

A Matplotlib-based 3D voxel visualization was generated for Case 157 using the existing ground-truth and Stage 3 prediction masks.

File: `evaluation/visual_results_stage3/case157_stage3_3d_visualization.png`

Case 157 gated Dice: 0.47766.

The visualization displays the ground-truth pancreas, ground-truth lesion, and predicted lesion.

## 6. Important limitations

- Performance varies considerably between cases; some cases have zero Dice.
- The Stage 1 pancreas localization gate can miss parts of the pancreas and lesions.
- Stage 2 lesion probabilities are weak on some cases, including Case 098.
- Connected components are image-derived candidates, not clinically verified individual lesions.
- These experimental segmentation results are not clinically validated and must not be used as a standalone diagnostic tool.
- The reported patch-based Stage 2 validation metrics and full-volume Stage 3 metrics use different evaluation protocols and should not be directly compared.

## 7. Conclusion

The project demonstrates an end-to-end experimental 3D segmentation workflow combining multi-organ localization, Task07-specific fine-tuning, pancreas-gated lesion prediction, quantitative evaluation, and 3D visualization. The gated evaluation improves micro Dice and precision over the ungated baseline, while recall decreases. Further improvement in localization and lesion detection is required before clinical use.

## 8. Reproducibility

The original checkpoints and evaluation exports are retained. The Stage 3 evaluation and visualization can be reproduced using the scripts and paths documented in this report.
