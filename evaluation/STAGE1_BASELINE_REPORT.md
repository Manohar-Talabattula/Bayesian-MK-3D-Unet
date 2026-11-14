# Stage-1 Bayesian MK-3D-UNet Baseline Report

## 1. Objective

Stage-1 performs 3D multi-organ anatomical segmentation using a Bayesian MK-3D-UNet model trained on harmonized BTCV, AMOS22, and TotalSegmentator-derived CT data.

The Stage-1 model predicts 13 classes:

| ID | Class |
|---:|---|
| 0 | Background |
| 1 | Spleen |
| 2 | Right Kidney |
| 3 | Left Kidney |
| 4 | Gallbladder |
| 5 | Esophagus |
| 6 | Liver |
| 7 | Stomach |
| 8 | Aorta |
| 9 | Inferior Vena Cava |
| 10 | Pancreas |
| 11 | Right Adrenal Gland |
| 12 | Left Adrenal Gland |

## 2. Dataset

### BTCV

Training cases:
- 30 CT volumes
- 30 segmentation labels

### AMOS22

Training:
- 200 CT volumes
- 200 segmentation labels

Validation:
- 100 CT volumes
- 100 segmentation labels

### TotalSegmentator-derived dataset

Training:
- 1082 cases

Validation:
- 57 cases

The Stage-1 validation set therefore contains:

- 100 AMOS22 validation volumes
- 57 TotalSegmentator validation volumes
- Total: 157 validation volumes

Some TotalSegmentator training cases had no voxels belonging to the selected common Stage-1 organs. These empty processed cases were removed from the training set; the raw dataset was not modified.

## 3. Unified Label Strategy

Only anatomical structures common to all three datasets were selected for the initial single-head Stage-1 model.

### BTCV mapping

0 -> 0
1 -> 1
2 -> 2
3 -> 3
4 -> 4
5 -> 5
6 -> 6
7 -> 7
8 -> 8
9 -> 9
11 -> 10
12 -> 11
13 -> 12

BTCV label 10 (portal vein + splenic vein) was excluded.

### AMOS22 mapping

0 -> 0
1 -> 1
2 -> 2
3 -> 3
4 -> 4
5 -> 5
6 -> 6
7 -> 7
8 -> 8
9 -> 9
10 -> 10
11 -> 11
12 -> 12

AMOS22 labels 13, 14 and 15 were excluded.

### TotalSegmentator mapping

0 -> 0
1 -> 1
2 -> 2
3 -> 3
4 -> 4
15 -> 5
5 -> 6
6 -> 7
52 -> 8
63 -> 9
7 -> 10
8 -> 11
9 -> 12

Other TotalSegmentator structures were excluded from Stage-1.

## 4. Preprocessing

Target voxel spacing:

1.5 x 1.5 x 3.0 mm

Intensity clipping:

-1000 to 1000 HU

Normalization:

Min-max normalization to approximately [0, 1]

Training patch size:

96 x 96 x 64

The processed validation data was verified to contain:

- 157 images
- 157 corresponding labels
- target spacing of 1.5 x 1.5 x 3.0 mm
- label IDs within 0-12

## 5. Model

Model:

Bayesian MK-3D-UNet

Input channels:

1

Output classes:

13

Base channels:

16

Dropout:

0.1

Trainable parameters:

15,111,462

The model contains:

- Multi-kernel 3D convolution blocks
- 3x3, 5x5 and 7x7 convolution branches
- Encoder-decoder structure
- Residual connections
- Deep supervision
- Monte-Carlo dropout support

## 6. Training Configuration

Experiment:

checkpoints_exp5_samples4

Training patch:

96 x 96 x 64

Batch size:

1

Foreground crop samples:

4

Optimizer:

AdamW

Learning rate:

1e-4

Weight decay:

1e-5

Loss:

DiceCELoss

Deep supervision weight:

0.2

Scheduler:

CosineAnnealingLR

Training was continued through 100 epochs.

The best checkpoint occurred at:

Epoch 71

Frozen baseline checkpoint:

training/checkpoints_exp5_samples4/stage1_baseline_epoch71.pth

This checkpoint must not be overwritten.

## 7. Quantitative Results

Best validation checkpoint:

Epoch 71

Best validation Dice:

0.6144059392724447

Mean foreground Dice from full-volume evaluation:

0.6357

Mean foreground IoU:

0.5166

Validation volumes evaluated:

157 / 157

## 8. Organ-wise Results

| Organ | Dice | IoU |
|---|---:|---:|
| Spleen | 0.7141 | 0.6111 |
| Right Kidney | 0.7631 | 0.6660 |
| Left Kidney | 0.7396 | 0.6363 |
| Gallbladder | 0.3521 | 0.2548 |
| Esophagus | 0.6083 | 0.4637 |
| Liver | 0.8007 | 0.7145 |
| Stomach | 0.6223 | 0.4951 |
| Aorta | 0.7555 | 0.6351 |
| Inferior Vena Cava | 0.7303 | 0.6000 |
| Pancreas | 0.6072 | 0.4573 |
| Right Adrenal Gland | 0.4948 | 0.3532 |
| Left Adrenal Gland | 0.4407 | 0.3116 |

## 9. Qualitative Evaluation

Five representative AMOS22 validation cases were visualized:

- amos22_amos_0008
- amos22_amos_0013
- amos22_amos_0018
- amos22_amos_0022
- amos22_amos_0029

Visualizations include:

1. CT image
2. Ground truth segmentation
3. Model prediction
4. Prediction overlay

The visual results show that the model has learned meaningful anatomical localization.

Large structures such as the liver and kidneys are generally better represented than smaller structures such as the gallbladder and adrenal glands.

Some predictions contain small disconnected regions and boundary differences relative to the ground truth.

These qualitative observations are consistent with the organ-wise quantitative metrics.

## 10. Full-Volume Inference

Full-volume inference was performed on the complete Stage-1 validation set using sliding-window inference.

Inference configuration:

- ROI size: 96 x 96 x 64
- Sliding-window overlap: 0.25
- Sliding-window batch size: 1
- Gaussian blending

Results:

- Total validation volumes: 157
- Successfully processed: 157
- Failed: 0

Therefore, full-volume inference was successfully completed for 157/157 validation volumes.

The generated segmentation predictions were saved under:

evaluation/inference_results/

## 11. Additional Quantitative Metrics

Additional voxel-level metrics were calculated across the complete validation set.

| Organ | Precision | Recall | Specificity | F1 |
|---|---:|---:|---:|---:|
| Spleen | 0.5674 | 0.5667 | 0.9987 | 0.5670 |
| Right Kidney | 0.4155 | 0.7888 | 0.9981 | 0.5443 |
| Left Kidney | 0.4018 | 0.5963 | 0.9984 | 0.4801 |
| Gallbladder | 0.4008 | 0.1165 | 0.9999 | 0.1806 |
| Esophagus | 0.4441 | 0.4650 | 0.9998 | 0.4543 |
| Liver | 0.6295 | 0.9171 | 0.9899 | 0.7465 |
| Stomach | 0.2671 | 0.6274 | 0.9919 | 0.3747 |
| Aorta | 0.2985 | 0.7029 | 0.9963 | 0.4191 |
| Inferior Vena Cava | 0.3865 | 0.7157 | 0.9990 | 0.5020 |
| Pancreas | 0.2590 | 0.5683 | 0.9986 | 0.3558 |
| Right Adrenal Gland | 0.4006 | 0.1653 | 1.0000 | 0.2340 |
| Left Adrenal Gland | 0.3233 | 0.1853 | 1.0000 | 0.2356 |

Mean aggregated foreground metrics:

- Precision: 0.3995
- Recall: 0.5346
- Specificity: 0.9976
- F1: 0.4245

The aggregated F1 is based on global TP/FP/FN aggregation and therefore should not be directly compared with the mean per-case foreground Dice of 0.6357.

The high specificity is expected because background voxels greatly outnumber foreground voxels in medical image segmentation.

## 12. Bayesian MC-Dropout Uncertainty Analysis

Monte-Carlo Dropout inference was performed to estimate predictive uncertainty.

The complete validation set was processed:

- Total cases: 157
- Successful cases: 157
- Failed cases: 0

Normalized uncertainty statistics:

- Mean case uncertainty: 0.031010
- Median case uncertainty: 0.025752
- Mean 95th percentile: 0.190672
- Mean 99th percentile: 0.413672

Average percentage of voxels above normalized uncertainty thresholds:

| Normalized Uncertainty Threshold | Percentage of Voxels |
|---:|---:|
| >= 0.10 | 9.790114% |
| >= 0.25 | 4.261353% |
| >= 0.50 | 0.838475% |
| >= 0.75 | 0.069930% |
| >= 0.90 | 0.004963% |

The uncertainty maps were normalized independently for each validation volume.

Therefore, a maximum normalized uncertainty of 1.0 is expected for each successfully analyzed case.

The uncertainty analysis is an additional analysis and does not modify the frozen Stage-1 checkpoint.

Uncertainty outputs were saved under:

evaluation/inference_results/

## 13. Model Export

The frozen epoch-71 Stage-1 model was exported as a clean inference artifact.

Source checkpoint:

training/checkpoints_exp5_samples4/stage1_baseline_epoch71.pth

Exported model:

evaluation/model_export/stage1_bayesian_mk_unet3d_epoch71.pth

The exported artifact contains:

- model_state
- model_config
- training_info
- stage1_info

Export verification confirmed:

- Epoch: 71
- Best validation Dice: 0.6144059392724447
- Total parameters: 15,111,462
- Trainable parameters: 15,111,462

The exported model was successfully written and verified.

## 14. Stage-1 Completion Status

The complete Stage-1 workflow has been executed:

Dataset preparation
-> Label harmonization
-> Preprocessing
-> DataLoader verification
-> Bayesian MK-3D-UNet implementation
-> Training
-> Validation
-> Quantitative evaluation
-> Organ-wise evaluation
-> Visual evaluation
-> Full-volume inference
-> MC-Dropout uncertainty analysis
-> Additional metrics
-> Model export

The epoch-71 checkpoint remains the frozen Stage-1 baseline.

Future experiments must use separate checkpoint directories and must not overwrite:

training/checkpoints_exp5_samples4/stage1_baseline_epoch71.pth