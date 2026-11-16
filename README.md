# Bayesian MK-3D-UNet
### Bayesian Multi-Kernel 3D U-Net for Anatomical and Pathology Segmentation in CT

<p align="center">
  <b>3D Medical Image Segmentation · Bayesian Deep Learning · Multi-Stage Inference</b>
</p>

<p align="center">
  <a href="https://github.com/Manohar-Talabattula/Bayesian-MK-3D-Unet">GitHub Repository</a>
</p>

---

## Overview

**Bayesian MK-3D-UNet** is a research-oriented 3D medical image segmentation project built around a multi-kernel Bayesian U-Net architecture. The system was developed to learn abdominal anatomy from volumetric CT scans and use anatomical localization to support downstream pathology segmentation.

The project brings together:

- A multi-kernel 3D encoder–decoder network with Bayesian/MC-dropout capability.
- Multi-dataset anatomical learning using BTCV, AMOS22 and TotalSegmentator.
- 3D preprocessing, patch-based training, validation and checkpoint management.
- Downstream pathology segmentation workflows for LiTS, KiTS19 and the Medical Segmentation Decathlon Task07 Pancreas dataset.
- Anatomy-guided inference using spatial gates derived from Stage 1 predictions.
- Full-volume sliding-window inference, quantitative evaluation and NIfTI export.
- A Streamlit application with separate pancreas and liver/kidney inference modes.

The implementation is intended for research and experimentation. It is **not a medical device**, does not provide a diagnosis, and has not been clinically validated.

## Project at a glance

| Component | Implementation |
|---|---|
| Core architecture | BayesianMKUNet3D |
| Architecture design | Multi-kernel 3D U-Net with 3×3, 5×5 and 7×7 convolution branches |
| Encoder/decoder | Four encoder levels, bottleneck and four decoder levels |
| Trainable parameters | 15,111,462 |
| Anatomical classes | 12 foreground organs + background |
| Main framework | PyTorch and MONAI |
| Medical image format | NIfTI (`.nii`, `.nii.gz`) |
| Downstream networks | 3-class 3D pathology segmentation; PlainConvUNet in the downstream pipelines |
| Inference | Full-volume / sliding-window inference |
| Application | Streamlit |
| Development environment | Python 3.12.10, MONAI 1.6.0, Windows |
| GPU used during development | NVIDIA RTX 4000 Ada Generation, approximately 19.99 GB VRAM |

## System architecture

The project is organized around anatomical learning followed by two downstream pathology workflows. The pancreas workflow and the LiTS/KiTS liver-kidney workflow are separate fine-tuning/evaluation paths; their checkpoints and metrics should not be conflated.

```text
                         3D CT VOLUMES
                               |
                     Canonical orientation
                     Resampling / intensity
                         normalization
                               |
                 +-------------+-------------+
                 |                           |
                 v                           v
          STAGE 1: ANATOMY              DOWNSTREAM DATA
          Bayesian MK-3D-UNet            (task-specific)
          12 organs + background               |
                 |                    +---------+----------+
                 |                    |                    |
                 v                    v                    v
          Anatomical masks       LiTS + KiTS19       Task07 Pancreas
                 |                    |                    |
                 v                    v                    v
          Spatial anatomy gate   Liver/kidney        Pancreas-specific
                 |               pathology model     fine-tuning
                 |                    |                    |
                 v                    v                    v
          Task-specific         Anatomy-guided       Stage 3 pancreas
          inference             full-volume          cascade inference
                               segmentation                |
                                    |                       |
                                    +-----------+-----------+
                                                |
                                     Masks / probabilities /
                                      metrics / JSON reports
                                                |
                                         Streamlit UI
```

### Bayesian MK-3D-UNet design

The core model is implemented in `models/bayesian_mk_unet3d.py` as `BayesianMKUNet3D`.

Key design elements:

- 3D convolutions for volumetric CT data.
- Parallel multi-kernel feature extraction using 3×3, 5×5 and 7×7 convolution branches.
- Four encoder levels and a bottleneck.
- Four decoder levels with multi-scale feature reconstruction.
- Main segmentation output and auxiliary deep-supervision outputs.
- Dropout-based stochastic modelling to support Monte Carlo inference and uncertainty-oriented experimentation.
- 15,111,462 trainable parameters in the recorded model configuration.

The Stage 1 model uses one input channel and 13 output classes: background plus 12 anatomical structures.

### Stage 1 anatomical label mapping

| Label | Structure |
|---:|---|
| 0 | Background |
| 1 | Spleen |
| 2 | Right kidney |
| 3 | Left kidney |
| 4 | Gallbladder |
| 5 | Esophagus |
| 6 | Liver |
| 7 | Stomach |
| 8 | Aorta |
| 9 | Inferior vena cava |
| 10 | Pancreas |
| 11 | Right adrenal gland |
| 12 | Left adrenal gland |

The downstream LiTS/KiTS anatomy gate uses Stage 1 classes **2, 3 and 6** (right kidney, left kidney and liver). The Task07 pancreas cascade uses the pancreas anatomical prediction.

---

## Data and preprocessing

The project uses public research datasets. The datasets themselves and trained checkpoint binaries are not included in this repository.

### Stage 1: multi-organ anatomy

Datasets used:

- **BTCV** — abdominal multi-organ CT segmentation.
- **AMOS22** — multi-organ abdominal CT segmentation.
- **TotalSegmentator** — CT images with multi-structure segmentation annotations.

Recorded processed Stage 1 dataset:

| Split | Volumes |
|---|---:|
| Training | 1,242 |
| Validation | 157 |
| **Total** | **1,399** |

The processed training set was checked for empty labels. Seventy empty processed training examples were excluded from the training set; the final processed training split contained 1,242 non-empty labels. The raw source data was not modified by this filtering step.

Recorded Stage 1 preprocessing configuration:

| Operation | Setting |
|---|---|
| Orientation | Canonical NIfTI orientation |
| Target spacing | 1.5 × 1.5 × 3.0 mm |
| CT intensity clipping | [-1000, 1000] |
| Intensity normalization | Min-max normalization to [0, 1] |
| Training patch | 96 × 96 × 64 voxels |
| Sampling | Foreground-centred positive crop sampling |
| Samples per training volume | 2 in the recorded configuration |

The Stage 1 data loader uses MONAI transforms for NIfTI loading, channel handling, patch extraction, spatial padding, augmentation and tensor conversion. The data loader consumes the already-preprocessed volumes rather than repeating the full preprocessing pipeline.

### Downstream data

Two downstream tracks were developed:

1. **LiTS + KiTS19 liver/kidney pathology track** — uses the Stage 1 liver/kidney anatomy prediction to constrain pathology predictions spatially.
2. **Task07 Pancreas track** — adapts the downstream segmentation model to pancreas and lesion classes and evaluates a pancreas-guided cascade.

For the LiTS/KiTS preprocessing investigation, the shared resampling pipeline canonicalizes image and label orientation, resamples images with linear interpolation and labels with nearest-neighbour interpolation, and uses target spacing of **[2.5, 1.5, 1.5] mm**. CT intensity normalization maps the processed image into [0, 1].

The 69-case LiTS/KiTS evaluation subset reported below comprised 6 LiTS Batch 1 cases, 21 LiTS Batch 2 cases and 42 KiTS19 cases. This is the verified evaluation coverage; it should not be interpreted as a claim that every available source volume was used for training.

---

## Training and downstream workflows

### 1. Stage 1 — Multi-organ anatomical segmentation

Stage 1 trains `BayesianMKUNet3D` to segment 12 abdominal structures plus background.

Recorded training configuration:

| Parameter | Value |
|---|---|
| Input channels | 1 |
| Output classes | 13 |
| Patch size | 96 × 96 × 64 |
| Batch size | 1 |
| Samples per volume | 2 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-5 |
| Loss | MONAI DiceCELoss |
| Deep supervision weight | 0.2 |
| Learning-rate schedule | CosineAnnealingLR |
| Precision | Automatic mixed precision when CUDA is available |
| Validation | Full-volume sliding-window validation |

The training pipeline includes checkpoint resume support and separate latest/best checkpoint handling. The Stage 1 checkpoint used by the downstream anatomy-gating workflow is:

```text
training/checkpoints_exp5_samples4/best.pth
```

The recorded Stage 1 experiment used for the downstream pipeline reached a best validation Dice of **0.6144** on 157 validation volumes. This figure belongs to that recorded experiment and should be interpreted with its own validation configuration.

### 2. Downstream track A — LiTS and KiTS19 pathology segmentation

This workflow combines:

- Stage 1 anatomical localization.
- A 3-class pathology segmentation model.
- Spatial mapping of the Stage 1 anatomy mask to the downstream model grid.
- Physical-space dilation of the anatomy gate.
- Probability thresholding and gate-constrained pathology output.

The Stage 1 gate selects classes **2, 3 and 6** (right kidney, left kidney and liver). The evaluated anatomy-gating configuration used a **15 mm** gate margin and a pathology probability threshold of **0.65**.

The downstream Stage 2 model checkpoint used for the recorded anatomy-gate evaluation is:

```text
datasets/stage2/models/stage2_best_balanced.pth
```

The corresponding Stage 1 checkpoint is:

```text
training/checkpoints_exp5_samples4/best.pth
```

#### 69-case anatomy-gating evaluation

The evaluation compared pathology predictions before and after applying the Stage 1 anatomy gate.

| Metric | Baseline | Anatomy-gated |
|---|---:|---:|
| Dice | 0.5143 | **0.5808** |
| IoU | 0.3461 | **0.4092** |
| Precision | 0.3965 | **0.5691** |
| Recall | **0.7315** | 0.5929 |
| Predicted / ground-truth volume ratio | 1.8447 | 1.0417 |

**Observed result:** anatomy gating improved micro Dice by approximately **0.0665**, improved IoU and precision, and brought the aggregate predicted-to-ground-truth volume ratio closer to 1.0. Recall decreased, reflecting the precision–recall trade-off introduced by spatial gating.

Dataset-wise micro Dice:

| Evaluation subset | Cases | Baseline Dice | Gated Dice |
|---|---:|---:|---:|
| LiTS Batch 1 | 6 | 0.4299 | **0.6392** |
| LiTS Batch 2 | 21 | 0.4592 | **0.6437** |
| KiTS19 | 42 | 0.5390 | **0.5581** |
| **Overall** | **69** | **0.5143** | **0.5808** |

The evaluation also generated representative case exports for visual comparison, including CT images, ground-truth masks, baseline predictions, gated predictions, pathology probability maps and expanded anatomy gates.

### 3. Downstream track B — Task07 Pancreas fine-tuning

A separate downstream model was fine-tuned for the Medical Segmentation Decathlon **Task07 Pancreas** dataset.

The 3 output classes are:

| Class | Meaning |
|---:|---|
| 0 | Background |
| 1 | Pancreas |
| 2 | Pathology / lesion |

Recorded implementation details include:

- 3D segmentation with a 3-class output.
- Patch size of 32 × 96 × 96.
- Lesion-aware patch sampling.
- Deterministic lesion-centred validation.
- Median lesion coordinates for validation patch placement.
- Deep supervision.
- Initialization from the Stage 1 anatomical model where configured by the fine-tuning pipeline.

The fine-tuned checkpoint used by the Stage 3 Task07 inference workflow is:

```text
datasets/stage2/models/task07_finetune_v2/latest_task07_finetune.pth
```

The project also contains best-checkpoint variants from the Task07 fine-tuning experiments. Use the exact checkpoint specified by the relevant evaluation script/report when reproducing an experiment.

A patch-level lesion Dice of approximately **0.8228** was recorded in a lesion-focused experiment. Patch-level validation and full-volume evaluation are different protocols and their Dice values should not be compared as if they were the same test.

### 4. Stage 3 — Task07 full-volume pancreas cascade

Stage 3 evaluates the Task07 pancreas model over complete 3D volumes using sliding-window inference and anatomy-guided post-processing.

Main evaluation/export script:

```text
evaluation/run_task07_stage3_export_v1.py
```

Core operations:

1. Load the Stage 1 anatomical checkpoint.
2. Load the Task07 fine-tuned pathology checkpoint.
3. Run full-volume 3D sliding-window inference.
4. Use Gaussian blending to combine overlapping windows.
5. Extract the lesion probability channel (class 2).
6. Apply the pancreas anatomy gate with a 15 mm margin.
7. Threshold the pathology probability map.
8. Compute segmentation metrics and lesion-volume statistics.
9. Export NIfTI masks, per-case JSON reports and aggregate CSV/JSON summaries.

The recorded 56-case evaluation used a pathology threshold of **0.80** and a pancreas gate margin of **15 mm**.

#### 56-case Stage 3 evaluation

| Metric | Baseline | Anatomy-gated |
|---|---:|---:|
| Dice | 0.0950 | **0.2614** |
| IoU | 0.0498 | **0.1504** |
| Precision | 0.0539 | **0.3222** |
| Recall | **0.3988** | 0.2199 |
| Predicted / ground-truth volume ratio | 7.399 | 0.683 |

The cascade increased micro Dice by approximately **0.1665** and substantially improved precision while reducing the aggregate predicted volume. Recall decreased, which is an important characteristic of this gate configuration and should be considered when interpreting the results.

Additional recorded Stage 3 evaluation details:

- 56 Task07 cases evaluated.
- 33 cases had non-zero gated Dice.
- 23 cases had zero gated Dice.
- 9 cases achieved Dice ≥ 0.50.
- True-positive voxels: 20,007.
- False-positive voxels: 42,093.
- False-negative voxels: 70,958.

The inference/export pipeline also calculates predicted lesion volume in millilitres, connected-component statistics and per-case measurements. Component listings use a minimum component-size filter for reporting; this reporting filter does not remove those voxels from the total predicted-mask volume.

#### Example full-volume inference

A separate inference smoke test on a KiTS19 CT volume produced:

| Output | Result |
|---|---:|
| Output mask shape | 188 × 314 × 204 |
| Probability map shape | 188 × 314 × 204 |
| Predicted voxels | 5,349 |
| Predicted volume | 30.088 mL |
| Reported connected components | 6 |

This is an inference demonstration on one volume, not an evaluation metric or a claim of ground-truth accuracy for that individual scan.

---

## Inference application

The project includes a Streamlit application in:

```text
inference/app.py
```

Prediction functions are organized in:

```text
inference/predict.py
```

The application exposes two selectable workflows.

### Pancreas (Task07)

- Loads the Stage 1 and Task07 downstream checkpoints.
- Runs pancreas-guided lesion inference.
- Uses the pancreas workflow's configured threshold and gate margin.
- Produces a predicted lesion mask, probability map and JSON report.

### Liver/Kidney (LiTS + KiTS19)

- Loads the Stage 1 anatomical model and the LiTS/KiTS pathology model.
- Uses Stage 1 classes 2, 3 and 6 for the anatomy gate.
- Runs the pathology model and applies the physical-space anatomy gate.
- Exposes threshold and gate-margin controls.
- Produces a predicted pathology mask, probability map and JSON report.

### Application outputs

The interface displays prediction measurements and provides download controls for:

- Segmentation mask (`.nii.gz`).
- Pathology probability map (`.nii.gz`).
- Structured prediction report (`.json`).

The report includes model/checkpoint details, inference settings, anatomy-gate information, predicted voxel counts and volume measurements, and component-level summaries where available.

The Streamlit dual-mode interface was tested in a browser using a KiTS19 CT volume. The browser output showed the same 5,349 predicted voxels and 30.088 mL predicted volume as the standalone inference test.

### Run locally

From the repository root, activate your virtual environment and launch:

```powershell
# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Launch the Streamlit application
streamlit run inference\app.py
```

Open the local URL printed by Streamlit, select the workflow, upload a supported NIfTI CT volume and run inference.

---

## Repository structure

The repository is organized by project responsibility. The following is a high-level guide; individual scripts and experiment outputs may evolve.

```text
Bayesian-MK-3D-Unet/
│
├── configs/                 # Experiment and model configurations
├── datasets/                # Dataset loaders and preprocessing interfaces
├── models/                  # Bayesian MK-3D-UNet architecture
│   └── bayesian_mk_unet3d.py
├── preprocessing/            # CT/label preprocessing and dataset utilities
├── training/                 # Stage 1 training and checkpoint utilities
├── stage3/                   # Downstream Task07 training/inference components
├── evaluation/               # Evaluation, diagnostics, exports and reports
│   └── run_task07_stage3_export_v1.py
├── inference/
│   ├── app.py                # Streamlit dual-workflow application
│   └── predict.py            # Inference loaders and prediction functions
├── requirements.txt
├── .gitignore
└── README.md
```

Large datasets, NIfTI scans, model checkpoints and generated outputs are intentionally managed outside ordinary source-code version control. The repository's `.gitignore` excludes large or generated artifacts such as virtual environments, medical image volumes and model weights.

---

## Installation

### Requirements

The recorded development environment:

- Windows 10/11
- Python 3.12.10
- PyTorch with CUDA support
- MONAI 1.6.0
- NVIDIA GPU (the development GPU was an RTX 4000 Ada with approximately 20 GB VRAM)

Install the project dependencies:

```powershell
# Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install a PyTorch build compatible with your CUDA environment if it is not already provided by `requirements.txt`. GPU support depends on the installed PyTorch/CUDA combination and the available NVIDIA driver.

### Data and checkpoints

Place the datasets and checkpoints in the paths expected by the project scripts. The checkpoint files are not distributed through this repository.

Important checkpoint locations used by the recorded workflows:

```text
training/checkpoints_exp5_samples4/best.pth
datasets/stage2/models/stage2_best_balanced.pth
datasets/stage2/models/task07_finetune_v2/latest_task07_finetune.pth
```

Check the relevant script's configuration before running an experiment, because different downstream tracks use different checkpoints.

---

## Evaluation outputs and reproducibility

The evaluation scripts produce structured outputs for quantitative and qualitative analysis, including:

- Per-case metric summaries.
- Aggregate JSON evaluation reports.
- CSV case summaries.
- NIfTI prediction masks.
- Pathology probability maps.
- Ground-truth and anatomy-gate comparison volumes.
- Representative visualization panels.
- Connected-component and volume measurements.

The project includes evaluation artifacts for both the 69-case LiTS/KiTS anatomy-gating experiment and the 56-case Task07 pancreas cascade experiment.

For reproducible comparisons, keep the following fixed and record them with every run:

- Model architecture and checkpoint path.
- Dataset split and case list.
- Preprocessing configuration and target spacing.
- Sliding-window ROI and overlap.
- Probability threshold.
- Anatomy-gate classes and physical margin.
- Metric aggregation method (micro/voxel-level versus per-case mean).

Do not compare patch-centred validation Dice directly with full-volume Dice. They measure different inference protocols.

---

## Results summary

| Experiment | Evaluation set | Main result |
|---|---:|---|
| Stage 1 anatomy | 157 validation volumes | Best recorded validation Dice: **0.6144** |
| LiTS/KiTS baseline pathology | 69 cases | Dice **0.5143** |
| LiTS/KiTS anatomy-gated pathology | 69 cases | Dice **0.5808**, precision **0.5691** |
| Task07 pancreas baseline | 56 cases | Dice **0.0950** |
| Task07 pancreas anatomy-gated cascade | 56 cases | Dice **0.2614**, precision **0.3222** |
| KiTS19 application smoke test | 1 CT volume | 5,349 predicted voxels; 30.088 mL |

The aggregate results show that anatomical gating improved precision and Dice in the reported experiments, while reducing recall. This trade-off is reported explicitly rather than treating the gate as an unconditional improvement for every case.

## Limitations and research scope

- This project is a research prototype and has not undergone clinical validation.
- Segmentation outputs are model predictions, not radiologist-confirmed findings.
- Performance varies across cases and datasets.
- Anatomy gating can reduce false positives but can also exclude true-positive voxels, reducing recall.
- The 69-case LiTS/KiTS results describe the evaluated subset, not every available dataset case.
- Patch-level and full-volume evaluation results use different protocols.
- Uncertainty-oriented dropout capability does not by itself establish calibrated uncertainty estimates.
- Dataset-specific splits, checkpoints and preprocessing must be checked before reproducing any reported metric.

## Technology stack

- **Language:** Python
- **Deep learning:** PyTorch
- **Medical imaging / transforms:** MONAI, NiBabel
- **Numerical processing:** NumPy
- **Data and metrics:** Pandas, scikit-learn (where used by evaluation utilities)
- **Visualization:** Matplotlib
- **Interactive inference:** Streamlit
- **Version control:** Git and GitHub
- **Input/output:** NIfTI medical imaging volumes

## Acknowledgements

This project builds on publicly available medical imaging datasets and open-source deep-learning and medical-imaging tools. Please consult and cite the original dataset publications and licenses when using the corresponding data.

## Disclaimer

This repository is intended for academic and research use only. It is not intended for clinical diagnosis, treatment planning or other medical decision-making. Predictions must not be used as a substitute for professional medical review.

---

**Repository:** https://github.com/Manohar-Talabattula/Bayesian-MK-3D-Unet
