# Bayesian MK-3D-UNet
## 3D Abdominal Organ and Pancreatic Lesion Segmentation

A research-oriented 3D medical image segmentation project using a Bayesian MK-3D-UNet architecture and a cascaded two-model inference pipeline.

## Project Pipeline

### Stage 1 ? Multi-organ segmentation
Predicts multiple abdominal organ classes and provides pancreas localization.

### Stage 2 ? Pancreatic lesion segmentation
Fine-tuned for Medical Segmentation Decathlon Task07 Pancreas to predict lesion probabilities using 3D image patches.

### Stage 3 ? Cascaded inference
Uses the Stage 1 pancreas prediction as a spatial gate, applies Stage 2 lesion prediction, probability thresholding, and connected-component filtering.

## Model Checkpoints

| Model | Checkpoint |
|---|---|
| Stage 1 | `training/checkpoints_exp5_samples4/best.pth` |
| Stage 2 | `datasets/stage2/models/task07_finetune_v2/latest_task07_finetune.pth` |

Trained checkpoints and datasets are not included in this repository. Both checkpoints must be placed at the paths above before inference.

## Environment

Tested with Python 3.12.10, PyTorch 2.13.0+cu130, MONAI 1.6.0, NiBabel 5.4.2, Streamlit 1.67.0, and a CUDA-enabled NVIDIA GPU.

## Installation

Clone the repository:

    git clone https://github.com/Manohar-Talabattula/Bayesian-MK-3D-Unet.git
    cd MK_3D_UNET_PROJECT

Create a virtual environment and activate it in PowerShell:

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1

Install dependencies:

    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt

A compatible CUDA-enabled PyTorch build is recommended. If necessary, install the appropriate build using the official PyTorch installation instructions.

## Running the Application

From the project root:

    streamlit run inference/app.py

Open the local URL displayed by Streamlit.

### Application workflow
1. Upload a compatible CT volume in `.nii` or `.nii.gz` format.
2. Configure the probability threshold and pancreas gate margin.
3. Run the cascaded segmentation pipeline.
4. Review predicted lesion volume, voxel count, localization information, and connected components.
5. Download the predicted mask, probability map, and JSON report.

Both trained checkpoints must be available for inference.

## Stage 3 Evaluation

Evaluated on 56 cases from the Medical Segmentation Decathlon Task07 Pancreas dataset.

| Parameter | Value |
|---|---:|
| Evaluated cases | 56 |
| Probability threshold | 0.80 |
| Pancreas gate margin | 15 mm |
| Sliding-window overlap | 0.50 |
| Minimum component size | 10 voxels |

### Aggregate results

| Metric | Ungated baseline | Cascaded model |
|---|---:|---:|
| Micro Dice | 0.09496 | 0.26142 |
| Precision | 0.05390 | 0.32217 |
| Recall | 0.39880 | 0.21994 |

The cascade improves aggregate Dice and precision compared with the ungated baseline, but recall decreases because the pancreas gate can exclude lesion voxels.

Additional results:
- Mean case Dice: 0.2037
- Median case Dice: 0.0623
- Nonzero Dice cases: 33
- Zero Dice cases: 23
- Cases with Dice >= 0.50: 9

## Visualization and Report

- Case 157 visualization: `evaluation/visual_results_stage3/case157_stage3_3d_visualization.png`
- Evaluation report: `evaluation/FINAL_STAGE3_PROJECT_REPORT.md`

## Project Structure

- `configs/` ? model configurations
- `preprocessing/` ? data preparation
- `training/` ? model training
- `stage3/` ? cascade components
- `datasets/` ? dataset and Stage 2 checkpoint location
- `evaluation/` ? evaluation scripts, results, and visualizations
- `inference/` ? prediction pipeline and Streamlit application

## Limitations

- The pancreas localization gate may miss parts of the pancreas or lesions.
- Lesion detection performance varies substantially between cases.
- Some evaluated cases have zero Dice score.
- Patch-based validation and full-volume evaluation use different protocols.
- Predicted components are not clinically confirmed lesions.
- Performance may vary with CT acquisition, image spacing, orientation, and preprocessing.

## Intended Use

This project is intended for academic research and experimentation. It is not intended for clinical diagnosis, treatment planning, or independent medical decision-making.
