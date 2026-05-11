# Publication Strategy

## Honest Position

The current work should not be claimed as state of the art. The final sampled-slice Dice is `0.7988`, while stronger BraTS-style systems commonly use 3D nnU-Net variants, transformer-based networks, or ensembles. The publishable angle is therefore not highest accuracy.

The strongest positioning is:

```text
An efficient, interpretable, reproducible, low-compute BraTS 2023 glioma MRI segmentation framework.
```

## Recommended Title

**Lightweight AI-Assisted Glioma MRI Segmentation Using a Computationally Efficient 2D U-Net Framework on BraTS 2023**

Alternative:

**Low-Compute Brain Tumor MRI Segmentation with Visual Error Analysis and Explainable Prediction Overlays**

## Core Contribution Claims

### 1. Tumor-Aware Slice Sampling

The project uses sampled axial slices and can distinguish tumor-focused training from tumor + non-tumor continuation. This should be framed as a computational strategy for low-resource training.

Suggested claim:

> We propose a tumor-aware slice sampling and continuation strategy for low-compute glioma segmentation experiments.

### 2. Low-Compute Lightweight Model

The model is intentionally small and can run locally on limited GPU hardware.

Suggested claim:

> The proposed framework provides an accessible baseline for researchers without high-end 3D segmentation infrastructure.

### 3. Explainable Visual Review

The project generates prediction overlays, ground-truth overlays, and error maps.

Suggested claim:

> The framework includes visual interpretability outputs that support clinical-style review and qualitative error analysis.

### 4. Patient-Level Evaluation Tooling

The repository now includes patient-level evaluation with Dice, IoU, precision, recall, specificity, Hausdorff-95, inference time, and GPU memory.

Suggested claim:

> We provide patient-level evaluation tooling to reduce slice-level leakage risk and support stronger reporting.

## Completed Supporting Results

| Result Type | Value |
|---|---:|
| Final sampled-slice test Dice | 0.7988 |
| Best validation Dice | 0.8142 |
| Quick patient-level Dice, 5 cases | 0.8716 |
| Quick patient-level IoU, 5 cases | 0.7725 |
| Quick patient-level Hausdorff-95 | 5.2751 voxels |
| Mean inference time | 2.1934 sec/case |
| Peak GPU memory | 46.7314 MB |

## Needed Before Formal Submission

Minimum:

1. Run patient-level evaluation over the full held-out test partition.
2. Add a true ablation table.
3. Add literature comparison table.
4. Avoid claiming SOTA.
5. Clearly report limitations.

Stronger:

1. Add Hausdorff-95 for full test partition.
2. Report WT/TC/ET separately if multi-class labels are implemented.
3. Add 3D baseline comparison if compute is available.
4. Train with denser slice stride, such as 12 or 8.

## Target Venues

Potentially realistic:

- IEEE Access
- Springer medical imaging / healthcare AI venues
- Informatics and applied AI journals
- Student research conference/workshop

Not realistic with current results alone:

- MICCAI main conference
- CVPR
- Nature-level medical AI journals
- SOTA BraTS challenge paper

## Safe Paper Wording

Use:

> lightweight
> low-compute
> interpretable
> reproducible baseline
> visual error analysis
> accessible MRI segmentation framework

Avoid:

> state of the art
> highest accuracy
> clinical diagnosis
> expert-level performance
> validated for patient care
