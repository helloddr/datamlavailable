# Lightweight AI-Assisted Glioma MRI Segmentation Using a Low-Compute 2D U-Net Framework on BraTS 2023

## Abstract

This project presents a lightweight brain tumor MRI segmentation pipeline using the BraTS 2023 adult glioma dataset. The workflow validates multi-modal NIfTI MRI cases, trains a compact 2D U-Net model, evaluates segmentation with Dice-style metrics, and generates visual prediction overlays for research review. The final trained model achieved a held-out sampled-slice test Dice of 0.7988. Additional case-level visual analysis showed Dice scores of 0.8963 and 0.7040 on two selected BraTS cases. The contribution is positioned as a low-compute, reproducible baseline and visual error analysis framework, not as a state-of-the-art BraTS solution.

## 1. Introduction

Automated glioma segmentation from multi-modal MRI is important for treatment planning, longitudinal monitoring, and quantitative neuro-oncology research. Strong BraTS challenge systems often use 3D nnU-Net variants, transformer-based models, or ensembles that require significant GPU memory and training time. This work explores a more accessible low-compute alternative based on a compact 2D U-Net.

The goal is not to exceed state-of-the-art BraTS performance. Instead, the project focuses on an efficient and interpretable baseline that can run on limited hardware, produce visual overlays, and support research manuscript preparation.

## 2. Dataset

Dataset: ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.

Validated dataset properties:

| Property | Value |
|---|---:|
| Cases | 1251 |
| MRI volume shape | 240 x 240 x 155 |
| Modalities | T1c, T1n, T2-FLAIR, T2w |
| Segmentation label | Whole tumor binary mask from BraTS seg |

## 3. Methodology

### 3.1 Preprocessing

Each modality was loaded from NIfTI format. Non-zero voxels were normalized per volume using z-score normalization and clipped to a fixed range. Four modalities were stacked as input channels.

### 3.2 Tumor-Aware Slice Sampling

To reduce compute cost, slices were sampled using a configurable stride. The final continuation stage used `slice_stride=24` and included both tumor and non-tumor slices.

Sampled training composition:

| Slice Type | Count | Percentage |
|---|---:|---:|
| Tumor slices | 3354 | 38.3% |
| Non-tumor slices | 5403 | 61.7% |
| Total sampled slices | 8757 | 100% |

### 3.3 Model

The segmentation model is a compact 2D U-Net:

- Input channels: 4
- Output channels: 1 binary tumor mask
- Encoder-decoder structure with skip connections
- Loss: Dice + BCE
- Optimizer: AdamW

### 3.4 Training Stages

Stage 1 trained on tumor-containing slices and saved `brats_tumor_only_best_model.pt`.

Stage 2 resumed from the tumor-only checkpoint and trained with tumor and non-tumor slices. The final model was saved as `brats_final_best_model.pt`.

## 4. Results

Final sampled split:

| Split | Slices |
|---|---:|
| Train | 7007 |
| Validation | 875 |
| Test | 875 |

Final training history:

| Epoch | Train Loss | Validation Loss | Validation Dice |
|---:|---:|---:|---:|
| 4 | 0.1786 | 0.1558 | 0.7075 |
| 5 | 0.1138 | 0.1942 | 0.6559 |
| 6 | 0.1045 | 0.0963 | 0.8142 |

Final held-out sampled-slice test Dice: **0.7988**.

## 5. Visual Error Analysis

The pipeline generated overlay figures showing:

- MRI slice
- Ground-truth tumor mask
- AI-predicted tumor mask
- Prediction overlay
- Error map with false positives and false negatives

Selected case-level results:

| Case | Dice | Predicted Tumor Voxels | Ground Truth Tumor Voxels |
|---|---:|---:|---:|
| BraTS-GLI-00000-000 | 0.8963 | 59,840 | 57,305 |
| BraTS-GLI-00002-000 | 0.7040 | 114,740 | 190,594 |

## 6. Quick Patient-Level Evaluation

A patient-level evaluation utility was added to reduce the weakness of slice-level reporting. A quick post-training evaluation was run on 5 held-out patient cases from a deterministic patient-level split.

Aggregate quick patient-level metrics:

| Metric | Value |
|---|---:|
| Dice | 0.8716 |
| IoU | 0.7725 |
| Precision | 0.8129 |
| Recall | 0.9395 |
| Specificity | 0.9976 |
| Accuracy | 0.9970 |
| Hausdorff-95 | 5.2751 voxels |
| Mean inference time | 2.1934 seconds/case |
| Peak GPU memory | 46.7314 MB |

Case-level Dice summary:

| Statistic | Dice |
|---|---:|
| Mean | 0.8673 |
| Standard deviation | 0.0448 |
| Minimum | 0.8012 |
| Maximum | 0.9388 |

This quick run is encouraging, but a full patient-level test evaluation should be run before formal submission.

## 7. Ablation Study Plan

The current completed run supports a preliminary ablation narrative, but a formal ablation should be run before submission.

| Configuration | Purpose | Current Status |
|---|---|---|
| Tumor-only training | Learn foreground tumor representation | Completed; checkpoint reached Dice 0.8978 during Stage 1 validation |
| Tumor + non-tumor continuation | Improve background discrimination | Completed; final sampled-slice test Dice 0.7988 |
| Visual error analysis | Assess false positive and false negative regions | Completed for selected cases |
| Quick patient-level evaluation | Reduce slice-level validation weakness | Completed on 5 held-out cases |
| Full patient-level evaluation | Stronger publication result | Recommended next |

Recommended formal ablation table:

| Experiment | Expected Question |
|---|---|
| No tumor-aware sampling | Does sampling matter? |
| Tumor-only slices | How much foreground learning helps? |
| Tumor + non-tumor slices | Does background exposure improve robustness? |
| Slice stride 24 vs 12 vs 8 | Does denser slice coverage improve Dice? |
| 2D U-Net vs larger U-Net | Is performance limited by capacity? |

## 8. Comparison Positioning

This method should be compared as a lightweight baseline rather than a SOTA model.

| Method Type | Expected Accuracy | Compute Cost | Notes |
|---|---:|---:|---|
| 3D nnU-Net / ensemble | Higher | Very high | Strong BraTS baseline/SOTA family |
| Transformer 3D models | Higher | Very high | More memory-intensive |
| This 2D U-Net framework | Moderate | Low | Efficient baseline with visual interpretability |

## 9. Clinical Interpretability / Explainability

The project includes prediction overlays, ground-truth overlays, and error maps. These visual outputs support clinical-style review by showing where the model agrees or disagrees with reference masks.

The explainability contribution is:

- MRI slice visualization
- predicted tumor mask visualization
- ground-truth comparison
- red/blue error maps for false positives and false negatives
- montage summaries for quick review

## 10. Limitations

- The current final training used sampled slices rather than all slices.
- The original final training report used a slice-level sampled split.
- A patient-level evaluation tool has been added and tested on a quick 5-case held-out subset.
- The model predicts a binary whole-tumor mask and does not separately report WT/TC/ET regions.
- Hausdorff-95 is not yet included in the final report.
- This is not a clinical diagnostic system.

## 11. Future Work

Recommended next steps:

1. Run full patient-level evaluation across the entire held-out patient test partition.
2. Add IoU, precision, recall, specificity, and Hausdorff-95 reporting.
3. Evaluate WT, TC, and ET subregions separately.
4. Compare against 3D U-Net, nnU-Net, and SwinUNETR literature.
5. Train with smaller slice stride or 3D patches when stronger GPU resources are available.

## 12. Conclusion

The project produced a working, low-compute BraTS MRI tumor segmentation pipeline with a final sampled-slice test Dice of 0.7988 and interpretable visual outputs. The strongest publication direction is a lightweight, reproducible, explainable baseline for resource-constrained MRI segmentation research.
