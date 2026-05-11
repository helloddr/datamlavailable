# Project Status

## Current Final Model

Final model path on local machine:

```text
C:\Users\dhara\Desktop\Research paper\paper2\brats_final_best_model.pt
```

Final sampled-slice result:

- Best validation Dice: `0.8142`
- Final test Dice: `0.7988`
- Final model rating: `79.88 / 100`

## Best Publication Position

This should be framed as:

```text
Lightweight, low-compute, visually interpretable BraTS 2023 glioma segmentation baseline
```

Do not claim state of the art.

## Main Strengths

- End-to-end BraTS NIfTI loading and validation.
- Four-modality MRI input: T1c, T1n, T2-FLAIR, T2w.
- Compact 2D U-Net model.
- Tumor-aware slice sampling.
- GPU-compatible local training.
- Report generation.
- Overlay and error-map visualization.
- PDF report generation.

## Main Weaknesses

- Slice-based split in the completed run.
- 2D model instead of 3D model.
- Sampled slices rather than all slices.
- Binary whole-tumor mask only.
- No Hausdorff-95 yet.

## Added Improvement Tooling

`src/evaluate_brats_patient_level.py` was added to evaluate saved models by patient-level test cases without retraining.

Example:

```powershell
python -m src.evaluate_brats_patient_level `
  --data-dir "C:\Users\dhara\Desktop\Research paper\paper2\reasearch paper2\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData" `
  --model-path "C:\Users\dhara\Desktop\Research paper\paper2\brats_final_best_model.pt" `
  --output-dir "C:\Users\dhara\Desktop\Research paper\paper2\patient_level_eval" `
  --max-test-cases 20
```

Outputs:

- `patient_level_summary.json`
- `patient_level_case_metrics.csv`
- overlay images

## Quick Patient-Level Evaluation Result

Ran on 5 held-out patient test cases without retraining.

| Metric | Value |
|---|---:|
| Aggregate Dice | 0.8716 |
| Aggregate IoU | 0.7725 |
| Precision | 0.8129 |
| Recall | 0.9395 |
| Specificity | 0.9976 |
| Accuracy | 0.9970 |

Case Dice mean: `0.8673`

Local output:

```text
C:\Users\dhara\Desktop\Research paper\paper2\patient_level_eval_quick
```

## Next Best Work

1. Run full patient-level evaluation over the complete held-out patient test partition.
2. Add Hausdorff-95.
3. Run ablation table.
4. Improve manuscript with literature comparison.
5. If compute is available, retrain with smaller slice stride or 3D architecture.
