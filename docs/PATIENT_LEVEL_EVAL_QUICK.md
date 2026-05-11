# Quick Patient-Level Evaluation

This evaluation was run after training, using the saved final model without retraining.

## Command

```powershell
python -m src.evaluate_brats_patient_level `
  --data-dir "C:\Users\dhara\Desktop\Research paper\paper2\reasearch paper2\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData" `
  --model-path "C:\Users\dhara\Desktop\Research paper\paper2\brats_final_best_model.pt" `
  --output-dir "C:\Users\dhara\Desktop\Research paper\paper2\patient_level_eval_quick" `
  --max-test-cases 5 `
  --save-overlays 5
```

## Split

The evaluator uses deterministic patient-level splitting with seed `42`.

| Split | Cases |
|---|---:|
| Train partition | 1001 |
| Validation partition | 125 |
| Test partition evaluated in this quick run | 5 |

## Aggregate Metrics Across 5 Held-Out Patient Cases

| Metric | Value |
|---|---:|
| Dice | 0.8716 |
| IoU | 0.7725 |
| Precision | 0.8129 |
| Recall | 0.9395 |
| Specificity | 0.9976 |
| Accuracy | 0.9970 |

## Case-Level Dice Summary

| Statistic | Dice |
|---|---:|
| Mean | 0.8673 |
| Std | 0.0448 |
| Min | 0.8012 |
| Max | 0.9388 |

## Notes

This is a quick evaluation on 5 held-out patient cases, not the final full test partition. It is useful for manuscript development and debugging, but a full patient-level evaluation should be run before formal submission.

Outputs were saved locally to:

```text
C:\Users\dhara\Desktop\Research paper\paper2\patient_level_eval_quick
```
