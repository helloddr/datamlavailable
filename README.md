# Lightweight BraTS MRI Tumor Segmentation Pipeline

End-to-end research pipeline for BraTS-style brain tumor MRI segmentation, visual prediction analysis, and AI-assisted research report generation.

This project is for research and manuscript preparation only. It is not a clinical diagnostic system, and generated reports must be reviewed by a qualified clinician/research supervisor before use.

## Current Project Position

This work is best positioned as a **lightweight, low-compute, interpretable BraTS 2023 baseline**, not as a state-of-the-art BraTS challenge method.

Current final local result:

```text
Final sampled-slice test Dice: 0.7988
Best validation Dice: 0.8142
Model type: compact 2D U-Net
Dataset: BraTS 2023 GLI training data
Cases validated: 1251
```

Publication angle:

```text
Efficient, reproducible, visually interpretable MRI tumor segmentation for resource-constrained settings.
```

## Workflow

```text
BraTS NIfTI MRI dataset
  -> data validation
  -> tumor-aware slice sampling
  -> train/validation/test split
  -> 2D U-Net model training
  -> saved model artifact
  -> patient/case evaluation
  -> visual overlays and error maps
  -> AI-assisted structured report
```

## BraTS Dataset Format

Your current local dataset was found here:

```text
C:\Users\dhara\Desktop\Research paper\paper2\reasearch paper2\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData
```

It is BraTS NIfTI format, not PNG/JPG class-folder format. Each case has:

```text
case/
  *-t1c.nii.gz
  *-t1n.nii.gz
  *-t2f.nii.gz
  *-t2w.nii.gz
  *-seg.nii.gz
```

For this dataset, use `brats_pipeline_all_in_one.py` or the modules under `src/`.

## Patient Input Format

Create a JSON file like:

```json
{
  "patient_id": "research_case_001",
  "age": 52,
  "sex": "Female",
  "clinical_history": "Headache and dizziness",
  "modality": "MRI",
  "sequence": "T1 contrast",
  "notes": "Research sample. De-identified."
}
```

Do not store names, phone numbers, addresses, dates of birth, hospital IDs, or other direct identifiers in this repository.

## Setup

```powershell
py -3.11 -m venv .venv311
.\.venv311\Scripts\Activate.ps1
pip install -r requirements.txt
```

Python 3.11 or 3.12 is recommended for this PyTorch workflow.

## Validate Dataset

```powershell
python -m src.validate_dataset --data-dir data/raw/train
```

For your BraTS dataset:

```powershell
python -m src.validate_brats_dataset --data-dir "C:\Users\dhara\Desktop\Research paper\paper2\reasearch paper2\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
```

## Train / Continue Training

```powershell
python -m src.train --data-dir data/raw/train --epochs 5
```

To use ImageNet transfer learning, add `--pretrained`. That may require internet access the first time torchvision downloads the weights.

The trained model will be saved under `artifacts/`.

For a lightweight BraTS tumor segmentation run:

```powershell
python -m src.train_brats_segmentation `
  --data-dir "C:\Users\dhara\Desktop\Research paper\paper2\reasearch paper2\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData" `
  --epochs 2 `
  --batch-size 4 `
  --max-cases 50
```

Remove `--max-cases 50` when you are ready to train on all 1,251 cases.

The project also includes the all-in-one script used for the completed run:

```powershell
python brats_pipeline_all_in_one.py train `
  --data-dir "C:\Users\dhara\Desktop\Research paper\paper2\reasearch paper2\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData" `
  --epochs 6 `
  --batch-size 2 `
  --slice-stride 24 `
  --include-empty-slices `
  --validation-split 0.10 `
  --test-split 0.10
```

## Patient-Level Evaluation

For stronger publication reporting, use the patient-level evaluator without retraining:

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
- overlay images for selected cases

## Generate Report For One MRI

```powershell
python -m src.predict_report `
  --model-path artifacts/mri_classifier.pt `
  --image-path data/incoming/case001.png `
  --patient-json data/incoming/case001.json `
  --output-path reports/case001_report.md
```

For one BraTS patient/case folder:

```powershell
python -m src.predict_brats_report `
  --model-path artifacts/brats_tumor_segmenter.pt `
  --case-dir "C:\Users\dhara\Desktop\Research paper\paper2\reasearch paper2\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData\BraTS-GLI-00000-000" `
  --patient-json data/incoming/example_patient.json `
  --output-path reports/brats_case_report.md
```

## Paper Use

For a research paper, document:

- Dataset source and inclusion/exclusion criteria
- Number of cases and sampled slices
- Train/validation/test split method, ideally patient-level
- Model architecture
- Image preprocessing
- Metrics such as Dice, IoU, precision, recall, specificity, and Hausdorff-95 if available
- Hardware and software versions
- Ethical approval/de-identification process for patient data

See:

- `docs/MANUSCRIPT_DRAFT.md`
- `docs/PROJECT_STATUS.md`
