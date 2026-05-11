# Kaggle Private Dataset Upload Steps

The Kaggle CLI is now installed on this machine. To upload your local BraTS dataset to Kaggle GPU, the remaining requirement is your Kaggle API token.

## 1. Kaggle CLI Status

```powershell
.\.venv311\Scripts\kaggle.exe --version
```

Expected output:

```text
Kaggle CLI 2.1.0
```

## 2. Create Kaggle API Token

1. Open Kaggle in your browser.
2. Go to Account settings.
3. Create API token.
4. Kaggle downloads `kaggle.json`.
5. Move it to:

```text
C:\Users\dhara\.kaggle\kaggle.json
```

The folder already exists.

## 3. Edit Dataset Metadata

Open:

```text
kaggle_dataset_metadata\dataset-metadata.json
```

Replace:

```text
REPLACE_WITH_YOUR_KAGGLE_USERNAME
```

with your Kaggle username.

Also copy that metadata file into:

```text
C:\Users\dhara\Desktop\Research paper\paper2\reasearch paper2\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData\dataset-metadata.json
```

## 4. Upload Private Dataset

Run from the project folder:

```powershell
.\.venv311\Scripts\kaggle.exe datasets create `
  -p "C:\Users\dhara\Desktop\Research paper\paper2\reasearch paper2\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData" `
  --dir-mode zip `
  --private
```

This uploads the dataset privately to your Kaggle account. It may take a long time because the dataset is about 13.4 GB.

## 5. Train In Kaggle Notebook

After upload, create a Kaggle notebook, enable GPU, attach your private dataset, upload `code.py`, then run the commands from:

```text
kaggle_gpu_commands.txt
```
