# Free GPU Setup For BraTS Training

Your local machine is CPU-only, so `--amp` will not help locally. The flag works only when PyTorch sees a CUDA GPU.

Check GPU:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
```

## Best Free Option: Kaggle

Kaggle is usually the best fit for this project because your BraTS folder is about 13.4 GB. Upload the dataset once as a private Kaggle Dataset, attach it to a Notebook, enable GPU, and run training there.

### Kaggle Steps

1. Go to Kaggle.
2. Create a new private Dataset.
3. Upload this folder:

```text
C:\Users\dhara\Desktop\Research paper\paper2\reasearch paper2\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData
```

4. Create a new Kaggle Notebook.
5. In notebook settings, enable GPU.
6. Upload `code.py` into the notebook, or paste its content into a cell.
7. Run:

```bash
!pip install nibabel matplotlib tqdm
!nvidia-smi
!python code.py validate --data-dir "/kaggle/input/YOUR_DATASET_NAME/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
!python code.py train \
  --data-dir "/kaggle/input/YOUR_DATASET_NAME/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData" \
  --epochs 25 \
  --batch-size 8 \
  --slice-stride 4 \
  --include-empty-slices \
  --amp \
  --output-path /kaggle/working/brats_full_best_model.pt \
  --metrics-path /kaggle/working/brats_full_metrics.json
```

If GPU memory fails, reduce `--batch-size 8` to `--batch-size 4`.

## Second Free Option: Google Colab

Colab can work if your Google Drive has enough space for the 13.4 GB dataset. Upload the dataset to Drive, mount Drive in Colab, enable GPU, then run:

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
!pip install nibabel matplotlib tqdm
!nvidia-smi
!python "/content/drive/MyDrive/paper2/code.py" validate --data-dir "/content/drive/MyDrive/paper2/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
!python "/content/drive/MyDrive/paper2/code.py" train \
  --data-dir "/content/drive/MyDrive/paper2/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData" \
  --epochs 25 \
  --batch-size 8 \
  --slice-stride 4 \
  --include-empty-slices \
  --amp \
  --output-path "/content/drive/MyDrive/paper2/brats_full_best_model.pt" \
  --metrics-path "/content/drive/MyDrive/paper2/brats_full_metrics.json"
```

## Important Notes

- `--amp` means automatic mixed precision. It speeds up training and reduces GPU memory use on CUDA GPUs.
- `--amp` does nothing useful on your current local CPU-only machine.
- Free GPU sessions can disconnect, so save outputs to `/kaggle/working` or Google Drive.
- The code saves the best model by validation Dice, so even if a later epoch is worse, the best checkpoint remains.
