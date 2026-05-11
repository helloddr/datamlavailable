from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch

WORKSPACE = Path(r"C:\Users\dhara\Documents\git ddr\datamlavailable")
sys.path.insert(0, str(WORKSPACE))

from brats_train_local import BraTSCase, MODALITIES, SmallUNet, load_case_modalities


CASE_ID = "BraTS-GLI-00002-000"
PREFIX = "second_case"


def dice_score(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    denominator = prediction.sum() + target.sum()
    if denominator == 0:
        return 1.0
    return float(2 * np.logical_and(prediction, target).sum() / denominator)


def main() -> None:
    paper = Path(r"C:\Users\dhara\Desktop\Research paper\paper2")
    source_case_dir = (
        paper
        / "reasearch paper2"
        / "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
        / CASE_ID
    )
    model_path = paper / "brats_final_best_model.pt"
    out_dir = paper / "test_patient_result_images"
    out_dir.mkdir(parents=True, exist_ok=True)

    modalities = {}
    for modality in MODALITIES:
        matches = sorted(source_case_dir.glob(f"*{modality}.nii.gz"))
        if not matches:
            raise FileNotFoundError(f"Missing {modality} file in {source_case_dir}")
        modalities[modality] = matches[0]

    seg_matches = sorted(source_case_dir.glob("*seg.nii.gz"))
    if not seg_matches:
        raise FileNotFoundError(f"Missing segmentation file in {source_case_dir}")
    ground_truth = np.asarray(nib.load(str(seg_matches[0])).dataobj) > 0

    case = BraTSCase(CASE_ID, source_case_dir, modalities, None)
    volume = load_case_modalities(case)

    checkpoint = torch.load(model_path, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallUNet(in_channels=len(MODALITIES)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    predictions = []
    pred_voxels_per_slice = []
    with torch.no_grad():
        for z_index in range(volume.shape[3]):
            image = torch.from_numpy(volume[:, :, :, z_index].copy()).float().unsqueeze(0).to(device)
            probability = torch.sigmoid(model(image)).squeeze().cpu().numpy()
            prediction = probability >= 0.5
            predictions.append(prediction)
            pred_voxels_per_slice.append(int(prediction.sum()))

    prediction_volume = np.stack(predictions, axis=2)
    case_dice = dice_score(prediction_volume, ground_truth)
    top_slices = np.argsort(pred_voxels_per_slice)[-16:][::-1]
    t2f_index = MODALITIES.index("t2f")

    for rank, z_index in enumerate(top_slices, start=1):
        mri = volume[t2f_index, :, :, z_index]
        low, high = np.percentile(mri, [1, 99])
        display = np.clip((mri - low) / (high - low), 0, 1) if high > low else np.zeros_like(mri)
        prediction = predictions[z_index]
        target = ground_truth[:, :, z_index]
        false_positive = np.logical_and(prediction, ~target)
        false_negative = np.logical_and(~prediction, target)
        slice_dice = dice_score(prediction, target)

        fig, axes = plt.subplots(1, 5, figsize=(18, 4))
        axes[0].imshow(display, cmap="gray")
        axes[0].set_title(f"{CASE_ID} slice {z_index}")
        axes[1].imshow(target, cmap="Greens")
        axes[1].set_title("Ground truth")
        axes[2].imshow(prediction, cmap="Reds")
        axes[2].set_title("AI prediction")
        axes[3].imshow(display, cmap="gray")
        axes[3].imshow(target, cmap="Greens", alpha=0.35)
        axes[3].imshow(prediction, cmap="Reds", alpha=0.35)
        axes[3].set_title(f"Overlay Dice {slice_dice:.3f}")
        axes[4].imshow(display, cmap="gray")
        axes[4].imshow(false_positive, cmap="Reds", alpha=0.45)
        axes[4].imshow(false_negative, cmap="Blues", alpha=0.45)
        axes[4].set_title("Errors red FP blue FN")
        for axis in axes:
            axis.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"{PREFIX}_compare_{rank:02d}_slice_{z_index:03d}.png", dpi=160)
        plt.close(fig)

    fig, axes = plt.subplots(3, 4, figsize=(14, 10))
    for axis, z_index in zip(axes.ravel(), top_slices[:12]):
        mri = volume[t2f_index, :, :, z_index]
        low, high = np.percentile(mri, [1, 99])
        display = np.clip((mri - low) / (high - low), 0, 1) if high > low else np.zeros_like(mri)
        axis.imshow(display, cmap="gray")
        axis.imshow(predictions[z_index], cmap="Reds", alpha=0.45)
        axis.set_title(f"Slice {z_index}")
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / f"{PREFIX}_prediction_overlay_montage.png", dpi=160)
    plt.close(fig)

    (out_dir / f"{PREFIX}_ground_truth_comparison.txt").write_text(
        "\n".join(
            [
                f"case_id={CASE_ID}",
                f"model={model_path}",
                f"case_dir={source_case_dir}",
                f"device={device}",
                f"case_dice={case_dice:.6f}",
                f"predicted_tumor_voxels={int(prediction_volume.sum())}",
                f"ground_truth_tumor_voxels={int(ground_truth.sum())}",
                f"saved_comparison_images={len(top_slices)}",
                "top_slices=" + ",".join(str(int(z_index)) for z_index in top_slices),
            ]
        ),
        encoding="utf-8",
    )

    print(f"saved_dir={out_dir}")
    print(f"case_id={CASE_ID}")
    print(f"case_dice={case_dice:.6f}")
    print(f"predicted_tumor_voxels={int(prediction_volume.sum())}")
    print(f"ground_truth_tumor_voxels={int(ground_truth.sum())}")


if __name__ == "__main__":
    main()
