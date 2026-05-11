from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

WORKSPACE = Path(r"C:\Users\dhara\Documents\git ddr\datamlavailable")
sys.path.insert(0, str(WORKSPACE))

from brats_train_local import BraTSCase, MODALITIES, SmallUNet, load_case_modalities


def main() -> None:
    paper = Path(r"C:\Users\dhara\Desktop\Research paper\paper2")
    case_dir = paper / "test_patient"
    model_path = paper / "brats_final_best_model.pt"
    out_dir = paper / "test_patient_result_images"
    out_dir.mkdir(parents=True, exist_ok=True)

    modalities = {}
    for modality in MODALITIES:
        matches = sorted(case_dir.glob(f"*{modality}.nii.gz"))
        if not matches:
            raise FileNotFoundError(f"Missing {modality} file in {case_dir}")
        modalities[modality] = matches[0]

    case = BraTSCase(case_dir.name, case_dir, modalities, None)
    volume = load_case_modalities(case)

    checkpoint = torch.load(model_path, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallUNet(in_channels=len(MODALITIES)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scores = []
    predictions = []
    with torch.no_grad():
        for z_index in range(volume.shape[3]):
            image = torch.from_numpy(volume[:, :, :, z_index].copy()).float().unsqueeze(0).to(device)
            probability = torch.sigmoid(model(image)).squeeze().cpu().numpy()
            prediction = probability >= 0.5
            scores.append(int(prediction.sum()))
            predictions.append(prediction)

    best_slices = np.argsort(scores)[-8:][::-1]
    t2f_index = MODALITIES.index("t2f")

    for rank, z_index in enumerate(best_slices, start=1):
        mri = volume[t2f_index, :, :, z_index]
        low, high = np.percentile(mri, [1, 99])
        display = np.clip((mri - low) / (high - low), 0, 1) if high > low else np.zeros_like(mri)
        prediction = predictions[z_index]

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(display, cmap="gray")
        axes[0].set_title(f"MRI T2-FLAIR slice {z_index}")
        axes[1].imshow(prediction, cmap="gray")
        axes[1].set_title("Predicted tumor mask")
        axes[2].imshow(display, cmap="gray")
        axes[2].imshow(prediction, cmap="Reds", alpha=0.45)
        axes[2].set_title("Prediction overlay")
        for axis in axes:
            axis.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"test_patient_prediction_{rank:02d}_slice_{z_index:03d}.png", dpi=160)
        plt.close(fig)

    (out_dir / "test_patient_prediction_summary.txt").write_text(
        "\n".join(
            [
                f"model={model_path}",
                f"case_dir={case_dir}",
                f"device={device}",
                f"total_predicted_tumor_voxels={sum(scores)}",
                f"saved_images={len(best_slices)}",
                "top_slices=" + ",".join(str(int(z_index)) for z_index in best_slices),
            ]
        ),
        encoding="utf-8",
    )

    print(f"saved_images_dir={out_dir}")
    print(f"total_predicted_tumor_voxels={sum(scores)}")
    print("top_slices=" + ",".join(str(int(z_index)) for z_index in best_slices))


if __name__ == "__main__":
    main()
