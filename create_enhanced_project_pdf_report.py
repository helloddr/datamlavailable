from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


PAPER_DIR = Path(r"C:\Users\dhara\Desktop\Research paper\paper2")
RESULT_DIR = PAPER_DIR / "test_patient_result_images"
OUTPUT_PDF = PAPER_DIR / "BraTS_MRI_AI_Project_Report_With_Images.pdf"


def add_text_page(pdf: PdfPages, title: str, body: str) -> None:
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.patch.set_facecolor("white")
    plt.axis("off")
    fig.text(0.08, 0.95, title, fontsize=17, fontweight="bold", va="top")

    y = 0.90
    for paragraph in body.split("\n"):
        if not paragraph.strip():
            y -= 0.018
            continue
        for line in textwrap.wrap(paragraph, width=92):
            fig.text(0.08, y, line, fontsize=10.5, va="top")
            y -= 0.018
            if y < 0.06:
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)
                fig = plt.figure(figsize=(8.27, 11.69))
                fig.patch.set_facecolor("white")
                plt.axis("off")
                y = 0.95
        y -= 0.008

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_image_page(pdf: PdfPages, title: str, image_path: Path) -> None:
    if not image_path.exists():
        return
    image = plt.imread(str(image_path))
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("white")
    plt.axis("off")
    fig.text(0.04, 0.96, title, fontsize=15, fontweight="bold", va="top")
    fig.text(0.04, 0.925, str(image_path.name), fontsize=9.5, va="top")
    ax = fig.add_axes([0.04, 0.05, 0.92, 0.84])
    ax.imshow(image)
    ax.axis("off")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    metrics = json.loads((PAPER_DIR / "brats_final_metrics.json").read_text(encoding="utf-8"))
    test_one = (RESULT_DIR / "test_patient_ground_truth_comparison.txt").read_text(encoding="utf-8")
    test_two = (RESULT_DIR / "second_case_ground_truth_comparison.txt").read_text(encoding="utf-8")

    summary = f"""AI-Assisted Brain Tumor MRI Segmentation Project Report

This report documents the complete MRI tumor segmentation workflow built for the BraTS 2023 GLI dataset. The pipeline validates the dataset, trains a 2D U-Net segmentation model, evaluates validation/test Dice scores, produces prediction overlays, and generates AI-assisted research report outputs.

Final model file:
C:\\Users\\dhara\\Desktop\\Research paper\\paper2\\brats_final_best_model.pt

Dataset:
ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData

Dataset validation:
- Cases: 1251
- Modalities: T1c, T1n, T2-FLAIR, T2w
- Segmentation mask: seg
- Volume size: 240 x 240 x 155

Final training setup:
- Sampled slices: 8757
- Train slices: 7007
- Validation slices: 875
- Test slices: 875
- Split: 80% train, 10% validation, 10% test
- Device: CUDA local GPU
- Model: SmallUNet2D

Final quantitative result:
- Best validation Dice: {metrics["best_dice"]:.6f}
- Final test loss: {metrics["final_test_metrics"]["loss"]:.6f}
- Final test Dice: {metrics["final_test_metrics"]["dice_score"]:.6f}
- Final model score: {metrics["final_test_metrics"]["dice_score"] * 100:.2f} / 100
"""

    training = """Training History:

Epoch 4:
- Train loss: 0.1786
- Validation loss: 0.1558
- Dice: 0.7075

Epoch 5:
- Train loss: 0.1138
- Validation loss: 0.1942
- Dice: 0.6559

Epoch 6:
- Train loss: 0.1045
- Validation loss: 0.0963
- Dice: 0.8142

The final held-out test Dice was 0.7988. The final model was saved as brats_final_best_model.pt.
"""

    case_results = f"""Detailed Case Testing Results:

Test Case 1:
{test_one}

Test Case 1 interpretation:
The model produced a strong segmentation result on this case, with Dice 0.896257.

Test Case 2:
{test_two}

Test Case 2 interpretation:
The model produced a moderate segmentation result on this case, with Dice 0.703990.
"""

    limitations = """Limitations:

This work is for research and manuscript preparation only. It is not a clinical diagnostic tool.

Key limitations:
- Training used sampled slices, not every available slice.
- The split was slice-based, not strict patient-level.
- The model is 2D rather than a full 3D segmentation network.
- Performance varies across cases.
- The report confidence proxy is based on tumor fraction and is not a calibrated clinical probability.

Conclusion:

The project successfully trained and tested an MRI tumor segmentation model. It produced final quantitative metrics, prediction overlays, ground-truth comparison images, and a complete saved model. The final model to use for demonstration is brats_final_best_model.pt.
"""

    image_pages = [
        ("Final Training Result Plot", PAPER_DIR / "brats_final_results.png"),
        ("Test Case 1 Prediction Montage", RESULT_DIR / "prediction_overlay_montage.png"),
        ("Test Case 1 Detailed Comparison 1", RESULT_DIR / "compare_gt_prediction_01_slice_074.png"),
        ("Test Case 1 Detailed Comparison 2", RESULT_DIR / "compare_gt_prediction_02_slice_073.png"),
        ("Test Case 1 Detailed Comparison 3", RESULT_DIR / "compare_gt_prediction_03_slice_077.png"),
        ("Test Case 1 Detailed Comparison 4", RESULT_DIR / "compare_gt_prediction_04_slice_071.png"),
        ("Test Case 2 Prediction Montage", RESULT_DIR / "second_case_prediction_overlay_montage.png"),
        ("Test Case 2 Detailed Comparison 1", RESULT_DIR / "second_case_compare_01_slice_081.png"),
        ("Test Case 2 Detailed Comparison 2", RESULT_DIR / "second_case_compare_02_slice_080.png"),
        ("Test Case 2 Detailed Comparison 3", RESULT_DIR / "second_case_compare_03_slice_076.png"),
        ("Test Case 2 Detailed Comparison 4", RESULT_DIR / "second_case_compare_04_slice_082.png"),
    ]

    with PdfPages(OUTPUT_PDF) as pdf:
        add_text_page(pdf, "Project Summary", summary)
        add_text_page(pdf, "Training Details", training)
        add_text_page(pdf, "Case Testing Results", case_results)
        for title, image_path in image_pages:
            add_image_page(pdf, title, image_path)
        add_text_page(pdf, "Limitations and Conclusion", limitations)

    print(f"saved_pdf={OUTPUT_PDF}")


if __name__ == "__main__":
    main()
