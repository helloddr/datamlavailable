from __future__ import annotations

import json
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


PAPER_DIR = Path(r"C:\Users\dhara\Desktop\Research paper\paper2")
OUTPUT_PDF = PAPER_DIR / "BraTS_MRI_AI_Project_Report.pdf"


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
        wrapped = textwrap.wrap(paragraph, width=92)
        for line in wrapped:
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
    fig.text(0.04, 0.96, title, fontsize=16, fontweight="bold", va="top")
    ax = fig.add_axes([0.04, 0.05, 0.92, 0.84])
    ax.imshow(image)
    ax.axis("off")
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    metrics_path = PAPER_DIR / "brats_final_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    test_one = (PAPER_DIR / "test_patient_result_images" / "test_patient_ground_truth_comparison.txt").read_text(
        encoding="utf-8"
    )
    test_two = (PAPER_DIR / "test_patient_result_images" / "second_case_ground_truth_comparison.txt").read_text(
        encoding="utf-8"
    )

    intro = """Project Title: AI-Assisted Brain Tumor MRI Segmentation and Report Generation Using BraTS 2023 GLI Data

Project Goal: This work built an end-to-end data engineering and deep learning pipeline to load BraTS MRI data, train a tumor segmentation model, evaluate it, generate prediction images, and create an AI-assisted research report.

Dataset: ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData
Dataset location: C:\\Users\\dhara\\Desktop\\Research paper\\paper2\\reasearch paper2\\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData

Dataset validation:
- Total cases: 1251
- MRI volume shape: 240 x 240 x 155
- Modalities used: T1c, T1n, T2-FLAIR, T2w
- Segmentation mask: seg
- Validation status: passed

Main pipeline file: C:\\Users\\dhara\\Desktop\\Research paper\\paper2\\code.py
Final model file: C:\\Users\\dhara\\Desktop\\Research paper\\paper2\\brats_final_best_model.pt
"""

    method = """Method Summary:

The model is a 2D Small U-Net segmentation network. It receives four MRI modalities as input channels and predicts a binary tumor mask. The model was trained using a Dice + BCE loss with AdamW optimization.

Training was performed in two stages. Stage 1 trained on tumor-containing slices and produced brats_tumor_only_best_model.pt. Stage 2 resumed from that checkpoint and continued learning with tumor and non-tumor slices. The final saved model is brats_final_best_model.pt.

Final Stage 2 setup:
- Cases used: 1251
- Sampled slices used: 8757
- Train slices: 7007
- Validation slices: 875
- Test slices: 875
- Split ratio: 80% train, 10% validation, 10% test
- Device: CUDA local GPU
- AMP: false
- Slice stride: 24

Sampled slice composition:
- Tumor slices used: 3354
- Non-tumor slices used: 5403
- Tumor ratio: 38.3%
- Non-tumor ratio: 61.7%
"""

    results = f"""Final Training and Test Results:

Best validation Dice: {metrics["best_dice"]:.6f}
Final test loss: {metrics["final_test_metrics"]["loss"]:.6f}
Final test Dice: {metrics["final_test_metrics"]["dice_score"]:.6f}
Final model rating as percentage: {metrics["final_test_metrics"]["dice_score"] * 100:.2f} / 100

Epoch history:
- Epoch 4: train loss 0.1786, validation loss 0.1558, Dice 0.7075
- Epoch 5: train loss 0.1138, validation loss 0.1942, Dice 0.6559
- Epoch 6: train loss 0.1045, validation loss 0.0963, Dice 0.8142

Saved outputs:
- brats_final_best_model.pt
- brats_final_metrics.json
- brats_final_results.png
- brats_safe_tumor_plus_nontumor_result_images
- test_patient_result_images
"""

    test_results = f"""External-Style Test Results:

Test Case 1:
{test_one}

Interpretation: This case produced a strong segmentation result with Dice 0.896257.

Test Case 2:
{test_two}

Interpretation: This case produced a moderate segmentation result with Dice 0.703990.
"""

    limitations = """Limitations and Research Use:

This project is for research and manuscript preparation only. It is not a clinical diagnostic system. The model should not be used for real patient diagnosis without expert clinical validation.

Important limitations:
- Training used sampled slices with slice_stride=24, not every MRI slice.
- The split is slice-based, not a strict patient-level split.
- The model is a 2D U-Net, not a full 3D medical segmentation model.
- Performance varies across cases.
- The report confidence proxy is based on predicted tumor fraction, not a calibrated clinical probability.

Conclusion:

The project successfully produced a complete brain tumor MRI segmentation workflow. The final model reached a final test Dice score of 0.798834. It generated visual MRI prediction overlays and AI-assisted research reports. Selected case-level evaluation showed strong performance on one case (Dice 0.896257) and moderate performance on another case (Dice 0.703990). The final model to use for application or research demonstration is brats_final_best_model.pt.
"""

    with PdfPages(OUTPUT_PDF) as pdf:
        add_text_page(pdf, "BraTS MRI AI Project Report", intro)
        add_text_page(pdf, "Methodology", method)
        add_text_page(pdf, "Final Results", results)
        add_text_page(pdf, "Case Testing Results", test_results)
        add_image_page(pdf, "Training Results Plot", PAPER_DIR / "brats_final_results.png")
        add_image_page(
            pdf,
            "Test Patient Prediction Montage",
            PAPER_DIR / "test_patient_result_images" / "prediction_overlay_montage.png",
        )
        add_image_page(
            pdf,
            "Second Case Prediction Montage",
            PAPER_DIR / "test_patient_result_images" / "second_case_prediction_overlay_montage.png",
        )
        add_text_page(pdf, "Limitations and Conclusion", limitations)

    print(f"saved_pdf={OUTPUT_PDF}")


if __name__ == "__main__":
    main()
