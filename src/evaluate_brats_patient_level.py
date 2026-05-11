from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm

from .brats_data import MODALITIES, BraTSCase, find_brats_cases, load_case_modalities, load_case_segmentation
from .segmentation_model import SmallUNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Patient-level BraTS evaluation for a saved 2D segmentation model.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-split", type=float, default=0.10)
    parser.add_argument("--test-split", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-test-cases", type=int, default=None)
    parser.add_argument("--save-overlays", type=int, default=8)
    return parser.parse_args()


def patient_level_split(
    cases: list[BraTSCase],
    validation_split: float,
    test_split: float,
    seed: int,
) -> tuple[list[BraTSCase], list[BraTSCase], list[BraTSCase]]:
    if validation_split < 0 or test_split <= 0 or validation_split + test_split >= 1:
        raise ValueError("Use non-negative validation split, positive test split, and total split below 1.")

    shuffled = list(cases)
    random.Random(seed).shuffle(shuffled)
    validation_size = max(1, int(len(shuffled) * validation_split))
    test_size = max(1, int(len(shuffled) * test_split))
    train_size = len(shuffled) - validation_size - test_size
    if train_size < 1:
        raise ValueError("Dataset is too small for requested patient-level split.")

    train_cases = shuffled[:train_size]
    validation_cases = shuffled[train_size : train_size + validation_size]
    test_cases = shuffled[train_size + validation_size :]
    return train_cases, validation_cases, test_cases


def confusion_counts(prediction: np.ndarray, target: np.ndarray) -> dict[str, int]:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    tp = int(np.logical_and(prediction, target).sum())
    fp = int(np.logical_and(prediction, ~target).sum())
    fn = int(np.logical_and(~prediction, target).sum())
    tn = int(np.logical_and(~prediction, ~target).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def metrics_from_counts(counts: dict[str, int]) -> dict[str, float]:
    tp = counts["tp"]
    fp = counts["fp"]
    fn = counts["fn"]
    tn = counts["tn"]
    eps = 1e-8
    dice = (2 * tp) / max(2 * tp + fp + fn, eps)
    iou = tp / max(tp + fp + fn, eps)
    precision = tp / max(tp + fp, eps)
    recall = tp / max(tp + fn, eps)
    specificity = tn / max(tn + fp, eps)
    accuracy = (tp + tn) / max(tp + fp + fn + tn, eps)
    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "accuracy": float(accuracy),
    }


def load_model(model_path: Path, device: torch.device) -> SmallUNet:
    checkpoint = torch.load(model_path, map_location=device)
    model = SmallUNet(in_channels=len(MODALITIES)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict_case(model: SmallUNet, case: BraTSCase, device: torch.device, threshold: float) -> np.ndarray:
    volume = load_case_modalities(case)
    predictions = []
    with torch.no_grad():
        for z_index in range(volume.shape[3]):
            image = torch.from_numpy(volume[:, :, :, z_index].copy()).float().unsqueeze(0).to(device)
            probability = torch.sigmoid(model(image)).squeeze().cpu().numpy()
            predictions.append(probability >= threshold)
    return np.stack(predictions, axis=2)


def save_overlay(case: BraTSCase, prediction: np.ndarray, target: np.ndarray, output_path: Path) -> None:
    volume = load_case_modalities(case)
    t2f_index = MODALITIES.index("t2f")
    pred_counts = prediction.sum(axis=(0, 1))
    top_slices = np.argsort(pred_counts)[-4:][::-1]

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for column, z_index in enumerate(top_slices):
        mri = volume[t2f_index, :, :, z_index]
        low, high = np.percentile(mri, [1, 99])
        display = np.clip((mri - low) / (high - low), 0, 1) if high > low else np.zeros_like(mri)
        axes[0, column].imshow(display, cmap="gray")
        axes[0, column].imshow(prediction[:, :, z_index], cmap="Reds", alpha=0.45)
        axes[0, column].set_title(f"Prediction z={z_index}")
        axes[1, column].imshow(display, cmap="gray")
        axes[1, column].imshow(target[:, :, z_index], cmap="Greens", alpha=0.35)
        axes[1, column].imshow(prediction[:, :, z_index], cmap="Reds", alpha=0.35)
        axes[1, column].set_title("GT green / Pred red")

    for axis in axes.ravel():
        axis.axis("off")
    fig.suptitle(case.case_id)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def main() -> None:
    args = parse_args()
    cases = find_brats_cases(args.data_dir, require_segmentation=True)
    train_cases, validation_cases, test_cases = patient_level_split(
        cases,
        validation_split=args.validation_split,
        test_split=args.test_split,
        seed=args.seed,
    )
    if args.max_test_cases is not None:
        test_cases = test_cases[: args.max_test_cases]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_path, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = args.output_dir / "overlays"

    rows = []
    aggregate_counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for case_index, case in enumerate(tqdm(test_cases, desc="patient_eval"), start=1):
        prediction = predict_case(model, case, device, args.threshold)
        target = load_case_segmentation(case).astype(bool)
        counts = confusion_counts(prediction, target)
        metrics = metrics_from_counts(counts)
        for key in aggregate_counts:
            aggregate_counts[key] += counts[key]
        row = {
            "case_id": case.case_id,
            **counts,
            **metrics,
            "predicted_tumor_voxels": int(prediction.sum()),
            "ground_truth_tumor_voxels": int(target.sum()),
        }
        rows.append(row)
        if case_index <= args.save_overlays:
            save_overlay(case, prediction, target, overlay_dir / f"{case.case_id}_overlay.png")

    csv_path = args.output_dir / "patient_level_case_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()) if rows else ["case_id"])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "data_dir": str(args.data_dir),
        "model_path": str(args.model_path),
        "device": str(device),
        "threshold": args.threshold,
        "seed": args.seed,
        "split": {
            "train_cases": len(train_cases),
            "validation_cases": len(validation_cases),
            "test_cases_evaluated": len(test_cases),
            "validation_split": args.validation_split,
            "test_split": args.test_split,
        },
        "aggregate_counts": aggregate_counts,
        "aggregate_metrics": metrics_from_counts(aggregate_counts),
        "case_metric_summary": {
            metric: summarize([float(row[metric]) for row in rows])
            for metric in ["dice", "iou", "precision", "recall", "specificity", "accuracy"]
        },
        "case_metrics_csv": str(csv_path),
        "overlay_dir": str(overlay_dir),
    }
    summary_path = args.output_dir / "patient_level_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
