from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .brats_data import BraTSCase, MODALITIES, find_brats_cases, load_case_modalities
from .report import build_research_report
from .segmentation_model import SmallUNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BraTS tumor segmentation and generate a research report.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--case-dir", type=Path, required=True, help="Folder containing t1c/t1n/t2f/t2w NIfTI files.")
    parser.add_argument("--patient-json", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, default=Path("reports/brats_case_report.md"))
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def load_patient_details(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_single_case(case_dir: Path) -> BraTSCase:
    cases = find_brats_cases(case_dir.parent, require_segmentation=False)
    for case in cases:
        if case.case_dir.resolve() == case_dir.resolve():
            return case
    raise ValueError(f"Could not load BraTS case folder: {case_dir}")


def predict_tumor_fraction(model_path: Path, case_dir: Path, threshold: float) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    model = SmallUNet(in_channels=len(MODALITIES)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    case = load_single_case(case_dir)
    volume = load_case_modalities(case)
    tumor_voxels = 0
    total_voxels = 0

    with torch.no_grad():
        for slice_number in range(volume.shape[3]):
            image = torch.from_numpy(volume[:, :, :, slice_number].copy()).float().unsqueeze(0).to(device)
            logits = model(image)
            prediction = (torch.sigmoid(logits) >= threshold).float()
            tumor_voxels += int(prediction.sum().item())
            total_voxels += int(prediction.numel())

    tumor_fraction = tumor_voxels / max(total_voxels, 1)
    return {
        "case_id": case.case_id,
        "tumor_voxels": tumor_voxels,
        "total_voxels": total_voxels,
        "tumor_fraction": tumor_fraction,
    }


def main() -> None:
    args = parse_args()
    result = predict_tumor_fraction(args.model_path, args.case_dir, args.threshold)
    patient_details = load_patient_details(args.patient_json)
    predicted_label = "tumor_present" if result["tumor_voxels"] > 0 else "no_tumor_detected"
    confidence = min(0.99, max(0.01, result["tumor_fraction"] * 10))
    probabilities = {
        "tumor_present": confidence,
        "no_tumor_detected": 1.0 - confidence,
    }
    report = build_research_report(
        patient_details=patient_details,
        image_path=str(args.case_dir),
        predicted_label=predicted_label,
        confidence=confidence,
        probabilities=probabilities,
    )
    report += (
        "\n## Segmentation Summary\n\n"
        f"- Case ID: {result['case_id']}\n"
        f"- Predicted tumor voxels: {result['tumor_voxels']}\n"
        f"- Total evaluated voxels: {result['total_voxels']}\n"
        f"- Predicted tumor fraction: {result['tumor_fraction']:.6f}\n"
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(report, encoding="utf-8")
    print(f"predicted_label={predicted_label}")
    print(f"tumor_voxels={result['tumor_voxels']}")
    print(f"saved_report={args.output_path}")


if __name__ == "__main__":
    main()
