from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from .config import REPORT_DIR
from .data import build_transforms
from .model import load_checkpoint
from .report import build_research_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict one MRI image and generate a research report.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--patient-json", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, default=REPORT_DIR / "mri_report.md")
    return parser.parse_args()


def load_patient_details(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Patient details JSON not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def predict(model_path: Path, image_path: Path) -> tuple[str, float, dict[str, float]]:
    if not image_path.exists():
        raise FileNotFoundError(f"MRI image not found: {image_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names = load_checkpoint(str(model_path), device)
    transform = build_transforms(train=False)

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probabilities_tensor = torch.softmax(logits, dim=1).squeeze(0).cpu()

    probabilities = {class_name: float(probabilities_tensor[index]) for index, class_name in enumerate(class_names)}
    predicted_index = int(torch.argmax(probabilities_tensor).item())
    predicted_label = class_names[predicted_index]
    confidence = probabilities[predicted_label]
    return predicted_label, confidence, probabilities


def main() -> None:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    patient_details = load_patient_details(args.patient_json)
    predicted_label, confidence, probabilities = predict(args.model_path, args.image_path)
    report = build_research_report(
        patient_details=patient_details,
        image_path=str(args.image_path),
        predicted_label=predicted_label,
        confidence=confidence,
        probabilities=probabilities,
    )

    args.output_path.write_text(report, encoding="utf-8")
    print(f"predicted_label={predicted_label}")
    print(f"confidence={confidence:.4f}")
    print(f"saved_report={args.output_path}")


if __name__ == "__main__":
    main()
