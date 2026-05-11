from __future__ import annotations

from datetime import datetime, timezone


def format_probability(value: float) -> str:
    return f"{value * 100:.2f}%"


def build_research_report(
    patient_details: dict,
    image_path: str,
    predicted_label: str,
    confidence: float,
    probabilities: dict[str, float],
) -> str:
    patient_id = patient_details.get("patient_id", "unknown_research_case")
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    probability_lines = "\n".join(
        f"- {label}: {format_probability(probability)}"
        for label, probability in sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    )

    return f"""# AI-Assisted MRI Research Report

## Research Case

- Patient/case ID: {patient_id}
- Generated at: {generated_at}
- MRI image: {image_path}
- Modality: {patient_details.get("modality", "Not provided")}
- Sequence: {patient_details.get("sequence", "Not provided")}

## Patient Details

- Age: {patient_details.get("age", "Not provided")}
- Sex: {patient_details.get("sex", "Not provided")}
- Clinical history: {patient_details.get("clinical_history", "Not provided")}
- Notes: {patient_details.get("notes", "Not provided")}

## Model Output

- Predicted class: {predicted_label}
- Model confidence: {format_probability(confidence)}

## Class Probabilities

{probability_lines}

## Research Interpretation Draft

The trained MRI classification model assigned the highest probability to `{predicted_label}` for this de-identified research case. This output should be interpreted as a model-derived research signal, not as a clinical diagnosis. The confidence score reflects the model probability distribution for the classes available during training and may not represent real-world diagnostic certainty.

## Limitations

- This report is generated for research and manuscript preparation only.
- Model performance depends on dataset quality, class balance, scanner variability, preprocessing, and external validation.
- The output must be reviewed by a qualified clinician or research supervisor before being included in formal research material.
- No direct patient identifiers should be stored with this report.
"""
