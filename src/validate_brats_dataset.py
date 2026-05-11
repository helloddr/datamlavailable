from __future__ import annotations

import argparse
import json
from pathlib import Path

from .brats_data import MODALITIES, find_brats_cases, load_nifti


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate local BraTS NIfTI dataset structure.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, default=Path("artifacts/brats_dataset_validation.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = find_brats_cases(args.data_dir, require_segmentation=True)
    first_case = cases[0]
    shapes = {modality: list(load_nifti(path).shape) for modality, path in first_case.modalities.items()}
    shapes["seg"] = list(load_nifti(first_case.segmentation).shape) if first_case.segmentation else []

    result = {
        "data_dir": str(args.data_dir),
        "case_count": len(cases),
        "required_modalities": list(MODALITIES),
        "first_case_id": first_case.case_id,
        "first_case_shapes": shapes,
        "passed": len(cases) > 0,
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
