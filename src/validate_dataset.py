from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from .data import summarize_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate MRI image dataset structure.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Folder containing class subfolders.")
    parser.add_argument("--output-path", type=Path, default=Path("artifacts/dataset_validation.json"))
    return parser.parse_args()


def validate_images(data_dir: Path) -> dict:
    summary = summarize_dataset(data_dir)
    class_counts = {class_name: 0 for class_name in summary.class_names}
    unreadable_files: list[str] = []

    for class_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        for image_path in sorted(path for path in class_dir.rglob("*") if path.is_file()):
            try:
                with Image.open(image_path) as image:
                    image.verify()
                class_counts[class_dir.name] = class_counts.get(class_dir.name, 0) + 1
            except Exception:
                unreadable_files.append(str(image_path))

    return {
        "data_dir": str(data_dir),
        "class_names": summary.class_names,
        "total_images": summary.total_images,
        "class_counts": class_counts,
        "unreadable_files": unreadable_files,
        "passed": len(unreadable_files) == 0 and summary.total_images > 0 and len(summary.class_names) >= 2,
    }


def main() -> None:
    args = parse_args()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    result = validate_images(args.data_dir)
    args.output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
