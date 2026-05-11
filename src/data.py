from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from torchvision import datasets, transforms

from .config import IMAGE_SIZE


@dataclass(frozen=True)
class DatasetSummary:
    data_dir: Path
    class_names: list[str]
    total_images: int


def build_transforms(train: bool = True) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=8),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def load_image_folder(data_dir: Path, train: bool = True) -> datasets.ImageFolder:
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {data_dir}")

    dataset = datasets.ImageFolder(root=str(data_dir), transform=build_transforms(train=train))
    if not dataset.classes:
        raise ValueError(f"No class folders found under: {data_dir}")

    return dataset


def summarize_dataset(data_dir: Path) -> DatasetSummary:
    dataset = load_image_folder(data_dir, train=False)
    return DatasetSummary(
        data_dir=Path(data_dir),
        class_names=list(dataset.classes),
        total_images=len(dataset),
    )
