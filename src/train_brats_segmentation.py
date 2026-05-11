from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .brats_data import BraTSSliceDataset, MODALITIES
from .config import ARTIFACT_DIR, DEFAULT_SEED
from .segmentation_model import SmallUNet, dice_score_from_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 2D BraTS tumor segmentation model.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--slice-stride", type=int, default=8)
    parser.add_argument("--include-empty-slices", action="store_true")
    parser.add_argument("--output-path", type=Path, default=ARTIFACT_DIR / "brats_tumor_segmenter.pt")
    parser.add_argument("--metrics-path", type=Path, default=ARTIFACT_DIR / "brats_training_metrics.json")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_epoch(model: nn.Module, loader: DataLoader, optimizer, criterion, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    for images, masks in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader, criterion, device: torch.device) -> dict:
    model.eval()
    total_loss = 0.0
    dice_scores = []
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="validate", leave=False):
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            total_loss += criterion(logits, masks).item() * images.size(0)
            dice_scores.append(dice_score_from_logits(logits, masks))
    return {
        "loss": total_loss / len(loader.dataset),
        "dice_score": sum(dice_scores) / max(len(dice_scores), 1),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    dataset = BraTSSliceDataset(
        args.data_dir,
        max_cases=args.max_cases,
        slice_stride=args.slice_stride,
        include_empty_slices=args.include_empty_slices,
    )
    validation_size = max(1, int(len(dataset) * args.validation_split))
    train_size = len(dataset) - validation_size
    if train_size < 1:
        raise ValueError("Dataset is too small for the requested validation split.")

    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size], generator=generator)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SmallUNet(in_channels=len(MODALITIES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        validation_metrics = evaluate(model, validation_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, **validation_metrics})
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"validation_loss={validation_metrics['loss']:.4f} dice={validation_metrics['dice_score']:.4f}"
        )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "modalities": list(MODALITIES),
            "model_type": "SmallUNet2D",
        },
        args.output_path,
    )
    args.metrics_path.write_text(
        json.dumps(
            {
                "data_dir": str(args.data_dir),
                "cases_used": len(dataset.cases),
                "slices_used": len(dataset),
                "train_slices": train_size,
                "validation_slices": validation_size,
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved_model={args.output_path}")
    print(f"saved_metrics={args.metrics_path}")


if __name__ == "__main__":
    main()
