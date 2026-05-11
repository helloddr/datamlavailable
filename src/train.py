from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report
from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .config import ARTIFACT_DIR, DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE, DEFAULT_SEED
from .data import load_image_folder, summarize_dataset
from .model import build_model, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an MRI image classifier.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Folder containing class subfolders.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--validation-split", type=float, default=0.2)
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet pretrained ResNet18 weights.")
    parser.add_argument("--output-path", type=Path, default=ARTIFACT_DIR / "mri_classifier.pt")
    parser.add_argument("--metrics-path", type=Path, default=ARTIFACT_DIR / "training_metrics.json")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0

    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, class_names: list[str]) -> dict:
    model.eval()
    all_predictions: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="validate", leave=False):
            images = images.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1).cpu().tolist()
            all_predictions.extend(predictions)
            all_labels.extend(labels.tolist())

    return {
        "accuracy": accuracy_score(all_labels, all_predictions),
        "classification_report": classification_report(
            all_labels,
            all_predictions,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_image_folder(args.data_dir, train=True)
    summary = summarize_dataset(args.data_dir)
    validation_size = max(1, int(len(dataset) * args.validation_split))
    train_size = len(dataset) - validation_size
    if train_size < 1:
        raise ValueError("Dataset is too small for the requested validation split.")

    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, validation_dataset = random_split(dataset, [train_size, validation_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=len(dataset.classes), pretrained=args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device)
        metrics = evaluate(model, validation_loader, device, dataset.classes)
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_accuracy": metrics["accuracy"]})
        print(f"epoch={epoch} train_loss={train_loss:.4f} validation_accuracy={metrics['accuracy']:.4f}")

    final_metrics = evaluate(model, validation_loader, device, dataset.classes)
    output = {
        "dataset": {
            "data_dir": str(summary.data_dir),
            "class_names": summary.class_names,
            "total_images": summary.total_images,
            "train_images": train_size,
            "validation_images": validation_size,
            "pretrained": args.pretrained,
        },
        "history": history,
        "final_metrics": final_metrics,
    }

    save_checkpoint(str(args.output_path), model, dataset.classes)
    args.metrics_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"saved_model={args.output_path}")
    print(f"saved_metrics={args.metrics_path}")


if __name__ == "__main__":
    main()
