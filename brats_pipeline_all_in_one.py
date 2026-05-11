from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm


MODALITIES = ("t1c", "t1n", "t2f", "t2w")
DEFAULT_DATA_DIR = Path(
    r"C:\Users\dhara\Desktop\Research paper\paper2\reasearch paper2\ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
)


@dataclass(frozen=True)
class BraTSCase:
    case_id: str
    case_dir: Path
    modalities: dict[str, Path]
    segmentation: Path | None


def find_brats_cases(data_dir: Path, require_segmentation: bool = True) -> list[BraTSCase]:
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"BraTS dataset folder not found: {data_dir}")

    cases: list[BraTSCase] = []
    for case_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        modalities = {}
        for modality in MODALITIES:
            matches = sorted(case_dir.glob(f"*{modality}.nii.gz"))
            if matches:
                modalities[modality] = matches[0]

        seg_matches = sorted(case_dir.glob("*seg.nii.gz"))
        segmentation = seg_matches[0] if seg_matches else None
        has_all_modalities = all(modality in modalities for modality in MODALITIES)
        has_required_seg = segmentation is not None or not require_segmentation

        if has_all_modalities and has_required_seg:
            cases.append(BraTSCase(case_dir.name, case_dir, modalities, segmentation))

    if not cases:
        raise ValueError(f"No valid BraTS cases found under: {data_dir}")
    return cases


def load_nifti(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).get_fdata(dtype=np.float32))


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    nonzero = volume[volume > 0]
    if nonzero.size == 0:
        return np.zeros_like(volume, dtype=np.float32)
    mean = float(nonzero.mean())
    std = float(nonzero.std()) or 1.0
    return np.clip((volume - mean) / std, -5, 5).astype(np.float32)


def load_case_modalities(case: BraTSCase) -> np.ndarray:
    volumes = [normalize_volume(load_nifti(case.modalities[modality])) for modality in MODALITIES]
    return np.stack(volumes, axis=0)


def load_case_segmentation(case: BraTSCase) -> np.ndarray:
    if case.segmentation is None:
        raise ValueError(f"Case has no segmentation file: {case.case_id}")
    return (load_nifti(case.segmentation) > 0).astype(np.float32)


class BraTSSliceDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        max_cases: int | None = None,
        slice_stride: int = 8,
        include_empty_slices: bool = False,
    ) -> None:
        self.cases = find_brats_cases(data_dir, require_segmentation=True)
        if max_cases is not None:
            self.cases = self.cases[:max_cases]

        print(
            f"indexing_cases={len(self.cases)} slice_stride={slice_stride} "
            f"include_empty_slices={include_empty_slices}",
            flush=True,
        )
        self.slice_index: list[tuple[int, int]] = []
        for case_index, case in enumerate(self.cases):
            segmentation = load_case_segmentation(case)
            for slice_number in range(0, segmentation.shape[2], slice_stride):
                has_tumor = bool(segmentation[:, :, slice_number].sum() > 0)
                if include_empty_slices or has_tumor:
                    self.slice_index.append((case_index, slice_number))
            if (case_index + 1) % 25 == 0 or (case_index + 1) == len(self.cases):
                print(
                    f"indexed_cases={case_index + 1}/{len(self.cases)} "
                    f"usable_slices={len(self.slice_index)}",
                    flush=True,
                )

        if not self.slice_index:
            raise ValueError("No usable slices found. Try --include-empty-slices or smaller --slice-stride.")

    def __len__(self) -> int:
        return len(self.slice_index)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        case_index, slice_number = self.slice_index[index]
        case = self.cases[case_index]
        image = load_case_modalities(case)[:, :, :, slice_number]
        mask = load_case_segmentation(case)[:, :, slice_number]
        return torch.from_numpy(image.copy()).float(), torch.from_numpy(mask.copy()).float().unsqueeze(0)


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SmallUNet(nn.Module):
    def __init__(self, in_channels: int = 4, out_channels: int = 1) -> None:
        super().__init__()
        self.down1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.bridge = DoubleConv(64, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(64, 32)
        self.out = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        bridge = self.bridge(self.pool2(d2))
        u2 = self.up2(bridge)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.conv2(u2)
        u1 = self.up1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.conv1(u1)
        return self.out(u1)


def dice_score_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    predictions = (torch.sigmoid(logits) >= threshold).float()
    intersection = (predictions * targets).sum()
    denominator = predictions.sum() + targets.sum()
    if denominator.item() == 0:
        return 1.0
    return float((2 * intersection / denominator).item())


class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, smooth: float = 1.0) -> None:
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        probabilities = torch.sigmoid(logits)
        intersection = (probabilities * targets).sum(dim=(1, 2, 3))
        denominator = probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice_loss = 1 - ((2 * intersection + self.smooth) / (denominator + self.smooth)).mean()
        return self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss


def validate_dataset(args: argparse.Namespace) -> None:
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


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    criterion,
    device: torch.device,
    scaler,
    use_amp: bool,
    gradient_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    for images, masks in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()

        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, masks)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader, criterion, device: torch.device, desc: str = "validate") -> dict:
    model.eval()
    total_loss = 0.0
    dice_scores = []
    with torch.no_grad():
        for images, masks in tqdm(loader, desc=desc, leave=False):
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            total_loss += criterion(logits, masks).item() * images.size(0)
            dice_scores.append(dice_score_from_logits(logits, masks))
    return {"loss": total_loss / len(loader.dataset), "dice_score": sum(dice_scores) / max(len(dice_scores), 1)}


def save_training_plot(history: list[dict], output_path: Path, test_metrics: dict | None = None) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib_not_installed=true skipped_plot=true")
        return

    if not history:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    validation_loss = [row["loss"] for row in history]
    validation_dice = [row["dice_score"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_loss, label="Train loss")
    axes[0].plot(epochs, validation_loss, label="Validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, validation_dice, label="Validation Dice", color="green")
    if test_metrics is not None:
        axes[1].axhline(test_metrics["dice_score"], label="Final test Dice", color="red", linestyle="--")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Dice score")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_test_sample_images(
    model: nn.Module,
    dataset: Dataset,
    device: torch.device,
    output_dir: Path,
    sample_count: int,
    threshold: float = 0.5,
) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib_not_installed=true skipped_sample_images=true")
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
    saved_paths: list[str] = []
    if sample_count <= 0:
        return saved_paths

    total_samples = len(dataset)
    if total_samples == 0:
        return saved_paths

    if sample_count >= total_samples:
        sample_indices = list(range(total_samples))
    else:
        sample_indices = np.linspace(0, total_samples - 1, sample_count, dtype=int).tolist()

    t2f_index = MODALITIES.index("t2f")
    with torch.no_grad():
        for image_index, dataset_index in enumerate(sample_indices, start=1):
            image, mask = dataset[dataset_index]
            logits = model(image.unsqueeze(0).to(device))
            probability = torch.sigmoid(logits).squeeze().cpu().numpy()
            prediction = probability >= threshold

            mri = image[t2f_index].numpy()
            low, high = np.percentile(mri, [1, 99])
            if high > low:
                mri_display = np.clip((mri - low) / (high - low), 0, 1)
            else:
                mri_display = np.zeros_like(mri)
            ground_truth = mask.squeeze(0).numpy() > 0

            fig, axes = plt.subplots(1, 4, figsize=(14, 4))
            axes[0].imshow(mri_display, cmap="gray")
            axes[0].set_title("MRI T2-FLAIR")
            axes[1].imshow(ground_truth, cmap="gray")
            axes[1].set_title("Ground Truth")
            axes[2].imshow(prediction, cmap="gray")
            axes[2].set_title("AI Prediction")
            axes[3].imshow(mri_display, cmap="gray")
            axes[3].imshow(ground_truth, cmap="Greens", alpha=0.35)
            axes[3].imshow(prediction, cmap="Reds", alpha=0.35)
            axes[3].set_title("Overlay")

            for axis in axes:
                axis.axis("off")

            fig.suptitle(f"Held-out test sample {image_index}")
            fig.tight_layout()
            sample_path = output_dir / f"test_sample_{image_index:02d}.png"
            fig.savefig(sample_path, dpi=160)
            plt.close(fig)
            saved_paths.append(str(sample_path))

    return saved_paths


def train_model(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    dataset = BraTSSliceDataset(
        args.data_dir,
        max_cases=args.max_cases,
        slice_stride=args.slice_stride,
        include_empty_slices=args.include_empty_slices,
    )
    if not 0 < args.validation_split < 1:
        raise ValueError("--validation-split must be between 0 and 1.")
    if not 0 <= args.test_split < 1:
        raise ValueError("--test-split must be between 0 and 1.")
    if args.validation_split + args.test_split >= 1:
        raise ValueError("--validation-split + --test-split must be less than 1.")

    validation_size = max(1, int(len(dataset) * args.validation_split))
    test_size = max(1, int(len(dataset) * args.test_split)) if args.test_split > 0 else 0
    train_size = len(dataset) - validation_size - test_size
    if train_size < 1:
        raise ValueError("Dataset is too small for the requested validation/test split.")
    print(
        f"dataset_ready cases={len(dataset.cases)} slices={len(dataset)} "
        f"train_slices={train_size} validation_slices={validation_size} test_slices={test_size}",
        flush=True,
    )

    generator = torch.Generator().manual_seed(args.seed)
    split_lengths = [train_size, validation_size] + ([test_size] if test_size else [])
    split_datasets = random_split(dataset, split_lengths, generator=generator)
    train_dataset = split_datasets[0]
    validation_dataset = split_datasets[1]
    test_dataset = split_datasets[2] if test_size else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"training_device={device}", flush=True)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
        if test_dataset is not None
        else None
    )

    model = SmallUNet(in_channels=len(MODALITIES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = DiceBCELoss(bce_weight=args.bce_weight)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=args.lr_patience,
    )
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 1
    best_dice = -1.0
    if args.resume_from and args.resume_from.exists():
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_dice = float(checkpoint.get("best_dice", -1.0))
        if args.reset_best_dice_on_resume:
            best_dice = -1.0
        print(f"resumed_from={args.resume_from} start_epoch={start_epoch} best_dice={best_dice:.4f}")

    history = []
    epochs_without_improvement = 0
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler,
            use_amp,
            args.gradient_clip,
        )
        validation_metrics = evaluate(model, validation_loader, criterion, device, desc="validate")
        validation_dice = validation_metrics["dice_score"]
        scheduler.step(validation_dice)
        current_lr = optimizer.param_groups[0]["lr"]
        improved = validation_dice > best_dice
        if improved:
            best_dice = validation_dice
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "modalities": list(MODALITIES),
                    "model_type": "SmallUNet2D",
                    "best_dice": best_dice,
                    "data_dir": str(args.data_dir),
                },
                args.output_path,
            )
        else:
            epochs_without_improvement += 1

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                **validation_metrics,
                "learning_rate": current_lr,
                "best_dice": best_dice,
                "saved_best_model": improved,
            }
        )
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"validation_loss={validation_metrics['loss']:.4f} "
            f"dice={validation_dice:.4f} best_dice={best_dice:.4f} lr={current_lr:.6g}"
        )

        if args.early_stopping_patience and epochs_without_improvement >= args.early_stopping_patience:
            print(f"early_stopping=true patience={args.early_stopping_patience}")
            break

    test_metrics = None
    sample_result_paths: list[str] = []
    if test_loader is not None and args.output_path.exists():
        best_checkpoint = torch.load(args.output_path, map_location=device)
        model.load_state_dict(best_checkpoint["model_state_dict"])
        test_metrics = evaluate(model, test_loader, criterion, device, desc="test")
        print(
            f"final_test_loss={test_metrics['loss']:.4f} "
            f"final_test_dice={test_metrics['dice_score']:.4f}",
            flush=True,
        )
        sample_result_paths = save_test_sample_images(
            model=model,
            dataset=test_dataset,
            device=device,
            output_dir=args.sample_results_dir,
            sample_count=args.num_sample_results,
            threshold=args.threshold,
        )

    if args.plot_path is not None:
        save_training_plot(history, args.plot_path, test_metrics=test_metrics)

    args.metrics_path.write_text(
        json.dumps(
            {
                "data_dir": str(args.data_dir),
                "device": str(device),
                "amp": use_amp,
                "cases_used": len(dataset.cases),
                "slices_used": len(dataset),
                "train_slices": train_size,
                "validation_slices": validation_size,
                "test_slices": test_size,
                "best_dice": best_dice,
                "final_test_metrics": test_metrics,
                "best_model_path": str(args.output_path),
                "plot_path": str(args.plot_path) if args.plot_path is not None else None,
                "sample_results_dir": str(args.sample_results_dir),
                "sample_result_paths": sample_result_paths,
                "history": history,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved_best_model={args.output_path}")
    print(f"saved_metrics={args.metrics_path}")
    if args.plot_path is not None:
        print(f"saved_plot={args.plot_path}")
    if sample_result_paths:
        print(f"saved_sample_results={args.sample_results_dir}")


def format_probability(value: float) -> str:
    return f"{value * 100:.2f}%"


def build_report(
    patient_details: dict,
    case_dir: Path,
    predicted_label: str,
    confidence: float,
    probabilities: dict[str, float],
    segmentation_summary: dict,
) -> str:
    patient_id = patient_details.get("patient_id", "unknown_research_case")
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    probability_lines = "\n".join(
        f"- {label}: {format_probability(probability)}"
        for label, probability in sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    )
    return f"""# AI-Assisted BraTS MRI Research Report

## Research Case

- Patient/case ID: {patient_id}
- Generated at: {generated_at}
- Case folder: {case_dir}
- Modality: {patient_details.get("modality", "MRI")}
- Sequence: {patient_details.get("sequence", "T1c, T1n, T2-FLAIR, T2w")}

## Patient Details

- Age: {patient_details.get("age", "Not provided")}
- Sex: {patient_details.get("sex", "Not provided")}
- Clinical history: {patient_details.get("clinical_history", "Not provided")}
- Notes: {patient_details.get("notes", "Not provided")}

## Model Output

- Predicted class: {predicted_label}
- Model confidence proxy: {format_probability(confidence)}

## Class Probabilities

{probability_lines}

## Segmentation Summary

- Case ID: {segmentation_summary["case_id"]}
- Predicted tumor voxels: {segmentation_summary["tumor_voxels"]}
- Total evaluated voxels: {segmentation_summary["total_voxels"]}
- Predicted tumor fraction: {segmentation_summary["tumor_fraction"]:.6f}

## Research Interpretation Draft

The trained BraTS MRI segmentation model produced a tumor-region mask for this de-identified research case. The tumor fraction is a model-derived research signal and should not be treated as diagnostic certainty.

## Limitations

- This report is generated for research and manuscript preparation only.
- This is not a clinical diagnostic report.
- Model performance depends on dataset quality, preprocessing, scanner variation, validation design, and expert review.
- A qualified clinician or research supervisor must review the output before it is used in formal research material.
- Do not store direct patient identifiers with this file.
"""


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


def predict_report(args: argparse.Namespace) -> None:
    result = predict_tumor_fraction(args.model_path, args.case_dir, args.threshold)
    patient_details = json.loads(args.patient_json.read_text(encoding="utf-8"))
    predicted_label = "tumor_present" if result["tumor_voxels"] > 0 else "no_tumor_detected"
    confidence = min(0.99, max(0.01, result["tumor_fraction"] * 10))
    probabilities = {"tumor_present": confidence, "no_tumor_detected": 1.0 - confidence}
    report = build_report(patient_details, args.case_dir, predicted_label, confidence, probabilities, result)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(report, encoding="utf-8")
    print(f"predicted_label={predicted_label}")
    print(f"tumor_voxels={result['tumor_voxels']}")
    print(f"saved_report={args.output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="All-in-one BraTS MRI research pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="Validate the BraTS dataset.")
    validate.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    validate.add_argument("--output-path", type=Path, default=Path("artifacts/brats_dataset_validation.json"))
    validate.set_defaults(func=validate_dataset)

    train = subparsers.add_parser("train", help="Train the tumor segmentation model.")
    train.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    train.add_argument("--epochs", type=int, default=25)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--bce-weight", type=float, default=0.35)
    train.add_argument("--validation-split", type=float, default=0.1)
    train.add_argument("--test-split", type=float, default=0.1)
    train.add_argument("--max-cases", type=int, default=None)
    train.add_argument("--slice-stride", type=int, default=4)
    train.add_argument("--include-empty-slices", action="store_true")
    train.add_argument("--num-workers", type=int, default=0)
    train.add_argument("--gradient-clip", type=float, default=1.0)
    train.add_argument("--lr-patience", type=int, default=3)
    train.add_argument("--early-stopping-patience", type=int, default=8)
    train.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA GPUs.")
    train.add_argument("--resume-from", type=Path, default=None)
    train.add_argument("--reset-best-dice-on-resume", action="store_true")
    train.add_argument("--output-path", type=Path, default=Path("artifacts/brats_tumor_segmenter.pt"))
    train.add_argument("--metrics-path", type=Path, default=Path("artifacts/brats_training_metrics.json"))
    train.add_argument("--plot-path", type=Path, default=Path("reports/brats_training_results.png"))
    train.add_argument("--sample-results-dir", type=Path, default=Path("reports/test_sample_results"))
    train.add_argument("--num-sample-results", type=int, default=8)
    train.add_argument("--threshold", type=float, default=0.5)
    train.add_argument("--seed", type=int, default=42)
    train.set_defaults(func=train_model)

    report = subparsers.add_parser("report", help="Predict one case and generate a research report.")
    report.add_argument("--model-path", type=Path, default=Path("artifacts/brats_tumor_segmenter.pt"))
    report.add_argument("--case-dir", type=Path, required=True)
    report.add_argument("--patient-json", type=Path, required=True)
    report.add_argument("--output-path", type=Path, default=Path("reports/brats_case_report.md"))
    report.add_argument("--threshold", type=float, default=0.5)
    report.set_defaults(func=predict_report)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
