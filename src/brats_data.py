from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset


MODALITIES = ("t1c", "t1n", "t2f", "t2w")


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
            cases.append(
                BraTSCase(
                    case_id=case_dir.name,
                    case_dir=case_dir,
                    modalities=modalities,
                    segmentation=segmentation,
                )
            )

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
    normalized = (volume - mean) / std
    return np.clip(normalized, -5, 5).astype(np.float32)


def load_case_modalities(case: BraTSCase) -> np.ndarray:
    volumes = [normalize_volume(load_nifti(case.modalities[modality])) for modality in MODALITIES]
    return np.stack(volumes, axis=0)


def load_case_segmentation(case: BraTSCase) -> np.ndarray:
    if case.segmentation is None:
        raise ValueError(f"Case has no segmentation file: {case.case_id}")
    segmentation = load_nifti(case.segmentation)
    return (segmentation > 0).astype(np.float32)


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

        self.slice_index: list[tuple[int, int]] = []
        for case_index, case in enumerate(self.cases):
            segmentation = load_case_segmentation(case)
            for slice_number in range(0, segmentation.shape[2], slice_stride):
                has_tumor = bool(segmentation[:, :, slice_number].sum() > 0)
                if include_empty_slices or has_tumor:
                    self.slice_index.append((case_index, slice_number))

        if not self.slice_index:
            raise ValueError("No usable training slices found. Try --include-empty-slices or a smaller --slice-stride.")

    def __len__(self) -> int:
        return len(self.slice_index)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        case_index, slice_number = self.slice_index[index]
        case = self.cases[case_index]
        image = load_case_modalities(case)[:, :, :, slice_number]
        mask = load_case_segmentation(case)[:, :, slice_number]

        image_tensor = torch.from_numpy(image.copy()).float()
        mask_tensor = torch.from_numpy(mask.copy()).float().unsqueeze(0)
        return image_tensor, mask_tensor
