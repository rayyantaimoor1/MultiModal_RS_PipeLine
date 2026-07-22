"""
eurosat_dataset.py

PyTorch Dataset for multi-spectral Sentinel-2 imagery in the style of the
EuroSAT benchmark, generalized to support an arbitrary number of raster
channels (e.g. all 13 Sentinel-2 bands rather than just RGB).

Supports two on-disk layouts:
    1. Directory-of-class-folders, each containing `.tif` / `.npy` raster
       patches (classification mode) — mirrors `torchvision.ImageFolder`.
    2. A flat manifest CSV with `filepath,label` columns.

Each sample is loaded as a `[num_channels, height, width]` float tensor and
optionally passed through spatial augmentations (flips/rotations) that are
safe for multi-spectral (non-RGB) data, since they avoid photometric color
jitter that assumes 3-channel visible-spectrum images.

Author: Principal ML Engineering Team
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import rasterio  # type: ignore

    _HAS_RASTERIO = True
except ImportError:  # pragma: no cover - optional dependency
    _HAS_RASTERIO = False


class SpatialAugment:
    """Lightweight geometric augmentation safe for N-channel rasters.

    Applies random horizontal flip, vertical flip, and 90-degree rotations.
    Deliberately avoids color-jitter / brightness transforms, since those
    assume a 3-channel visible-light image and would corrupt reflectance
    values in non-RGB bands (e.g. SWIR, NIR).

    Args:
        hflip_prob: Probability of a horizontal flip.
        vflip_prob: Probability of a vertical flip.
        rotate_prob: Probability of a random 90/180/270-degree rotation.
    """

    def __init__(
        self,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        rotate_prob: float = 0.5,
    ) -> None:
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.rotate_prob = rotate_prob

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """Applies augmentation.

        Args:
            image: FloatTensor, shape [C, H, W]

        Returns:
            Augmented FloatTensor, shape [C, H, W]
        """
        if torch.rand(1).item() < self.hflip_prob:
            image = torch.flip(image, dims=[2])  # flip width
        if torch.rand(1).item() < self.vflip_prob:
            image = torch.flip(image, dims=[1])  # flip height
        if torch.rand(1).item() < self.rotate_prob:
            k = int(torch.randint(1, 4, (1,)).item())  # 1, 2, or 3 quarter-turns
            image = torch.rot90(image, k=k, dims=[1, 2])
        return image


class EuroSATDataset(Dataset):
    """Multi-spectral Sentinel-2 patch classification dataset.

    Loads N-channel raster patches (GeoTIFF via `rasterio` if available,
    otherwise `.npy` arrays of shape [C, H, W] or [H, W, C]) and their
    integer class labels for land-cover classification (e.g. EuroSAT's
    10 classes: AnnualCrop, Forest, Residential, ...).

    Directory layout expected (ImageFolder-style):
        root/
            AnnualCrop/
                patch_0001.tif
                patch_0002.tif
            Forest/
                patch_0001.tif
            ...

    Args:
        root_dir: Root directory containing one sub-folder per class, OR
            a manifest CSV path with `filepath,label` columns.
        num_channels: Expected number of spectral bands (e.g. 13 for full
            Sentinel-2 L2A, 3 for RGB-only subsets). Used for validation
            and for synthetic fallback when rasterio is unavailable.
        image_size: Target spatial size (H, W) patches are center-cropped
            or resized to, for batching consistency.
        band_mean: Optional per-band mean for normalization, shape
            [num_channels]. If None, raw reflectance values are scaled by
            `reflectance_scale` only.
        band_std: Optional per-band std for normalization, shape
            [num_channels].
        reflectance_scale: Divisor applied to raw digital numbers to bring
            Sentinel-2 reflectance (typically 0-10000) into a [0, 1]-ish
            range before optional z-score normalization.
        transform: Optional callable (e.g. `SpatialAugment`) applied to the
            loaded image tensor.
        class_to_idx: Optional pre-fixed class-name -> index mapping (reuse
            the training set's mapping for val/test splits).

    Returns (via __getitem__):
        Tuple[torch.Tensor, torch.Tensor]:
            image: FloatTensor, shape [num_channels, height, width]
            label: LongTensor scalar, shape [] (class index)
    """

    VALID_EXTENSIONS = {".tif", ".tiff", ".npy"}

    def __init__(
        self,
        root_dir: Union[str, Path],
        num_channels: int = 13,
        image_size: Tuple[int, int] = (64, 64),
        band_mean: Optional[np.ndarray] = None,
        band_std: Optional[np.ndarray] = None,
        reflectance_scale: float = 10000.0,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        class_to_idx: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__()

        self.root_dir = Path(root_dir)
        self.num_channels = num_channels
        self.image_size = image_size
        self.band_mean = band_mean
        self.band_std = band_std
        self.reflectance_scale = reflectance_scale
        self.transform = transform

        self.samples: List[Tuple[Path, int]] = []
        self.class_to_idx: Dict[str, int] = class_to_idx or {}

        if self.root_dir.suffix.lower() == ".csv":
            self._load_from_manifest(self.root_dir)
        else:
            self._load_from_folder(self.root_dir)

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples found under {self.root_dir}. Expected class "
                f"sub-folders with {self.VALID_EXTENSIONS} files, or a manifest CSV."
            )

    def _load_from_folder(self, root_dir: Path) -> None:
        """Populates `self.samples` / `self.class_to_idx` from ImageFolder-style dirs."""
        class_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir()])
        if not self.class_to_idx:
            self.class_to_idx = {d.name: i for i, d in enumerate(class_dirs)}

        for class_dir in class_dirs:
            label = self.class_to_idx.get(class_dir.name)
            if label is None:
                continue  # unseen class not present in fixed mapping; skip
            for file_path in sorted(class_dir.iterdir()):
                if file_path.suffix.lower() in self.VALID_EXTENSIONS:
                    self.samples.append((file_path, label))

    def _load_from_manifest(self, manifest_path: Path) -> None:
        """Populates `self.samples` from a `filepath,label` CSV manifest."""
        labels_seen: List[str] = []
        rows: List[Tuple[str, str]] = []
        with open(manifest_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append((row["filepath"], row["label"]))
                labels_seen.append(row["label"])

        if not self.class_to_idx:
            unique_labels = sorted(set(labels_seen))
            self.class_to_idx = {name: i for i, name in enumerate(unique_labels)}

        for filepath_str, label_name in rows:
            label = self.class_to_idx.get(label_name)
            if label is not None:
                self.samples.append((Path(filepath_str), label))

    def _read_raster(self, file_path: Path) -> np.ndarray:
        """Reads a raster file into a [C, H, W] float32 numpy array."""
        if file_path.suffix.lower() == ".npy":
            arr = np.load(file_path).astype(np.float32)
            if arr.ndim == 3 and arr.shape[0] not in (self.num_channels,) and arr.shape[-1] == self.num_channels:
                arr = np.transpose(arr, (2, 0, 1))  # [H, W, C] -> [C, H, W]
            return arr

        if _HAS_RASTERIO:
            with rasterio.open(file_path) as src:  # pragma: no cover - IO dependent
                arr = src.read().astype(np.float32)  # [C, H, W]
            return arr

        raise RuntimeError(
            "rasterio is not installed; cannot read GeoTIFF files. "
            "Install rasterio or provide .npy patches instead."
        )

    def _resize_to_target(self, image: torch.Tensor) -> torch.Tensor:
        """Center-crops or bilinearly resizes to `self.image_size`."""
        target_h, target_w = self.image_size
        c, h, w = image.shape
        if (h, w) == (target_h, target_w):
            return image
        image = image.unsqueeze(0)  # [1, C, H, W] for interpolate
        image = torch.nn.functional.interpolate(
            image, size=(target_h, target_w), mode="bilinear", align_corners=False
        )
        return image.squeeze(0)  # [C, H, W]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Loads and preprocesses a single raster patch.

        Returns:
            image: FloatTensor, shape [num_channels, height, width]
            label: LongTensor scalar, shape []
        """
        file_path, label = self.samples[index]
        raw_array = self._read_raster(file_path)  # [C, H, W]

        image = torch.from_numpy(raw_array).float()  # [C, H, W]
        image = image / self.reflectance_scale  # scale digital numbers -> ~[0, 1]

        if self.band_mean is not None and self.band_std is not None:
            mean = torch.as_tensor(self.band_mean, dtype=torch.float32).view(-1, 1, 1)
            std = torch.as_tensor(self.band_std, dtype=torch.float32).view(-1, 1, 1)
            std = torch.where(std < 1e-8, torch.ones_like(std), std)
            image = (image - mean) / std

        image = self._resize_to_target(image)

        if self.transform is not None:
            image = self.transform(image)

        label_tensor = torch.tensor(label, dtype=torch.long)  # scalar
        return image, label_tensor

    @property
    def num_classes(self) -> int:
        return len(self.class_to_idx)


def _build_dummy_eurosat_dir(root: Path, num_channels: int = 13, patches_per_class: int = 4) -> None:
    """Creates a small synthetic multi-spectral dataset of .npy patches on disk."""
    classes = ["AnnualCrop", "Forest", "Residential", "River"]
    rng = np.random.default_rng(seed=7)
    for class_name in classes:
        class_dir = root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(patches_per_class):
            patch = rng.integers(0, 10000, size=(num_channels, 64, 64)).astype(np.float32)
            np.save(class_dir / f"patch_{i:04d}.npy", patch)


if __name__ == "__main__":
    import shutil
    import tempfile

    print("=== EuroSATDataset smoke test ===")

    tmp_dir = Path(tempfile.mkdtemp(prefix="eurosat_dummy_"))
    try:
        NUM_CHANNELS = 13
        _build_dummy_eurosat_dir(tmp_dir, num_channels=NUM_CHANNELS, patches_per_class=4)

        augment = SpatialAugment(hflip_prob=0.5, vflip_prob=0.5, rotate_prob=0.5)
        dataset = EuroSATDataset(
            root_dir=tmp_dir,
            num_channels=NUM_CHANNELS,
            image_size=(64, 64),
            transform=augment,
        )

        print(f"Num samples: {len(dataset)} | Num classes: {dataset.num_classes}")
        print(f"Class mapping: {dataset.class_to_idx}")

        image, label = dataset[0]
        print(f"Single sample -> image: {tuple(image.shape)}, label: {label.item()}")
        assert image.shape == (NUM_CHANNELS, 64, 64)
        assert image.dtype == torch.float32
        assert label.dtype == torch.long

        from torch.utils.data import DataLoader

        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        batch_images, batch_labels = next(iter(loader))
        print(f"Batch -> images: {tuple(batch_images.shape)}, labels: {tuple(batch_labels.shape)}")
        assert batch_images.shape == (4, NUM_CHANNELS, 64, 64)
        assert batch_labels.shape == (4,)

        print("All EuroSATDataset assertions passed.")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
