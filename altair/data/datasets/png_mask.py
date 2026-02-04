"""
PNG mask dataset for simple folder-based segmentation data.

Expected directory structure:
    images/
        001.png
        002.png
        ...
    masks/
        001.png
        002.png
        ...

Image and mask files should have matching names.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from altair.core.registry import DATASETS


@DATASETS.register("png")
class PNGMaskDataset(Dataset):
    """
    Dataset for PNG images with corresponding PNG masks.

    Assumes masks are single-channel images where pixel values represent
    class indices.

    Args:
        images_dir: Path to directory containing images.
        masks_dir: Path to directory containing masks.
        transform: Optional transform to apply to image and mask.
        image_suffix: File suffix for images (default: inferred).
        mask_suffix: File suffix for masks (default: inferred).

    Example:
        >>> dataset = PNGMaskDataset("data/images", "data/masks")
        >>> image, mask = dataset[0]
    """

    # Common image extensions
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}

    def __init__(
        self,
        images_dir: str | Path,
        masks_dir: str | Path,
        transform: Callable | None = None,
        image_suffix: str | None = None,
        mask_suffix: str | None = None,
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.transform = transform

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {self.masks_dir}")

        # Find all image files
        self.image_paths = self._find_images(image_suffix)
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {self.images_dir}")

        # Verify masks exist
        self.mask_paths = []
        for img_path in self.image_paths:
            mask_path = self._find_matching_mask(img_path, mask_suffix)
            if mask_path is None:
                raise FileNotFoundError(
                    f"No matching mask found for {img_path.name} in {self.masks_dir}"
                )
            self.mask_paths.append(mask_path)

    def _find_images(self, suffix: str | None = None) -> list[Path]:
        """Find all image files in the images directory."""
        if suffix:
            return sorted(self.images_dir.glob(f"*{suffix}"))

        images = []
        for ext in self.IMAGE_EXTENSIONS:
            images.extend(self.images_dir.glob(f"*{ext}"))
            images.extend(self.images_dir.glob(f"*{ext.upper()}"))

        return sorted(set(images))

    def _find_matching_mask(self, image_path: Path, mask_suffix: str | None = None) -> Path | None:
        """Find the mask file matching an image file."""
        stem = image_path.stem

        if mask_suffix:
            mask_path = self.masks_dir / f"{stem}{mask_suffix}"
            return mask_path if mask_path.exists() else None

        # Try common extensions
        for ext in self.IMAGE_EXTENSIONS:
            mask_path = self.masks_dir / f"{stem}{ext}"
            if mask_path.exists():
                return mask_path

        return None

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Get a sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with 'image' and 'mask' tensors.
        """
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)

        # Load mask
        mask_path = self.mask_paths[idx]
        mask = Image.open(mask_path)
        mask = np.array(mask)

        # Ensure mask is 2D (single channel)
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        return {
            "image": image,
            "mask": mask,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
        }

    def get_sample_paths(self, idx: int) -> tuple[Path, Path]:
        """Get the paths for a sample."""
        return self.image_paths[idx], self.mask_paths[idx]
