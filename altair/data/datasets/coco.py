"""
COCO format dataset for segmentation.

Expects COCO-style JSON annotations with segmentation polygons or RLE masks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from altair.core.registry import DATASETS


@DATASETS.register("coco")
class COCOSegmentationDataset(Dataset):
    """
    Dataset for COCO format segmentation annotations.

    Supports both polygon and RLE segmentation formats. Converts annotations
    to dense masks for training.

    Args:
        images_dir: Path to directory containing images.
        annotations_path: Path to COCO JSON annotation file.
        transform: Optional transform to apply to image and mask.

    Example:
        >>> dataset = COCOSegmentationDataset("data/images", "data/annotations.json")
        >>> sample = dataset[0]
        >>> print(sample["image"].shape, sample["mask"].shape)
    """

    def __init__(
        self,
        images_dir: str | Path,
        annotations_path: str | Path,
        transform: Callable | None = None,
    ):
        self.images_dir = Path(images_dir)
        self.annotations_path = Path(annotations_path)
        self.transform = transform

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.annotations_path.exists():
            raise FileNotFoundError(f"Annotations file not found: {self.annotations_path}")

        # Load COCO annotations
        with open(self.annotations_path) as f:
            self.coco_data = json.load(f)

        # Build indices
        self.images = {img["id"]: img for img in self.coco_data["images"]}
        self.categories = {cat["id"]: cat for cat in self.coco_data["categories"]}

        # Group annotations by image
        self.image_annotations: dict[int, list[dict]] = {}
        for ann in self.coco_data.get("annotations", []):
            img_id = ann["image_id"]
            if img_id not in self.image_annotations:
                self.image_annotations[img_id] = []
            self.image_annotations[img_id].append(ann)

        # List of image IDs with annotations
        self.image_ids = sorted(self.image_annotations.keys())

        if len(self.image_ids) == 0:
            raise ValueError("No images with annotations found in the dataset")

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Get a sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with 'image' and 'mask' tensors.
        """
        image_id = self.image_ids[idx]
        image_info = self.images[image_id]

        # Load image
        image_path = self.images_dir / image_info["file_name"]
        image = Image.open(image_path).convert("RGB")
        image = np.array(image)

        # Create mask from annotations
        height = image_info["height"]
        width = image_info["width"]
        mask = self._create_mask(image_id, height, width)

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        return {
            "image": image,
            "mask": mask,
            "image_path": str(image_path),
            "image_id": image_id,
        }

    def _create_mask(self, image_id: int, height: int, width: int) -> np.ndarray:
        """
        Create a segmentation mask from COCO annotations.

        Args:
            image_id: Image ID.
            height: Image height.
            width: Image width.

        Returns:
            Mask array of shape (H, W) with class indices.
        """
        mask = np.zeros((height, width), dtype=np.uint8)

        annotations = self.image_annotations.get(image_id, [])
        for ann in annotations:
            category_id = ann["category_id"]
            segmentation = ann.get("segmentation")

            if segmentation is None:
                continue

            # Handle RLE format
            if isinstance(segmentation, dict):
                ann_mask = self._decode_rle(segmentation, height, width)
            # Handle polygon format
            elif isinstance(segmentation, list):
                ann_mask = self._decode_polygons(segmentation, height, width)
            else:
                continue

            # Apply mask with category ID
            mask[ann_mask > 0] = category_id

        return mask

    def _decode_rle(self, rle: dict, height: int, width: int) -> np.ndarray:
        """Decode RLE segmentation to binary mask."""
        try:
            from pycocotools import mask as coco_mask

            if "counts" in rle:
                if isinstance(rle["counts"], list):
                    # Uncompressed RLE
                    rle = coco_mask.frPyObjects(rle, height, width)
                return coco_mask.decode(rle)
        except ImportError:
            pass

        # Fallback: manual RLE decoding
        counts = rle.get("counts", [])
        if isinstance(counts, str):
            # Compressed RLE - requires pycocotools
            raise ImportError(
                "pycocotools required for compressed RLE. "
                "Install with: pip install pycocotools"
            )

        mask = np.zeros(height * width, dtype=np.uint8)
        pos = 0
        for i, count in enumerate(counts):
            if i % 2 == 1:  # Odd indices are foreground
                mask[pos : pos + count] = 1
            pos += count

        return mask.reshape((height, width), order="F")

    def _decode_polygons(
        self, polygons: list[list[float]], height: int, width: int
    ) -> np.ndarray:
        """Decode polygon segmentation to binary mask."""
        try:
            from pycocotools import mask as coco_mask

            rles = coco_mask.frPyObjects(polygons, height, width)
            rle = coco_mask.merge(rles)
            return coco_mask.decode(rle)
        except ImportError:
            pass

        # Fallback: use PIL for polygon drawing
        from PIL import Image, ImageDraw

        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)

        for polygon in polygons:
            if len(polygon) < 6:  # Need at least 3 points
                continue
            # Convert flat list to list of tuples
            points = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]
            draw.polygon(points, fill=1)

        return np.array(mask)

    @property
    def num_classes(self) -> int:
        """Return the number of classes in the dataset."""
        return len(self.categories)

    @property
    def class_names(self) -> list[str]:
        """Return the list of class names."""
        return [cat["name"] for cat in self.coco_data["categories"]]
