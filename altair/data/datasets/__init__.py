"""Dataset implementations for Altair."""

from altair.data.datasets.png_mask import PNGMaskDataset
from altair.data.datasets.coco import COCOSegmentationDataset

__all__ = ["PNGMaskDataset", "COCOSegmentationDataset"]
