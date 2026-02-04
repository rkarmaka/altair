"""
Data loading and preprocessing for Altair.

This module provides dataset classes and data loaders for segmentation tasks.
Supports PNG mask format and COCO annotation format.

Example:
    >>> from altair.data import build_dataloaders
    >>> train_loader, val_loader = build_dataloaders(config.data, config.augmentations)
"""

from altair.core.registry import DATASETS

from altair.data.datasets import COCOSegmentationDataset, PNGMaskDataset

__all__ = [
    "DATASETS",
    "PNGMaskDataset",
    "COCOSegmentationDataset",
    "build_dataloaders",
]


def build_dataloaders(data_config, augmentation_config=None):
    """
    Build train and validation data loaders from configuration.

    Args:
        data_config: DataConfig instance with data paths and loader settings.
        augmentation_config: Optional AugmentationConfig for transforms.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    from torch.utils.data import DataLoader

    from altair.data.transforms import build_transforms

    # Build transforms
    train_transform = None
    val_transform = None
    if augmentation_config:
        train_transform = build_transforms(augmentation_config.train)
        val_transform = build_transforms(augmentation_config.val)

    # Build datasets
    if data_config.format == "png":
        train_dataset = PNGMaskDataset(
            images_dir=data_config.train.images,
            masks_dir=data_config.train.masks,
            transform=train_transform,
        )
        val_dataset = PNGMaskDataset(
            images_dir=data_config.val.images,
            masks_dir=data_config.val.masks,
            transform=val_transform,
        )
    else:  # coco
        train_dataset = COCOSegmentationDataset(
            images_dir=data_config.train.images,
            annotations_path=data_config.train.annotations,
            transform=train_transform,
        )
        val_dataset = COCOSegmentationDataset(
            images_dir=data_config.val.images,
            annotations_path=data_config.val.annotations,
            transform=val_transform,
        )

    # Build loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config.batch_size,
        shuffle=True,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=data_config.batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
    )

    return train_loader, val_loader
