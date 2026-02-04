"""
Data augmentation transforms for Altair.

Built on top of albumentations for efficient image augmentation.
"""

from __future__ import annotations

from typing import Any

import albumentations as A
from albumentations.pytorch import ToTensorV2

__all__ = [
    "build_transforms",
    "get_train_transforms",
    "get_val_transforms",
]


def build_transforms(augmentation_configs: list[dict[str, Any]]) -> A.Compose:
    """
    Build an albumentations transform pipeline from configuration.

    Args:
        augmentation_configs: List of augmentation configurations.
            Each config should have a 'name' key and optional parameters.

    Returns:
        Composed albumentations transform.

    Example:
        >>> configs = [
        ...     {"name": "resize", "height": 512, "width": 512},
        ...     {"name": "horizontal_flip", "p": 0.5},
        ... ]
        >>> transform = build_transforms(configs)
    """
    transforms = []

    for config in augmentation_configs:
        # Extract name and parameters
        config = dict(config)  # Copy to avoid modifying original
        name = config.pop("name")
        transform = _get_transform(name, config)
        transforms.append(transform)

    # Always add ToTensor at the end
    transforms.append(ToTensorV2())

    return A.Compose(transforms)


def _get_transform(name: str, params: dict[str, Any]) -> A.BasicTransform:
    """
    Get an albumentations transform by name.

    Args:
        name: Transform name (lowercase).
        params: Transform parameters.

    Returns:
        Albumentations transform instance.
    """
    # Map of common transform names to albumentations classes
    transform_map = {
        # Spatial transforms
        "resize": A.Resize,
        "random_crop": A.RandomCrop,
        "center_crop": A.CenterCrop,
        "crop": A.Crop,
        "pad": A.PadIfNeeded,
        "random_resized_crop": A.RandomResizedCrop,
        # Flip transforms
        "horizontal_flip": A.HorizontalFlip,
        "vertical_flip": A.VerticalFlip,
        "flip": A.Flip,
        # Rotation transforms
        "rotate": A.Rotate,
        "random_rotate_90": A.RandomRotate90,
        "safe_rotate": A.SafeRotate,
        # Affine transforms
        "shift_scale_rotate": A.ShiftScaleRotate,
        "affine": A.Affine,
        "perspective": A.Perspective,
        "elastic": A.ElasticTransform,
        "grid_distortion": A.GridDistortion,
        "optical_distortion": A.OpticalDistortion,
        # Color transforms
        "normalize": A.Normalize,
        "color_jitter": A.ColorJitter,
        "random_brightness_contrast": A.RandomBrightnessContrast,
        "hue_saturation_value": A.HueSaturationValue,
        "rgb_shift": A.RGBShift,
        "channel_shuffle": A.ChannelShuffle,
        "clahe": A.CLAHE,
        "blur": A.Blur,
        "gaussian_blur": A.GaussianBlur,
        "motion_blur": A.MotionBlur,
        "median_blur": A.MedianBlur,
        "gaussian_noise": A.GaussNoise,
        "iso_noise": A.ISONoise,
        "coarse_dropout": A.CoarseDropout,
        "random_gamma": A.RandomGamma,
        "to_gray": A.ToGray,
        "invert": A.InvertImg,
        "posterize": A.Posterize,
        "equalize": A.Equalize,
        "sharpen": A.Sharpen,
        "emboss": A.Emboss,
        "superpixels": A.Superpixels,
        # Weather effects
        "random_rain": A.RandomRain,
        "random_snow": A.RandomSnow,
        "random_fog": A.RandomFog,
        "random_sun_flare": A.RandomSunFlare,
        "random_shadow": A.RandomShadow,
    }

    # Normalize name
    name_lower = name.lower().replace("-", "_")

    if name_lower not in transform_map:
        raise ValueError(
            f"Unknown transform: {name}. "
            f"Available: {sorted(transform_map.keys())}"
        )

    transform_cls = transform_map[name_lower]

    # Handle special cases
    if name_lower == "normalize" and "mean" not in params:
        # Default ImageNet normalization
        params.setdefault("mean", [0.485, 0.456, 0.406])
        params.setdefault("std", [0.229, 0.224, 0.225])

    return transform_cls(**params)


# Common preset transforms
def get_train_transforms(height: int = 512, width: int = 512) -> A.Compose:
    """Get default training transforms."""
    return A.Compose([
        A.Resize(height, width),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms(height: int = 512, width: int = 512) -> A.Compose:
    """Get default validation transforms."""
    return A.Compose([
        A.Resize(height, width),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
