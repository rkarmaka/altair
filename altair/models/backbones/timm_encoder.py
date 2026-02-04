"""
Timm-based encoder backbones.

Wraps timm models to extract multi-scale features for encoder-decoder
architectures. Supports any timm model with feature extraction capability.

Reference:
    https://github.com/huggingface/pytorch-image-models
"""

from __future__ import annotations

from typing import Literal

import timm
import torch
import torch.nn as nn

from altair.core.registry import BACKBONES


@BACKBONES.register("timm")
class TimmEncoder(nn.Module):
    """
    Encoder backbone using timm models.

    Extracts multi-scale features from a pretrained timm model for use
    in encoder-decoder segmentation architectures.

    Args:
        name: Timm model name (e.g., 'resnet50', 'efficientnet_b0').
        pretrained: Whether to load pretrained weights.
        output_stride: Output stride (8, 16, or 32). Lower values preserve
            more spatial resolution but use more memory.
        features_only: Whether to return only features (always True for encoders).
        out_indices: Which feature levels to return (0=stem, 1-4=stages).

    Attributes:
        out_channels: List of output channel sizes for each feature level.
        feature_info: Information about each feature level.

    Example:
        >>> encoder = TimmEncoder("resnet50", pretrained=True)
        >>> x = torch.randn(1, 3, 256, 256)
        >>> features = encoder(x)
        >>> for i, f in enumerate(features):
        ...     print(f"Level {i}: {f.shape}")
    """

    def __init__(
        self,
        name: str = "resnet50",
        pretrained: bool = True,
        output_stride: Literal[8, 16, 32] = 32,
        out_indices: tuple[int, ...] = (0, 1, 2, 3, 4),
        in_channels: int = 3,
    ):
        super().__init__()

        self.name = name
        self._out_indices = out_indices

        # Create timm model with feature extraction
        self.model = timm.create_model(
            name,
            pretrained=pretrained,
            features_only=True,
            output_stride=output_stride,
            out_indices=out_indices,
            in_chans=in_channels,
        )

        # Get feature information
        self._feature_info = self.model.feature_info
        self._out_channels = [info["num_chs"] for info in self._feature_info]
        self._out_strides = [info["reduction"] for info in self._feature_info]

    @property
    def out_channels(self) -> list[int]:
        """Return output channels for each feature level."""
        return self._out_channels

    @property
    def out_strides(self) -> list[int]:
        """Return output strides for each feature level."""
        return self._out_strides

    @property
    def num_features(self) -> int:
        """Return number of feature levels."""
        return len(self._out_channels)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Extract multi-scale features.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            List of feature tensors at different scales.
        """
        return self.model(x)


def get_encoder(
    name: str,
    pretrained: bool = True,
    in_channels: int = 3,
    depth: int = 5,
    output_stride: int = 32,
) -> TimmEncoder:
    """
    Create a timm encoder by name.

    Args:
        name: Encoder name (timm model name).
        pretrained: Whether to load pretrained weights.
        in_channels: Number of input channels.
        depth: Number of encoder stages (1-5).
        output_stride: Output stride.

    Returns:
        TimmEncoder instance.

    Example:
        >>> encoder = get_encoder("resnet50", pretrained=True, depth=5)
        >>> print(encoder.out_channels)
        [64, 256, 512, 1024, 2048]
    """
    # Map depth to out_indices
    out_indices = tuple(range(depth))

    return TimmEncoder(
        name=name,
        pretrained=pretrained,
        output_stride=output_stride,
        out_indices=out_indices,
        in_channels=in_channels,
    )


# Common encoder configurations
ENCODER_CONFIGS = {
    # ResNet family
    "resnet18": {"out_channels": [64, 64, 128, 256, 512]},
    "resnet34": {"out_channels": [64, 64, 128, 256, 512]},
    "resnet50": {"out_channels": [64, 256, 512, 1024, 2048]},
    "resnet101": {"out_channels": [64, 256, 512, 1024, 2048]},
    "resnet152": {"out_channels": [64, 256, 512, 1024, 2048]},
    # ResNeXt
    "resnext50_32x4d": {"out_channels": [64, 256, 512, 1024, 2048]},
    "resnext101_32x8d": {"out_channels": [64, 256, 512, 1024, 2048]},
    # EfficientNet
    "efficientnet_b0": {"out_channels": [16, 24, 40, 112, 320]},
    "efficientnet_b1": {"out_channels": [16, 24, 40, 112, 320]},
    "efficientnet_b2": {"out_channels": [16, 24, 48, 120, 352]},
    "efficientnet_b3": {"out_channels": [24, 32, 48, 136, 384]},
    "efficientnet_b4": {"out_channels": [24, 32, 56, 160, 448]},
    # ConvNeXt
    "convnext_tiny": {"out_channels": [96, 96, 192, 384, 768]},
    "convnext_small": {"out_channels": [96, 96, 192, 384, 768]},
    "convnext_base": {"out_channels": [128, 128, 256, 512, 1024]},
    # MobileNet
    "mobilenetv3_large_100": {"out_channels": [16, 24, 40, 112, 960]},
    "mobilenetv3_small_100": {"out_channels": [16, 16, 24, 48, 576]},
}


def list_encoders() -> list[str]:
    """List available encoder names."""
    return list(ENCODER_CONFIGS.keys()) + ["... and any timm model"]
