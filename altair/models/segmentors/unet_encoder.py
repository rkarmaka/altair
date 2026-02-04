"""
UNet with pretrained encoder backbone.

Combines a pretrained encoder (from timm) with a UNet-style decoder
for segmentation tasks.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from altair.core.registry import MODELS
from altair.models.backbones.timm_encoder import get_encoder
from altair.models.decoders.unet_decoder import UNetDecoder


class SegmentationHead(nn.Module):
    """
    Segmentation head for final prediction.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels (classes).
        kernel_size: Kernel size for final convolution.
        upsampling: Upsampling factor to match input resolution.
        dropout: Dropout probability before final convolution.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        upsampling: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.upsampling = (
            nn.Upsample(scale_factor=upsampling, mode="bilinear", align_corners=False)
            if upsampling > 1
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        x = self.conv(x)
        x = self.upsampling(x)
        return x


@MODELS.register("unet_encoder")
@MODELS.register("unet_resnet")  # Alias for backward compatibility
class UNetEncoder(nn.Module):
    """
    UNet with pretrained encoder backbone.

    Combines a pretrained encoder from timm with a UNet-style decoder.
    This is the recommended architecture for production use due to
    pretrained weights and proven encoder architectures.

    Args:
        architecture: Architecture name (unused, for registry compatibility).
        task: Task type ('binary', 'multiclass', 'regression').
        num_classes: Number of output classes/channels.
        encoder: Encoder backbone name (any timm model).
        encoder_weights: Pretrained weights ('imagenet', None, or path).
        encoder_depth: Number of encoder stages (1-5).
        decoder_channels: List of decoder channel sizes.
        decoder_attention: Attention type in decoder ('scse' or None).
        dropout: Dropout probability.
        in_channels: Number of input channels.
        **kwargs: Additional arguments (ignored).

    Supported encoders:
        - ResNet: resnet18, resnet34, resnet50, resnet101, resnet152
        - ResNeXt: resnext50_32x4d, resnext101_32x8d
        - EfficientNet: efficientnet_b0 to efficientnet_b7
        - ConvNeXt: convnext_tiny, convnext_small, convnext_base
        - MobileNet: mobilenetv3_large_100, mobilenetv3_small_100
        - And many more from timm...

    Example:
        >>> model = UNetEncoder(
        ...     encoder="resnet50",
        ...     encoder_weights="imagenet",
        ...     num_classes=10,
        ... )
        >>> x = torch.randn(1, 3, 256, 256)
        >>> out = model(x)
        >>> out.shape
        torch.Size([1, 10, 256, 256])
    """

    def __init__(
        self,
        architecture: str = "unet_encoder",
        task: Literal["binary", "multiclass", "regression"] = "multiclass",
        num_classes: int = 1,
        encoder: str = "resnet50",
        encoder_weights: str | None = "imagenet",
        encoder_depth: int = 5,
        decoder_channels: list[int] | None = None,
        decoder_attention: Literal["scse"] | None = None,
        dropout: float = 0.0,
        drop_path: float = 0.0,  # Not used yet, for future ViT support
        in_channels: int = 3,
        norm_layer: str = "batchnorm",  # For compatibility
        activation: str = "relu",  # For compatibility
        **kwargs,
    ):
        super().__init__()

        self.task = task
        self.num_classes = num_classes
        self.encoder_name = encoder
        self.encoder_depth = encoder_depth

        # Default decoder channels
        if decoder_channels is None:
            decoder_channels = [256, 128, 64, 32, 16]

        # Adjust decoder channels to match encoder depth
        decoder_channels = decoder_channels[:encoder_depth]

        # Determine if pretrained
        pretrained = encoder_weights is not None and encoder_weights != "null"

        # Create encoder
        self.encoder = get_encoder(
            name=encoder,
            pretrained=pretrained,
            in_channels=in_channels,
            depth=encoder_depth,
        )

        # Create decoder
        self.decoder = UNetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            use_batchnorm=(norm_layer == "batchnorm"),
            attention_type=decoder_attention,
            dropout=dropout,
        )

        # Create segmentation head
        # Calculate upsampling factor to match input resolution
        # The decoder outputs at stride = 2 (after all upsampling)
        self.head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=num_classes,
            kernel_size=3,
            upsampling=2,  # Final 2x upsample to match input
            dropout=dropout,
        )

        # Initialize decoder and head weights
        self._init_weights()

    def _init_weights(self):
        """Initialize decoder and head weights."""
        for m in self.decoder.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        for m in self.head.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Output tensor of shape (B, num_classes, H, W).
        """
        # Get input size for final resize
        input_size = x.shape[2:]

        # Encoder
        features = self.encoder(x)

        # Decoder
        decoder_output = self.decoder(*features)

        # Segmentation head
        output = self.head(decoder_output)

        # Ensure output matches input size
        if output.shape[2:] != input_size:
            output = nn.functional.interpolate(
                output, size=input_size, mode="bilinear", align_corners=False
            )

        return output

    def get_output_activation(self) -> nn.Module | None:
        """
        Get the appropriate output activation for the task.

        Returns:
            Activation module or None for regression.
        """
        if self.task == "binary":
            return nn.Sigmoid()
        elif self.task == "multiclass":
            return nn.Softmax(dim=1)
        else:  # regression
            return None

    @property
    def encoder_output_channels(self) -> list[int]:
        """Return encoder output channels."""
        return self.encoder.out_channels

    def freeze_encoder(self) -> None:
        """Freeze encoder weights (for fine-tuning decoder only)."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder weights."""
        for param in self.encoder.parameters():
            param.requires_grad = True


# Convenience aliases for common configurations
@MODELS.register("unet_resnet18")
def unet_resnet18(**kwargs):
    """UNet with ResNet-18 encoder."""
    kwargs.setdefault("encoder", "resnet18")
    return UNetEncoder(**kwargs)


@MODELS.register("unet_resnet34")
def unet_resnet34(**kwargs):
    """UNet with ResNet-34 encoder."""
    kwargs.setdefault("encoder", "resnet34")
    return UNetEncoder(**kwargs)


@MODELS.register("unet_resnet50")
def unet_resnet50(**kwargs):
    """UNet with ResNet-50 encoder."""
    kwargs.setdefault("encoder", "resnet50")
    return UNetEncoder(**kwargs)


@MODELS.register("unet_resnet101")
def unet_resnet101(**kwargs):
    """UNet with ResNet-101 encoder."""
    kwargs.setdefault("encoder", "resnet101")
    return UNetEncoder(**kwargs)


@MODELS.register("unet_efficientnet_b0")
def unet_efficientnet_b0(**kwargs):
    """UNet with EfficientNet-B0 encoder."""
    kwargs.setdefault("encoder", "efficientnet_b0")
    return UNetEncoder(**kwargs)


@MODELS.register("unet_efficientnet_b4")
def unet_efficientnet_b4(**kwargs):
    """UNet with EfficientNet-B4 encoder."""
    kwargs.setdefault("encoder", "efficientnet_b4")
    return UNetEncoder(**kwargs)


@MODELS.register("unet_convnext_tiny")
def unet_convnext_tiny(**kwargs):
    """UNet with ConvNeXt-Tiny encoder."""
    kwargs.setdefault("encoder", "convnext_tiny")
    return UNetEncoder(**kwargs)


@MODELS.register("unet_mobilenet")
def unet_mobilenet(**kwargs):
    """UNet with MobileNetV3-Large encoder."""
    kwargs.setdefault("encoder", "mobilenetv3_large_100")
    return UNetEncoder(**kwargs)
