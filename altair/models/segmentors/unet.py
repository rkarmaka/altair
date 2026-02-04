"""
UNet segmentation model implementation.

The classic U-Net architecture with configurable depth, channels, and
regularization options.

Reference:
    Ronneberger et al., "U-Net: Convolutional Networks for Biomedical
    Image Segmentation", MICCAI 2015.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from altair.core.registry import MODELS


class ConvBlock(nn.Module):
    """
    Double convolution block used in UNet.

    Consists of two 3x3 convolutions, each followed by normalization
    and activation.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        norm_layer: Normalization layer type.
        activation: Activation function type.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_layer: str = "batchnorm",
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()

        # Get normalization layer
        norm_cls = self._get_norm_layer(norm_layer, out_channels)
        act_fn = self._get_activation(activation)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = norm_cls
        self.act1 = act_fn

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = self._get_norm_layer(norm_layer, out_channels)
        self.act2 = self._get_activation(activation)

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def _get_norm_layer(self, name: str, num_features: int) -> nn.Module:
        """Get normalization layer by name."""
        if name == "batchnorm":
            return nn.BatchNorm2d(num_features)
        elif name == "layernorm":
            return nn.GroupNorm(1, num_features)  # LayerNorm equivalent for 2D
        elif name == "groupnorm":
            return nn.GroupNorm(min(32, num_features), num_features)
        elif name == "instancenorm":
            return nn.InstanceNorm2d(num_features)
        else:
            raise ValueError(f"Unknown normalization: {name}")

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        if name == "relu":
            return nn.ReLU(inplace=True)
        elif name == "gelu":
            return nn.GELU()
        elif name == "silu":
            return nn.SiLU(inplace=True)
        elif name == "leaky_relu":
            return nn.LeakyReLU(0.1, inplace=True)
        else:
            raise ValueError(f"Unknown activation: {name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)

        x = self.dropout(x)
        return x


class EncoderBlock(nn.Module):
    """
    UNet encoder block: convolution followed by downsampling.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        norm_layer: Normalization layer type.
        activation: Activation function type.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_layer: str = "batchnorm",
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv = ConvBlock(in_channels, out_channels, norm_layer, activation, dropout)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            Tuple of (pooled features, skip connection features).
        """
        features = self.conv(x)
        pooled = self.pool(features)
        return pooled, features


class DecoderBlock(nn.Module):
    """
    UNet decoder block: upsampling followed by convolution.

    Args:
        in_channels: Number of input channels.
        skip_channels: Number of skip connection channels.
        out_channels: Number of output channels.
        norm_layer: Normalization layer type.
        activation: Activation function type.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        norm_layer: str = "batchnorm",
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(
            in_channels // 2 + skip_channels, out_channels, norm_layer, activation, dropout
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Forward pass with skip connection."""
        x = self.upsample(x)

        # Handle size mismatch due to odd dimensions
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


@MODELS.register("unet")
class UNet(nn.Module):
    """
    Classic UNet architecture for image segmentation.

    A symmetric encoder-decoder architecture with skip connections.
    The encoder progressively downsamples while the decoder upsamples,
    with skip connections preserving spatial information.

    Args:
        architecture: Architecture name (unused, for registry compatibility).
        task: Task type ('binary', 'multiclass', 'regression').
        num_classes: Number of output classes/channels.
        in_channels: Number of input channels (default: 3 for RGB).
        encoder_depth: Number of encoder stages (1-5).
        decoder_channels: List of decoder channel sizes.
        dropout: Dropout probability.
        norm_layer: Normalization layer type.
        activation: Activation function type.
        **kwargs: Additional arguments (ignored).

    Example:
        >>> model = UNet(num_classes=10, encoder_depth=4)
        >>> x = torch.randn(1, 3, 256, 256)
        >>> out = model(x)
        >>> out.shape
        torch.Size([1, 10, 256, 256])
    """

    def __init__(
        self,
        architecture: str = "unet",
        task: Literal["binary", "multiclass", "regression"] = "multiclass",
        num_classes: int = 1,
        in_channels: int = 3,
        encoder_depth: int = 5,
        decoder_channels: list[int] | None = None,
        dropout: float = 0.0,
        norm_layer: str = "batchnorm",
        activation: str = "relu",
        **kwargs,
    ):
        super().__init__()

        self.task = task
        self.num_classes = num_classes
        self.encoder_depth = encoder_depth

        # Default decoder channels
        if decoder_channels is None:
            decoder_channels = [256, 128, 64, 32, 16]

        # Ensure decoder channels match encoder depth
        decoder_channels = decoder_channels[: encoder_depth - 1]

        # Encoder channel progression: 64 -> 128 -> 256 -> 512 -> 1024
        encoder_channels = [64 * (2**i) for i in range(encoder_depth)]

        # Build encoder
        self.encoders = nn.ModuleList()
        in_ch = in_channels
        for out_ch in encoder_channels[:-1]:
            self.encoders.append(
                EncoderBlock(in_ch, out_ch, norm_layer, activation, dropout)
            )
            in_ch = out_ch

        # Bottleneck
        self.bottleneck = ConvBlock(
            encoder_channels[-2], encoder_channels[-1], norm_layer, activation, dropout
        )

        # Build decoder
        self.decoders = nn.ModuleList()
        in_ch = encoder_channels[-1]
        for i, out_ch in enumerate(decoder_channels):
            skip_ch = encoder_channels[-(i + 2)]
            self.decoders.append(
                DecoderBlock(in_ch, skip_ch, out_ch, norm_layer, activation, dropout)
            )
            in_ch = out_ch

        # Output head
        self.head = nn.Conv2d(decoder_channels[-1], num_classes, kernel_size=1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Output tensor of shape (B, num_classes, H, W).
        """
        # Encoder
        skips = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        # Output
        x = self.head(x)

        return x

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
