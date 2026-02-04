"""
UNet++ (Nested UNet) segmentation model implementation.

UNet++ uses dense skip connections and deep supervision to improve
segmentation performance.

Reference:
    Zhou et al., "UNet++: A Nested U-Net Architecture for Medical Image
    Segmentation", DLMIA 2018.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from altair.core.registry import MODELS
from altair.models.segmentors.unet import ConvBlock


class NestedDecoderBlock(nn.Module):
    """
    Nested decoder block for UNet++.

    Combines multiple skip connections with upsampled features.

    Args:
        in_channels: Number of input channels (from previous level).
        skip_channels: Number of channels from each skip connection.
        out_channels: Number of output channels.
        num_skips: Number of skip connections to concatenate.
        norm_layer: Normalization layer type.
        activation: Activation function type.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        num_skips: int = 1,
        norm_layer: str = "batchnorm",
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        total_in_channels = in_channels + skip_channels * num_skips
        self.conv = ConvBlock(total_in_channels, out_channels, norm_layer, activation, dropout)

    def forward(self, x: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor from previous decoder level.
            skips: List of skip connection tensors.

        Returns:
            Output tensor.
        """
        x = self.upsample(x)

        # Handle size mismatch
        target_size = skips[0].shape[2:]
        if x.shape[2:] != target_size:
            x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)

        x = torch.cat([x] + skips, dim=1)
        return self.conv(x)


@MODELS.register("unet++")
class UNetPlusPlus(nn.Module):
    """
    UNet++ (Nested UNet) architecture for image segmentation.

    UNet++ introduces dense skip connections between encoder and decoder,
    allowing the network to capture features at multiple semantic scales.

    Args:
        architecture: Architecture name (unused, for registry compatibility).
        task: Task type ('binary', 'multiclass', 'regression').
        num_classes: Number of output classes/channels.
        in_channels: Number of input channels (default: 3 for RGB).
        encoder_depth: Number of encoder stages (3-5).
        decoder_channels: List of decoder channel sizes.
        dropout: Dropout probability.
        norm_layer: Normalization layer type.
        activation: Activation function type.
        deep_supervision: Whether to use deep supervision during training.
        **kwargs: Additional arguments (ignored).

    Example:
        >>> model = UNetPlusPlus(num_classes=10, encoder_depth=4)
        >>> x = torch.randn(1, 3, 256, 256)
        >>> out = model(x)
        >>> out.shape
        torch.Size([1, 10, 256, 256])
    """

    def __init__(
        self,
        architecture: str = "unet++",
        task: Literal["binary", "multiclass", "regression"] = "multiclass",
        num_classes: int = 1,
        in_channels: int = 3,
        encoder_depth: int = 4,
        decoder_channels: list[int] | None = None,
        dropout: float = 0.0,
        norm_layer: str = "batchnorm",
        activation: str = "relu",
        deep_supervision: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.task = task
        self.num_classes = num_classes
        self.encoder_depth = encoder_depth
        self.deep_supervision = deep_supervision

        # Default decoder channels
        if decoder_channels is None:
            decoder_channels = [64, 128, 256, 512, 1024]

        # Use only as many channels as encoder depth
        channels = decoder_channels[:encoder_depth]

        # Store for reference
        self.channels = channels

        # Build encoder path (x_i,0 nodes)
        self.encoder_blocks = nn.ModuleList()
        self.pools = nn.ModuleList()

        prev_ch = in_channels
        for ch in channels:
            self.encoder_blocks.append(
                ConvBlock(prev_ch, ch, norm_layer, activation, dropout)
            )
            self.pools.append(nn.MaxPool2d(2))
            prev_ch = ch

        # Build nested decoder blocks
        # x_i,j where i is depth (row) and j is column
        # We need blocks for j >= 1
        self.nested_blocks = nn.ModuleDict()

        for j in range(1, encoder_depth):
            for i in range(encoder_depth - j):
                # Number of skip connections from same row
                num_skips = j

                # Input channels: from x_{i+1, j-1}
                in_ch = channels[i + 1] if j == 1 else channels[i]

                # Skip channels: each from x_{i, k} for k < j
                skip_ch = channels[i]

                # Output channels
                out_ch = channels[i]

                key = f"x_{i}_{j}"
                self.nested_blocks[key] = NestedDecoderBlock(
                    in_channels=in_ch,
                    skip_channels=skip_ch,
                    out_channels=out_ch,
                    num_skips=num_skips,
                    norm_layer=norm_layer,
                    activation=activation,
                    dropout=dropout,
                )

        # Output heads
        if deep_supervision:
            self.heads = nn.ModuleList([
                nn.Conv2d(channels[0], num_classes, kernel_size=1)
                for _ in range(encoder_depth - 1)
            ])
        else:
            self.head = nn.Conv2d(channels[0], num_classes, kernel_size=1)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            If deep_supervision is False: Output tensor of shape (B, num_classes, H, W).
            If deep_supervision is True: List of output tensors from each decoder column.
        """
        # Storage for all x_i,j nodes
        # x[i][j] = feature map at row i, column j
        features = [[None] * self.encoder_depth for _ in range(self.encoder_depth)]

        # Encoder path (column 0)
        for i in range(self.encoder_depth):
            if i == 0:
                features[i][0] = self.encoder_blocks[i](x)
            else:
                pooled = self.pools[i - 1](features[i - 1][0])
                features[i][0] = self.encoder_blocks[i](pooled)

        # Nested decoder path
        for j in range(1, self.encoder_depth):
            for i in range(self.encoder_depth - j):
                key = f"x_{i}_{j}"

                # Get input from previous level
                if j == 1:
                    # From encoder at i+1
                    input_features = features[i + 1][0]
                else:
                    # From decoder at i+1, j-1
                    input_features = features[i + 1][j - 1]

                # Collect skip connections from same row, previous columns
                skips = [features[i][k] for k in range(j)]

                features[i][j] = self.nested_blocks[key](input_features, skips)

        # Output
        if self.deep_supervision and self.training:
            # Return outputs from all columns at row 0
            outputs = []
            for j in range(1, self.encoder_depth):
                out = self.heads[j - 1](features[0][j])
                outputs.append(out)
            return outputs
        else:
            # Return final output (top-right corner)
            if self.deep_supervision:
                return self.heads[-1](features[0][self.encoder_depth - 1])
            else:
                return self.head(features[0][self.encoder_depth - 1])

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
