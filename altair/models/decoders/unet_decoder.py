"""
UNet decoder for encoder-decoder segmentation models.

Works with any encoder that provides multi-scale features.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    """Convolution + BatchNorm + ReLU block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        use_batchnorm: bool = True,
    ):
        super().__init__()

        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=not use_batchnorm,
            ),
        ]

        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))

        layers.append(nn.ReLU(inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SCSEModule(nn.Module):
    """
    Concurrent Spatial and Channel Squeeze & Excitation.

    Reference:
        Roy et al., "Concurrent Spatial and Channel Squeeze & Excitation
        in Fully Convolutional Networks", MICCAI 2018.
    """

    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()

        # Channel squeeze-excitation
        self.cse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )

        # Spatial squeeze-excitation
        self.sse = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cse = self.cse(x) * x
        sse = self.sse(x) * x
        return cse + sse


class DecoderBlock(nn.Module):
    """
    Single decoder block for UNet decoder.

    Upsamples input, concatenates with skip connection, and applies
    two convolution blocks.

    Args:
        in_channels: Number of input channels (from previous decoder block).
        skip_channels: Number of channels in skip connection.
        out_channels: Number of output channels.
        use_batchnorm: Whether to use batch normalization.
        attention_type: Type of attention ('scse' or None).
        dropout: Dropout probability.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_batchnorm: bool = True,
        attention_type: str | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.conv1 = ConvBNReLU(
            in_channels + skip_channels,
            out_channels,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = ConvBNReLU(
            out_channels,
            out_channels,
            use_batchnorm=use_batchnorm,
        )

        # Attention
        if attention_type == "scse":
            self.attention = SCSEModule(out_channels)
        else:
            self.attention = nn.Identity()

        # Dropout
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor from previous decoder block.
            skip: Skip connection tensor from encoder.

        Returns:
            Output tensor.
        """
        # Upsample
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        # Concatenate with skip connection
        if skip is not None:
            # Handle size mismatch
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=False
                )
            x = torch.cat([x, skip], dim=1)

        # Convolutions
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention(x)
        x = self.dropout(x)

        return x


class CenterBlock(nn.Module):
    """
    Center block between encoder and decoder.

    Optional block that processes the deepest encoder features.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batchnorm: bool = True,
    ):
        super().__init__()

        self.block = nn.Sequential(
            ConvBNReLU(in_channels, out_channels, use_batchnorm=use_batchnorm),
            ConvBNReLU(out_channels, out_channels, use_batchnorm=use_batchnorm),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetDecoder(nn.Module):
    """
    UNet decoder that works with encoder features.

    Takes multi-scale features from an encoder and progressively upsamples
    them while merging with skip connections.

    Args:
        encoder_channels: List of encoder output channels (low to high resolution).
        decoder_channels: List of decoder output channels.
        use_batchnorm: Whether to use batch normalization.
        attention_type: Type of attention in decoder blocks.
        dropout: Dropout probability in decoder blocks.
        center: Whether to use center block.

    Example:
        >>> decoder = UNetDecoder(
        ...     encoder_channels=[64, 256, 512, 1024, 2048],
        ...     decoder_channels=[256, 128, 64, 32, 16],
        ... )
        >>> features = [f1, f2, f3, f4, f5]  # From encoder
        >>> x = decoder(*features)
    """

    def __init__(
        self,
        encoder_channels: list[int],
        decoder_channels: list[int],
        use_batchnorm: bool = True,
        attention_type: str | None = None,
        dropout: float = 0.0,
        center: bool = False,
    ):
        super().__init__()

        # Reverse encoder channels (decoder goes from deep to shallow)
        encoder_channels = encoder_channels[::-1]

        # Center block
        if center:
            self.center = CenterBlock(
                encoder_channels[0],
                encoder_channels[0],
                use_batchnorm=use_batchnorm,
            )
        else:
            self.center = nn.Identity()

        # Calculate input channels for each decoder block
        # First block: encoder_channels[0] (deepest) + skip from encoder_channels[1]
        in_channels = [encoder_channels[0]] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]

        # Ensure we have the right number of channels
        num_blocks = len(decoder_channels)
        in_channels = in_channels[:num_blocks]
        skip_channels = skip_channels[:num_blocks]

        # Build decoder blocks
        self.blocks = nn.ModuleList()
        for i, (in_ch, skip_ch, out_ch) in enumerate(
            zip(in_channels, skip_channels, decoder_channels)
        ):
            self.blocks.append(
                DecoderBlock(
                    in_channels=in_ch,
                    skip_channels=skip_ch,
                    out_channels=out_ch,
                    use_batchnorm=use_batchnorm,
                    attention_type=attention_type,
                    dropout=dropout,
                )
            )

    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            *features: Encoder features from low to high resolution.
                features[0] = highest resolution (shallowest)
                features[-1] = lowest resolution (deepest)

        Returns:
            Decoded feature tensor.
        """
        # Reverse features (process from deep to shallow)
        features = features[::-1]

        # Start with deepest features
        x = self.center(features[0])

        # Decode with skip connections
        skips = list(features[1:]) + [None]
        for block, skip in zip(self.blocks, skips):
            x = block(x, skip)

        return x
