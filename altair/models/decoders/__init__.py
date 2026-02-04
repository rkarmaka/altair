"""
Decoder implementations for segmentation models.

Provides decoder modules that work with encoder features to produce
segmentation outputs.
"""

from altair.models.decoders.unet_decoder import (
    CenterBlock,
    DecoderBlock,
    SCSEModule,
    UNetDecoder,
)

__all__ = [
    "UNetDecoder",
    "DecoderBlock",
    "CenterBlock",
    "SCSEModule",
]
