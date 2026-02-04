"""
Segmentation model implementations.

Provides complete segmentation models including:
- UNet: Classic U-Net architecture
- UNet++: Nested U-Net with dense skip connections
- UNetEncoder: U-Net decoder with pretrained timm encoders
"""

from altair.models.segmentors.unet import UNet
from altair.models.segmentors.unet_plusplus import UNetPlusPlus
from altair.models.segmentors.unet_encoder import UNetEncoder

__all__ = ["UNet", "UNetPlusPlus", "UNetEncoder"]
