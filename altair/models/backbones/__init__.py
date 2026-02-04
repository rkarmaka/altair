"""
Backbone encoder implementations.

Provides encoder backbones for encoder-decoder segmentation models.
Currently supports timm-based encoders with pretrained weights.
"""

from altair.models.backbones.timm_encoder import (
    ENCODER_CONFIGS,
    TimmEncoder,
    get_encoder,
    list_encoders,
)

__all__ = [
    "TimmEncoder",
    "get_encoder",
    "list_encoders",
    "ENCODER_CONFIGS",
]
