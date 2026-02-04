"""
Model architectures for Altair.

This module provides segmentation model architectures including:
- UNet: Classic U-Net architecture
- UNet++: Nested U-Net with dense skip connections
- UNetEncoder: U-Net decoder with pretrained timm encoders

Example:
    >>> from altair.models import build_model
    >>> model = build_model(config.model)

Available architectures:
    - unet: Classic UNet
    - unet++: UNet++ with dense skip connections
    - unet_encoder, unet_resnet: UNet with timm encoder
    - unet_resnet18, unet_resnet34, unet_resnet50, unet_resnet101
    - unet_efficientnet_b0, unet_efficientnet_b4
    - unet_convnext_tiny
    - unet_mobilenet
"""

from altair.core.registry import MODELS

# Import submodules to trigger registration
from altair.models import backbones, decoders, segmentors
from altair.models.backbones import TimmEncoder, get_encoder, list_encoders
from altair.models.segmentors import UNet, UNetEncoder, UNetPlusPlus

__all__ = [
    "MODELS",
    "build_model",
    "UNet",
    "UNetPlusPlus",
    "UNetEncoder",
    "TimmEncoder",
    "get_encoder",
    "list_encoders",
]


def build_model(config):
    """
    Build a model from configuration.

    Args:
        config: ModelConfig instance, dict with model configuration,
            or any object with a model_dump() method.

    Returns:
        nn.Module: The constructed model.

    Example:
        >>> model = build_model({"architecture": "unet", "num_classes": 10})
        >>> model = build_model({"architecture": "unet_resnet50", "num_classes": 19})
    """
    if hasattr(config, "model_dump"):
        config = config.model_dump()
    elif hasattr(config, "dict"):
        config = config.dict()

    architecture = config.get("architecture", "unet")

    # Handle encoder-based architectures
    if architecture in ("unet_encoder", "unet_resnet") and "encoder" not in config:
        config["encoder"] = "resnet50"  # Default encoder

    return MODELS.build(architecture, **config)


def list_available_models() -> list[str]:
    """
    List all available model architectures.

    Returns:
        List of registered model names.
    """
    return MODELS.registered_names
