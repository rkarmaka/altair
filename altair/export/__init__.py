"""
Model export utilities for deployment.

Supports exporting models to ONNX and TorchScript formats for inference
in production environments.

Example:
    >>> from altair.export import export_onnx, export_torchscript
    >>> export_onnx(model, "model.onnx", input_shape=(1, 3, 512, 512))
    >>> export_torchscript(model, "model.pt", input_shape=(1, 3, 512, 512))
"""

from altair.export.exporter import (
    ModelExporter,
    export_onnx,
    export_torchscript,
    validate_onnx,
    validate_torchscript,
)

__all__ = [
    "ModelExporter",
    "export_onnx",
    "export_torchscript",
    "validate_onnx",
    "validate_torchscript",
]
