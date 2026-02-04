"""
Model exporter for ONNX and TorchScript formats.

Provides utilities for exporting trained segmentation models to formats
suitable for deployment in production environments.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from altair.utils.console import console, print_info, print_success, print_warning, status_spinner

logger = logging.getLogger(__name__)


@dataclass
class ExportConfig:
    """
    Configuration for model export.

    Attributes:
        input_shape: Input tensor shape (N, C, H, W).
        opset_version: ONNX opset version.
        dynamic_axes: Dynamic axes for ONNX export.
        simplify: Whether to simplify ONNX model.
        half: Whether to export in FP16.
        optimize: Whether to optimize TorchScript model.
    """

    input_shape: tuple[int, int, int, int] = (1, 3, 512, 512)
    opset_version: int = 17
    dynamic_axes: dict[str, dict[int, str]] | None = None
    simplify: bool = True
    half: bool = False
    optimize: bool = True


@dataclass
class ExportResult:
    """
    Result of model export.

    Attributes:
        path: Path to exported model.
        format: Export format ('onnx' or 'torchscript').
        input_shape: Input shape used for export.
        file_size_mb: File size in megabytes.
        metadata: Additional export metadata.
    """

    path: Path
    format: str
    input_shape: tuple[int, int, int, int]
    file_size_mb: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"ExportResult(format={self.format}, "
            f"path={self.path}, size={self.file_size_mb:.2f}MB)"
        )


class ModelExporter:
    """
    Export segmentation models to deployment formats.

    Supports ONNX and TorchScript export with validation.

    Args:
        model: PyTorch model to export.
        input_shape: Input tensor shape (N, C, H, W).
        device: Device to use for export.

    Example:
        >>> exporter = ModelExporter(model, input_shape=(1, 3, 512, 512))
        >>> result = exporter.to_onnx("model.onnx")
        >>> print(f"Exported to {result.path} ({result.file_size_mb:.2f}MB)")
    """

    def __init__(
        self,
        model: nn.Module,
        input_shape: tuple[int, int, int, int] = (1, 3, 512, 512),
        device: str | None = None,
    ):
        self.model = model
        self.input_shape = input_shape

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Prepare model
        self.model = self.model.to(self.device)
        self.model.eval()

    def to_onnx(
        self,
        output_path: str | Path,
        opset_version: int = 17,
        dynamic_axes: dict[str, dict[int, str]] | None = None,
        simplify: bool = True,
        validate: bool = True,
        input_names: list[str] | None = None,
        output_names: list[str] | None = None,
    ) -> ExportResult:
        """
        Export model to ONNX format.

        Args:
            output_path: Output file path.
            opset_version: ONNX opset version (default: 17).
            dynamic_axes: Dynamic axes for variable-size inputs.
            simplify: Whether to simplify the ONNX graph.
            validate: Whether to validate the exported model.
            input_names: Names for input tensors.
            output_names: Names for output tensors.

        Returns:
            ExportResult with export details.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Default names
        if input_names is None:
            input_names = ["input"]
        if output_names is None:
            output_names = ["output"]

        # Default dynamic axes for batch size and spatial dimensions
        if dynamic_axes is None:
            dynamic_axes = {
                "input": {0: "batch", 2: "height", 3: "width"},
                "output": {0: "batch", 2: "height", 3: "width"},
            }

        # Create dummy input
        dummy_input = torch.randn(self.input_shape, device=self.device)

        logger.info(f"Exporting model to ONNX format (opset {opset_version})")
        print_info(f"Exporting to ONNX (opset {opset_version})...")

        # Export
        torch.onnx.export(
            self.model,
            dummy_input,
            str(output_path),
            opset_version=opset_version,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )

        # Simplify if requested
        if simplify:
            self._simplify_onnx(output_path)

        # Validate if requested
        if validate:
            self._validate_onnx(output_path, dummy_input)

        # Get file size
        file_size_mb = output_path.stat().st_size / (1024 * 1024)

        logger.info(f"ONNX export complete: {output_path} ({file_size_mb:.2f} MB)")
        print_success(f"ONNX export complete: {file_size_mb:.2f} MB")

        return ExportResult(
            path=output_path,
            format="onnx",
            input_shape=self.input_shape,
            file_size_mb=file_size_mb,
            metadata={
                "opset_version": opset_version,
                "dynamic_axes": dynamic_axes,
                "simplified": simplify,
            },
        )

    def _simplify_onnx(self, path: Path) -> None:
        """Simplify ONNX model graph."""
        try:
            import onnx
            from onnxsim import simplify

            model = onnx.load(str(path))
            model_simp, check = simplify(model)

            if check:
                onnx.save(model_simp, str(path))
                logger.info("ONNX model simplified successfully")
                print_success("ONNX model simplified")
            else:
                logger.warning("ONNX simplification check failed, using original model")
                print_warning("ONNX simplification check failed, using original")

        except ImportError:
            logger.warning("onnxsim not installed, skipping simplification")
            print_warning(
                "onnxsim not installed, skipping simplification. "
                "Install with: pip install onnxsim"
            )
        except Exception as e:
            logger.warning(f"ONNX simplification failed: {e}")
            print_warning(f"ONNX simplification failed: {e}")

    def _validate_onnx(self, path: Path, dummy_input: torch.Tensor) -> None:
        """Validate ONNX model output."""
        try:
            import onnx
            import onnxruntime as ort

            # Check model
            model = onnx.load(str(path))
            onnx.checker.check_model(model)
            logger.info("ONNX model structure validated")
            print_success("ONNX model structure validated")

            # Compare outputs
            self.model.eval()
            with torch.no_grad():
                torch_output = self.model(dummy_input).cpu().numpy()

            # Run ONNX inference
            providers = ["CPUExecutionProvider"]
            if self.device.type == "cuda":
                providers.insert(0, "CUDAExecutionProvider")

            session = ort.InferenceSession(str(path), providers=providers)
            onnx_output = session.run(
                None,
                {"input": dummy_input.cpu().numpy()},
            )[0]

            # Compare
            if np.allclose(torch_output, onnx_output, rtol=1e-3, atol=1e-5):
                logger.info("ONNX output validation passed")
                print_success("ONNX output validation passed")
            else:
                max_diff = np.abs(torch_output - onnx_output).max()
                logger.warning(f"ONNX output differs from PyTorch (max diff: {max_diff:.6f})")
                print_warning(f"ONNX output differs from PyTorch (max diff: {max_diff:.6f})")

        except ImportError as e:
            logger.warning(f"ONNX validation skipped - missing dependency: {e}")
            print_warning(f"Validation skipped - missing dependency: {e}")
        except Exception as e:
            logger.error(f"ONNX validation failed: {e}")
            print_warning(f"ONNX validation failed: {e}")

    def to_torchscript(
        self,
        output_path: str | Path,
        method: str = "trace",
        optimize: bool = True,
        validate: bool = True,
    ) -> ExportResult:
        """
        Export model to TorchScript format.

        Args:
            output_path: Output file path.
            method: Export method ('trace' or 'script').
            optimize: Whether to optimize for inference.
            validate: Whether to validate the exported model.

        Returns:
            ExportResult with export details.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create dummy input
        dummy_input = torch.randn(self.input_shape, device=self.device)

        logger.info(f"Exporting model to TorchScript format (method={method})")
        print_info(f"Exporting to TorchScript ({method})...")

        self.model.eval()

        if method == "trace":
            with torch.no_grad():
                scripted = torch.jit.trace(self.model, dummy_input)
        elif method == "script":
            scripted = torch.jit.script(self.model)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'trace' or 'script'.")

        # Optimize if requested
        if optimize:
            scripted = torch.jit.optimize_for_inference(scripted)
            logger.info("TorchScript model optimized for inference")
            print_success("TorchScript model optimized for inference")

        # Save
        scripted.save(str(output_path))

        # Validate if requested
        if validate:
            self._validate_torchscript(output_path, dummy_input)

        # Get file size
        file_size_mb = output_path.stat().st_size / (1024 * 1024)

        logger.info(f"TorchScript export complete: {output_path} ({file_size_mb:.2f} MB)")
        print_success(f"TorchScript export complete: {file_size_mb:.2f} MB")

        return ExportResult(
            path=output_path,
            format="torchscript",
            input_shape=self.input_shape,
            file_size_mb=file_size_mb,
            metadata={
                "method": method,
                "optimized": optimize,
            },
        )

    def _validate_torchscript(self, path: Path, dummy_input: torch.Tensor) -> None:
        """Validate TorchScript model output."""
        try:
            # Load scripted model
            scripted = torch.jit.load(str(path), map_location=self.device)
            scripted.eval()

            # Compare outputs
            self.model.eval()
            with torch.no_grad():
                torch_output = self.model(dummy_input)
                script_output = scripted(dummy_input)

            # Compare
            if torch.allclose(torch_output, script_output, rtol=1e-3, atol=1e-5):
                logger.info("TorchScript output validation passed")
                print_success("TorchScript output validation passed")
            else:
                max_diff = (torch_output - script_output).abs().max().item()
                logger.warning(f"TorchScript output differs from PyTorch (max diff: {max_diff:.6f})")
                print_warning(
                    f"TorchScript output differs from PyTorch (max diff: {max_diff:.6f})"
                )

        except Exception as e:
            logger.error(f"TorchScript validation failed: {e}")
            print_warning(f"TorchScript validation failed: {e}")

    def to_half(self) -> "ModelExporter":
        """
        Convert model to FP16 for export.

        Returns:
            Self for method chaining.
        """
        self.model = self.model.half()
        logger.info("Model converted to FP16")
        return self

    def export_all(
        self,
        output_dir: str | Path,
        name: str = "model",
        formats: list[str] | None = None,
        **kwargs,
    ) -> dict[str, ExportResult]:
        """
        Export model to multiple formats.

        Args:
            output_dir: Output directory.
            name: Base name for exported files.
            formats: List of formats to export ('onnx', 'torchscript').
            **kwargs: Additional arguments passed to export functions.

        Returns:
            Dictionary mapping format to ExportResult.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if formats is None:
            formats = ["onnx", "torchscript"]

        results = {}

        for fmt in formats:
            if fmt == "onnx":
                path = output_dir / f"{name}.onnx"
                results["onnx"] = self.to_onnx(path, **kwargs)
            elif fmt == "torchscript":
                path = output_dir / f"{name}.pt"
                results["torchscript"] = self.to_torchscript(path, **kwargs)
            else:
                logger.warning(f"Unknown format: {fmt}")

        return results


def export_onnx(
    model: nn.Module,
    output_path: str | Path,
    input_shape: tuple[int, int, int, int] = (1, 3, 512, 512),
    opset_version: int = 17,
    dynamic_axes: dict[str, dict[int, str]] | None = None,
    simplify: bool = True,
    validate: bool = True,
    device: str | None = None,
) -> ExportResult:
    """
    Export a model to ONNX format.

    Convenience function for quick ONNX export.

    Args:
        model: PyTorch model to export.
        output_path: Output file path.
        input_shape: Input tensor shape (N, C, H, W).
        opset_version: ONNX opset version.
        dynamic_axes: Dynamic axes for variable-size inputs.
        simplify: Whether to simplify the ONNX graph.
        validate: Whether to validate the exported model.
        device: Device to use for export.

    Returns:
        ExportResult with export details.

    Example:
        >>> result = export_onnx(model, "model.onnx", input_shape=(1, 3, 512, 512))
        >>> print(f"Exported to {result.path}")
    """
    exporter = ModelExporter(model, input_shape=input_shape, device=device)
    return exporter.to_onnx(
        output_path,
        opset_version=opset_version,
        dynamic_axes=dynamic_axes,
        simplify=simplify,
        validate=validate,
    )


def export_torchscript(
    model: nn.Module,
    output_path: str | Path,
    input_shape: tuple[int, int, int, int] = (1, 3, 512, 512),
    method: str = "trace",
    optimize: bool = True,
    validate: bool = True,
    device: str | None = None,
) -> ExportResult:
    """
    Export a model to TorchScript format.

    Convenience function for quick TorchScript export.

    Args:
        model: PyTorch model to export.
        output_path: Output file path.
        input_shape: Input tensor shape (N, C, H, W).
        method: Export method ('trace' or 'script').
        optimize: Whether to optimize for inference.
        validate: Whether to validate the exported model.
        device: Device to use for export.

    Returns:
        ExportResult with export details.

    Example:
        >>> result = export_torchscript(model, "model.pt")
        >>> print(f"Exported to {result.path}")
    """
    exporter = ModelExporter(model, input_shape=input_shape, device=device)
    return exporter.to_torchscript(
        output_path,
        method=method,
        optimize=optimize,
        validate=validate,
    )


def validate_onnx(
    model_path: str | Path,
    input_shape: tuple[int, int, int, int] = (1, 3, 512, 512),
) -> dict[str, Any]:
    """
    Validate an ONNX model.

    Args:
        model_path: Path to ONNX model.
        input_shape: Input shape for testing.

    Returns:
        Dictionary with validation results.

    Example:
        >>> results = validate_onnx("model.onnx")
        >>> print(f"Valid: {results['valid']}")
    """
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        raise ImportError("onnx and onnxruntime are required for validation")

    model_path = Path(model_path)
    results = {
        "path": str(model_path),
        "valid": False,
        "errors": [],
        "warnings": [],
    }

    # Load and check model
    try:
        model = onnx.load(str(model_path))
        onnx.checker.check_model(model)
        results["valid"] = True
    except Exception as e:
        results["errors"].append(f"Model check failed: {e}")
        return results

    # Get model info
    results["opset_version"] = model.opset_import[0].version
    results["ir_version"] = model.ir_version

    # Get input/output info
    results["inputs"] = []
    for inp in model.graph.input:
        shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        results["inputs"].append({"name": inp.name, "shape": shape})

    results["outputs"] = []
    for out in model.graph.output:
        shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
        results["outputs"].append({"name": out.name, "shape": shape})

    # Test inference
    try:
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        output = session.run(None, {"input": dummy_input})
        results["output_shape"] = list(output[0].shape)
        results["inference_ok"] = True
    except Exception as e:
        results["inference_ok"] = False
        results["warnings"].append(f"Inference test failed: {e}")

    # File size
    results["file_size_mb"] = model_path.stat().st_size / (1024 * 1024)

    return results


def validate_torchscript(
    model_path: str | Path,
    input_shape: tuple[int, int, int, int] = (1, 3, 512, 512),
    device: str = "cpu",
) -> dict[str, Any]:
    """
    Validate a TorchScript model.

    Args:
        model_path: Path to TorchScript model.
        input_shape: Input shape for testing.
        device: Device to use for testing.

    Returns:
        Dictionary with validation results.

    Example:
        >>> results = validate_torchscript("model.pt")
        >>> print(f"Valid: {results['valid']}")
    """
    model_path = Path(model_path)
    results = {
        "path": str(model_path),
        "valid": False,
        "errors": [],
        "warnings": [],
    }

    # Load model
    try:
        model = torch.jit.load(str(model_path), map_location=device)
        model.eval()
        results["valid"] = True
    except Exception as e:
        results["errors"].append(f"Model load failed: {e}")
        return results

    # Test inference
    try:
        dummy_input = torch.randn(input_shape, device=device)
        with torch.no_grad():
            output = model(dummy_input)
        results["output_shape"] = list(output.shape)
        results["inference_ok"] = True
    except Exception as e:
        results["inference_ok"] = False
        results["warnings"].append(f"Inference test failed: {e}")

    # File size
    results["file_size_mb"] = model_path.stat().st_size / (1024 * 1024)

    return results


class ONNXInferenceSession:
    """
    ONNX inference session wrapper.

    Provides a simple interface for running inference on ONNX models.

    Args:
        model_path: Path to ONNX model.
        providers: Execution providers (e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider']).

    Example:
        >>> session = ONNXInferenceSession("model.onnx")
        >>> output = session(input_tensor)
    """

    def __init__(
        self,
        model_path: str | Path,
        providers: list[str] | None = None,
    ):
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime is required. Install with: pip install onnxruntime")

        if providers is None:
            providers = ["CPUExecutionProvider"]
            # Try to add CUDA provider
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers.insert(0, "CUDAExecutionProvider")

        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def __call__(self, input_tensor: np.ndarray | torch.Tensor) -> np.ndarray:
        """
        Run inference on input tensor.

        Args:
            input_tensor: Input array/tensor of shape (N, C, H, W).

        Returns:
            Output array.
        """
        if isinstance(input_tensor, torch.Tensor):
            input_tensor = input_tensor.cpu().numpy()

        if input_tensor.dtype != np.float32:
            input_tensor = input_tensor.astype(np.float32)

        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        return outputs[0]

    @property
    def input_shape(self) -> list[int | str]:
        """Get expected input shape."""
        return [d.dim_value if hasattr(d, "dim_value") else d.dim_param
                for d in self.session.get_inputs()[0].shape]

    @property
    def output_shape(self) -> list[int | str]:
        """Get expected output shape."""
        return [d.dim_value if hasattr(d, "dim_value") else d.dim_param
                for d in self.session.get_outputs()[0].shape]


class TorchScriptInferenceSession:
    """
    TorchScript inference session wrapper.

    Provides a simple interface for running inference on TorchScript models.

    Args:
        model_path: Path to TorchScript model.
        device: Device to use for inference.

    Example:
        >>> session = TorchScriptInferenceSession("model.pt", device="cuda")
        >>> output = session(input_tensor)
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.model = torch.jit.load(str(model_path), map_location=self.device)
        self.model.eval()

    def __call__(self, input_tensor: np.ndarray | torch.Tensor) -> torch.Tensor:
        """
        Run inference on input tensor.

        Args:
            input_tensor: Input array/tensor of shape (N, C, H, W).

        Returns:
            Output tensor.
        """
        if isinstance(input_tensor, np.ndarray):
            input_tensor = torch.from_numpy(input_tensor)

        input_tensor = input_tensor.to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)

        return output

    def to(self, device: str) -> "TorchScriptInferenceSession":
        """Move model to device."""
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        return self
