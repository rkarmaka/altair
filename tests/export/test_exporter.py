"""Tests for model exporter."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from altair.export.exporter import (
    ExportConfig,
    ExportResult,
    ModelExporter,
    export_onnx,
    export_torchscript,
    validate_torchscript,
)


# ============================================================================
# Fixtures
# ============================================================================


class SimpleConvNet(nn.Module):
    """Simple CNN for testing export."""

    def __init__(self, in_channels: int = 3, num_classes: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, num_classes, 1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


@pytest.fixture
def simple_model():
    """Create a simple model for testing."""
    model = SimpleConvNet(in_channels=3, num_classes=2)
    model.eval()
    return model


@pytest.fixture
def input_shape():
    """Default input shape."""
    return (1, 3, 64, 64)


@pytest.fixture
def dummy_input(input_shape):
    """Create dummy input tensor."""
    return torch.randn(input_shape)


# ============================================================================
# ExportConfig Tests
# ============================================================================


class TestExportConfig:
    """Tests for ExportConfig dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = ExportConfig()

        assert config.input_shape == (1, 3, 512, 512)
        assert config.opset_version == 17
        assert config.simplify is True
        assert config.half is False
        assert config.optimize is True

    def test_custom_values(self):
        """Should accept custom values."""
        config = ExportConfig(
            input_shape=(2, 3, 256, 256),
            opset_version=14,
            half=True,
        )

        assert config.input_shape == (2, 3, 256, 256)
        assert config.opset_version == 14
        assert config.half is True


# ============================================================================
# ExportResult Tests
# ============================================================================


class TestExportResult:
    """Tests for ExportResult dataclass."""

    def test_str_representation(self):
        """Should have readable string representation."""
        result = ExportResult(
            path=Path("model.onnx"),
            format="onnx",
            input_shape=(1, 3, 512, 512),
            file_size_mb=15.5,
        )

        result_str = str(result)
        assert "onnx" in result_str
        assert "model.onnx" in result_str
        assert "15.50MB" in result_str

    def test_metadata(self):
        """Should store metadata."""
        result = ExportResult(
            path=Path("model.pt"),
            format="torchscript",
            input_shape=(1, 3, 256, 256),
            file_size_mb=10.0,
            metadata={"method": "trace", "optimized": True},
        )

        assert result.metadata["method"] == "trace"
        assert result.metadata["optimized"] is True


# ============================================================================
# ModelExporter Tests
# ============================================================================


class TestModelExporter:
    """Tests for ModelExporter class."""

    def test_init(self, simple_model, input_shape):
        """Should initialize correctly."""
        exporter = ModelExporter(simple_model, input_shape=input_shape)

        assert exporter.input_shape == input_shape
        assert exporter.model is not None

    def test_init_default_device(self, simple_model):
        """Should use default device."""
        exporter = ModelExporter(simple_model)

        assert exporter.device is not None

    def test_model_in_eval_mode(self, simple_model, input_shape):
        """Model should be in eval mode after init."""
        simple_model.train()
        exporter = ModelExporter(simple_model, input_shape=input_shape)

        assert not exporter.model.training


class TestTorchScriptExport:
    """Tests for TorchScript export."""

    def test_basic_export_trace(self, simple_model, input_shape):
        """Should export using trace method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.pt"

            exporter = ModelExporter(simple_model, input_shape=input_shape, device="cpu")
            result = exporter.to_torchscript(output_path, method="trace", validate=False)

            assert result.path.exists()
            assert result.format == "torchscript"
            assert result.file_size_mb > 0

    def test_basic_export_script(self, simple_model, input_shape):
        """Should export using script method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.pt"

            exporter = ModelExporter(simple_model, input_shape=input_shape, device="cpu")
            result = exporter.to_torchscript(output_path, method="script", validate=False)

            assert result.path.exists()
            assert result.format == "torchscript"

    def test_export_with_validation(self, simple_model, input_shape):
        """Should validate exported model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.pt"

            exporter = ModelExporter(simple_model, input_shape=input_shape, device="cpu")
            result = exporter.to_torchscript(output_path, validate=True)

            assert result.path.exists()

    def test_export_with_optimization(self, simple_model, input_shape):
        """Should optimize for inference."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.pt"

            exporter = ModelExporter(simple_model, input_shape=input_shape, device="cpu")
            result = exporter.to_torchscript(output_path, optimize=True)

            assert result.path.exists()
            assert result.metadata["optimized"] is True

    def test_export_creates_directory(self, simple_model, input_shape):
        """Should create parent directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "nested" / "dir" / "model.pt"

            exporter = ModelExporter(simple_model, input_shape=input_shape, device="cpu")
            result = exporter.to_torchscript(output_path, validate=False)

            assert result.path.exists()
            assert output_path.parent.exists()

    def test_invalid_method_raises(self, simple_model, input_shape):
        """Should raise for invalid export method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.pt"

            exporter = ModelExporter(simple_model, input_shape=input_shape, device="cpu")

            with pytest.raises(ValueError, match="Unknown method"):
                exporter.to_torchscript(output_path, method="invalid")

    def test_exported_model_inference(self, simple_model, input_shape, dummy_input):
        """Exported model should produce correct output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.pt"

            exporter = ModelExporter(simple_model, input_shape=input_shape, device="cpu")
            exporter.to_torchscript(output_path, optimize=False)

            # Load and run inference
            loaded = torch.jit.load(str(output_path))
            loaded.eval()

            with torch.no_grad():
                original_output = simple_model(dummy_input)
                loaded_output = loaded(dummy_input)

            assert torch.allclose(original_output, loaded_output, atol=1e-5)


class TestONNXExport:
    """Tests for ONNX export."""

    def test_basic_export(self, simple_model, input_shape):
        """Should export to ONNX format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.onnx"

            exporter = ModelExporter(simple_model, input_shape=input_shape, device="cpu")
            result = exporter.to_onnx(output_path, simplify=False, validate=False)

            assert result.path.exists()
            assert result.format == "onnx"
            assert result.file_size_mb > 0

    def test_export_with_dynamic_axes(self, simple_model, input_shape):
        """Should export with dynamic axes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.onnx"

            dynamic_axes = {
                "input": {0: "batch", 2: "height", 3: "width"},
                "output": {0: "batch", 2: "height", 3: "width"},
            }

            exporter = ModelExporter(simple_model, input_shape=input_shape, device="cpu")
            result = exporter.to_onnx(
                output_path,
                dynamic_axes=dynamic_axes,
                simplify=False,
                validate=False,
            )

            assert result.path.exists()
            assert result.metadata["dynamic_axes"] == dynamic_axes

    def test_export_different_opset(self, simple_model, input_shape):
        """Should export with different opset version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.onnx"

            exporter = ModelExporter(simple_model, input_shape=input_shape, device="cpu")
            result = exporter.to_onnx(
                output_path,
                opset_version=14,
                simplify=False,
                validate=False,
            )

            assert result.path.exists()
            assert result.metadata["opset_version"] == 14

    def test_custom_input_output_names(self, simple_model, input_shape):
        """Should use custom input/output names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.onnx"

            exporter = ModelExporter(simple_model, input_shape=input_shape, device="cpu")
            result = exporter.to_onnx(
                output_path,
                input_names=["image"],
                output_names=["segmentation"],
                simplify=False,
                validate=False,
            )

            assert result.path.exists()


class TestExportAll:
    """Tests for exporting to multiple formats."""

    def test_export_all_formats(self, simple_model, input_shape):
        """Should export to all formats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = ModelExporter(simple_model, input_shape=input_shape, device="cpu")
            results = exporter.export_all(
                tmpdir,
                name="test_model",
                formats=["onnx", "torchscript"],
                simplify=False,
                validate=False,
            )

            assert "onnx" in results
            assert "torchscript" in results
            assert (Path(tmpdir) / "test_model.onnx").exists()
            assert (Path(tmpdir) / "test_model.pt").exists()

    def test_export_single_format(self, simple_model, input_shape):
        """Should export to single format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = ModelExporter(simple_model, input_shape=input_shape, device="cpu")
            results = exporter.export_all(
                tmpdir,
                formats=["torchscript"],
                validate=False,
            )

            assert "torchscript" in results
            assert "onnx" not in results


class TestHalfPrecision:
    """Tests for FP16 export."""

    def test_to_half(self, simple_model, input_shape):
        """Should convert model to FP16."""
        exporter = ModelExporter(simple_model, input_shape=input_shape, device="cpu")
        result = exporter.to_half()

        assert result is exporter  # Method chaining

        # Check model parameters are FP16
        for param in exporter.model.parameters():
            assert param.dtype == torch.float16


# ============================================================================
# Convenience Function Tests
# ============================================================================


class TestExportONNXFunction:
    """Tests for export_onnx convenience function."""

    def test_basic_usage(self, simple_model, input_shape):
        """Should export model to ONNX."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.onnx"

            result = export_onnx(
                simple_model,
                output_path,
                input_shape=input_shape,
                simplify=False,
                validate=False,
            )

            assert result.path.exists()
            assert result.format == "onnx"


class TestExportTorchScriptFunction:
    """Tests for export_torchscript convenience function."""

    def test_basic_usage(self, simple_model, input_shape):
        """Should export model to TorchScript."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.pt"

            result = export_torchscript(
                simple_model,
                output_path,
                input_shape=input_shape,
                validate=False,
            )

            assert result.path.exists()
            assert result.format == "torchscript"


class TestValidateTorchScript:
    """Tests for validate_torchscript function."""

    def test_validate_valid_model(self, simple_model, input_shape):
        """Should validate a valid TorchScript model."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.pt"

            # Export first
            export_torchscript(
                simple_model,
                output_path,
                input_shape=input_shape,
                validate=False,
            )

            # Validate
            results = validate_torchscript(output_path, input_shape=input_shape)

            assert results["valid"] is True
            assert results["inference_ok"] is True
            assert "output_shape" in results

    def test_validate_invalid_path(self):
        """Should handle invalid path."""
        results = validate_torchscript("/nonexistent/model.pt")

        assert results["valid"] is False
        assert len(results["errors"]) > 0


# ============================================================================
# Integration Tests
# ============================================================================


class TestExporterIntegration:
    """Integration tests for model export."""

    def test_export_and_inference_workflow(self, simple_model, input_shape, dummy_input):
        """Test complete export and inference workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Export to TorchScript
            ts_path = Path(tmpdir) / "model.pt"
            exporter = ModelExporter(simple_model, input_shape=input_shape, device="cpu")
            exporter.to_torchscript(ts_path, validate=True)

            # Load and compare
            loaded = torch.jit.load(str(ts_path))
            loaded.eval()

            with torch.no_grad():
                original = simple_model(dummy_input)
                exported = loaded(dummy_input)

            assert torch.allclose(original, exported, atol=1e-5)

    def test_different_input_shapes(self, simple_model):
        """Test export with different input shapes."""
        shapes = [
            (1, 3, 32, 32),
            (1, 3, 64, 64),
            (1, 3, 128, 128),
            (2, 3, 64, 64),  # Different batch size
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            for i, shape in enumerate(shapes):
                output_path = Path(tmpdir) / f"model_{i}.pt"

                exporter = ModelExporter(simple_model, input_shape=shape, device="cpu")
                result = exporter.to_torchscript(output_path, validate=False)

                assert result.path.exists()
                assert result.input_shape == shape
