"""Tests for UNet with encoder backbone."""

import pytest
import torch

from altair.core.registry import MODELS


class TestTimmEncoder:
    """Test cases for TimmEncoder backbone."""

    def test_resnet50_encoder(self):
        """Test ResNet-50 encoder."""
        from altair.models.backbones import get_encoder

        encoder = get_encoder("resnet50", pretrained=False, depth=5)

        assert encoder.num_features == 5
        assert len(encoder.out_channels) == 5

        x = torch.randn(1, 3, 256, 256)
        features = encoder(x)

        assert len(features) == 5
        # Check feature sizes decrease
        for i in range(1, len(features)):
            assert features[i].shape[2] <= features[i - 1].shape[2]

    def test_efficientnet_encoder(self):
        """Test EfficientNet encoder."""
        from altair.models.backbones import get_encoder

        encoder = get_encoder("efficientnet_b0", pretrained=False, depth=5)

        x = torch.randn(1, 3, 256, 256)
        features = encoder(x)

        assert len(features) == 5

    def test_encoder_depth(self):
        """Test different encoder depths."""
        from altair.models.backbones import get_encoder

        for depth in [3, 4, 5]:
            encoder = get_encoder("resnet34", pretrained=False, depth=depth)
            assert encoder.num_features == depth

            x = torch.randn(1, 3, 128, 128)
            features = encoder(x)
            assert len(features) == depth

    def test_custom_input_channels(self):
        """Test encoder with custom input channels."""
        from altair.models.backbones import get_encoder

        encoder = get_encoder("resnet18", pretrained=False, in_channels=1)

        x = torch.randn(1, 1, 128, 128)
        features = encoder(x)
        assert len(features) == 5

    def test_list_encoders(self):
        """Test listing available encoders."""
        from altair.models.backbones import list_encoders

        encoders = list_encoders()
        assert "resnet50" in encoders
        assert "efficientnet_b0" in encoders


class TestUNetDecoder:
    """Test cases for UNet decoder."""

    def test_decoder_forward(self):
        """Test decoder forward pass."""
        from altair.models.decoders import UNetDecoder

        encoder_channels = [64, 256, 512, 1024, 2048]
        decoder_channels = [256, 128, 64, 32, 16]

        decoder = UNetDecoder(
            encoder_channels=encoder_channels,
            decoder_channels=decoder_channels,
        )

        # Create mock encoder features
        features = [
            torch.randn(1, 64, 64, 64),    # stride 4
            torch.randn(1, 256, 32, 32),   # stride 8
            torch.randn(1, 512, 16, 16),   # stride 16
            torch.randn(1, 1024, 8, 8),    # stride 32
            torch.randn(1, 2048, 4, 4),    # stride 64
        ]

        out = decoder(*features)
        assert out.shape == (1, 16, 128, 128)

    def test_decoder_with_attention(self):
        """Test decoder with SCSE attention."""
        from altair.models.decoders import UNetDecoder

        decoder = UNetDecoder(
            encoder_channels=[64, 128, 256],
            decoder_channels=[128, 64, 32],
            attention_type="scse",
        )

        features = [
            torch.randn(1, 64, 32, 32),
            torch.randn(1, 128, 16, 16),
            torch.randn(1, 256, 8, 8),
        ]

        out = decoder(*features)
        assert out.shape[1] == 32


class TestSCSEModule:
    """Test cases for SCSE attention module."""

    def test_scse_forward(self):
        """Test SCSE module forward pass."""
        from altair.models.decoders import SCSEModule

        module = SCSEModule(in_channels=64, reduction=16)
        x = torch.randn(2, 64, 32, 32)
        out = module(x)

        assert out.shape == x.shape

    def test_scse_different_channels(self):
        """Test SCSE with different channel sizes."""
        from altair.models.decoders import SCSEModule

        for channels in [32, 64, 128, 256]:
            module = SCSEModule(in_channels=channels)
            x = torch.randn(1, channels, 16, 16)
            out = module(x)
            assert out.shape == x.shape


class TestUNetEncoder:
    """Test cases for UNetEncoder model."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        from altair.models.segmentors import UNetEncoder

        model = UNetEncoder(
            encoder="resnet18",
            encoder_weights=None,
            num_classes=10,
        )

        x = torch.randn(1, 3, 256, 256)
        out = model(x)

        assert out.shape == (1, 10, 256, 256)

    def test_different_encoders(self):
        """Test with different encoder backbones."""
        from altair.models.segmentors import UNetEncoder

        encoders = ["resnet18", "resnet34", "efficientnet_b0"]

        for encoder_name in encoders:
            model = UNetEncoder(
                encoder=encoder_name,
                encoder_weights=None,
                num_classes=5,
            )

            x = torch.randn(1, 3, 128, 128)
            out = model(x)
            assert out.shape == (1, 5, 128, 128)

    def test_binary_task(self):
        """Test binary segmentation."""
        from altair.models.segmentors import UNetEncoder

        model = UNetEncoder(
            encoder="resnet18",
            encoder_weights=None,
            task="binary",
            num_classes=1,
        )

        x = torch.randn(1, 3, 128, 128)
        out = model(x)
        assert out.shape == (1, 1, 128, 128)

    def test_multiclass_task(self):
        """Test multi-class segmentation."""
        from altair.models.segmentors import UNetEncoder

        model = UNetEncoder(
            encoder="resnet18",
            encoder_weights=None,
            task="multiclass",
            num_classes=20,
        )

        x = torch.randn(1, 3, 128, 128)
        out = model(x)
        assert out.shape == (1, 20, 128, 128)

    def test_regression_task(self):
        """Test regression task."""
        from altair.models.segmentors import UNetEncoder

        model = UNetEncoder(
            encoder="resnet18",
            encoder_weights=None,
            task="regression",
            num_classes=1,
        )

        x = torch.randn(1, 3, 128, 128)
        out = model(x)
        assert out.shape == (1, 1, 128, 128)

    def test_with_attention(self):
        """Test with SCSE attention."""
        from altair.models.segmentors import UNetEncoder

        model = UNetEncoder(
            encoder="resnet18",
            encoder_weights=None,
            num_classes=10,
            decoder_attention="scse",
        )

        x = torch.randn(1, 3, 128, 128)
        out = model(x)
        assert out.shape == (1, 10, 128, 128)

    def test_with_dropout(self):
        """Test with dropout."""
        from altair.models.segmentors import UNetEncoder

        model = UNetEncoder(
            encoder="resnet18",
            encoder_weights=None,
            num_classes=10,
            dropout=0.5,
        )
        model.train()

        x = torch.randn(1, 3, 128, 128)
        out = model(x)
        assert out.shape == (1, 10, 128, 128)

    def test_different_input_sizes(self):
        """Test with different input sizes."""
        from altair.models.segmentors import UNetEncoder

        model = UNetEncoder(
            encoder="resnet18",
            encoder_weights=None,
            num_classes=10,
            encoder_depth=4,
        )

        for size in [64, 128, 256, 512]:
            x = torch.randn(1, 3, size, size)
            out = model(x)
            assert out.shape == (1, 10, size, size)

    def test_non_square_input(self):
        """Test with non-square input."""
        from altair.models.segmentors import UNetEncoder

        model = UNetEncoder(
            encoder="resnet18",
            encoder_weights=None,
            num_classes=10,
            encoder_depth=4,
        )

        x = torch.randn(1, 3, 128, 256)
        out = model(x)
        assert out.shape == (1, 10, 128, 256)

    def test_custom_decoder_channels(self):
        """Test with custom decoder channels."""
        from altair.models.segmentors import UNetEncoder

        model = UNetEncoder(
            encoder="resnet18",
            encoder_weights=None,
            num_classes=10,
            decoder_channels=[128, 64, 32, 16, 8],
        )

        x = torch.randn(1, 3, 256, 256)
        out = model(x)
        assert out.shape == (1, 10, 256, 256)

    def test_encoder_depth(self):
        """Test with different encoder depths."""
        from altair.models.segmentors import UNetEncoder

        for depth in [3, 4, 5]:
            model = UNetEncoder(
                encoder="resnet18",
                encoder_weights=None,
                num_classes=10,
                encoder_depth=depth,
            )

            x = torch.randn(1, 3, 128, 128)
            out = model(x)
            assert out.shape == (1, 10, 128, 128)

    def test_freeze_encoder(self):
        """Test freezing encoder weights."""
        from altair.models.segmentors import UNetEncoder

        model = UNetEncoder(
            encoder="resnet18",
            encoder_weights=None,
            num_classes=10,
        )

        # Initially all params require grad
        encoder_params = list(model.encoder.parameters())
        assert all(p.requires_grad for p in encoder_params)

        # Freeze encoder
        model.freeze_encoder()
        assert all(not p.requires_grad for p in encoder_params)

        # Unfreeze encoder
        model.unfreeze_encoder()
        assert all(p.requires_grad for p in encoder_params)

    def test_output_activation(self):
        """Test output activation getter."""
        from altair.models.segmentors import UNetEncoder

        binary = UNetEncoder(encoder="resnet18", encoder_weights=None, task="binary", num_classes=1)
        assert isinstance(binary.get_output_activation(), torch.nn.Sigmoid)

        multi = UNetEncoder(encoder="resnet18", encoder_weights=None, task="multiclass", num_classes=10)
        assert isinstance(multi.get_output_activation(), torch.nn.Softmax)

        reg = UNetEncoder(encoder="resnet18", encoder_weights=None, task="regression", num_classes=1)
        assert reg.get_output_activation() is None


class TestRegisteredModels:
    """Test registered model variants."""

    def test_unet_encoder_registered(self):
        """Test that unet_encoder is registered."""
        assert "unet_encoder" in MODELS
        assert "unet_resnet" in MODELS

    def test_convenience_models_registered(self):
        """Test that convenience models are registered."""
        expected = [
            "unet_resnet18",
            "unet_resnet34",
            "unet_resnet50",
            "unet_resnet101",
            "unet_efficientnet_b0",
            "unet_efficientnet_b4",
            "unet_convnext_tiny",
            "unet_mobilenet",
        ]
        for name in expected:
            assert name in MODELS

    def test_build_convenience_models(self):
        """Test building convenience models."""
        models_to_test = ["unet_resnet18", "unet_efficientnet_b0"]

        for name in models_to_test:
            model = MODELS.build(name, num_classes=5, encoder_weights=None)
            x = torch.randn(1, 3, 64, 64)
            out = model(x)
            assert out.shape == (1, 5, 64, 64)


class TestBuildModel:
    """Test build_model function with encoder models."""

    def test_build_unet_resnet(self):
        """Test building UNet with ResNet encoder."""
        from altair.models import build_model

        config = {
            "architecture": "unet_resnet",
            "encoder": "resnet18",
            "encoder_weights": None,
            "num_classes": 10,
        }

        model = build_model(config)
        x = torch.randn(1, 3, 128, 128)
        out = model(x)
        assert out.shape == (1, 10, 128, 128)

    def test_build_unet_encoder_default_encoder(self):
        """Test building UNet encoder with default encoder."""
        from altair.models import build_model

        config = {
            "architecture": "unet_encoder",
            "encoder_weights": None,
            "num_classes": 10,
        }

        model = build_model(config)
        assert model.encoder_name == "resnet50"

    @pytest.mark.slow
    def test_gradient_flow(self):
        """Test gradient flow through the model."""
        from altair.models.segmentors import UNetEncoder

        model = UNetEncoder(
            encoder="resnet18",
            encoder_weights=None,
            num_classes=10,
        )

        x = torch.randn(1, 3, 64, 64, requires_grad=True)
        out = model(x)
        loss = out.mean()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
