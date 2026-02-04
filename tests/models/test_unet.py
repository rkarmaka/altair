"""Tests for UNet model."""

import pytest
import torch

from altair.models.segmentors.unet import ConvBlock, DecoderBlock, EncoderBlock, UNet


class TestConvBlock:
    """Test cases for ConvBlock."""

    def test_output_shape(self):
        """Test that output shape matches expected."""
        block = ConvBlock(in_channels=3, out_channels=64)
        x = torch.randn(2, 3, 64, 64)
        out = block(x)
        assert out.shape == (2, 64, 64, 64)

    def test_different_norm_layers(self):
        """Test different normalization layers."""
        for norm in ["batchnorm", "groupnorm", "instancenorm"]:
            block = ConvBlock(in_channels=3, out_channels=64, norm_layer=norm)
            x = torch.randn(2, 3, 32, 32)
            out = block(x)
            assert out.shape == (2, 64, 32, 32)

    def test_different_activations(self):
        """Test different activation functions."""
        for act in ["relu", "gelu", "silu", "leaky_relu"]:
            block = ConvBlock(in_channels=3, out_channels=64, activation=act)
            x = torch.randn(2, 3, 32, 32)
            out = block(x)
            assert out.shape == (2, 64, 32, 32)

    def test_with_dropout(self):
        """Test block with dropout."""
        block = ConvBlock(in_channels=3, out_channels=64, dropout=0.5)
        block.train()
        x = torch.randn(2, 3, 32, 32)
        out = block(x)
        assert out.shape == (2, 64, 32, 32)


class TestEncoderBlock:
    """Test cases for EncoderBlock."""

    def test_output_shapes(self):
        """Test encoder output shapes (pooled and skip)."""
        block = EncoderBlock(in_channels=3, out_channels=64)
        x = torch.randn(2, 3, 64, 64)
        pooled, skip = block(x)

        assert pooled.shape == (2, 64, 32, 32)  # Halved spatial dims
        assert skip.shape == (2, 64, 64, 64)  # Original spatial dims


class TestDecoderBlock:
    """Test cases for DecoderBlock."""

    def test_output_shape(self):
        """Test decoder output shape with skip connection."""
        block = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)
        x = torch.randn(2, 128, 16, 16)
        skip = torch.randn(2, 64, 32, 32)
        out = block(x, skip)

        assert out.shape == (2, 64, 32, 32)

    def test_handles_size_mismatch(self):
        """Test that decoder handles odd-sized tensors."""
        block = DecoderBlock(in_channels=128, skip_channels=64, out_channels=64)
        x = torch.randn(2, 128, 15, 15)  # Odd size
        skip = torch.randn(2, 64, 31, 31)  # Different odd size
        out = block(x, skip)

        assert out.shape == (2, 64, 31, 31)


class TestUNet:
    """Test cases for UNet model."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        model = UNet(num_classes=10)
        x = torch.randn(2, 3, 256, 256)
        out = model(x)

        assert out.shape == (2, 10, 256, 256)

    def test_binary_task(self):
        """Test binary segmentation output."""
        model = UNet(task="binary", num_classes=1)
        x = torch.randn(2, 3, 128, 128)
        out = model(x)

        assert out.shape == (2, 1, 128, 128)

    def test_multiclass_task(self):
        """Test multi-class segmentation output."""
        model = UNet(task="multiclass", num_classes=20)
        x = torch.randn(2, 3, 128, 128)
        out = model(x)

        assert out.shape == (2, 20, 128, 128)

    def test_regression_task(self):
        """Test regression output."""
        model = UNet(task="regression", num_classes=1)
        x = torch.randn(2, 3, 128, 128)
        out = model(x)

        assert out.shape == (2, 1, 128, 128)

    def test_custom_encoder_depth(self):
        """Test model with different encoder depths."""
        for depth in [3, 4, 5]:
            model = UNet(num_classes=10, encoder_depth=depth)
            x = torch.randn(1, 3, 128, 128)
            out = model(x)
            assert out.shape == (1, 10, 128, 128)

    def test_custom_decoder_channels(self):
        """Test model with custom decoder channels."""
        channels = [512, 256, 128, 64]
        model = UNet(
            num_classes=10,
            encoder_depth=5,
            decoder_channels=channels,
        )
        x = torch.randn(1, 3, 128, 128)
        out = model(x)
        assert out.shape == (1, 10, 128, 128)

    def test_with_dropout(self):
        """Test model with dropout."""
        model = UNet(num_classes=10, dropout=0.5)
        model.train()
        x = torch.randn(1, 3, 128, 128)
        out = model(x)
        assert out.shape == (1, 10, 128, 128)

    def test_different_input_sizes(self):
        """Test model with different input sizes."""
        model = UNet(num_classes=10, encoder_depth=4)

        for size in [64, 128, 256, 512]:
            x = torch.randn(1, 3, size, size)
            out = model(x)
            assert out.shape == (1, 10, size, size)

    def test_non_square_input(self):
        """Test model with non-square input."""
        model = UNet(num_classes=10, encoder_depth=4)
        x = torch.randn(1, 3, 128, 256)
        out = model(x)
        assert out.shape == (1, 10, 128, 256)

    def test_get_output_activation(self):
        """Test output activation getter."""
        binary_model = UNet(task="binary", num_classes=1)
        assert isinstance(binary_model.get_output_activation(), torch.nn.Sigmoid)

        multi_model = UNet(task="multiclass", num_classes=10)
        assert isinstance(multi_model.get_output_activation(), torch.nn.Softmax)

        reg_model = UNet(task="regression", num_classes=1)
        assert reg_model.get_output_activation() is None

    def test_registered_in_models_registry(self):
        """Test that UNet is registered in MODELS registry."""
        from altair.core.registry import MODELS

        assert "unet" in MODELS
        model = MODELS.build("unet", num_classes=5)
        assert isinstance(model, UNet)

    @pytest.mark.slow
    def test_gradient_flow(self):
        """Test that gradients flow through the model."""
        model = UNet(num_classes=10)
        x = torch.randn(1, 3, 64, 64, requires_grad=True)
        out = model(x)
        loss = out.mean()
        loss.backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape
