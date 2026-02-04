"""Tests for the Registry pattern implementation."""

import pytest

from altair.core.registry import Registry


class TestRegistry:
    """Test cases for the Registry class."""

    def test_create_registry(self):
        """Test creating a new registry."""
        registry = Registry("test")
        assert registry.name == "test"
        assert len(registry) == 0
        assert registry.registered_names == []

    def test_register_class_with_decorator(self):
        """Test registering a class using the decorator."""
        registry = Registry("models")

        @registry.register("my_model")
        class MyModel:
            def __init__(self, num_classes: int):
                self.num_classes = num_classes

        assert "my_model" in registry
        assert len(registry) == 1

    def test_register_class_without_name(self):
        """Test registering a class using class name as key."""
        registry = Registry("models")

        @registry.register()
        class AutoNamedModel:
            pass

        assert "AutoNamedModel" in registry

    def test_get_registered_class(self):
        """Test retrieving a registered class."""
        registry = Registry("losses")

        @registry.register("dice")
        class DiceLoss:
            pass

        retrieved = registry.get("dice")
        assert retrieved is DiceLoss

    def test_get_unregistered_raises_keyerror(self):
        """Test that getting unregistered name raises KeyError."""
        registry = Registry("models")

        with pytest.raises(KeyError, match="not found in models registry"):
            registry.get("nonexistent")

    def test_build_creates_instance(self):
        """Test building an instance from the registry."""
        registry = Registry("models")

        @registry.register("unet")
        class UNet:
            def __init__(self, num_classes: int, encoder: str = "resnet"):
                self.num_classes = num_classes
                self.encoder = encoder

        model = registry.build("unet", num_classes=10, encoder="efficientnet")
        assert isinstance(model, UNet)
        assert model.num_classes == 10
        assert model.encoder == "efficientnet"

    def test_build_with_defaults(self):
        """Test building with default arguments."""
        registry = Registry("models")

        @registry.register("simple")
        class SimpleModel:
            def __init__(self, num_classes: int = 5):
                self.num_classes = num_classes

        model = registry.build("simple")
        assert model.num_classes == 5

    def test_duplicate_registration_raises_error(self):
        """Test that registering duplicate name raises ValueError."""
        registry = Registry("models")

        @registry.register("duplicate")
        class FirstModel:
            pass

        with pytest.raises(ValueError, match="already registered"):

            @registry.register("duplicate")
            class SecondModel:
                pass

    def test_contains_operator(self):
        """Test the 'in' operator for checking registration."""
        registry = Registry("models")

        @registry.register("exists")
        class ExistingModel:
            pass

        assert "exists" in registry
        assert "nonexistent" not in registry

    def test_len_operator(self):
        """Test len() returns correct count."""
        registry = Registry("test")

        assert len(registry) == 0

        @registry.register("first")
        class First:
            pass

        assert len(registry) == 1

        @registry.register("second")
        class Second:
            pass

        assert len(registry) == 2

    def test_registered_names_property(self):
        """Test that registered_names returns all registered names."""
        registry = Registry("test")

        @registry.register("alpha")
        class Alpha:
            pass

        @registry.register("beta")
        class Beta:
            pass

        names = registry.registered_names
        assert set(names) == {"alpha", "beta"}

    def test_repr(self):
        """Test string representation of registry."""
        registry = Registry("models")

        @registry.register("unet")
        class UNet:
            pass

        repr_str = repr(registry)
        assert "models" in repr_str
        assert "unet" in repr_str

    def test_register_function(self):
        """Test registering a function instead of a class."""
        registry = Registry("transforms")

        @registry.register("normalize")
        def normalize(x, mean=0, std=1):
            return (x - mean) / std

        func = registry.get("normalize")
        result = func(10, mean=5, std=5)
        assert result == 1.0

    def test_register_with_special_characters(self):
        """Test registering with names containing special characters."""
        registry = Registry("models")

        @registry.register("unet++")
        class UNetPlusPlus:
            pass

        @registry.register("resnet-50")
        class ResNet50:
            pass

        assert "unet++" in registry
        assert "resnet-50" in registry


class TestGlobalRegistries:
    """Test the global registry instances."""

    def test_global_registries_exist(self):
        """Test that global registries are available."""
        from altair.core.registry import (
            BACKBONES,
            DATASETS,
            DECODERS,
            LOSSES,
            MODELS,
            OPTIMIZERS,
            SCHEDULERS,
            TRANSFORMS,
        )

        assert isinstance(MODELS, Registry)
        assert isinstance(BACKBONES, Registry)
        assert isinstance(DECODERS, Registry)
        assert isinstance(LOSSES, Registry)
        assert isinstance(OPTIMIZERS, Registry)
        assert isinstance(SCHEDULERS, Registry)
        assert isinstance(DATASETS, Registry)
        assert isinstance(TRANSFORMS, Registry)

    def test_global_registries_have_correct_names(self):
        """Test that global registries have descriptive names."""
        from altair.core.registry import LOSSES, MODELS

        assert MODELS.name == "models"
        assert LOSSES.name == "losses"
