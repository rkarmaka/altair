"""
Registry pattern for registering and building components.

The Registry class provides a decorator-based registration system that allows
components (models, losses, optimizers, etc.) to be registered by name and
instantiated from configuration.

Example:
    >>> from altair.core.registry import Registry
    >>> MODELS = Registry("models")
    >>>
    >>> @MODELS.register("unet")
    ... class UNet:
    ...     def __init__(self, num_classes: int):
    ...         self.num_classes = num_classes
    >>>
    >>> model = MODELS.build("unet", num_classes=10)
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

T = TypeVar("T")


class Registry:
    """
    A registry for storing and retrieving classes/functions by name.

    The registry pattern enables configuration-driven object instantiation,
    making it easy to swap components (models, losses, etc.) via config files.

    Attributes:
        name: Human-readable name for this registry (for error messages).

    Example:
        >>> LOSSES = Registry("losses")
        >>> @LOSSES.register("dice")
        ... class DiceLoss:
        ...     pass
        >>> loss_cls = LOSSES.get("dice")
    """

    def __init__(self, name: str) -> None:
        """
        Initialize a new registry.

        Args:
            name: A descriptive name for this registry (e.g., "models", "losses").
        """
        self._name = name
        self._registry: dict[str, type | Callable[..., Any]] = {}

    @property
    def name(self) -> str:
        """Return the registry name."""
        return self._name

    @property
    def registered_names(self) -> list[str]:
        """Return a list of all registered component names."""
        return list(self._registry.keys())

    def register(self, name: str | None = None) -> Callable[[type[T]], type[T]]:
        """
        Register a class or function with this registry.

        Can be used as a decorator with or without arguments:
            @REGISTRY.register()
            class MyClass: ...

            @REGISTRY.register("custom_name")
            class MyClass: ...

        Args:
            name: Optional name to register under. If None, uses the class/function name.

        Returns:
            A decorator that registers the class/function.

        Raises:
            ValueError: If the name is already registered.
        """

        def decorator(cls: type[T]) -> type[T]:
            key = name if name is not None else cls.__name__
            if key in self._registry:
                raise ValueError(
                    f"'{key}' is already registered in {self._name} registry. "
                    f"Registered names: {self.registered_names}"
                )
            self._registry[key] = cls
            return cls

        return decorator

    def get(self, name: str) -> type | Callable[..., Any]:
        """
        Retrieve a registered class/function by name.

        Args:
            name: The registered name to look up.

        Returns:
            The registered class or function.

        Raises:
            KeyError: If the name is not found in the registry.
        """
        if name not in self._registry:
            raise KeyError(
                f"'{name}' not found in {self._name} registry. "
                f"Available: {self.registered_names}"
            )
        return self._registry[name]

    def build(self, name: str, **kwargs: Any) -> Any:
        """
        Build an instance of a registered class.

        This is a convenience method that combines get() and instantiation.

        Args:
            name: The registered name to instantiate.
            **kwargs: Arguments to pass to the class constructor.

        Returns:
            An instance of the registered class.

        Raises:
            KeyError: If the name is not found in the registry.
            TypeError: If instantiation fails due to invalid arguments.

        Example:
            >>> model = MODELS.build("unet", num_classes=10, encoder="resnet50")
        """
        cls = self.get(name)
        try:
            return cls(**kwargs)
        except TypeError as e:
            raise TypeError(
                f"Failed to instantiate '{name}' from {self._name} registry: {e}. "
                f"Provided args: {list(kwargs.keys())}"
            ) from e

    def __contains__(self, name: str) -> bool:
        """Check if a name is registered."""
        return name in self._registry

    def __len__(self) -> int:
        """Return the number of registered components."""
        return len(self._registry)

    def __repr__(self) -> str:
        """Return a string representation of the registry."""
        return f"Registry(name='{self._name}', registered={self.registered_names})"


# Global registries for different component types
MODELS = Registry("models")
BACKBONES = Registry("backbones")
DECODERS = Registry("decoders")
LOSSES = Registry("losses")
OPTIMIZERS = Registry("optimizers")
SCHEDULERS = Registry("schedulers")
DATASETS = Registry("datasets")
TRANSFORMS = Registry("transforms")
