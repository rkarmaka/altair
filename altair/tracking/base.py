"""Base tracker interface."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class Tracker(ABC):
    """
    Abstract base class for experiment trackers.

    Defines the interface that all tracking backends must implement.
    """

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """
        Log metrics to the tracker.

        Args:
            metrics: Dictionary of metric name to value.
            step: Optional step/epoch number.
        """
        pass

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None:
        """
        Log parameters/hyperparameters to the tracker.

        Args:
            params: Dictionary of parameter name to value.
        """
        pass

    @abstractmethod
    def log_artifact(self, path: str | Path) -> None:
        """
        Log an artifact (file) to the tracker.

        Args:
            path: Path to the artifact file.
        """
        pass

    @abstractmethod
    def finish(self) -> None:
        """Finish tracking and clean up resources."""
        pass

    def log_image(
        self,
        image,
        name: str,
        step: int | None = None,
    ) -> None:
        """
        Log an image to the tracker.

        Args:
            image: Image array or PIL Image.
            name: Name for the image.
            step: Optional step number.
        """
        # Default implementation does nothing
        pass
