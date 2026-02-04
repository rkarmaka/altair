"""MLflow tracking integration."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from altair.core.run import Run
from altair.tracking.base import Tracker

logger = logging.getLogger(__name__)


class MLflowTracker(Tracker):
    """
    MLflow experiment tracker.

    Provides integration with MLflow for logging metrics, parameters,
    and artifacts.

    Args:
        config: TrackingConfig with MLflow settings.
        run: Run object for the experiment.

    Example:
        >>> tracker = MLflowTracker(config, run)
        >>> tracker.log_metrics({"loss": 0.5}, step=1)
        >>> tracker.log_params({"lr": 0.001})
    """

    def __init__(self, config, run: Run):
        self.config = config
        self.run = run
        self._mlflow_run = None

        # Initialize MLflow
        self._setup_mlflow()

    def _setup_mlflow(self) -> None:
        """Set up MLflow tracking."""
        try:
            import mlflow

            # Set tracking URI
            mlflow.set_tracking_uri(self.config.uri)

            # Set experiment
            mlflow.set_experiment(self.run.project)

            # Start run
            self._mlflow_run = mlflow.start_run(run_name=self.run.name)

            # Log initial config
            self._log_config()

            logger.info(
                f"MLflow tracking initialized: {self.config.uri}, "
                f"run_id={self._mlflow_run.info.run_id}"
            )

        except ImportError:
            logger.warning("MLflow not installed. Tracking disabled.")
        except Exception as e:
            logger.warning(f"Failed to initialize MLflow: {e}")

    def _log_config(self) -> None:
        """Log configuration as parameters."""
        try:
            import mlflow

            # Flatten config for logging
            params = self._flatten_dict(self.run.config)

            # MLflow has a limit on param value length
            for key, value in params.items():
                str_value = str(value)
                if len(str_value) > 250:
                    str_value = str_value[:247] + "..."
                mlflow.log_param(key, str_value)

        except Exception as e:
            logger.warning(f"Failed to log config to MLflow: {e}")

    def _flatten_dict(
        self, d: dict[str, Any], parent_key: str = "", sep: str = "."
    ) -> dict[str, Any]:
        """Flatten a nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics to MLflow."""
        try:
            import mlflow

            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            logger.warning(f"Failed to log metrics to MLflow: {e}")

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters to MLflow."""
        try:
            import mlflow

            for key, value in params.items():
                mlflow.log_param(key, value)
        except Exception as e:
            logger.warning(f"Failed to log params to MLflow: {e}")

    def log_artifact(self, path: str | Path) -> None:
        """Log an artifact to MLflow."""
        try:
            import mlflow

            mlflow.log_artifact(str(path))
        except Exception as e:
            logger.warning(f"Failed to log artifact to MLflow: {e}")

    def log_image(self, image, name: str, step: int | None = None) -> None:
        """Log an image to MLflow."""
        try:
            import mlflow
            import numpy as np
            from PIL import Image

            # Convert to PIL Image if needed
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)

            # Save temporarily and log
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                image.save(f.name)
                mlflow.log_artifact(f.name, artifact_path="images")

        except Exception as e:
            logger.warning(f"Failed to log image to MLflow: {e}")

    def finish(self) -> None:
        """End the MLflow run."""
        try:
            import mlflow

            if self._mlflow_run:
                mlflow.end_run()
                logger.info("MLflow run ended")
        except Exception as e:
            logger.warning(f"Failed to end MLflow run: {e}")
