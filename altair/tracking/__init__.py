"""
Experiment tracking for Altair.

Provides integration with MLflow and other tracking backends.
"""

from altair.tracking.base import Tracker
from altair.tracking.mlflow import MLflowTracker


def build_tracker(config, run):
    """
    Build a tracker from configuration.

    Args:
        config: TrackingConfig instance.
        run: Run object for the experiment.

    Returns:
        Tracker instance.
    """
    if config.backend == "mlflow":
        return MLflowTracker(config, run)
    elif config.backend == "none":
        return NoOpTracker()
    else:
        raise ValueError(f"Unknown tracking backend: {config.backend}")


class NoOpTracker(Tracker):
    """Tracker that does nothing (for disabled tracking)."""

    def log_metrics(self, metrics, step=None):
        pass

    def log_params(self, params):
        pass

    def log_artifact(self, path):
        pass

    def finish(self):
        pass


__all__ = ["Tracker", "MLflowTracker", "build_tracker"]
