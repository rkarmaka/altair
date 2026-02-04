"""
Altair: A flexible, config-driven segmentation framework.

This package provides a simple, intuitive API for training, evaluating,
and deploying segmentation models.

Example:
    >>> import altair as alt
    >>> run = alt.train("configs/unet_resnet50.yaml")
    >>> results = alt.evaluate(run.id, data="path/to/test")
    >>> masks = alt.predict(run.id, images="path/to/images")
"""

from altair.api import evaluate, export, list_runs, load, predict, train
from altair.core.config import Config
from altair.core.run import Run

__version__ = "0.1.0"
__all__ = [
    # Core API functions
    "train",
    "evaluate",
    "predict",
    "load",
    "list_runs",
    "export",
    # Core classes
    "Config",
    "Run",
]
