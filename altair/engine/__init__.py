"""
Training and evaluation engine for Altair.

This module provides the core training loop, evaluation, prediction,
and metrics computation functionality.
"""

from altair.engine.evaluator import EvaluationResults, Evaluator, evaluate_model
from altair.engine.metrics import (
    BinaryMetrics,
    ConfusionMatrix,
    MetricResults,
    RegressionMetrics,
    SegmentationMetrics,
    build_metrics,
)
from altair.engine.predictor import Prediction, PredictionResults, Predictor
from altair.engine.trainer import Trainer

__all__ = [
    # Training
    "Trainer",
    # Evaluation
    "Evaluator",
    "EvaluationResults",
    "evaluate_model",
    # Prediction
    "Predictor",
    "Prediction",
    "PredictionResults",
    # Metrics
    "SegmentationMetrics",
    "BinaryMetrics",
    "RegressionMetrics",
    "ConfusionMatrix",
    "MetricResults",
    "build_metrics",
]
