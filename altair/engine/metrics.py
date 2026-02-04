"""
Segmentation metrics for Altair.

Provides comprehensive metrics for evaluating segmentation models including:
- Standard metrics: IoU, Dice, Precision, Recall, F1
- Per-class metrics
- Integration with segmentation-evaluation package for advanced metrics

Example:
    >>> from altair.engine.metrics import SegmentationMetrics
    >>> metrics = SegmentationMetrics(num_classes=10)
    >>> metrics.update(predictions, targets)
    >>> results = metrics.compute()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass
class MetricResults:
    """
    Container for metric results.

    Attributes:
        metrics: Dictionary of aggregate metrics.
        per_class: Dictionary of per-class metrics.
        per_sample: List of per-sample metric dictionaries.
    """

    metrics: dict[str, float] = field(default_factory=dict)
    per_class: dict[str, dict[int, float]] = field(default_factory=dict)
    per_sample: list[dict[str, float]] = field(default_factory=list)

    def __getitem__(self, key: str) -> float:
        return self.metrics[key]

    def get(self, key: str, default: float | None = None) -> float | None:
        return self.metrics.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "metrics": self.metrics,
            "per_class": self.per_class,
        }


class ConfusionMatrix:
    """
    Confusion matrix accumulator for segmentation metrics.

    Efficiently accumulates true positives, false positives, and false
    negatives for computing IoU, Dice, Precision, Recall.

    Args:
        num_classes: Number of classes including background.
        ignore_index: Class index to ignore (e.g., unlabeled pixels).
    """

    def __init__(self, num_classes: int, ignore_index: int | None = None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        """Reset accumulated statistics."""
        self._tp = np.zeros(self.num_classes, dtype=np.int64)
        self._fp = np.zeros(self.num_classes, dtype=np.int64)
        self._fn = np.zeros(self.num_classes, dtype=np.int64)
        self._total_pixels = 0
        self._correct_pixels = 0

    def update(
        self,
        pred: np.ndarray | torch.Tensor,
        target: np.ndarray | torch.Tensor,
    ) -> None:
        """
        Update confusion matrix with a batch of predictions.

        Args:
            pred: Predicted labels of shape (N, H, W) or (H, W).
            target: Ground truth labels of shape (N, H, W) or (H, W).
        """
        # Convert to numpy if tensor
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()

        # Ensure batch dimension
        if pred.ndim == 2:
            pred = pred[np.newaxis, ...]
        if target.ndim == 2:
            target = target[np.newaxis, ...]

        # Create mask for valid pixels
        if self.ignore_index is not None:
            valid_mask = target != self.ignore_index
        else:
            valid_mask = np.ones_like(target, dtype=bool)

        # Update pixel accuracy
        self._total_pixels += valid_mask.sum()
        self._correct_pixels += ((pred == target) & valid_mask).sum()

        # Update per-class statistics
        for cls in range(self.num_classes):
            pred_cls = (pred == cls) & valid_mask
            target_cls = (target == cls) & valid_mask

            self._tp[cls] += (pred_cls & target_cls).sum()
            self._fp[cls] += (pred_cls & ~target_cls).sum()
            self._fn[cls] += (~pred_cls & target_cls).sum()

    def compute_iou(self) -> tuple[np.ndarray, float]:
        """
        Compute IoU (Intersection over Union) per class and mean.

        Returns:
            Tuple of (per-class IoU array, mean IoU).
        """
        denominator = self._tp + self._fp + self._fn
        iou = np.zeros(self.num_classes, dtype=np.float64)

        valid = denominator > 0
        iou[valid] = self._tp[valid] / denominator[valid]

        mean_iou = iou[valid].mean() if valid.any() else 0.0
        return iou, float(mean_iou)

    def compute_dice(self) -> tuple[np.ndarray, float]:
        """
        Compute Dice coefficient per class and mean.

        Returns:
            Tuple of (per-class Dice array, mean Dice).
        """
        denominator = 2 * self._tp + self._fp + self._fn
        dice = np.zeros(self.num_classes, dtype=np.float64)

        valid = denominator > 0
        dice[valid] = 2 * self._tp[valid] / denominator[valid]

        mean_dice = dice[valid].mean() if valid.any() else 0.0
        return dice, float(mean_dice)

    def compute_precision(self) -> tuple[np.ndarray, float]:
        """
        Compute precision per class and mean.

        Returns:
            Tuple of (per-class precision array, mean precision).
        """
        denominator = self._tp + self._fp
        precision = np.zeros(self.num_classes, dtype=np.float64)

        valid = denominator > 0
        precision[valid] = self._tp[valid] / denominator[valid]

        mean_precision = precision[valid].mean() if valid.any() else 0.0
        return precision, float(mean_precision)

    def compute_recall(self) -> tuple[np.ndarray, float]:
        """
        Compute recall per class and mean.

        Returns:
            Tuple of (per-class recall array, mean recall).
        """
        denominator = self._tp + self._fn
        recall = np.zeros(self.num_classes, dtype=np.float64)

        valid = denominator > 0
        recall[valid] = self._tp[valid] / denominator[valid]

        mean_recall = recall[valid].mean() if valid.any() else 0.0
        return recall, float(mean_recall)

    def compute_f1(self) -> tuple[np.ndarray, float]:
        """
        Compute F1 score per class and mean.

        Returns:
            Tuple of (per-class F1 array, mean F1).
        """
        precision, _ = self.compute_precision()
        recall, _ = self.compute_recall()

        denominator = precision + recall
        f1 = np.zeros(self.num_classes, dtype=np.float64)

        valid = denominator > 0
        f1[valid] = 2 * precision[valid] * recall[valid] / denominator[valid]

        mean_f1 = f1[valid].mean() if valid.any() else 0.0
        return f1, float(mean_f1)

    def compute_pixel_accuracy(self) -> float:
        """Compute overall pixel accuracy."""
        if self._total_pixels == 0:
            return 0.0
        return float(self._correct_pixels / self._total_pixels)


class SegmentationMetrics:
    """
    Comprehensive segmentation metrics calculator.

    Computes standard metrics and optionally integrates with the
    segmentation-evaluation package for advanced metrics.

    Args:
        num_classes: Number of classes including background.
        ignore_index: Class index to ignore.
        task: Task type ('binary', 'multiclass', 'regression').
        class_names: Optional list of class names for reporting.
        use_advanced_metrics: Whether to use segmentation-evaluation package.
        eval_config: Configuration for advanced metrics.

    Example:
        >>> metrics = SegmentationMetrics(num_classes=10)
        >>> for pred, target in dataloader:
        ...     metrics.update(pred, target)
        >>> results = metrics.compute()
        >>> print(results.metrics)
    """

    def __init__(
        self,
        num_classes: int,
        ignore_index: int | None = None,
        task: str = "multiclass",
        class_names: list[str] | None = None,
        use_advanced_metrics: bool = True,
        eval_config: dict[str, Any] | None = None,
    ):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.task = task
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.use_advanced_metrics = use_advanced_metrics
        self.eval_config = eval_config or {}

        # Initialize confusion matrix
        self._confusion_matrix = ConfusionMatrix(num_classes, ignore_index)

        # Storage for advanced metrics (per-sample)
        self._advanced_results: list[dict[str, Any]] = []

        # Check if segmentation-evaluation package is available
        self._has_seg_eval = False
        if use_advanced_metrics:
            try:
                from metrics.evaluate import evaluate_segmentation
                self._has_seg_eval = True
                self._evaluate_segmentation = evaluate_segmentation
            except ImportError:
                pass

    def reset(self) -> None:
        """Reset all accumulated metrics."""
        self._confusion_matrix.reset()
        self._advanced_results.clear()

    def update(
        self,
        pred: np.ndarray | torch.Tensor,
        target: np.ndarray | torch.Tensor,
    ) -> None:
        """
        Update metrics with predictions and targets.

        Args:
            pred: Predictions of shape (N, H, W) or (H, W).
                For multiclass, should be class indices.
                For binary, should be 0/1 values.
            target: Ground truth of shape (N, H, W) or (H, W).
        """
        # Convert to numpy
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()

        # Update confusion matrix
        self._confusion_matrix.update(pred, target)

        # Compute advanced metrics if available
        if self._has_seg_eval:
            self._compute_advanced_metrics(pred, target)

    def _compute_advanced_metrics(
        self,
        pred: np.ndarray,
        target: np.ndarray,
    ) -> None:
        """Compute advanced metrics using segmentation-evaluation package."""
        # Ensure batch dimension
        if pred.ndim == 2:
            pred = pred[np.newaxis, ...]
        if target.ndim == 2:
            target = target[np.newaxis, ...]

        for p, t in zip(pred, target):
            try:
                result = self._evaluate_segmentation(
                    y_true=t,
                    y_pred=p,
                    thresh=self.eval_config.get("thresh", 0.5),
                    iou_high=self.eval_config.get("iou_high", 0.5),
                    iou_low=self.eval_config.get("iou_low", 0.05),
                    soft_pq_method=self.eval_config.get("soft_pq_method", "sqrt"),
                )
                self._advanced_results.append(result)
            except Exception:
                # Skip on error
                pass

    def compute(self) -> MetricResults:
        """
        Compute all metrics.

        Returns:
            MetricResults containing all computed metrics.
        """
        results = MetricResults()

        # Compute standard metrics
        iou_per_class, mean_iou = self._confusion_matrix.compute_iou()
        dice_per_class, mean_dice = self._confusion_matrix.compute_dice()
        precision_per_class, mean_precision = self._confusion_matrix.compute_precision()
        recall_per_class, mean_recall = self._confusion_matrix.compute_recall()
        f1_per_class, mean_f1 = self._confusion_matrix.compute_f1()
        pixel_acc = self._confusion_matrix.compute_pixel_accuracy()

        # Aggregate metrics
        results.metrics = {
            "mIoU": mean_iou,
            "mDice": mean_dice,
            "mPrecision": mean_precision,
            "mRecall": mean_recall,
            "mF1": mean_f1,
            "pixel_accuracy": pixel_acc,
        }

        # Per-class metrics
        results.per_class = {
            "IoU": {i: float(v) for i, v in enumerate(iou_per_class)},
            "Dice": {i: float(v) for i, v in enumerate(dice_per_class)},
            "Precision": {i: float(v) for i, v in enumerate(precision_per_class)},
            "Recall": {i: float(v) for i, v in enumerate(recall_per_class)},
            "F1": {i: float(v) for i, v in enumerate(f1_per_class)},
        }

        # Add named per-class metrics if names are available
        if self.class_names:
            for metric_name, values in results.per_class.items():
                for cls_idx, value in values.items():
                    if cls_idx < len(self.class_names):
                        results.metrics[f"{metric_name}/{self.class_names[cls_idx]}"] = value

        # Add advanced metrics if available
        if self._advanced_results:
            results.metrics.update(self._aggregate_advanced_metrics())

        return results

    def _aggregate_advanced_metrics(self) -> dict[str, float]:
        """Aggregate advanced metrics from per-sample results."""
        if not self._advanced_results:
            return {}

        metrics = {}

        # Extract and average soft_pq
        soft_pq_values = []
        for r in self._advanced_results:
            if isinstance(r, dict) and "soft_pq" in r:
                soft_pq_values.append(r["soft_pq"])
        if soft_pq_values:
            metrics["soft_pq"] = float(np.mean(soft_pq_values))

        # Extract and average mean_average_precision
        map_values = []
        for r in self._advanced_results:
            if isinstance(r, dict) and "mean_average_precision" in r:
                map_values.append(r["mean_average_precision"])
        if map_values:
            metrics["mAP"] = float(np.mean(map_values))

        return metrics


class BinaryMetrics:
    """
    Metrics for binary segmentation tasks.

    Specialized metric calculator for binary (foreground/background)
    segmentation with threshold-based metrics.

    Args:
        threshold: Threshold for converting probabilities to binary.
        ignore_index: Pixel value to ignore.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        ignore_index: int | None = None,
    ):
        self.threshold = threshold
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        """Reset accumulated statistics."""
        self._tp = 0
        self._fp = 0
        self._fn = 0
        self._tn = 0

    def update(
        self,
        pred: np.ndarray | torch.Tensor,
        target: np.ndarray | torch.Tensor,
    ) -> None:
        """
        Update metrics with predictions and targets.

        Args:
            pred: Predictions (probabilities or binary) of shape (N, H, W) or (H, W).
            target: Binary ground truth of shape (N, H, W) or (H, W).
        """
        # Convert to numpy
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()

        # Apply threshold if needed
        if pred.dtype == np.float32 or pred.dtype == np.float64:
            pred = (pred > self.threshold).astype(np.uint8)

        # Create valid mask
        if self.ignore_index is not None:
            valid = target != self.ignore_index
        else:
            valid = np.ones_like(target, dtype=bool)

        pred = pred.astype(bool)
        target = target.astype(bool)

        # Update statistics
        self._tp += ((pred & target) & valid).sum()
        self._fp += ((pred & ~target) & valid).sum()
        self._fn += ((~pred & target) & valid).sum()
        self._tn += ((~pred & ~target) & valid).sum()

    def compute(self) -> MetricResults:
        """Compute all binary metrics."""
        results = MetricResults()

        # IoU (Jaccard)
        iou_denom = self._tp + self._fp + self._fn
        iou = self._tp / iou_denom if iou_denom > 0 else 0.0

        # Dice (F1)
        dice_denom = 2 * self._tp + self._fp + self._fn
        dice = 2 * self._tp / dice_denom if dice_denom > 0 else 0.0

        # Precision
        prec_denom = self._tp + self._fp
        precision = self._tp / prec_denom if prec_denom > 0 else 0.0

        # Recall (Sensitivity)
        recall_denom = self._tp + self._fn
        recall = self._tp / recall_denom if recall_denom > 0 else 0.0

        # Specificity
        spec_denom = self._tn + self._fp
        specificity = self._tn / spec_denom if spec_denom > 0 else 0.0

        # Accuracy
        total = self._tp + self._tn + self._fp + self._fn
        accuracy = (self._tp + self._tn) / total if total > 0 else 0.0

        results.metrics = {
            "IoU": float(iou),
            "Dice": float(dice),
            "Precision": float(precision),
            "Recall": float(recall),
            "Specificity": float(specificity),
            "Accuracy": float(accuracy),
            "TP": int(self._tp),
            "FP": int(self._fp),
            "FN": int(self._fn),
            "TN": int(self._tn),
        }

        return results


class RegressionMetrics:
    """
    Metrics for regression segmentation tasks.

    For continuous output tasks like depth estimation.
    """

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """Reset accumulated statistics."""
        self._sum_squared_error = 0.0
        self._sum_absolute_error = 0.0
        self._sum_target = 0.0
        self._sum_target_squared = 0.0
        self._count = 0

    def update(
        self,
        pred: np.ndarray | torch.Tensor,
        target: np.ndarray | torch.Tensor,
    ) -> None:
        """Update metrics with predictions and targets."""
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()

        pred = pred.flatten()
        target = target.flatten()

        self._sum_squared_error += ((pred - target) ** 2).sum()
        self._sum_absolute_error += np.abs(pred - target).sum()
        self._sum_target += target.sum()
        self._sum_target_squared += (target ** 2).sum()
        self._count += len(pred)

    def compute(self) -> MetricResults:
        """Compute regression metrics."""
        results = MetricResults()

        if self._count == 0:
            return results

        mse = self._sum_squared_error / self._count
        mae = self._sum_absolute_error / self._count
        rmse = np.sqrt(mse)

        # R² score
        mean_target = self._sum_target / self._count
        ss_tot = self._sum_target_squared - self._count * mean_target ** 2
        ss_res = self._sum_squared_error
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        results.metrics = {
            "MSE": float(mse),
            "MAE": float(mae),
            "RMSE": float(rmse),
            "R2": float(r2),
        }

        return results


def build_metrics(
    task: str,
    num_classes: int,
    eval_config: dict[str, Any] | None = None,
    class_names: list[str] | None = None,
) -> SegmentationMetrics | BinaryMetrics | RegressionMetrics:
    """
    Build appropriate metrics for the task.

    Args:
        task: Task type ('binary', 'multiclass', 'regression').
        num_classes: Number of classes.
        eval_config: Configuration for evaluation.
        class_names: Optional class names.

    Returns:
        Appropriate metrics calculator.
    """
    if task == "binary":
        return BinaryMetrics(
            threshold=eval_config.get("thresh", 0.5) if eval_config else 0.5
        )
    elif task == "regression":
        return RegressionMetrics()
    else:
        return SegmentationMetrics(
            num_classes=num_classes,
            task=task,
            class_names=class_names,
            eval_config=eval_config,
        )
