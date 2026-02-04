"""Tests for segmentation metrics."""

import numpy as np
import pytest
import torch

from altair.engine.metrics import (
    BinaryMetrics,
    ConfusionMatrix,
    MetricResults,
    RegressionMetrics,
    SegmentationMetrics,
    build_metrics,
)


class TestConfusionMatrix:
    """Test cases for ConfusionMatrix."""

    def test_perfect_prediction(self):
        """Test metrics with perfect predictions."""
        cm = ConfusionMatrix(num_classes=3)

        pred = np.array([[0, 1, 2], [0, 1, 2]])
        target = np.array([[0, 1, 2], [0, 1, 2]])

        cm.update(pred, target)

        iou_per_class, mean_iou = cm.compute_iou()
        assert mean_iou == 1.0
        assert all(iou == 1.0 for iou in iou_per_class)

        dice_per_class, mean_dice = cm.compute_dice()
        assert mean_dice == 1.0

        assert cm.compute_pixel_accuracy() == 1.0

    def test_zero_overlap(self):
        """Test metrics with no overlap."""
        cm = ConfusionMatrix(num_classes=2)

        pred = np.zeros((4, 4), dtype=np.int64)
        target = np.ones((4, 4), dtype=np.int64)

        cm.update(pred, target)

        _, mean_iou = cm.compute_iou()
        assert mean_iou == 0.0

        assert cm.compute_pixel_accuracy() == 0.0

    def test_partial_overlap(self):
        """Test metrics with partial overlap."""
        cm = ConfusionMatrix(num_classes=2)

        # Half overlap for class 1
        pred = np.array([[1, 1, 0, 0], [1, 1, 0, 0]])
        target = np.array([[1, 1, 1, 1], [0, 0, 0, 0]])

        cm.update(pred, target)

        iou_per_class, _ = cm.compute_iou()

        # Class 1: intersection=2, union=6 -> IoU=2/6=0.333
        assert abs(iou_per_class[1] - 2 / 6) < 1e-6

    def test_batch_accumulation(self):
        """Test that metrics accumulate correctly across batches."""
        cm = ConfusionMatrix(num_classes=2)

        # First batch
        pred1 = np.array([[0, 1], [0, 1]])
        target1 = np.array([[0, 1], [0, 1]])
        cm.update(pred1, target1)

        # Second batch
        pred2 = np.array([[1, 0], [1, 0]])
        target2 = np.array([[1, 0], [1, 0]])
        cm.update(pred2, target2)

        # Should still be perfect
        _, mean_iou = cm.compute_iou()
        assert mean_iou == 1.0

    def test_ignore_index(self):
        """Test that ignore_index is respected."""
        cm = ConfusionMatrix(num_classes=2, ignore_index=255)

        pred = np.array([[0, 1, 0], [1, 0, 1]])
        target = np.array([[0, 255, 0], [1, 255, 1]])

        cm.update(pred, target)

        # Only non-ignored pixels should be counted
        assert cm.compute_pixel_accuracy() == 1.0

    def test_reset(self):
        """Test that reset clears accumulated stats."""
        cm = ConfusionMatrix(num_classes=2)

        pred = np.ones((4, 4), dtype=np.int64)
        target = np.ones((4, 4), dtype=np.int64)

        cm.update(pred, target)
        assert cm.compute_pixel_accuracy() == 1.0

        cm.reset()

        # After reset, no predictions
        assert cm._total_pixels == 0

    def test_precision_recall(self):
        """Test precision and recall computation."""
        cm = ConfusionMatrix(num_classes=2)

        # Predictions: some true positives, some false positives
        pred = np.array([[1, 1, 1, 0], [1, 1, 0, 0]])
        target = np.array([[1, 1, 0, 0], [1, 0, 0, 0]])

        cm.update(pred, target)

        precision, _ = cm.compute_precision()
        recall, _ = cm.compute_recall()

        # Class 1: TP=3, FP=2, FN=0
        # Precision = 3/(3+2) = 0.6
        # Recall = 3/(3+0) = 1.0
        assert abs(precision[1] - 0.6) < 1e-6
        assert abs(recall[1] - 1.0) < 1e-6

    def test_tensor_input(self):
        """Test with PyTorch tensor input."""
        cm = ConfusionMatrix(num_classes=3)

        pred = torch.tensor([[0, 1, 2], [0, 1, 2]])
        target = torch.tensor([[0, 1, 2], [0, 1, 2]])

        cm.update(pred, target)

        _, mean_iou = cm.compute_iou()
        assert mean_iou == 1.0


class TestSegmentationMetrics:
    """Test cases for SegmentationMetrics."""

    def test_multiclass_metrics(self):
        """Test multiclass segmentation metrics."""
        metrics = SegmentationMetrics(num_classes=5, use_advanced_metrics=False)

        # Create test data
        pred = np.random.randint(0, 5, (4, 32, 32))
        target = pred.copy()  # Perfect prediction

        metrics.update(pred, target)
        results = metrics.compute()

        assert results.metrics["mIoU"] == 1.0
        assert results.metrics["mDice"] == 1.0
        assert results.metrics["pixel_accuracy"] == 1.0

    def test_per_class_metrics(self):
        """Test that per-class metrics are computed."""
        metrics = SegmentationMetrics(num_classes=3, use_advanced_metrics=False)

        pred = np.array([[0, 1, 2], [0, 1, 2]])
        target = np.array([[0, 1, 2], [0, 1, 2]])

        metrics.update(pred, target)
        results = metrics.compute()

        assert "IoU" in results.per_class
        assert len(results.per_class["IoU"]) == 3

    def test_class_names(self):
        """Test metrics with class names."""
        class_names = ["background", "tumor", "vessel"]
        metrics = SegmentationMetrics(
            num_classes=3,
            class_names=class_names,
            use_advanced_metrics=False,
        )

        pred = np.array([[0, 1, 2]])
        target = np.array([[0, 1, 2]])

        metrics.update(pred, target)
        results = metrics.compute()

        # Should have named metrics
        assert "IoU/background" in results.metrics
        assert "IoU/tumor" in results.metrics

    def test_reset(self):
        """Test metrics reset."""
        metrics = SegmentationMetrics(num_classes=3, use_advanced_metrics=False)

        pred = np.ones((2, 8, 8), dtype=np.int64)
        target = np.ones((2, 8, 8), dtype=np.int64)

        metrics.update(pred, target)
        metrics.reset()

        # After reset, update with different data
        metrics.update(np.zeros((2, 8, 8), dtype=np.int64), np.zeros((2, 8, 8), dtype=np.int64))
        results = metrics.compute()

        # Should reflect only the second update
        assert results.metrics["pixel_accuracy"] == 1.0


class TestBinaryMetrics:
    """Test cases for BinaryMetrics."""

    def test_perfect_prediction(self):
        """Test binary metrics with perfect predictions."""
        metrics = BinaryMetrics()

        pred = np.array([[1, 1, 0, 0], [1, 0, 0, 0]])
        target = np.array([[1, 1, 0, 0], [1, 0, 0, 0]])

        metrics.update(pred, target)
        results = metrics.compute()

        assert results.metrics["IoU"] == 1.0
        assert results.metrics["Dice"] == 1.0
        assert results.metrics["Precision"] == 1.0
        assert results.metrics["Recall"] == 1.0
        assert results.metrics["Accuracy"] == 1.0

    def test_probability_thresholding(self):
        """Test thresholding of probability outputs."""
        metrics = BinaryMetrics(threshold=0.5)

        pred = np.array([[0.9, 0.8, 0.2, 0.1]])  # Float probabilities
        target = np.array([[1, 1, 0, 0]])

        metrics.update(pred, target)
        results = metrics.compute()

        assert results.metrics["Accuracy"] == 1.0

    def test_custom_threshold(self):
        """Test with custom threshold."""
        metrics = BinaryMetrics(threshold=0.7)

        pred = np.array([[0.8, 0.6, 0.3]])  # 0.6 < 0.7, so becomes 0
        target = np.array([[1, 1, 0]])

        metrics.update(pred, target)
        results = metrics.compute()

        # 0.8 -> 1 (correct), 0.6 -> 0 (wrong), 0.3 -> 0 (correct)
        # TP=1, FN=1, TN=1, FP=0
        assert results.metrics["TP"] == 1
        assert results.metrics["FN"] == 1

    def test_specificity(self):
        """Test specificity computation."""
        metrics = BinaryMetrics()

        # All background correctly predicted
        pred = np.zeros((4, 4), dtype=np.int64)
        target = np.zeros((4, 4), dtype=np.int64)

        metrics.update(pred, target)
        results = metrics.compute()

        assert results.metrics["Specificity"] == 1.0

    def test_tensor_input(self):
        """Test with tensor input."""
        metrics = BinaryMetrics()

        pred = torch.tensor([[1, 0, 1, 0]])
        target = torch.tensor([[1, 0, 1, 0]])

        metrics.update(pred, target)
        results = metrics.compute()

        assert results.metrics["Accuracy"] == 1.0


class TestRegressionMetrics:
    """Test cases for RegressionMetrics."""

    def test_perfect_prediction(self):
        """Test regression metrics with perfect predictions."""
        metrics = RegressionMetrics()

        pred = np.array([[1.0, 2.0, 3.0]])
        target = np.array([[1.0, 2.0, 3.0]])

        metrics.update(pred, target)
        results = metrics.compute()

        assert results.metrics["MSE"] == 0.0
        assert results.metrics["MAE"] == 0.0
        assert results.metrics["RMSE"] == 0.0

    def test_known_error(self):
        """Test with known error values."""
        metrics = RegressionMetrics()

        pred = np.array([[1.0, 2.0, 3.0]])
        target = np.array([[2.0, 3.0, 4.0]])  # All off by 1

        metrics.update(pred, target)
        results = metrics.compute()

        assert results.metrics["MAE"] == 1.0
        assert results.metrics["MSE"] == 1.0
        assert results.metrics["RMSE"] == 1.0

    def test_tensor_input(self):
        """Test with tensor input."""
        metrics = RegressionMetrics()

        pred = torch.tensor([[1.0, 2.0]])
        target = torch.tensor([[1.0, 2.0]])

        metrics.update(pred, target)
        results = metrics.compute()

        assert results.metrics["MAE"] == 0.0


class TestBuildMetrics:
    """Test cases for build_metrics function."""

    def test_build_binary(self):
        """Test building binary metrics."""
        metrics = build_metrics(task="binary", num_classes=1)
        assert isinstance(metrics, BinaryMetrics)

    def test_build_multiclass(self):
        """Test building multiclass metrics."""
        metrics = build_metrics(task="multiclass", num_classes=10)
        assert isinstance(metrics, SegmentationMetrics)

    def test_build_regression(self):
        """Test building regression metrics."""
        metrics = build_metrics(task="regression", num_classes=1)
        assert isinstance(metrics, RegressionMetrics)

    def test_build_with_config(self):
        """Test building with eval config."""
        config = {"thresh": 0.7}
        metrics = build_metrics(task="binary", num_classes=1, eval_config=config)
        assert metrics.threshold == 0.7


class TestMetricResults:
    """Test cases for MetricResults dataclass."""

    def test_getitem(self):
        """Test dictionary-like access."""
        results = MetricResults(metrics={"mIoU": 0.8, "mDice": 0.85})
        assert results["mIoU"] == 0.8

    def test_get_with_default(self):
        """Test get with default value."""
        results = MetricResults(metrics={"mIoU": 0.8})
        assert results.get("mIoU") == 0.8
        assert results.get("nonexistent") is None
        assert results.get("nonexistent", 0.0) == 0.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        results = MetricResults(
            metrics={"mIoU": 0.8},
            per_class={"IoU": {0: 0.7, 1: 0.9}},
        )
        d = results.to_dict()
        assert "metrics" in d
        assert "per_class" in d


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_prediction(self):
        """Test with no positive predictions."""
        metrics = BinaryMetrics()

        pred = np.zeros((4, 4), dtype=np.int64)
        target = np.ones((4, 4), dtype=np.int64)

        metrics.update(pred, target)
        results = metrics.compute()

        # Should not crash, precision is 0 (no positive predictions)
        assert results.metrics["Precision"] == 0.0
        assert results.metrics["Recall"] == 0.0

    def test_single_class_present(self):
        """Test when only one class is present."""
        cm = ConfusionMatrix(num_classes=5)

        pred = np.zeros((4, 4), dtype=np.int64)
        target = np.zeros((4, 4), dtype=np.int64)

        cm.update(pred, target)
        iou_per_class, mean_iou = cm.compute_iou()

        # Only class 0 has predictions
        assert iou_per_class[0] == 1.0
        # Other classes have 0 IoU but shouldn't affect mean
        assert mean_iou == 1.0  # Only valid classes count

    def test_large_batch(self):
        """Test with larger batch size."""
        metrics = SegmentationMetrics(num_classes=10, use_advanced_metrics=False)

        pred = np.random.randint(0, 10, (32, 256, 256))
        target = np.random.randint(0, 10, (32, 256, 256))

        metrics.update(pred, target)
        results = metrics.compute()

        assert "mIoU" in results.metrics
        assert 0 <= results.metrics["mIoU"] <= 1.0
