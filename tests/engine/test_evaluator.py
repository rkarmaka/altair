"""Tests for evaluator module."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from altair.engine.evaluator import EvaluationResults


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_predictions():
    """Create sample predictions."""
    return [
        np.random.randint(0, 3, (64, 64), dtype=np.uint8)
        for _ in range(5)
    ]


@pytest.fixture
def sample_images():
    """Create sample images."""
    np.random.seed(42)
    return [
        np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        for _ in range(5)
    ]


@pytest.fixture
def sample_ground_truths():
    """Create sample ground truth masks."""
    return [
        np.random.randint(0, 3, (64, 64), dtype=np.uint8)
        for _ in range(5)
    ]


@pytest.fixture
def sample_metrics():
    """Create sample per-sample metrics."""
    return [
        {"mIoU": 0.8 + i * 0.02, "pixel_accuracy": 0.9 + i * 0.01, "image_path": f"img_{i}.png"}
        for i in range(5)
    ]


@pytest.fixture
def evaluation_results(sample_predictions, sample_metrics):
    """Create evaluation results."""
    return EvaluationResults(
        metrics={"mIoU": 0.85, "mDice": 0.89, "pixel_accuracy": 0.92},
        per_class_metrics={"IoU": {0: 0.95, 1: 0.82, 2: 0.78}},
        per_sample_metrics=sample_metrics,
        predictions=sample_predictions,
        config={"thresh": 0.5},
    )


# ============================================================================
# EvaluationResults Tests
# ============================================================================


class TestEvaluationResults:
    """Tests for EvaluationResults dataclass."""

    def test_getitem(self, evaluation_results):
        """Should access metrics via getitem."""
        assert evaluation_results["mIoU"] == 0.85
        assert evaluation_results["mDice"] == 0.89

    def test_get_with_default(self, evaluation_results):
        """Should return default for missing metrics."""
        assert evaluation_results.get("nonexistent", 0.0) == 0.0
        assert evaluation_results.get("mIoU") == 0.85

    def test_to_dict(self, evaluation_results):
        """Should convert to dictionary."""
        d = evaluation_results.to_dict()

        assert "metrics" in d
        assert "per_class_metrics" in d
        assert "per_sample_metrics" in d
        assert "config" in d
        assert d["metrics"]["mIoU"] == 0.85

    def test_to_csv(self, evaluation_results):
        """Should save per-sample metrics to CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "results.csv"
            evaluation_results.to_csv(csv_path)

            assert csv_path.exists()

            # Read and verify
            with open(csv_path) as f:
                lines = f.readlines()

            assert len(lines) == 6  # Header + 5 samples

    def test_to_json(self, evaluation_results):
        """Should save results to JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "results.json"
            evaluation_results.to_json(json_path)

            assert json_path.exists()

            with open(json_path) as f:
                data = json.load(f)

            assert data["metrics"]["mIoU"] == 0.85
            assert len(data["per_sample_metrics"]) == 5

    def test_summary(self, evaluation_results):
        """Should generate summary string."""
        summary = evaluation_results.summary()

        assert "Evaluation Results" in summary
        assert "mIoU" in summary
        assert "0.85" in summary

    def test_creates_parent_directories(self, evaluation_results):
        """Should create parent directories for output files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nested_path = Path(tmpdir) / "nested" / "dir" / "results.json"
            evaluation_results.to_json(nested_path)

            assert nested_path.exists()


class TestEvaluationResultsExport:
    """Tests for EvaluationResults.export_samples method."""

    def test_export_samples_basic(
        self, evaluation_results, sample_images, sample_ground_truths
    ):
        """Should export samples with visualizations."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = evaluation_results.export_samples(
                output_dir=tmpdir,
                images=sample_images,
                ground_truths=sample_ground_truths,
                n_samples=3,
            )

            assert output_path.exists()
            # Check summary file
            summary_path = output_path / "samples_summary.json"
            assert summary_path.exists()

    def test_export_samples_respects_limit(
        self, evaluation_results, sample_images, sample_ground_truths
    ):
        """Should respect n_samples limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            evaluation_results.export_samples(
                output_dir=tmpdir,
                images=sample_images,
                ground_truths=sample_ground_truths,
                n_samples=2,
            )

            summary_path = Path(tmpdir) / "samples_summary.json"
            with open(summary_path) as f:
                summary = json.load(f)

            assert summary["num_samples"] == 2

    def test_export_samples_without_predictions_warns(self):
        """Should warn when no predictions stored."""
        results = EvaluationResults(
            metrics={"mIoU": 0.85},
            predictions=[],  # Empty
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = results.export_samples(output_dir=tmpdir, n_samples=5)
            # Should return path but not create samples
            assert output_path.exists()

    def test_export_samples_custom_palette(
        self, evaluation_results, sample_images, sample_ground_truths
    ):
        """Should use custom palette."""
        palette = [[0, 0, 0], [255, 0, 0], [0, 255, 0]]

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = evaluation_results.export_samples(
                output_dir=tmpdir,
                images=sample_images,
                ground_truths=sample_ground_truths,
                n_samples=2,
                palette=palette,
            )

            assert output_path.exists()

    def test_export_samples_custom_alpha(
        self, evaluation_results, sample_images, sample_ground_truths
    ):
        """Should use custom alpha."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = evaluation_results.export_samples(
                output_dir=tmpdir,
                images=sample_images,
                ground_truths=sample_ground_truths,
                n_samples=2,
                alpha=0.7,
            )

            assert output_path.exists()


# ============================================================================
# Integration Tests
# ============================================================================


class TestEvaluatorIntegration:
    """Integration tests for evaluator components."""

    def test_full_results_workflow(self, sample_predictions, sample_metrics):
        """Test complete results workflow."""
        # Create results
        results = EvaluationResults(
            metrics={"mIoU": 0.82, "mDice": 0.87},
            per_class_metrics={"IoU": {0: 0.9, 1: 0.75, 2: 0.8}},
            per_sample_metrics=sample_metrics,
            predictions=sample_predictions,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save all formats
            results.to_csv(Path(tmpdir) / "results.csv")
            results.to_json(Path(tmpdir) / "results.json")

            # Verify files
            assert (Path(tmpdir) / "results.csv").exists()
            assert (Path(tmpdir) / "results.json").exists()

            # Load and verify JSON
            with open(Path(tmpdir) / "results.json") as f:
                data = json.load(f)

            assert data["metrics"]["mIoU"] == 0.82
            assert len(data["per_sample_metrics"]) == 5

    def test_empty_per_sample_metrics_csv(self):
        """Should handle empty per-sample metrics gracefully."""
        results = EvaluationResults(
            metrics={"mIoU": 0.85},
            per_sample_metrics=[],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "results.csv"
            results.to_csv(csv_path)

            # Should not create file (warning logged)
            assert not csv_path.exists()
