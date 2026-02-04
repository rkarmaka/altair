"""Tests for visualization utilities."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from altair.utils.visualization import (
    DEFAULT_PALETTE,
    MEDICAL_PALETTE,
    CITYSCAPES_PALETTE,
    SampleExporter,
    create_comparison,
    create_error_map,
    create_overlay,
    get_palette,
    mask_to_rgb,
    save_prediction,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_image():
    """Create a sample RGB image."""
    np.random.seed(42)
    return np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)


@pytest.fixture
def sample_mask():
    """Create a sample segmentation mask."""
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:30, 10:30] = 1
    mask[35:55, 35:55] = 2
    return mask


@pytest.fixture
def binary_mask():
    """Create a binary mask."""
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[16:48, 16:48] = 1
    return mask


@pytest.fixture
def sample_prediction():
    """Create a sample prediction mask (slightly different from ground truth)."""
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[12:32, 8:28] = 1  # Shifted region 1
    mask[33:53, 37:57] = 2  # Shifted region 2
    return mask


# ============================================================================
# Palette Tests
# ============================================================================


class TestGetPalette:
    """Tests for get_palette function."""

    def test_default_palette(self):
        """Should return default palette."""
        palette = get_palette("default")
        assert palette == DEFAULT_PALETTE

    def test_medical_palette(self):
        """Should return medical palette."""
        palette = get_palette("medical")
        assert palette == MEDICAL_PALETTE

    def test_cityscapes_palette(self):
        """Should return cityscapes palette."""
        palette = get_palette("cityscapes")
        assert palette == CITYSCAPES_PALETTE

    def test_unknown_palette_returns_default(self):
        """Should return default for unknown palette names."""
        palette = get_palette("unknown")
        assert palette == DEFAULT_PALETTE

    def test_extends_palette_if_needed(self):
        """Should extend palette if num_classes exceeds palette length."""
        palette = get_palette("default", num_classes=50)
        assert len(palette) >= 50

    def test_extended_palette_has_valid_colors(self):
        """Extended palette colors should be valid RGB values."""
        palette = get_palette("default", num_classes=100)
        for color in palette:
            assert len(color) == 3
            assert all(0 <= c <= 255 for c in color)

    def test_palette_is_copy(self):
        """Returned palette should be a copy, not original."""
        palette = get_palette("default")
        palette.append([123, 123, 123])
        assert len(DEFAULT_PALETTE) < len(palette)


# ============================================================================
# Mask to RGB Tests
# ============================================================================


class TestMaskToRgb:
    """Tests for mask_to_rgb function."""

    def test_basic_conversion(self, sample_mask):
        """Should convert mask to RGB image."""
        rgb = mask_to_rgb(sample_mask)

        assert rgb.shape == (64, 64, 3)
        assert rgb.dtype == np.uint8

    def test_uses_palette_colors(self, sample_mask):
        """Should use palette colors for each class."""
        palette = [[0, 0, 0], [255, 0, 0], [0, 255, 0]]
        rgb = mask_to_rgb(sample_mask, palette=palette)

        # Background should be black
        assert np.all(rgb[0, 0] == [0, 0, 0])
        # Class 1 region should be red
        assert np.all(rgb[20, 20] == [255, 0, 0])
        # Class 2 region should be green
        assert np.all(rgb[45, 45] == [0, 255, 0])

    def test_handles_binary_mask(self, binary_mask):
        """Should handle binary masks correctly."""
        rgb = mask_to_rgb(binary_mask)

        assert rgb.shape == (64, 64, 3)

    def test_clips_values_beyond_palette(self):
        """Should clip mask values beyond palette range."""
        mask = np.array([[0, 1], [2, 100]], dtype=np.uint8)
        palette = [[0, 0, 0], [255, 0, 0], [0, 255, 0]]

        rgb = mask_to_rgb(mask, palette=palette)
        # Value 100 should be clipped to max palette index (2)
        assert np.all(rgb[1, 1] == [0, 255, 0])

    def test_auto_generates_palette(self):
        """Should auto-generate palette if not provided."""
        mask = np.array([[0, 1, 2, 3, 4]], dtype=np.uint8)
        rgb = mask_to_rgb(mask)

        assert rgb.shape == (1, 5, 3)


# ============================================================================
# Create Overlay Tests
# ============================================================================


class TestCreateOverlay:
    """Tests for create_overlay function."""

    def test_basic_overlay(self, sample_image, sample_mask):
        """Should create overlay with mask on image."""
        overlay = create_overlay(sample_image, sample_mask)

        assert overlay.shape == sample_image.shape
        assert overlay.dtype == np.uint8

    def test_alpha_blending(self, sample_image, binary_mask):
        """Should blend image and mask based on alpha."""
        # Full transparency (alpha=0) should return original image
        overlay_0 = create_overlay(sample_image, binary_mask, alpha=0)
        assert np.allclose(overlay_0, sample_image, atol=1)

        # Full opacity (alpha=1) should return mask colors
        overlay_1 = create_overlay(sample_image, binary_mask, alpha=1)
        mask_rgb = mask_to_rgb(binary_mask)
        assert np.allclose(overlay_1, mask_rgb, atol=1)

    def test_custom_palette(self, sample_image, sample_mask):
        """Should use custom palette for overlay."""
        palette = [[255, 255, 255], [0, 128, 0], [0, 0, 128]]
        overlay = create_overlay(sample_image, sample_mask, palette=palette)

        assert overlay.shape == sample_image.shape

    def test_ignore_index(self, sample_image, sample_mask):
        """Should keep ignore_index transparent."""
        # Set ignore_index to 0 (background)
        overlay = create_overlay(sample_image, sample_mask, alpha=1.0, ignore_index=0)

        # Background regions should be original image
        bg_region = sample_mask == 0
        assert np.allclose(overlay[bg_region], sample_image[bg_region], atol=1)

    def test_handles_float_image(self, sample_mask):
        """Should handle float images (0-1 range)."""
        float_image = np.random.rand(64, 64, 3).astype(np.float32)
        overlay = create_overlay(float_image, sample_mask)

        assert overlay.dtype == np.uint8
        assert overlay.max() <= 255


# ============================================================================
# Create Comparison Tests
# ============================================================================


class TestCreateComparison:
    """Tests for create_comparison function."""

    def test_basic_comparison(self, sample_image, sample_mask, sample_prediction):
        """Should create side-by-side comparison."""
        comparison = create_comparison(sample_image, sample_mask, sample_prediction)

        # Width should be 3x original (image + GT + pred)
        assert comparison.shape == (64, 64 * 3, 3)
        assert comparison.dtype == np.uint8

    def test_comparison_contains_original(self, sample_image, sample_mask, sample_prediction):
        """First panel should be original image."""
        comparison = create_comparison(sample_image, sample_mask, sample_prediction, alpha=0)

        original_panel = comparison[:, :64, :]
        assert np.allclose(original_panel, sample_image, atol=1)


# ============================================================================
# Create Error Map Tests
# ============================================================================


class TestCreateErrorMap:
    """Tests for create_error_map function."""

    def test_basic_error_map(self, sample_mask, sample_prediction):
        """Should create error map."""
        error_map = create_error_map(sample_mask, sample_prediction)

        assert error_map.shape == (64, 64, 3)
        assert error_map.dtype == np.uint8

    def test_correct_predictions_green(self, sample_mask):
        """Correct predictions should be green."""
        # Same mask as GT means all correct
        error_map = create_error_map(sample_mask, sample_mask)

        # All non-background should be green (default correct color)
        mask_region = sample_mask > 0
        assert np.all(error_map[mask_region] == [0, 255, 0])

    def test_incorrect_predictions_red(self, sample_mask):
        """Incorrect predictions should be red."""
        wrong_pred = np.ones_like(sample_mask) * 2  # All class 2
        error_map = create_error_map(sample_mask, wrong_pred)

        # Regions where GT != pred should be red
        error_regions = sample_mask != wrong_pred
        assert np.all(error_map[error_regions] == [255, 0, 0])

    def test_custom_colors(self, sample_mask, sample_prediction):
        """Should use custom colors."""
        error_map = create_error_map(
            sample_mask,
            sample_prediction,
            correct_color=(0, 0, 255),  # Blue for correct
            error_color=(255, 255, 0),  # Yellow for error
        )

        correct = sample_mask == sample_prediction
        error = ~correct

        if np.any(correct):
            assert np.all(error_map[correct] == [0, 0, 255])
        if np.any(error):
            assert np.all(error_map[error] == [255, 255, 0])

    def test_ignore_index(self, sample_mask, sample_prediction):
        """Should ignore specified class."""
        error_map = create_error_map(sample_mask, sample_prediction, ignore_index=0)

        # Background should be black (ignored)
        bg_region = sample_mask == 0
        assert np.all(error_map[bg_region] == [0, 0, 0])


# ============================================================================
# Save Prediction Tests
# ============================================================================


class TestSavePrediction:
    """Tests for save_prediction function."""

    def test_saves_mask(self, sample_image, sample_prediction):
        """Should save mask as PNG."""
        with tempfile.TemporaryDirectory() as tmpdir:
            saved = save_prediction(
                sample_image, sample_prediction, tmpdir, "test",
                save_mask=True, save_overlay=False, save_comparison=False,
            )

            assert "mask" in saved
            assert saved["mask"].exists()
            assert saved["mask"].suffix == ".png"

    def test_saves_overlay(self, sample_image, sample_prediction):
        """Should save overlay image."""
        with tempfile.TemporaryDirectory() as tmpdir:
            saved = save_prediction(
                sample_image, sample_prediction, tmpdir, "test",
                save_mask=False, save_overlay=True, save_comparison=False,
            )

            assert "overlay" in saved
            assert saved["overlay"].exists()

    def test_saves_comparison_with_gt(self, sample_image, sample_mask, sample_prediction):
        """Should save comparison when ground truth provided."""
        with tempfile.TemporaryDirectory() as tmpdir:
            saved = save_prediction(
                sample_image, sample_prediction, tmpdir, "test",
                ground_truth=sample_mask,
                save_mask=False, save_overlay=False, save_comparison=True,
            )

            assert "comparison" in saved
            assert saved["comparison"].exists()

    def test_no_comparison_without_gt(self, sample_image, sample_prediction):
        """Should not save comparison without ground truth."""
        with tempfile.TemporaryDirectory() as tmpdir:
            saved = save_prediction(
                sample_image, sample_prediction, tmpdir, "test",
                save_mask=False, save_overlay=False, save_comparison=True,
            )

            assert "comparison" not in saved

    def test_creates_output_directory(self, sample_image, sample_prediction):
        """Should create output directory if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "nested" / "dir"
            saved = save_prediction(
                sample_image, sample_prediction, output_dir, "test",
                save_mask=True, save_overlay=False, save_comparison=False,
            )

            assert output_dir.exists()
            assert saved["mask"].exists()


# ============================================================================
# Sample Exporter Tests
# ============================================================================


class TestSampleExporter:
    """Tests for SampleExporter class."""

    def test_basic_export(self, sample_image, sample_prediction):
        """Should export sample."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = SampleExporter(tmpdir)
            result = exporter.add_sample(sample_image, sample_prediction)

            assert result is True
            assert len(exporter.samples) == 1

    def test_max_samples_limit(self, sample_image, sample_prediction):
        """Should respect max_samples limit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = SampleExporter(tmpdir, max_samples=2)

            exporter.add_sample(sample_image, sample_prediction)
            exporter.add_sample(sample_image, sample_prediction)
            result = exporter.add_sample(sample_image, sample_prediction)

            assert result is False
            assert len(exporter.samples) == 2

    def test_stores_metrics(self, sample_image, sample_prediction):
        """Should store per-sample metrics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = SampleExporter(tmpdir)
            metrics = {"IoU": 0.85, "Dice": 0.91}

            exporter.add_sample(sample_image, sample_prediction, metrics=metrics)

            assert exporter.samples[0]["metrics"] == metrics

    def test_stores_image_path(self, sample_image, sample_prediction):
        """Should store image path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = SampleExporter(tmpdir)

            exporter.add_sample(
                sample_image, sample_prediction,
                image_path="/path/to/image.png"
            )

            assert exporter.samples[0]["image_path"] == "/path/to/image.png"

    def test_save_summary(self, sample_image, sample_prediction):
        """Should save summary JSON."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = SampleExporter(tmpdir)
            exporter.add_sample(sample_image, sample_prediction)
            exporter.add_sample(sample_image, sample_prediction)

            summary_path = exporter.save_summary()

            assert summary_path.exists()
            with open(summary_path) as f:
                summary = json.load(f)

            assert summary["num_samples"] == 2
            assert len(summary["samples"]) == 2

    def test_create_grid(self, sample_image, sample_prediction):
        """Should create grid image."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = SampleExporter(tmpdir)

            for _ in range(4):
                exporter.add_sample(sample_image, sample_prediction)

            grid = exporter.create_grid(cols=2, cell_size=(32, 32))

            # 4 images, 2 cols = 2 rows
            assert grid.shape == (64, 64, 3)

    def test_save_grid(self, sample_image, sample_prediction):
        """Should save grid to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = SampleExporter(tmpdir)

            for _ in range(4):
                exporter.add_sample(sample_image, sample_prediction)

            grid_path = exporter.save_grid()

            assert grid_path.exists()
            assert grid_path.name == "samples_grid.png"

    def test_empty_grid_returns_none(self):
        """Should return None for empty grid."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = SampleExporter(tmpdir)
            grid = exporter.create_grid()

            assert grid is None

    def test_empty_grid_save_raises(self):
        """Should raise error when saving empty grid."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = SampleExporter(tmpdir)

            with pytest.raises(ValueError, match="No samples"):
                exporter.save_grid()

    def test_with_ground_truth(self, sample_image, sample_mask, sample_prediction):
        """Should export with ground truth."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = SampleExporter(tmpdir)

            exporter.add_sample(
                sample_image, sample_prediction,
                ground_truth=sample_mask,
            )

            # Should have comparison file
            assert "comparison" in exporter.samples[0]["files"]

    def test_custom_palette(self, sample_image, sample_prediction):
        """Should use custom palette."""
        palette = [[0, 0, 0], [255, 128, 0], [0, 128, 255]]

        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = SampleExporter(tmpdir, palette=palette)
            exporter.add_sample(sample_image, sample_prediction)

            assert len(exporter.samples) == 1

    def test_custom_alpha(self, sample_image, sample_prediction):
        """Should use custom alpha."""
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = SampleExporter(tmpdir, alpha=0.7)
            exporter.add_sample(sample_image, sample_prediction)

            assert len(exporter.samples) == 1


# ============================================================================
# Integration Tests
# ============================================================================


class TestVisualizationIntegration:
    """Integration tests for visualization utilities."""

    def test_full_workflow(self, sample_image, sample_mask, sample_prediction):
        """Test complete visualization workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create exporter
            exporter = SampleExporter(tmpdir, max_samples=5)

            # Add samples with metrics
            for i in range(3):
                metrics = {"IoU": 0.8 + i * 0.05, "Dice": 0.85 + i * 0.05}
                exporter.add_sample(
                    sample_image,
                    sample_prediction,
                    ground_truth=sample_mask,
                    metrics=metrics,
                    image_path=f"image_{i}.png",
                )

            # Save summary
            summary_path = exporter.save_summary()
            assert summary_path.exists()

            # Save grid
            grid_path = exporter.save_grid()
            assert grid_path.exists()

            # Verify files
            for sample in exporter.samples:
                for filepath in sample["files"].values():
                    assert Path(filepath).exists()

    def test_large_mask_values(self, sample_image):
        """Test with many classes."""
        # Create mask with 15 classes
        mask = np.random.randint(0, 15, (64, 64), dtype=np.uint8)

        # Should handle many classes
        rgb = mask_to_rgb(mask)
        assert rgb.shape == (64, 64, 3)

        overlay = create_overlay(sample_image, mask)
        assert overlay.shape == (64, 64, 3)
