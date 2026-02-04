"""
Visualization utilities for segmentation results.

Provides functions for visualizing predictions, creating overlays,
and exporting sample results.

Example:
    >>> from altair.utils.visualization import visualize_prediction, create_overlay
    >>> overlay = create_overlay(image, mask, alpha=0.5)
    >>> visualize_prediction(image, mask, prediction, save_path="result.png")
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

# Default color palette for segmentation masks (RGB)
# First color is typically background
DEFAULT_PALETTE = [
    [0, 0, 0],        # 0: Background (black)
    [255, 0, 0],      # 1: Red
    [0, 255, 0],      # 2: Green
    [0, 0, 255],      # 3: Blue
    [255, 255, 0],    # 4: Yellow
    [255, 0, 255],    # 5: Magenta
    [0, 255, 255],    # 6: Cyan
    [255, 128, 0],    # 7: Orange
    [128, 0, 255],    # 8: Purple
    [0, 128, 255],    # 9: Light Blue
    [255, 128, 128],  # 10: Light Red
    [128, 255, 128],  # 11: Light Green
    [128, 128, 255],  # 12: Light Blue
    [255, 255, 128],  # 13: Light Yellow
    [255, 128, 255],  # 14: Light Magenta
    [128, 255, 255],  # 15: Light Cyan
    [192, 192, 192],  # 16: Silver
    [128, 128, 128],  # 17: Gray
    [64, 64, 64],     # 18: Dark Gray
    [255, 192, 203],  # 19: Pink
]

# Medical imaging palette (more distinguishable)
MEDICAL_PALETTE = [
    [0, 0, 0],        # Background
    [255, 0, 0],      # Lesion/Tumor (Red)
    [0, 255, 0],      # Healthy tissue (Green)
    [0, 0, 255],      # Vessel (Blue)
    [255, 255, 0],    # Necrosis (Yellow)
    [255, 0, 255],    # Edema (Magenta)
    [0, 255, 255],    # Enhancement (Cyan)
]

# Cityscapes-like palette for urban scenes
CITYSCAPES_PALETTE = [
    [128, 64, 128],   # Road
    [244, 35, 232],   # Sidewalk
    [70, 70, 70],     # Building
    [102, 102, 156],  # Wall
    [190, 153, 153],  # Fence
    [153, 153, 153],  # Pole
    [250, 170, 30],   # Traffic Light
    [220, 220, 0],    # Traffic Sign
    [107, 142, 35],   # Vegetation
    [152, 251, 152],  # Terrain
    [70, 130, 180],   # Sky
    [220, 20, 60],    # Person
    [255, 0, 0],      # Rider
    [0, 0, 142],      # Car
    [0, 0, 70],       # Truck
    [0, 60, 100],     # Bus
    [0, 80, 100],     # Train
    [0, 0, 230],      # Motorcycle
    [119, 11, 32],    # Bicycle
]


def get_palette(name: str = "default", num_classes: int | None = None) -> list[list[int]]:
    """
    Get a color palette by name.

    Args:
        name: Palette name ('default', 'medical', 'cityscapes').
        num_classes: Number of classes (will extend palette if needed).

    Returns:
        List of RGB color values.
    """
    palettes = {
        "default": DEFAULT_PALETTE,
        "medical": MEDICAL_PALETTE,
        "cityscapes": CITYSCAPES_PALETTE,
    }

    palette = palettes.get(name, DEFAULT_PALETTE).copy()

    # Extend palette if needed
    if num_classes and len(palette) < num_classes:
        np.random.seed(42)  # Reproducible colors
        while len(palette) < num_classes:
            palette.append(list(np.random.randint(0, 256, 3)))

    return palette


def mask_to_rgb(
    mask: np.ndarray,
    palette: list[list[int]] | None = None,
    num_classes: int | None = None,
) -> np.ndarray:
    """
    Convert a class mask to an RGB image.

    Args:
        mask: Class mask of shape (H, W) with integer class indices.
        palette: Color palette (list of RGB values).
        num_classes: Number of classes (for palette generation).

    Returns:
        RGB image of shape (H, W, 3).
    """
    if palette is None:
        max_class = mask.max() if num_classes is None else num_classes - 1
        palette = get_palette("default", max_class + 1)

    palette = np.array(palette, dtype=np.uint8)

    # Ensure mask values are within palette range
    mask = np.clip(mask, 0, len(palette) - 1)

    return palette[mask]


def create_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.5,
    palette: list[list[int]] | None = None,
    ignore_index: int | None = None,
) -> np.ndarray:
    """
    Create an overlay of a segmentation mask on an image.

    Args:
        image: RGB image of shape (H, W, 3).
        mask: Class mask of shape (H, W).
        alpha: Transparency of the overlay (0=image only, 1=mask only).
        palette: Color palette for the mask.
        ignore_index: Class index to keep transparent.

    Returns:
        Blended image of shape (H, W, 3).
    """
    # Ensure image is uint8
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)

    # Convert mask to RGB
    mask_rgb = mask_to_rgb(mask, palette)

    # Create alpha mask
    if ignore_index is not None:
        alpha_mask = (mask != ignore_index).astype(np.float32) * alpha
        alpha_mask = alpha_mask[:, :, np.newaxis]
    else:
        alpha_mask = alpha

    # Blend
    overlay = (image * (1 - alpha_mask) + mask_rgb * alpha_mask).astype(np.uint8)

    return overlay


def create_comparison(
    image: np.ndarray,
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    palette: list[list[int]] | None = None,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Create a side-by-side comparison of ground truth and prediction.

    Args:
        image: RGB image of shape (H, W, 3).
        ground_truth: Ground truth mask of shape (H, W).
        prediction: Predicted mask of shape (H, W).
        palette: Color palette for masks.
        alpha: Overlay transparency.

    Returns:
        Comparison image of shape (H, W*3, 3) showing:
        [Original | Ground Truth Overlay | Prediction Overlay]
    """
    # Create overlays
    gt_overlay = create_overlay(image, ground_truth, alpha, palette)
    pred_overlay = create_overlay(image, prediction, alpha, palette)

    # Concatenate horizontally
    comparison = np.concatenate([image, gt_overlay, pred_overlay], axis=1)

    return comparison


def create_error_map(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    correct_color: tuple[int, int, int] = (0, 255, 0),
    error_color: tuple[int, int, int] = (255, 0, 0),
    ignore_index: int | None = None,
) -> np.ndarray:
    """
    Create an error map showing correct and incorrect predictions.

    Args:
        ground_truth: Ground truth mask of shape (H, W).
        prediction: Predicted mask of shape (H, W).
        correct_color: RGB color for correct predictions.
        error_color: RGB color for incorrect predictions.
        ignore_index: Class index to ignore.

    Returns:
        Error map image of shape (H, W, 3).
    """
    h, w = ground_truth.shape
    error_map = np.zeros((h, w, 3), dtype=np.uint8)

    correct_mask = ground_truth == prediction
    error_mask = ~correct_mask

    if ignore_index is not None:
        ignore_mask = ground_truth == ignore_index
        correct_mask = correct_mask & ~ignore_mask
        error_mask = error_mask & ~ignore_mask

    error_map[correct_mask] = correct_color
    error_map[error_mask] = error_color

    return error_map


def visualize_prediction(
    image: np.ndarray,
    ground_truth: np.ndarray | None = None,
    prediction: np.ndarray | None = None,
    palette: list[list[int]] | None = None,
    alpha: float = 0.5,
    show_error_map: bool = False,
    figsize: tuple[int, int] = (15, 5),
    save_path: str | Path | None = None,
    title: str | None = None,
) -> np.ndarray | None:
    """
    Visualize segmentation prediction with matplotlib.

    Args:
        image: RGB image of shape (H, W, 3).
        ground_truth: Optional ground truth mask of shape (H, W).
        prediction: Optional predicted mask of shape (H, W).
        palette: Color palette for masks.
        alpha: Overlay transparency.
        show_error_map: Whether to show error map.
        figsize: Figure size for matplotlib.
        save_path: Path to save the figure.
        title: Optional title for the figure.

    Returns:
        Figure as numpy array if save_path is None.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for visualization. Install with: pip install matplotlib")

    # Determine number of subplots
    n_plots = 1  # Original image
    if ground_truth is not None:
        n_plots += 1
    if prediction is not None:
        n_plots += 1
    if show_error_map and ground_truth is not None and prediction is not None:
        n_plots += 1

    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    idx = 0

    # Original image
    axes[idx].imshow(image)
    axes[idx].set_title("Image")
    axes[idx].axis("off")
    idx += 1

    # Ground truth
    if ground_truth is not None:
        gt_overlay = create_overlay(image, ground_truth, alpha, palette)
        axes[idx].imshow(gt_overlay)
        axes[idx].set_title("Ground Truth")
        axes[idx].axis("off")
        idx += 1

    # Prediction
    if prediction is not None:
        pred_overlay = create_overlay(image, prediction, alpha, palette)
        axes[idx].imshow(pred_overlay)
        axes[idx].set_title("Prediction")
        axes[idx].axis("off")
        idx += 1

    # Error map
    if show_error_map and ground_truth is not None and prediction is not None:
        error_map = create_error_map(ground_truth, prediction)
        axes[idx].imshow(error_map)
        axes[idx].set_title("Error Map (Green=Correct, Red=Error)")
        axes[idx].axis("off")

    if title:
        fig.suptitle(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return None
    else:
        # Convert figure to numpy array
        fig.canvas.draw()
        img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return img_array


def save_prediction(
    image: np.ndarray,
    prediction: np.ndarray,
    output_dir: str | Path,
    filename: str,
    ground_truth: np.ndarray | None = None,
    palette: list[list[int]] | None = None,
    alpha: float = 0.5,
    save_mask: bool = True,
    save_overlay: bool = True,
    save_comparison: bool = True,
) -> dict[str, Path]:
    """
    Save prediction results to files.

    Args:
        image: RGB image of shape (H, W, 3).
        prediction: Predicted mask of shape (H, W).
        output_dir: Output directory.
        filename: Base filename (without extension).
        ground_truth: Optional ground truth mask.
        palette: Color palette for visualization.
        alpha: Overlay transparency.
        save_mask: Whether to save raw mask as PNG.
        save_overlay: Whether to save overlay image.
        save_comparison: Whether to save comparison image.

    Returns:
        Dictionary mapping output type to file path.
    """
    from PIL import Image

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = {}

    # Save raw mask
    if save_mask:
        mask_path = output_dir / f"{filename}_mask.png"
        Image.fromarray(prediction.astype(np.uint8)).save(mask_path)
        saved_files["mask"] = mask_path

    # Save overlay
    if save_overlay:
        overlay = create_overlay(image, prediction, alpha, palette)
        overlay_path = output_dir / f"{filename}_overlay.png"
        Image.fromarray(overlay).save(overlay_path)
        saved_files["overlay"] = overlay_path

    # Save comparison (if ground truth available)
    if save_comparison and ground_truth is not None:
        comparison = create_comparison(image, ground_truth, prediction, palette, alpha)
        comparison_path = output_dir / f"{filename}_comparison.png"
        Image.fromarray(comparison).save(comparison_path)
        saved_files["comparison"] = comparison_path

    return saved_files


class SampleExporter:
    """
    Export sample predictions with visualizations.

    Useful for exporting random samples during evaluation or training.

    Args:
        output_dir: Directory to save samples.
        palette: Color palette for visualization.
        alpha: Overlay transparency.
        max_samples: Maximum number of samples to export.

    Example:
        >>> exporter = SampleExporter("samples/", max_samples=10)
        >>> for image, mask, pred in dataloader:
        ...     exporter.add_sample(image, mask, pred)
        >>> exporter.save_summary()
    """

    def __init__(
        self,
        output_dir: str | Path,
        palette: list[list[int]] | None = None,
        alpha: float = 0.5,
        max_samples: int = 20,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.palette = palette
        self.alpha = alpha
        self.max_samples = max_samples

        self.samples: list[dict] = []
        self._sample_count = 0

    def add_sample(
        self,
        image: np.ndarray,
        prediction: np.ndarray,
        ground_truth: np.ndarray | None = None,
        metrics: dict[str, float] | None = None,
        image_path: str | None = None,
    ) -> bool:
        """
        Add a sample to export.

        Args:
            image: RGB image.
            prediction: Predicted mask.
            ground_truth: Optional ground truth mask.
            metrics: Optional per-sample metrics.
            image_path: Optional original image path.

        Returns:
            True if sample was added, False if max_samples reached.
        """
        if self._sample_count >= self.max_samples:
            return False

        self._sample_count += 1
        filename = f"sample_{self._sample_count:04d}"

        # Save files
        saved_files = save_prediction(
            image=image,
            prediction=prediction,
            output_dir=self.output_dir,
            filename=filename,
            ground_truth=ground_truth,
            palette=self.palette,
            alpha=self.alpha,
        )

        # Store sample info
        sample_info = {
            "filename": filename,
            "files": {k: str(v) for k, v in saved_files.items()},
        }

        if metrics:
            sample_info["metrics"] = metrics
        if image_path:
            sample_info["image_path"] = image_path

        self.samples.append(sample_info)

        return True

    def save_summary(self) -> Path:
        """
        Save a summary JSON file with all sample information.

        Returns:
            Path to the summary file.
        """
        import json

        summary_path = self.output_dir / "samples_summary.json"
        with open(summary_path, "w") as f:
            json.dump(
                {
                    "num_samples": len(self.samples),
                    "samples": self.samples,
                },
                f,
                indent=2,
            )

        return summary_path

    def create_grid(
        self,
        cols: int = 4,
        cell_size: tuple[int, int] = (256, 256),
    ) -> np.ndarray | None:
        """
        Create a grid image of all samples.

        Args:
            cols: Number of columns in the grid.
            cell_size: Size of each cell (width, height).

        Returns:
            Grid image as numpy array.
        """
        from PIL import Image

        if not self.samples:
            return None

        # Load overlay images
        overlays = []
        for sample in self.samples:
            overlay_path = sample["files"].get("overlay")
            if overlay_path:
                img = Image.open(overlay_path)
                img = img.resize(cell_size, Image.Resampling.LANCZOS)
                overlays.append(np.array(img))

        if not overlays:
            return None

        # Calculate grid dimensions
        n = len(overlays)
        rows = (n + cols - 1) // cols

        # Create grid
        grid_h = rows * cell_size[1]
        grid_w = cols * cell_size[0]
        grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

        for i, overlay in enumerate(overlays):
            row = i // cols
            col = i % cols
            y = row * cell_size[1]
            x = col * cell_size[0]
            grid[y:y + cell_size[1], x:x + cell_size[0]] = overlay

        return grid

    def save_grid(self, filename: str = "samples_grid.png", **kwargs) -> Path:
        """
        Save a grid image of all samples.

        Args:
            filename: Output filename.
            **kwargs: Arguments for create_grid.

        Returns:
            Path to the grid image.
        """
        from PIL import Image

        grid = self.create_grid(**kwargs)
        if grid is None:
            raise ValueError("No samples to create grid from")

        grid_path = self.output_dir / filename
        Image.fromarray(grid).save(grid_path)

        return grid_path
