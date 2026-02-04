"""
Predictor for running inference on images.

Provides a simple interface for running model predictions on images
or directories of images with visualization support.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn

from altair.core.run import Run
from altair.utils.console import console, print_info, print_success

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    """
    Single prediction result.

    Attributes:
        image_path: Path to the input image.
        mask: Predicted segmentation mask (H, W) with class indices.
        probabilities: Optional probability maps (C, H, W).
        original_size: Original image size (H, W).
        metrics: Optional per-sample metrics.
    """

    image_path: Path
    mask: np.ndarray
    probabilities: np.ndarray | None = None
    original_size: tuple[int, int] | None = None
    metrics: dict[str, float] = field(default_factory=dict)

    def get_mask_at_original_size(self) -> np.ndarray:
        """Get mask resized to original image size."""
        mask = self.mask
        if self.original_size and mask.shape != self.original_size:
            mask_img = Image.fromarray(mask.astype(np.uint8))
            mask_img = mask_img.resize(
                (self.original_size[1], self.original_size[0]),
                Image.NEAREST,
            )
            mask = np.array(mask_img)
        return mask

    def save(self, output_dir: str | Path, suffix: str = "_mask") -> Path:
        """
        Save the prediction mask as PNG.

        Args:
            output_dir: Directory to save the mask.
            suffix: Suffix to add to the filename.

        Returns:
            Path to the saved mask file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        stem = self.image_path.stem
        output_path = output_dir / f"{stem}{suffix}.png"

        mask = self.get_mask_at_original_size()
        Image.fromarray(mask.astype(np.uint8)).save(output_path)

        return output_path

    def save_overlay(
        self,
        output_dir: str | Path,
        alpha: float = 0.5,
        suffix: str = "_overlay",
        palette: list[list[int]] | None = None,
    ) -> Path:
        """
        Save prediction overlaid on the original image.

        Args:
            output_dir: Directory to save the overlay.
            alpha: Transparency of the mask overlay.
            suffix: Suffix to add to the filename.
            palette: Optional color palette for classes.

        Returns:
            Path to the saved overlay file.
        """
        from altair.utils.visualization import create_overlay

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load original image
        image = Image.open(self.image_path).convert("RGB")
        image = np.array(image)

        # Get mask at original size
        mask = self.get_mask_at_original_size()

        # Create overlay
        overlay = create_overlay(image, mask, alpha, palette)

        # Save
        stem = self.image_path.stem
        output_path = output_dir / f"{stem}{suffix}.png"
        Image.fromarray(overlay).save(output_path)

        return output_path

    def save_all(
        self,
        output_dir: str | Path,
        alpha: float = 0.5,
        palette: list[list[int]] | None = None,
        save_mask: bool = True,
        save_overlay: bool = True,
        save_probabilities: bool = False,
    ) -> dict[str, Path]:
        """
        Save all prediction outputs.

        Args:
            output_dir: Directory to save outputs.
            alpha: Transparency for overlay.
            palette: Color palette for visualization.
            save_mask: Whether to save raw mask.
            save_overlay: Whether to save overlay image.
            save_probabilities: Whether to save probability maps.

        Returns:
            Dictionary mapping output type to file path.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_files = {}
        stem = self.image_path.stem

        if save_mask:
            saved_files["mask"] = self.save(output_dir)

        if save_overlay:
            saved_files["overlay"] = self.save_overlay(output_dir, alpha, palette=palette)

        if save_probabilities and self.probabilities is not None:
            # Save probability maps as numpy file
            prob_path = output_dir / f"{stem}_probs.npy"
            np.save(prob_path, self.probabilities)
            saved_files["probabilities"] = prob_path

        return saved_files

    def visualize(
        self,
        ground_truth: np.ndarray | None = None,
        palette: list[list[int]] | None = None,
        alpha: float = 0.5,
        show_error_map: bool = False,
        save_path: str | Path | None = None,
    ):
        """
        Visualize the prediction.

        Args:
            ground_truth: Optional ground truth mask.
            palette: Color palette for visualization.
            alpha: Overlay transparency.
            show_error_map: Whether to show error map.
            save_path: Path to save visualization.

        Returns:
            Figure as numpy array if save_path is None.
        """
        from altair.utils.visualization import visualize_prediction

        image = Image.open(self.image_path).convert("RGB")
        image = np.array(image)
        mask = self.get_mask_at_original_size()

        return visualize_prediction(
            image=image,
            ground_truth=ground_truth,
            prediction=mask,
            palette=palette,
            alpha=alpha,
            show_error_map=show_error_map,
            save_path=save_path,
        )


class PredictionResults:
    """
    Container for multiple predictions.

    Supports iteration and batch operations.

    Example:
        >>> results = predictor.predict("images/")
        >>> for pred in results:
        ...     print(pred.image_path, pred.mask.shape)
        >>> results.save_all("outputs/")
    """

    def __init__(self, predictions: list[Prediction]):
        self.predictions = predictions

    def __len__(self) -> int:
        return len(self.predictions)

    def __getitem__(self, idx: int) -> Prediction:
        return self.predictions[idx]

    def __iter__(self) -> Iterator[Prediction]:
        return iter(self.predictions)

    def save_all(
        self,
        output_dir: str | Path,
        save_mask: bool = True,
        save_overlay: bool = True,
        alpha: float = 0.5,
        palette: list[list[int]] | None = None,
    ) -> list[dict[str, Path]]:
        """
        Save all predictions to a directory.

        Args:
            output_dir: Output directory.
            save_mask: Whether to save raw masks.
            save_overlay: Whether to save overlay images.
            alpha: Overlay transparency.
            palette: Color palette for visualization.

        Returns:
            List of dictionaries mapping output type to file path.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved = []
        for pred in self.predictions:
            files = pred.save_all(
                output_dir,
                alpha=alpha,
                palette=palette,
                save_mask=save_mask,
                save_overlay=save_overlay,
            )
            saved.append(files)

        logger.info(f"Saved {len(saved)} predictions to {output_dir}")
        return saved

    def export_samples(
        self,
        output_dir: str | Path,
        n_samples: int = 10,
        alpha: float = 0.5,
        palette: list[list[int]] | None = None,
    ) -> Path:
        """
        Export a subset of samples with visualizations.

        Args:
            output_dir: Output directory.
            n_samples: Number of samples to export.
            alpha: Overlay transparency.
            palette: Color palette.

        Returns:
            Path to output directory.
        """
        from altair.utils.visualization import SampleExporter

        exporter = SampleExporter(
            output_dir=output_dir,
            palette=palette,
            alpha=alpha,
            max_samples=n_samples,
        )

        for pred in self.predictions[:n_samples]:
            image = Image.open(pred.image_path).convert("RGB")
            image = np.array(image)
            mask = pred.get_mask_at_original_size()

            exporter.add_sample(
                image=image,
                prediction=mask,
                metrics=pred.metrics,
                image_path=str(pred.image_path),
            )

        exporter.save_summary()

        try:
            exporter.save_grid()
        except Exception as e:
            logger.warning(f"Could not save grid: {e}")

        return Path(output_dir)

    def to_numpy(self) -> list[np.ndarray]:
        """Get all masks as a list of numpy arrays."""
        return [pred.mask for pred in self.predictions]


class Predictor:
    """
    Predictor for running inference on images.

    Args:
        run: Run object containing configuration.
        checkpoint: Path to model checkpoint.
        device: Device to use for inference.
        palette: Optional color palette for visualizations.

    Example:
        >>> predictor = Predictor(run, checkpoint)
        >>> results = predictor.predict("path/to/images")
        >>> results.save_all("outputs/", save_overlay=True)
    """

    def __init__(
        self,
        run: Run,
        checkpoint: Path,
        device: str | None = None,
        palette: list[list[int]] | None = None,
    ):
        self.run = run
        self.checkpoint = checkpoint
        self.palette = palette

        # Set device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Using device: {self.device}")

        # Load model
        self.model = self._load_model()
        self.model.eval()

        # Get config
        self.config = run.config

        # Setup transforms
        self._setup_transforms()

    def _load_model(self) -> nn.Module:
        """Load model from checkpoint."""
        from altair.models import build_model

        model = build_model(self.config["model"])
        checkpoint_data = torch.load(self.checkpoint, map_location=self.device)
        model.load_state_dict(checkpoint_data["model_state_dict"])
        model = model.to(self.device)

        logger.info(f"Loaded model from {self.checkpoint}")
        return model

    def _setup_transforms(self) -> None:
        """Setup inference transforms."""
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        # Use validation transforms for inference
        aug_config = self.config.get("augmentations", {}).get("val", [])

        transforms = []
        for aug in aug_config:
            name = aug.get("name", "")
            if name == "resize":
                transforms.append(A.Resize(aug["height"], aug["width"]))
            elif name == "normalize":
                transforms.append(
                    A.Normalize(
                        mean=aug.get("mean", [0.485, 0.456, 0.406]),
                        std=aug.get("std", [0.229, 0.224, 0.225]),
                    )
                )

        transforms.append(ToTensorV2())
        self.transform = A.Compose(transforms)

    def predict(
        self,
        images: str | Path | list[str | Path],
        batch_size: int = 1,
        output_dir: Path | None = None,
        save_overlay: bool = False,
        alpha: float = 0.5,
    ) -> PredictionResults:
        """
        Run inference on images.

        Args:
            images: Path to image, directory of images, or list of paths.
            batch_size: Batch size for inference.
            output_dir: Optional directory to save predictions.
            save_overlay: Whether to save overlay images.
            alpha: Overlay transparency.

        Returns:
            PredictionResults with all predictions.
        """
        # Collect image paths
        image_paths = self._collect_image_paths(images)

        predictions = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Predicting"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("predict", total=len(image_paths))

            for i in range(0, len(image_paths), batch_size):
                batch_paths = image_paths[i : i + batch_size]
                batch_preds = self._predict_batch(batch_paths)
                predictions.extend(batch_preds)
                progress.update(task, advance=len(batch_paths))

        results = PredictionResults(predictions)

        # Save if output directory specified
        if output_dir:
            logger.info(f"Saving {len(predictions)} predictions to {output_dir}")
            print_info(f"Saving predictions to [path]{output_dir}[/path]...")
            results.save_all(
                output_dir,
                save_overlay=save_overlay,
                alpha=alpha,
                palette=self.palette,
            )
            logger.info(f"Saved {len(predictions)} predictions successfully")
            print_success(f"Saved {len(predictions)} predictions")

        return results

    def _collect_image_paths(
        self, images: str | Path | list[str | Path]
    ) -> list[Path]:
        """Collect all image paths from input."""
        if isinstance(images, (str, Path)):
            path = Path(images)
            if path.is_file():
                return [path]
            elif path.is_dir():
                extensions = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"}
                paths = []
                for ext in extensions:
                    paths.extend(path.glob(f"*{ext}"))
                    paths.extend(path.glob(f"*{ext.upper()}"))
                return sorted(set(paths))
            else:
                raise FileNotFoundError(f"Path not found: {path}")
        else:
            return [Path(p) for p in images]

    @torch.no_grad()
    def _predict_batch(self, image_paths: list[Path]) -> list[Prediction]:
        """Run inference on a batch of images."""
        predictions = []

        for image_path in image_paths:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            original_size = (image.height, image.width)
            image_np = np.array(image)

            # Apply transforms
            transformed = self.transform(image=image_np)
            image_tensor = transformed["image"].unsqueeze(0).to(self.device)

            # Forward pass
            output = self.model(image_tensor)

            # Convert to prediction
            task = self.config["model"]["task"]
            thresh = self.config.get("evaluation", {}).get("thresh", 0.5)

            if task == "binary":
                probs = torch.sigmoid(output)
                mask = (probs > thresh).squeeze().cpu().numpy().astype(np.uint8)
                probs = probs.squeeze().cpu().numpy()
            elif task == "multiclass":
                probs = torch.softmax(output, dim=1)
                mask = output.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)
                probs = probs.squeeze().cpu().numpy()
            else:  # regression
                mask = output.squeeze().cpu().numpy()
                probs = None

            predictions.append(
                Prediction(
                    image_path=image_path,
                    mask=mask,
                    probabilities=probs,
                    original_size=original_size,
                )
            )

        return predictions

    def predict_single(
        self,
        image: str | Path | np.ndarray,
    ) -> Prediction:
        """
        Run inference on a single image.

        Args:
            image: Path to image or numpy array.

        Returns:
            Single Prediction object.
        """
        if isinstance(image, (str, Path)):
            results = self.predict([Path(image)])
            return results[0]

        # Handle numpy array input
        original_size = (image.shape[0], image.shape[1])

        # Apply transforms
        transformed = self.transform(image=image)
        image_tensor = transformed["image"].unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image_tensor)

        task = self.config["model"]["task"]
        thresh = self.config.get("evaluation", {}).get("thresh", 0.5)

        if task == "binary":
            probs = torch.sigmoid(output)
            mask = (probs > thresh).squeeze().cpu().numpy().astype(np.uint8)
            probs = probs.squeeze().cpu().numpy()
        elif task == "multiclass":
            probs = torch.softmax(output, dim=1)
            mask = output.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)
            probs = probs.squeeze().cpu().numpy()
        else:
            mask = output.squeeze().cpu().numpy()
            probs = None

        return Prediction(
            image_path=Path("array_input"),
            mask=mask,
            probabilities=probs,
            original_size=original_size,
        )
