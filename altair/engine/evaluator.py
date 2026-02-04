"""
Evaluator for model evaluation on datasets.

Computes comprehensive segmentation metrics including standard metrics
(IoU, Dice, etc.) and advanced metrics from the segmentation-evaluation package.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn
from torch.utils.data import DataLoader

from altair.core.run import Run
from altair.engine.metrics import (
    BinaryMetrics,
    MetricResults,
    RegressionMetrics,
    SegmentationMetrics,
    build_metrics,
)
from altair.utils.console import console, print_info, print_success

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResults:
    """
    Container for evaluation results.

    Attributes:
        metrics: Dictionary of aggregate metrics.
        per_class_metrics: Dictionary of per-class metrics.
        per_sample_metrics: List of per-sample metric dictionaries.
        predictions: List of prediction arrays (if stored).
        config: Evaluation configuration used.
    """

    metrics: dict[str, float] = field(default_factory=dict)
    per_class_metrics: dict[str, dict[int, float]] = field(default_factory=dict)
    per_sample_metrics: list[dict[str, Any]] = field(default_factory=list)
    predictions: list[np.ndarray] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> float:
        return self.metrics[key]

    def get(self, key: str, default: float | None = None) -> float | None:
        return self.metrics.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        """Convert results to dictionary."""
        return {
            "metrics": self.metrics,
            "per_class_metrics": self.per_class_metrics,
            "per_sample_metrics": self.per_sample_metrics,
            "config": self.config,
        }

    def to_csv(self, path: str | Path) -> None:
        """
        Save per-sample metrics to CSV.

        Args:
            path: Output CSV file path.
        """
        import csv

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if not self.per_sample_metrics:
            logger.warning("No per-sample metrics to save")
            return

        fieldnames = list(self.per_sample_metrics[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.per_sample_metrics)

        logger.info(f"Saved per-sample metrics to {path}")

    def to_json(self, path: str | Path) -> None:
        """
        Save results to JSON.

        Args:
            path: Output JSON file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

        logger.info(f"Saved results to {path}")

    def summary(self) -> str:
        """Generate a human-readable summary of results."""
        lines = ["Evaluation Results", "=" * 40]

        # Main metrics
        for name, value in sorted(self.metrics.items()):
            if "/" not in name:  # Skip per-class metrics in summary
                lines.append(f"  {name}: {value:.4f}")

        return "\n".join(lines)

    def print_summary(self) -> None:
        """Print evaluation summary to console."""
        print(self.summary())

    def export_samples(
        self,
        output_dir: str | Path,
        images: list[np.ndarray] | None = None,
        ground_truths: list[np.ndarray] | None = None,
        n_samples: int = 10,
        alpha: float = 0.5,
        palette: list[list[int]] | None = None,
    ) -> Path:
        """
        Export sample predictions with visualizations.

        Args:
            output_dir: Output directory.
            images: List of input images (RGB arrays).
            ground_truths: List of ground truth masks.
            n_samples: Number of samples to export.
            alpha: Overlay transparency.
            palette: Color palette for visualization.

        Returns:
            Path to output directory.
        """
        from altair.utils.visualization import SampleExporter

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not self.predictions:
            logger.warning("No predictions stored. Run evaluate with store_predictions=True")
            return output_dir

        exporter = SampleExporter(
            output_dir=output_dir,
            palette=palette,
            alpha=alpha,
            max_samples=n_samples,
        )

        n_to_export = min(n_samples, len(self.predictions))

        for i in range(n_to_export):
            pred = self.predictions[i]
            sample_metrics = self.per_sample_metrics[i] if i < len(self.per_sample_metrics) else {}

            # Get image if available
            image = images[i] if images and i < len(images) else None
            gt = ground_truths[i] if ground_truths and i < len(ground_truths) else None
            image_path = sample_metrics.get("image_path", f"sample_{i}")

            exporter.add_sample(
                image=image,
                prediction=pred,
                ground_truth=gt,
                metrics=sample_metrics,
                image_path=str(image_path),
            )

        exporter.save_summary()

        try:
            exporter.save_grid()
        except Exception as e:
            logger.warning(f"Could not save grid: {e}")

        logger.info(f"Exported {n_to_export} samples to {output_dir}")
        return output_dir


class Evaluator:
    """
    Evaluator for segmentation models.

    Computes comprehensive metrics on a dataset including:
    - Standard metrics: mIoU, mDice, Precision, Recall, F1
    - Per-class metrics
    - Advanced metrics from segmentation-evaluation package
    - Per-sample metrics for detailed analysis

    Args:
        run: Run object containing configuration.
        checkpoint: Path to model checkpoint.
        data_path: Optional override for data path.
        batch_size: Optional override for batch size.
        device: Device to use for evaluation.
        store_predictions: Whether to store predictions in results.

    Example:
        >>> evaluator = Evaluator(run, checkpoint)
        >>> results = evaluator.evaluate()
        >>> print(results.metrics)
        >>> results.to_csv("results.csv")
    """

    def __init__(
        self,
        run: Run,
        checkpoint: Path,
        data_path: Path | None = None,
        batch_size: int | None = None,
        device: str | None = None,
        store_predictions: bool = False,
        palette: list[list[int]] | None = None,
    ):
        self.run = run
        self.checkpoint = checkpoint
        self.data_path = data_path
        self.batch_size = batch_size
        self.store_predictions = store_predictions
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

        # Get config from run
        self.config = run.config

        # Get task and num_classes
        self.task = self.config["model"]["task"]
        self.num_classes = self.config["model"]["num_classes"]

        # Get evaluation config
        self.eval_config = self.config.get("evaluation", {})

    def _load_model(self) -> nn.Module:
        """Load model from checkpoint."""
        from altair.models import build_model

        # Build model architecture
        model = build_model(self.config["model"])

        # Load weights
        checkpoint_data = torch.load(self.checkpoint, map_location=self.device)
        model.load_state_dict(checkpoint_data["model_state_dict"])
        model = model.to(self.device)

        logger.info(f"Loaded model from {self.checkpoint}")
        return model

    def _create_dataloader(self) -> DataLoader:
        """Create dataloader for evaluation."""
        from altair.core.config import Config
        from altair.data import build_dataloaders

        config = Config.from_dict(self.config)

        # Override batch size if provided
        if self.batch_size:
            config.data.batch_size = self.batch_size

        # Build dataloaders (we only need val loader)
        _, val_loader = build_dataloaders(config.data, config.augmentations)
        return val_loader

    def evaluate(
        self,
        dataloader: DataLoader | None = None,
        export_samples: bool = False,
        export_dir: str | Path | None = None,
        n_export_samples: int = 10,
    ) -> EvaluationResults:
        """
        Run evaluation on the dataset.

        Args:
            dataloader: Optional dataloader. If not provided, creates one from config.
            export_samples: Whether to export sample visualizations.
            export_dir: Directory for sample exports (required if export_samples=True).
            n_export_samples: Number of samples to export.

        Returns:
            EvaluationResults with computed metrics.
        """
        if dataloader is None:
            dataloader = self._create_dataloader()

        # Initialize metrics
        metrics_calculator = build_metrics(
            task=self.task,
            num_classes=self.num_classes,
            eval_config=self.eval_config,
        )

        # Results container
        results = EvaluationResults(config=self.eval_config)
        per_sample_results = []

        # Storage for sample export
        stored_images = []
        stored_masks = []
        samples_collected = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]Evaluating"),
            BarColumn(bar_width=30),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("eval", total=len(dataloader))

            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    images = batch["image"].to(self.device)
                    masks = batch["mask"]

                    # Forward pass
                    outputs = self.model(images)

                    # Convert outputs to predictions
                    predictions = self._outputs_to_predictions(outputs)

                    # Convert to numpy
                    predictions_np = predictions.cpu().numpy()
                    masks_np = masks.numpy()

                    # Update aggregate metrics
                    metrics_calculator.update(predictions_np, masks_np)

                    # Compute per-sample metrics
                    for i in range(len(predictions_np)):
                        sample_metrics = self._compute_sample_metrics(
                            predictions_np[i], masks_np[i]
                        )
                        sample_metrics["batch_idx"] = batch_idx
                        sample_metrics["sample_idx"] = i

                        # Add image path if available
                        if "image_path" in batch:
                            if isinstance(batch["image_path"], (list, tuple)):
                                sample_metrics["image_path"] = batch["image_path"][i]
                            else:
                                sample_metrics["image_path"] = batch["image_path"]

                        per_sample_results.append(sample_metrics)

                        # Collect samples for export
                        if export_samples and samples_collected < n_export_samples:
                            # Denormalize image for visualization
                            img = images[i].cpu().numpy().transpose(1, 2, 0)
                            img = self._denormalize_image(img)
                            stored_images.append(img)
                            stored_masks.append(masks_np[i])
                            samples_collected += 1

                    # Store predictions if requested
                    if self.store_predictions:
                        results.predictions.extend(predictions_np)

                    progress.update(task, advance=1)

        # Compute final metrics
        metric_results = metrics_calculator.compute()

        # Populate results
        results.metrics = metric_results.metrics
        results.per_class_metrics = metric_results.per_class
        results.per_sample_metrics = per_sample_results

        # Export samples if requested
        if export_samples and export_dir:
            results.export_samples(
                output_dir=export_dir,
                images=stored_images,
                ground_truths=stored_masks,
                n_samples=n_export_samples,
                palette=self.palette,
            )

        return results

    def _denormalize_image(self, image: np.ndarray) -> np.ndarray:
        """Denormalize image for visualization."""
        # Default ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        # Get normalization from config if available
        aug_config = self.config.get("augmentations", {}).get("val", [])
        for aug in aug_config:
            if aug.get("name") == "normalize":
                mean = np.array(aug.get("mean", mean))
                std = np.array(aug.get("std", std))
                break

        # Denormalize
        image = image * std + mean
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
        return image

    def _outputs_to_predictions(self, outputs: torch.Tensor) -> torch.Tensor:
        """Convert model outputs to predictions."""
        if self.task == "binary":
            # Sigmoid + threshold
            probs = torch.sigmoid(outputs)
            predictions = (probs > self.eval_config.get("thresh", 0.5)).long()
            predictions = predictions.squeeze(1)  # Remove channel dim
        elif self.task == "multiclass":
            # Argmax
            predictions = outputs.argmax(dim=1)
        else:  # regression
            predictions = outputs.squeeze(1)

        return predictions

    def _compute_sample_metrics(
        self,
        pred: np.ndarray,
        target: np.ndarray,
    ) -> dict[str, float]:
        """Compute metrics for a single sample."""
        metrics = {}

        if self.task == "binary":
            # Binary metrics
            pred_binary = pred.astype(bool)
            target_binary = target.astype(bool)

            intersection = (pred_binary & target_binary).sum()
            union = (pred_binary | target_binary).sum()
            pred_sum = pred_binary.sum()
            target_sum = target_binary.sum()

            metrics["IoU"] = float(intersection / union) if union > 0 else 1.0
            metrics["Dice"] = (
                float(2 * intersection / (pred_sum + target_sum))
                if (pred_sum + target_sum) > 0
                else 1.0
            )

        elif self.task == "multiclass":
            # Per-sample mIoU
            ious = []
            for cls in range(self.num_classes):
                pred_cls = pred == cls
                target_cls = target == cls

                intersection = (pred_cls & target_cls).sum()
                union = (pred_cls | target_cls).sum()

                if union > 0:
                    ious.append(intersection / union)

            metrics["mIoU"] = float(np.mean(ious)) if ious else 0.0

            # Pixel accuracy
            metrics["pixel_accuracy"] = float((pred == target).mean())

        else:  # regression
            metrics["MSE"] = float(((pred - target) ** 2).mean())
            metrics["MAE"] = float(np.abs(pred - target).mean())

        return metrics

    def evaluate_with_advanced_metrics(
        self,
        dataloader: DataLoader | None = None,
    ) -> EvaluationResults:
        """
        Run evaluation with advanced metrics from segmentation-evaluation package.

        This method computes additional metrics like Soft Panoptic Quality
        and Mean Average Precision.

        Args:
            dataloader: Optional dataloader.

        Returns:
            EvaluationResults with all metrics.
        """
        # Run standard evaluation
        results = self.evaluate(dataloader)

        # Try to compute advanced metrics
        try:
            from metrics.evaluate import evaluate_segmentation

            logger.info("Computing advanced metrics...")

            soft_pq_values = []
            map_values = []

            # Recompute with advanced metrics
            if dataloader is None:
                dataloader = self._create_dataloader()

            with Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]Advanced Metrics"),
                BarColumn(bar_width=30),
                MofNCompleteColumn(),
                TextColumn("•"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("adv_metrics", total=len(dataloader))
                with torch.no_grad():
                    for batch in dataloader:
                        images = batch["image"].to(self.device)
                        masks = batch["mask"].numpy()

                        outputs = self.model(images)
                        predictions = self._outputs_to_predictions(outputs)
                        predictions_np = predictions.cpu().numpy()

                        for pred, target in zip(predictions_np, masks):
                            try:
                                adv_result = evaluate_segmentation(
                                    y_true=target,
                                    y_pred=pred,
                                    thresh=self.eval_config.get("thresh", 0.5),
                                    iou_high=self.eval_config.get("iou_high", 0.5),
                                    iou_low=self.eval_config.get("iou_low", 0.05),
                                    soft_pq_method=self.eval_config.get("soft_pq_method", "sqrt"),
                                )

                                if "soft_pq" in adv_result:
                                    soft_pq_values.append(adv_result["soft_pq"])
                                if "mean_average_precision" in adv_result:
                                    map_values.append(adv_result["mean_average_precision"])

                            except Exception as e:
                                logger.debug(f"Advanced metric computation failed: {e}")

                        progress.update(task, advance=1)

            # Add advanced metrics to results
            if soft_pq_values:
                results.metrics["soft_pq"] = float(np.mean(soft_pq_values))
            if map_values:
                results.metrics["mAP"] = float(np.mean(map_values))

            logger.info("Advanced metrics computed successfully")

        except ImportError:
            logger.warning(
                "segmentation-evaluation package not found. "
                "Install with: pip install segmentation-evaluation"
            )

        return results


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    task: str = "multiclass",
    num_classes: int = 2,
    device: str | None = None,
    eval_config: dict[str, Any] | None = None,
) -> MetricResults:
    """
    Evaluate a model on a dataloader.

    Convenience function for quick evaluation without creating an Evaluator.

    Args:
        model: Model to evaluate.
        dataloader: DataLoader with evaluation data.
        task: Task type ('binary', 'multiclass', 'regression').
        num_classes: Number of classes.
        device: Device to use.
        eval_config: Evaluation configuration.

    Returns:
        MetricResults with computed metrics.

    Example:
        >>> results = evaluate_model(model, val_loader, num_classes=10)
        >>> print(results.metrics)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    model = model.to(device)
    model.eval()

    eval_config = eval_config or {}
    metrics_calculator = build_metrics(
        task=task,
        num_classes=num_classes,
        eval_config=eval_config,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Evaluating"),
        BarColumn(bar_width=30),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        eval_task = progress.add_task("eval", total=len(dataloader))
        with torch.no_grad():
            for batch in dataloader:
                images = batch["image"].to(device)
                masks = batch["mask"]

                outputs = model(images)

                # Convert to predictions
                if task == "binary":
                    thresh = eval_config.get("thresh", 0.5)
                    predictions = (torch.sigmoid(outputs) > thresh).long().squeeze(1)
                elif task == "multiclass":
                    predictions = outputs.argmax(dim=1)
                else:
                    predictions = outputs.squeeze(1)

                metrics_calculator.update(predictions.cpu().numpy(), masks.numpy())
                progress.update(eval_task, advance=1)

    return metrics_calculator.compute()
