"""
Trainer class for model training.

The Trainer handles the complete training loop including:
- Model initialization
- Optimizer and scheduler setup
- Training and validation loops
- Checkpointing
- Logging and experiment tracking
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from torch.optim.lr_scheduler import LRScheduler

from altair.core.config import Config
from altair.core.run import Run, RunStatus
from altair.data import build_dataloaders
from altair.engine.losses import build_loss
from altair.engine.metrics import build_metrics
from altair.models import build_model
from altair.utils.console import (
    console,
    create_training_progress,
    format_duration,
    print_epoch_summary,
    print_header,
    print_info,
    print_model_summary,
    print_success,
    print_training_complete,
    print_training_start,
    print_warning,
)

logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer for segmentation models.

    Handles the complete training pipeline including model setup, optimization,
    training/validation loops, checkpointing, and experiment tracking.

    Args:
        config: Experiment configuration (Config object, dict, or path).
        resume: Optional path to checkpoint or run ID to resume from.

    Example:
        >>> trainer = Trainer("configs/unet.yaml")
        >>> run = trainer.fit()
        >>> print(run.metrics)
    """

    def __init__(
        self,
        config: Config | dict | str | Path,
        resume: str | Path | None = None,
    ):
        # Load configuration
        if isinstance(config, (str, Path)):
            self.config = Config.from_yaml(config)
        elif isinstance(config, dict):
            self.config = Config.from_dict(config)
        else:
            self.config = config

        self.resume = resume

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Set random seed
        if self.config.experiment.seed is not None:
            self._set_seed(self.config.experiment.seed)

        # Initialize components (lazy)
        self._model: nn.Module | None = None
        self._optimizer: torch.optim.Optimizer | None = None
        self._scheduler: Any | None = None
        self._loss_fn: nn.Module | None = None
        self._scaler: GradScaler | None = None
        self._train_loader = None
        self._val_loader = None
        self._tracker = None
        self._run: Run | None = None

    def _set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        import random

        import numpy as np

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @property
    def model(self) -> nn.Module:
        """Get or create the model."""
        if self._model is None:
            self._model = build_model(self.config.model)
            self._model = self._model.to(self.device)
            logger.info(f"Created model: {self.config.model.architecture}")
        return self._model

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        """Get or create the optimizer."""
        if self._optimizer is None:
            opt_config = self.config.training.optimizer
            if opt_config.name == "adam":
                self._optimizer = torch.optim.Adam(
                    self.model.parameters(),
                    lr=opt_config.lr,
                    weight_decay=opt_config.weight_decay,
                    betas=opt_config.betas,
                )
            elif opt_config.name == "adamw":
                self._optimizer = torch.optim.AdamW(
                    self.model.parameters(),
                    lr=opt_config.lr,
                    weight_decay=opt_config.weight_decay,
                    betas=opt_config.betas,
                )
            elif opt_config.name == "sgd":
                self._optimizer = torch.optim.SGD(
                    self.model.parameters(),
                    lr=opt_config.lr,
                    weight_decay=opt_config.weight_decay,
                    momentum=opt_config.momentum,
                )
            else:
                raise ValueError(f"Unknown optimizer: {opt_config.name}")
        return self._optimizer

    @property
    def scheduler(self) -> LRScheduler | None:
        """Get or create the learning rate scheduler."""
        if self._scheduler is None:
            sched_config = self.config.training.scheduler
            if sched_config.name == "none":
                self._scheduler = None
            elif sched_config.name == "cosine":
                self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.config.training.epochs - sched_config.warmup_epochs,
                    eta_min=sched_config.min_lr,
                )
            elif sched_config.name == "step":
                self._scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=sched_config.step_size,
                    gamma=sched_config.gamma,
                )
            elif sched_config.name == "plateau":
                self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer,
                    mode="min",
                    patience=sched_config.patience,
                    factor=sched_config.gamma,
                )
        return self._scheduler

    @property
    def loss_fn(self) -> nn.Module:
        """Get or create the loss function."""
        if self._loss_fn is None:
            self._loss_fn = build_loss(
                self.config.training.loss,
                self.config.training.loss_weights,
                self.config.model.task,
                self.config.model.num_classes,
            )
        return self._loss_fn

    @property
    def run(self) -> Run:
        """Get or create the run."""
        if self._run is None:
            self._run = Run.create(
                name=self.config.experiment.name,
                project=self.config.experiment.project,
                config=self.config.to_dict(),
                output_dir=self.config.experiment.output_dir,
            )
        return self._run

    def _setup_data(self) -> None:
        """Set up data loaders."""
        if self._train_loader is None:
            self._train_loader, self._val_loader = build_dataloaders(
                self.config.data,
                self.config.augmentations,
            )
            logger.info(
                f"Created data loaders: {len(self._train_loader)} train batches, "
                f"{len(self._val_loader)} val batches"
            )

    def _setup_tracking(self) -> None:
        """Set up experiment tracking."""
        if self._tracker is None and self.config.tracking.backend != "none":
            from altair.tracking import build_tracker

            self._tracker = build_tracker(self.config.tracking, self.run)
            logger.info(f"Set up tracking: {self.config.tracking.backend}")

    def fit(self) -> Run:
        """
        Run the full training loop.

        Returns:
            Run object with training results and metadata.
        """
        start_time = time.time()

        # Print header
        print_header("Training Setup")

        # Setup
        print_info("Loading data...")
        self._setup_data()
        logger.info(
            f"Created data loaders: {len(self._train_loader)} train batches, "
            f"{len(self._val_loader)} val batches"
        )
        print_success(
            f"Loaded {len(self._train_loader)} train batches, "
            f"{len(self._val_loader)} val batches"
        )

        self._setup_tracking()

        # Print model summary
        num_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model: {self.config.model.architecture} with {num_params:,} parameters")
        print_model_summary(
            architecture=self.config.model.architecture,
            encoder=self.config.model.encoder or "custom",
            num_classes=self.config.model.num_classes,
            task=self.config.model.task,
            num_params=num_params,
        )

        # Initialize AMP scaler if enabled
        use_amp = self.config.training.amp and self.device.type == "cuda"
        self._scaler = GradScaler() if use_amp else None

        if use_amp:
            logger.info("Mixed precision training enabled (AMP)")
            print_info("Mixed precision training enabled (AMP)")

        # Track best metric for checkpointing
        best_metric = None
        start_epoch = 1
        ckpt_config = self.config.checkpointing

        # Handle checkpoint resumption
        if self.resume is not None:
            print_info(f"Resuming from checkpoint: {self.resume}")
            checkpoint_data = self._load_checkpoint(self.resume)
            start_epoch = checkpoint_data.get("epoch", 0) + 1
            # Restore best metric from checkpoint if available
            if checkpoint_data.get("metrics"):
                best_metric = checkpoint_data["metrics"].get(ckpt_config.monitor)
            logger.info(f"Resuming from epoch {start_epoch}, best_metric={best_metric}")
            print_success(f"Resumed from epoch {start_epoch - 1}")

        # Update run status
        self.run.update_status(RunStatus.RUNNING)
        logger.info(f"Starting training run: {self.run.id}")

        # Print training start info
        print_training_start(
            run_id=self.run.id,
            epochs=self.config.training.epochs,
            device=str(self.device),
            output_dir=str(self.run.output_dir),
        )

        console.print()
        print_header("Training Progress")

        try:
            for epoch in range(start_epoch, self.config.training.epochs + 1):
                # Training
                train_metrics = self._train_epoch(epoch, use_amp)

                # Validation
                val_metrics = self._validate_epoch(epoch, use_amp)

                # Combine metrics
                metrics = {**train_metrics, **val_metrics, "epoch": epoch}

                # Log metrics
                if self._tracker:
                    self._tracker.log_metrics(metrics, step=epoch)

                # Learning rate scheduling
                if self.scheduler is not None:
                    if isinstance(
                        self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                    ):
                        self.scheduler.step(val_metrics.get("val/loss", 0))
                    else:
                        self.scheduler.step()

                # Checkpointing
                current_metric = metrics.get(ckpt_config.monitor)
                is_best = False
                if current_metric is not None:
                    is_best = self._is_better(current_metric, best_metric, ckpt_config.mode)
                    if is_best:
                        best_metric = current_metric
                        if ckpt_config.save_best:
                            self._save_checkpoint("best.pt", metrics, epoch)

                if ckpt_config.save_last:
                    self._save_checkpoint("last.pt", metrics, epoch)

                if (
                    ckpt_config.save_every_n_epochs
                    and epoch % ckpt_config.save_every_n_epochs == 0
                ):
                    self._save_checkpoint(f"epoch_{epoch:04d}.pt", metrics, epoch)

                # Update run metrics
                self.run.update_metrics(metrics)

                # Log and print epoch summary
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Epoch {epoch}/{self.config.training.epochs} - "
                    f"train_loss: {train_metrics['train/loss']:.4f}, "
                    f"val_loss: {val_metrics['val/loss']:.4f}, "
                    f"lr: {lr:.2e}"
                    + (" [BEST]" if is_best else "")
                )
                display_metrics = {k.replace("val/", ""): v for k, v in val_metrics.items() if k != "val/loss"}
                print_epoch_summary(
                    epoch=epoch,
                    total_epochs=self.config.training.epochs,
                    train_loss=train_metrics["train/loss"],
                    val_loss=val_metrics["val/loss"],
                    metrics=display_metrics,
                    lr=lr,
                    best=is_best,
                )

            # Training complete
            self.run.update_status(RunStatus.COMPLETED)
            logger.info(f"Training completed successfully. Best metric: {best_metric}")

            # Print completion summary
            duration = format_duration(time.time() - start_time)
            console.print()
            print_training_complete(
                run_id=self.run.id,
                best_metric=best_metric or 0.0,
                best_checkpoint=str(self.run.best_checkpoint) if self.run.best_checkpoint else "N/A",
                duration=duration,
            )

        except KeyboardInterrupt:
            console.print()
            logger.warning("Training interrupted by user")
            print_warning("Training interrupted by user")
            self.run.update_status(RunStatus.INTERRUPTED)
            self._save_checkpoint("interrupted.pt", {}, epoch)

        except Exception as e:
            console.print()
            logger.error(f"Training failed with error: {e}", exc_info=True)
            from altair.utils.console import print_error
            print_error(f"Training failed: {e}")
            self.run.update_status(RunStatus.FAILED)
            raise

        finally:
            if self._tracker:
                self._tracker.finish()

        return self.run

    def _train_epoch(self, epoch: int, use_amp: bool) -> dict[str, float]:
        """Run one training epoch."""
        from rich.progress import Progress, TextColumn, BarColumn, MofNCompleteColumn

        self.model.train()
        total_loss = 0.0
        num_batches = len(self._train_loader)

        with Progress(
            TextColumn("  [cyan]Train[/cyan]"),
            BarColumn(bar_width=25),
            MofNCompleteColumn(),
            TextColumn("loss: {task.fields[loss]:.4f}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("train", total=num_batches, loss=0.0)

            for batch_idx, batch in enumerate(self._train_loader):
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)

                # Forward pass
                self.optimizer.zero_grad()

                with autocast(enabled=use_amp):
                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, masks)

                # Backward pass
                if use_amp and self._scaler is not None:
                    self._scaler.scale(loss).backward()
                    if self.config.training.gradient_clip:
                        self._scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.training.gradient_clip
                        )
                    self._scaler.step(self.optimizer)
                    self._scaler.update()
                else:
                    loss.backward()
                    if self.config.training.gradient_clip:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.training.gradient_clip
                        )
                    self.optimizer.step()

                total_loss += loss.item()
                progress.update(task, advance=1, loss=loss.item())

        return {"train/loss": total_loss / num_batches}

    @torch.no_grad()
    def _validate_epoch(self, epoch: int, use_amp: bool) -> dict[str, float]:
        """Run one validation epoch with metric computation."""
        from rich.progress import Progress, TextColumn, BarColumn, MofNCompleteColumn

        self.model.eval()
        total_loss = 0.0
        num_batches = len(self._val_loader)

        # Initialize metrics calculator
        metrics_calculator = build_metrics(
            task=self.config.model.task,
            num_classes=self.config.model.num_classes,
            eval_config=self.config.evaluation.model_dump() if hasattr(self.config, 'evaluation') else {},
        )

        with Progress(
            TextColumn("  [green]Val[/green]  "),
            BarColumn(bar_width=25),
            MofNCompleteColumn(),
            TextColumn("loss: {task.fields[loss]:.4f}"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("val", total=num_batches, loss=0.0)

            for batch in self._val_loader:
                images = batch["image"].to(self.device)
                masks = batch["mask"].to(self.device)

                with autocast(enabled=use_amp):
                    outputs = self.model(images)
                    loss = self.loss_fn(outputs, masks)

                total_loss += loss.item()

                # Convert outputs to predictions for metrics
                predictions = self._outputs_to_predictions(outputs)

                # Update metrics
                metrics_calculator.update(predictions.cpu(), masks.cpu())

                progress.update(task, advance=1, loss=loss.item())

        # Compute final metrics
        metric_results = metrics_calculator.compute()

        # Build results dictionary
        val_metrics = {"val/loss": total_loss / num_batches}

        # Add computed metrics with val/ prefix
        for name, value in metric_results.metrics.items():
            # Skip per-class metrics in main log (they're stored separately)
            if "/" not in name:
                val_metrics[f"val/{name}"] = value

        return val_metrics

    def _outputs_to_predictions(self, outputs: torch.Tensor) -> torch.Tensor:
        """Convert model outputs to predictions."""
        task = self.config.model.task
        if task == "binary":
            # Sigmoid + threshold
            thresh = 0.5
            if hasattr(self.config, 'evaluation'):
                thresh = self.config.evaluation.thresh
            probs = torch.sigmoid(outputs)
            predictions = (probs > thresh).long()
            predictions = predictions.squeeze(1)  # Remove channel dim
        elif task == "multiclass":
            # Argmax
            predictions = outputs.argmax(dim=1)
        else:  # regression
            predictions = outputs.squeeze(1)

        return predictions

    def _is_better(
        self, current: float, best: float | None, mode: str
    ) -> bool:
        """Check if current metric is better than best."""
        if best is None:
            return True
        if mode == "max":
            return current > best
        return current < best

    def _save_checkpoint(self, filename: str, metrics: dict, epoch: int) -> None:
        """Save a checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config.to_dict(),
            "metrics": metrics,
            "run_id": self.run.id,
            "epoch": epoch,
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        if self._scaler is not None:
            checkpoint["scaler_state_dict"] = self._scaler.state_dict()

        path = self.run.checkpoints_dir / filename
        torch.save(checkpoint, path)
        logger.debug(f"Saved checkpoint: {path}")

    def _load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        """
        Load a checkpoint for resuming training.

        Args:
            path: Path to checkpoint file.

        Returns:
            Dictionary containing checkpoint data including epoch and metrics.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        logger.info(f"Loading checkpoint from: {path}")
        checkpoint = torch.load(path, map_location=self.device)

        # Restore model state
        self.model.load_state_dict(checkpoint["model_state_dict"])
        logger.debug("Restored model state")

        # Restore optimizer state
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.debug("Restored optimizer state")

        # Restore scheduler state if available
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            logger.debug("Restored scheduler state")

        # Restore scaler state if available
        if self._scaler is not None and "scaler_state_dict" in checkpoint:
            self._scaler.load_state_dict(checkpoint["scaler_state_dict"])
            logger.debug("Restored scaler state")

        return checkpoint
