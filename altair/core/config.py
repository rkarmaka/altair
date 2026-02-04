"""
Configuration schemas for Altair experiments.

This module defines Pydantic models for all configuration options. Configs can be
loaded from YAML files or created programmatically.

Example:
    >>> from altair.core.config import Config
    >>> config = Config.from_yaml("configs/unet_resnet50.yaml")
    >>> print(config.model.architecture)
    'unet_resnet'
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class TaskType(str, Enum):
    """Supported segmentation task types."""

    BINARY = "binary"
    MULTICLASS = "multiclass"
    REGRESSION = "regression"


class NormLayer(str, Enum):
    """Supported normalization layers."""

    BATCHNORM = "batchnorm"
    LAYERNORM = "layernorm"
    GROUPNORM = "groupnorm"
    INSTANCENORM = "instancenorm"


class Activation(str, Enum):
    """Supported activation functions."""

    RELU = "relu"
    GELU = "gelu"
    SILU = "silu"
    LEAKY_RELU = "leaky_relu"


class BaseConfig(BaseModel):
    """Base configuration with common settings."""

    model_config = ConfigDict(
        extra="forbid",  # Raise error on unknown fields
        validate_assignment=True,  # Validate on attribute assignment
        use_enum_values=True,  # Use enum values instead of enum objects
    )


class ExperimentConfig(BaseConfig):
    """
    Experiment-level configuration.

    Attributes:
        name: Unique name for this experiment run.
        project: Project name for grouping related experiments.
        output_dir: Directory to save outputs (checkpoints, logs, etc.).
        seed: Random seed for reproducibility. If None, uses random seed.
    """

    name: str = Field(..., description="Unique experiment name")
    project: str = Field(default="default", description="Project name for grouping")
    output_dir: Path = Field(default=Path("experiments"), description="Output directory")
    seed: int | None = Field(default=42, description="Random seed (None for random)")


class ModelConfig(BaseConfig):
    """
    Model architecture configuration.

    Attributes:
        architecture: Model architecture name (e.g., 'unet', 'unet++', 'unet_resnet').
        task: Task type determining output activation and loss.
        num_classes: Number of output classes/channels.
        encoder: Encoder backbone name (for encoder-decoder architectures).
        encoder_weights: Pretrained weights ('imagenet', None, or path).
        encoder_depth: Number of encoder stages.
        decoder_channels: Channel sizes for decoder stages.
        decoder_attention: Attention mechanism in decoder (None or 'scse').
        dropout: Dropout probability in decoder.
        drop_path: Stochastic depth probability.
        norm_layer: Normalization layer type.
        activation: Activation function type.
    """

    architecture: str = Field(..., description="Model architecture name")
    task: TaskType = Field(default=TaskType.MULTICLASS, description="Task type")
    num_classes: int = Field(default=1, ge=1, description="Number of output classes")

    # Encoder settings
    encoder: str | None = Field(default=None, description="Encoder backbone name")
    encoder_weights: str | None = Field(
        default="imagenet", description="Pretrained weights or None"
    )
    encoder_depth: int = Field(default=5, ge=1, le=7, description="Encoder depth")

    # Decoder settings
    decoder_channels: list[int] = Field(
        default=[256, 128, 64, 32, 16], description="Decoder channel sizes"
    )
    decoder_attention: Literal["scse"] | None = Field(
        default=None, description="Decoder attention type"
    )

    # Regularization
    dropout: float = Field(default=0.0, ge=0.0, le=1.0, description="Dropout probability")
    drop_path: float = Field(default=0.0, ge=0.0, le=1.0, description="Stochastic depth rate")

    # Architecture components
    norm_layer: NormLayer = Field(default=NormLayer.BATCHNORM, description="Normalization layer")
    activation: Activation = Field(default=Activation.RELU, description="Activation function")

    @field_validator("decoder_channels")
    @classmethod
    def validate_decoder_channels(cls, v: list[int]) -> list[int]:
        """Ensure decoder channels are positive and decreasing."""
        if not all(c > 0 for c in v):
            raise ValueError("All decoder channels must be positive")
        return v


class DataSourceConfig(BaseConfig):
    """Configuration for a single data source (train/val/test)."""

    images: Path = Field(..., description="Path to images directory")
    masks: Path | None = Field(default=None, description="Path to masks (for PNG format)")
    annotations: Path | None = Field(default=None, description="Path to COCO JSON annotations")


class DataConfig(BaseConfig):
    """
    Data loading and processing configuration.

    Attributes:
        format: Data format ('png' or 'coco').
        train: Training data source configuration.
        val: Validation data source configuration.
        test: Optional test data source configuration.
        batch_size: Batch size for training.
        num_workers: Number of data loading workers.
        pin_memory: Whether to pin memory for faster GPU transfer.
    """

    format: Literal["png", "coco"] = Field(default="png", description="Data format")
    train: DataSourceConfig = Field(..., description="Training data configuration")
    val: DataSourceConfig = Field(..., description="Validation data configuration")
    test: DataSourceConfig | None = Field(default=None, description="Test data configuration")

    batch_size: int = Field(default=16, ge=1, description="Batch size")
    num_workers: int = Field(default=4, ge=0, description="Number of data loading workers")
    pin_memory: bool = Field(default=True, description="Pin memory for GPU transfer")

    @model_validator(mode="after")
    def validate_data_format(self) -> "DataConfig":
        """Validate that appropriate paths are provided for the format."""
        if self.format == "png":
            if self.train.masks is None:
                raise ValueError("PNG format requires 'masks' path in train config")
            if self.val.masks is None:
                raise ValueError("PNG format requires 'masks' path in val config")
        elif self.format == "coco":
            if self.train.annotations is None:
                raise ValueError("COCO format requires 'annotations' path in train config")
            if self.val.annotations is None:
                raise ValueError("COCO format requires 'annotations' path in val config")
        return self

    def validate_paths(self, check_existence: bool = True) -> list[Path]:
        """
        Validate that data paths exist on the filesystem.

        Args:
            check_existence: If True, verify each path exists. If False, just
                collect and return the paths without checking.

        Returns:
            List of validated paths.

        Raises:
            FileNotFoundError: If check_existence is True and any paths are missing.
        """
        paths_to_check: list[tuple[str, Path]] = []

        # Collect paths based on format
        for split_name, source in [("train", self.train), ("val", self.val)]:
            paths_to_check.append((f"{split_name}.images", source.images))

            if self.format == "png" and source.masks is not None:
                paths_to_check.append((f"{split_name}.masks", source.masks))
            elif self.format == "coco" and source.annotations is not None:
                paths_to_check.append((f"{split_name}.annotations", source.annotations))

        # Add test paths if present
        if self.test is not None:
            paths_to_check.append(("test.images", self.test.images))
            if self.format == "png" and self.test.masks is not None:
                paths_to_check.append(("test.masks", self.test.masks))
            elif self.format == "coco" and self.test.annotations is not None:
                paths_to_check.append(("test.annotations", self.test.annotations))

        if check_existence:
            missing = []
            for name, path in paths_to_check:
                if not path.exists():
                    missing.append(f"  - {name}: {path}")

            if missing:
                raise FileNotFoundError(
                    f"The following data paths do not exist:\n" + "\n".join(missing)
                )

        return [path for _, path in paths_to_check]


class OptimizerConfig(BaseConfig):
    """
    Optimizer configuration.

    Attributes:
        name: Optimizer name ('adam', 'adamw', 'sgd').
        lr: Learning rate.
        weight_decay: Weight decay (L2 regularization).
        betas: Adam beta parameters.
        momentum: SGD momentum.
    """

    name: Literal["adam", "adamw", "sgd"] = Field(default="adamw", description="Optimizer name")
    lr: float = Field(default=1e-4, gt=0, description="Learning rate")
    weight_decay: float = Field(default=0.01, ge=0, description="Weight decay")
    betas: tuple[float, float] = Field(default=(0.9, 0.999), description="Adam betas")
    momentum: float = Field(default=0.9, ge=0, le=1, description="SGD momentum")


class SchedulerConfig(BaseConfig):
    """
    Learning rate scheduler configuration.

    Attributes:
        name: Scheduler name ('cosine', 'step', 'plateau', 'none').
        warmup_epochs: Number of warmup epochs.
        min_lr: Minimum learning rate for cosine scheduler.
        step_size: Step size for step scheduler.
        gamma: Decay factor for step scheduler.
        patience: Patience for plateau scheduler.
    """

    name: Literal["cosine", "step", "plateau", "none"] = Field(
        default="cosine", description="Scheduler name"
    )
    warmup_epochs: int = Field(default=5, ge=0, description="Warmup epochs")
    min_lr: float = Field(default=1e-6, ge=0, description="Minimum learning rate")

    # Step scheduler
    step_size: int = Field(default=30, ge=1, description="Step size for step scheduler")
    gamma: float = Field(default=0.1, gt=0, le=1, description="Decay factor")

    # Plateau scheduler
    patience: int = Field(default=10, ge=1, description="Patience for plateau scheduler")


class TrainingConfig(BaseConfig):
    """
    Training loop configuration.

    Attributes:
        epochs: Number of training epochs.
        optimizer: Optimizer configuration.
        scheduler: Learning rate scheduler configuration.
        loss: Loss function name, combination (e.g., "ce+dice"), or list of losses.
        loss_weights: Weights for combined losses.
        amp: Whether to use automatic mixed precision.
        gradient_clip: Maximum gradient norm (None to disable).
        accumulation_steps: Gradient accumulation steps.
    """

    epochs: int = Field(default=100, ge=1, description="Number of epochs")
    optimizer: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)

    loss: str | list[str] = Field(
        default="ce+dice",
        description="Loss function name, combination (e.g., 'ce+dice'), or list of losses"
    )
    loss_weights: dict[str, float] = Field(
        default_factory=lambda: {"ce": 1.0, "dice": 1.0}, description="Loss weights"
    )

    amp: bool = Field(default=True, description="Use automatic mixed precision")
    gradient_clip: float | None = Field(default=1.0, description="Gradient clipping norm")
    accumulation_steps: int = Field(default=1, ge=1, description="Gradient accumulation steps")

    @field_validator("loss", mode="before")
    @classmethod
    def normalize_loss(cls, v: str | list[str]) -> str | list[str]:
        """Accept both old underscore format and new plus format."""
        # If it's already a list, return as-is
        if isinstance(v, list):
            return v
        # Keep as string - parsing happens in build_loss
        return v


class AugmentationItemConfig(BaseConfig):
    """Single augmentation configuration."""

    name: str = Field(..., description="Augmentation name")
    p: float = Field(default=1.0, ge=0, le=1, description="Probability")

    # Allow extra fields for augmentation-specific parameters
    model_config = ConfigDict(extra="allow")


class AugmentationConfig(BaseConfig):
    """
    Data augmentation configuration.

    Attributes:
        train: List of augmentations for training.
        val: List of augmentations for validation.
    """

    train: list[AugmentationItemConfig] = Field(default_factory=list)
    val: list[AugmentationItemConfig] = Field(default_factory=list)


class TrackingConfig(BaseConfig):
    """
    Experiment tracking configuration.

    Attributes:
        backend: Tracking backend ('mlflow', 'wandb', 'none').
        uri: Tracking server URI or local path.
        log_every_n_steps: Logging frequency in steps.
        log_images: Whether to log sample images.
        log_images_every_n_epochs: Image logging frequency.
    """

    backend: Literal["mlflow", "wandb", "none"] = Field(
        default="mlflow", description="Tracking backend"
    )
    uri: str = Field(default="mlruns", description="Tracking URI")
    log_every_n_steps: int = Field(default=50, ge=1, description="Logging frequency")
    log_images: bool = Field(default=True, description="Log sample images")
    log_images_every_n_epochs: int = Field(default=5, ge=1, description="Image logging frequency")


class CheckpointConfig(BaseConfig):
    """
    Model checkpointing configuration.

    Attributes:
        save_best: Whether to save the best model.
        save_last: Whether to save the last model.
        save_every_n_epochs: Save checkpoint every N epochs (None to disable).
        monitor: Metric to monitor for best model.
        mode: Whether to maximize or minimize the monitored metric.
    """

    save_best: bool = Field(default=True, description="Save best model")
    save_last: bool = Field(default=True, description="Save last model")
    save_every_n_epochs: int | None = Field(default=None, description="Periodic save frequency")
    monitor: str = Field(default="val/mIoU", description="Metric to monitor")
    mode: Literal["min", "max"] = Field(default="max", description="Optimization direction")


class EvaluationConfig(BaseConfig):
    """
    Evaluation metrics configuration.

    These parameters are passed to the segmentation-evaluation package.

    Attributes:
        thresh: Matching threshold for metrics.
        iou_high: Upper IoU bound for soft PQ.
        iou_low: Lower IoU bound for soft PQ.
        soft_pq_method: Method for soft PQ calculation.
    """

    thresh: float = Field(default=0.5, ge=0, le=1, description="Matching threshold")
    iou_high: float = Field(default=0.5, ge=0, le=1, description="Upper IoU bound")
    iou_low: float = Field(default=0.05, ge=0, le=1, description="Lower IoU bound")
    soft_pq_method: Literal["sqrt", "linear"] = Field(
        default="sqrt", description="Soft PQ method"
    )


class Config(BaseConfig):
    """
    Complete experiment configuration.

    This is the top-level configuration object that contains all settings
    for an experiment. It can be loaded from YAML or created programmatically.

    Example:
        >>> config = Config.from_yaml("configs/experiment.yaml")
        >>> config = Config(
        ...     experiment=ExperimentConfig(name="test"),
        ...     model=ModelConfig(architecture="unet", num_classes=10),
        ...     data=DataConfig(...),
        ...     training=TrainingConfig(),
        ... )
    """

    experiment: ExperimentConfig
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    augmentations: AugmentationConfig = Field(default_factory=AugmentationConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    checkpointing: CheckpointConfig = Field(default_factory=CheckpointConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """
        Load configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            A Config instance with all settings from the file.

        Raises:
            FileNotFoundError: If the config file doesn't exist.
            ValidationError: If the config file has invalid values.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        return cls.model_validate(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """
        Create configuration from a dictionary.

        Args:
            data: Dictionary with configuration values.

        Returns:
            A Config instance.
        """
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        """
        Save configuration to a YAML file.

        Args:
            path: Path to save the configuration.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert configuration to a dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        return self.model_dump()

    def validate_data_paths(self, check_existence: bool = True) -> list[Path]:
        """
        Validate that data paths exist on the filesystem.

        This is a convenience method that delegates to self.data.validate_paths().

        Args:
            check_existence: If True, verify each path exists. If False, just
                collect and return the paths without checking.

        Returns:
            List of validated paths.

        Raises:
            FileNotFoundError: If check_existence is True and any paths are missing.
        """
        return self.data.validate_paths(check_existence=check_existence)
