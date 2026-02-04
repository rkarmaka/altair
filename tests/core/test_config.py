"""Tests for the configuration system."""

import pytest
from pydantic import ValidationError

from altair.core.config import (
    AugmentationConfig,
    CheckpointConfig,
    Config,
    DataConfig,
    DataSourceConfig,
    EvaluationConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizerConfig,
    SchedulerConfig,
    TaskType,
    TrackingConfig,
    TrainingConfig,
)


class TestExperimentConfig:
    """Test cases for ExperimentConfig."""

    def test_minimal_config(self):
        """Test creating config with only required fields."""
        config = ExperimentConfig(name="test")
        assert config.name == "test"
        assert config.project == "default"
        assert config.seed == 42

    def test_full_config(self):
        """Test creating config with all fields."""
        config = ExperimentConfig(
            name="my-experiment",
            project="segmentation",
            output_dir="outputs/",
            seed=123,
        )
        assert config.name == "my-experiment"
        assert config.project == "segmentation"
        assert config.seed == 123

    def test_none_seed_allowed(self):
        """Test that seed can be None for random initialization."""
        config = ExperimentConfig(name="test", seed=None)
        assert config.seed is None


class TestModelConfig:
    """Test cases for ModelConfig."""

    def test_minimal_model_config(self):
        """Test creating model config with only required fields."""
        config = ModelConfig(architecture="unet")
        assert config.architecture == "unet"
        assert config.task == "multiclass"
        assert config.num_classes == 1
        assert config.dropout == 0.0

    def test_binary_task(self):
        """Test binary segmentation task configuration."""
        config = ModelConfig(
            architecture="unet",
            task=TaskType.BINARY,
            num_classes=1,
        )
        assert config.task == "binary"

    def test_regression_task(self):
        """Test regression task configuration."""
        config = ModelConfig(
            architecture="unet",
            task=TaskType.REGRESSION,
            num_classes=1,
        )
        assert config.task == "regression"

    def test_encoder_config(self):
        """Test encoder-based model configuration."""
        config = ModelConfig(
            architecture="unet_resnet",
            encoder="resnet50",
            encoder_weights="imagenet",
            encoder_depth=5,
        )
        assert config.encoder == "resnet50"
        assert config.encoder_weights == "imagenet"

    def test_regularization_params(self):
        """Test regularization parameters."""
        config = ModelConfig(
            architecture="unet",
            dropout=0.3,
            drop_path=0.1,
        )
        assert config.dropout == 0.3
        assert config.drop_path == 0.1

    def test_dropout_validation(self):
        """Test that dropout must be between 0 and 1."""
        with pytest.raises(ValidationError):
            ModelConfig(architecture="unet", dropout=1.5)

        with pytest.raises(ValidationError):
            ModelConfig(architecture="unet", dropout=-0.1)

    def test_num_classes_validation(self):
        """Test that num_classes must be positive."""
        with pytest.raises(ValidationError):
            ModelConfig(architecture="unet", num_classes=0)

    def test_decoder_channels(self):
        """Test custom decoder channels."""
        channels = [512, 256, 128, 64, 32]
        config = ModelConfig(architecture="unet", decoder_channels=channels)
        assert config.decoder_channels == channels


class TestDataConfig:
    """Test cases for DataConfig."""

    def test_png_format_config(self):
        """Test PNG format data configuration."""
        config = DataConfig(
            format="png",
            train=DataSourceConfig(
                images="data/train/images",
                masks="data/train/masks",
            ),
            val=DataSourceConfig(
                images="data/val/images",
                masks="data/val/masks",
            ),
        )
        assert config.format == "png"
        assert config.train.masks is not None

    def test_coco_format_config(self):
        """Test COCO format data configuration."""
        config = DataConfig(
            format="coco",
            train=DataSourceConfig(
                images="data/train",
                annotations="data/train.json",
            ),
            val=DataSourceConfig(
                images="data/val",
                annotations="data/val.json",
            ),
        )
        assert config.format == "coco"
        assert config.train.annotations is not None

    def test_png_requires_masks(self):
        """Test that PNG format requires masks path."""
        with pytest.raises(ValidationError, match="PNG format requires 'masks'"):
            DataConfig(
                format="png",
                train=DataSourceConfig(images="data/train/images"),
                val=DataSourceConfig(images="data/val/images", masks="data/val/masks"),
            )

    def test_coco_requires_annotations(self):
        """Test that COCO format requires annotations path."""
        with pytest.raises(ValidationError, match="COCO format requires 'annotations'"):
            DataConfig(
                format="coco",
                train=DataSourceConfig(images="data/train"),
                val=DataSourceConfig(images="data/val", annotations="data/val.json"),
            )

    def test_dataloader_params(self):
        """Test dataloader parameters."""
        config = DataConfig(
            format="png",
            train=DataSourceConfig(images="train/img", masks="train/mask"),
            val=DataSourceConfig(images="val/img", masks="val/mask"),
            batch_size=32,
            num_workers=8,
            pin_memory=False,
        )
        assert config.batch_size == 32
        assert config.num_workers == 8
        assert config.pin_memory is False


class TestTrainingConfig:
    """Test cases for TrainingConfig."""

    def test_default_training_config(self):
        """Test default training configuration."""
        config = TrainingConfig()
        assert config.epochs == 100
        assert config.amp is True
        assert config.gradient_clip == 1.0

    def test_optimizer_config(self):
        """Test optimizer configuration."""
        config = TrainingConfig(
            optimizer=OptimizerConfig(
                name="sgd",
                lr=0.01,
                momentum=0.9,
            )
        )
        assert config.optimizer.name == "sgd"
        assert config.optimizer.lr == 0.01

    def test_scheduler_config(self):
        """Test scheduler configuration."""
        config = TrainingConfig(
            scheduler=SchedulerConfig(
                name="step",
                step_size=10,
                gamma=0.1,
            )
        )
        assert config.scheduler.name == "step"
        assert config.scheduler.step_size == 10

    def test_loss_config(self):
        """Test loss configuration."""
        config = TrainingConfig(
            loss="focal_dice",
            loss_weights={"focal": 0.5, "dice": 1.0},
        )
        assert config.loss == "focal_dice"
        assert config.loss_weights["focal"] == 0.5


class TestConfig:
    """Test cases for the main Config class."""

    def test_from_dict(self, sample_config_dict):
        """Test creating config from dictionary."""
        config = Config.from_dict(sample_config_dict)
        assert config.experiment.name == "test-experiment"
        assert config.model.architecture == "unet"

    def test_from_yaml(self, sample_config_yaml):
        """Test loading config from YAML file."""
        config = Config.from_yaml(sample_config_yaml)
        assert config.experiment.name == "test-experiment"

    def test_to_dict(self, sample_config_dict):
        """Test converting config to dictionary."""
        config = Config.from_dict(sample_config_dict)
        result = config.to_dict()
        assert result["experiment"]["name"] == "test-experiment"
        assert result["model"]["architecture"] == "unet"

    def test_to_yaml(self, sample_config_dict, temp_dir):
        """Test saving config to YAML file."""
        config = Config.from_dict(sample_config_dict)
        output_path = temp_dir / "output.yaml"
        config.to_yaml(output_path)

        # Load and verify
        loaded = Config.from_yaml(output_path)
        assert loaded.experiment.name == config.experiment.name

    def test_missing_config_file(self):
        """Test that missing config file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            Config.from_yaml("nonexistent.yaml")

    def test_invalid_config_raises_validation_error(self):
        """Test that invalid config raises ValidationError."""
        with pytest.raises(ValidationError):
            Config.from_dict(
                {
                    "experiment": {"name": "test"},
                    "model": {"architecture": "unet"},
                    # Missing required 'data' field
                }
            )

    def test_extra_fields_rejected(self, sample_config_dict):
        """Test that unknown fields are rejected."""
        sample_config_dict["unknown_field"] = "value"
        with pytest.raises(ValidationError):
            Config.from_dict(sample_config_dict)

    def test_multiclass_config(self, sample_multiclass_config_dict):
        """Test multiclass segmentation configuration."""
        config = Config.from_dict(sample_multiclass_config_dict)
        assert config.model.task == "multiclass"
        assert config.model.num_classes == 10
        assert config.model.encoder == "resnet50"
        assert config.model.dropout == 0.2

    def test_defaults_applied(self, sample_config_dict):
        """Test that defaults are applied for optional fields."""
        config = Config.from_dict(sample_config_dict)

        # Training defaults
        assert config.training.epochs == 100
        assert config.training.amp is True

        # Tracking defaults
        assert config.tracking.backend == "mlflow"

        # Checkpoint defaults
        assert config.checkpointing.save_best is True


class TestAugmentationConfig:
    """Test cases for AugmentationConfig."""

    def test_empty_augmentations(self):
        """Test config with no augmentations."""
        config = AugmentationConfig()
        assert config.train == []
        assert config.val == []

    def test_augmentation_items(self):
        """Test augmentation item configuration."""
        config = AugmentationConfig(
            train=[
                {"name": "resize", "height": 512, "width": 512},
                {"name": "horizontal_flip", "p": 0.5},
            ],
            val=[
                {"name": "resize", "height": 512, "width": 512},
            ],
        )
        assert len(config.train) == 2
        assert config.train[0].name == "resize"
        assert config.train[1].p == 0.5


class TestEvaluationConfig:
    """Test cases for EvaluationConfig."""

    def test_default_evaluation_config(self):
        """Test default evaluation parameters."""
        config = EvaluationConfig()
        assert config.thresh == 0.5
        assert config.iou_high == 0.5
        assert config.iou_low == 0.05
        assert config.soft_pq_method == "sqrt"

    def test_custom_evaluation_config(self):
        """Test custom evaluation parameters."""
        config = EvaluationConfig(
            thresh=0.7,
            iou_high=0.7,
            iou_low=0.1,
            soft_pq_method="linear",
        )
        assert config.thresh == 0.7
        assert config.soft_pq_method == "linear"
