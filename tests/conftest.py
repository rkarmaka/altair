"""
Pytest fixtures for Altair tests.

This module provides shared fixtures for testing, including sample
configurations, temporary directories, and mock data.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config_dict():
    """Return a minimal valid configuration dictionary."""
    return {
        "experiment": {
            "name": "test-experiment",
            "project": "test-project",
        },
        "model": {
            "architecture": "unet",
            "task": "binary",
            "num_classes": 1,
        },
        "data": {
            "format": "png",
            "train": {
                "images": "data/train/images",
                "masks": "data/train/masks",
            },
            "val": {
                "images": "data/val/images",
                "masks": "data/val/masks",
            },
        },
    }


@pytest.fixture
def sample_config_yaml(temp_dir, sample_config_dict):
    """Create a sample YAML config file and return its path."""
    config_path = temp_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(sample_config_dict, f)
    return config_path


@pytest.fixture
def sample_multiclass_config_dict():
    """Return a multiclass segmentation configuration."""
    return {
        "experiment": {
            "name": "multiclass-test",
            "project": "test-project",
        },
        "model": {
            "architecture": "unet_resnet",
            "task": "multiclass",
            "num_classes": 10,
            "encoder": "resnet50",
            "encoder_weights": "imagenet",
            "dropout": 0.2,
        },
        "data": {
            "format": "coco",
            "train": {
                "images": "data/train",
                "annotations": "data/train.json",
            },
            "val": {
                "images": "data/val",
                "annotations": "data/val.json",
            },
            "batch_size": 8,
        },
        "training": {
            "epochs": 50,
            "optimizer": {
                "name": "adamw",
                "lr": 0.0001,
            },
        },
    }


@pytest.fixture
def sample_image():
    """Create a sample RGB image as numpy array."""
    return np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)


@pytest.fixture
def sample_mask():
    """Create a sample segmentation mask as numpy array."""
    return np.random.randint(0, 5, (256, 256), dtype=np.uint8)


@pytest.fixture
def sample_binary_mask():
    """Create a sample binary mask as numpy array."""
    return np.random.randint(0, 2, (256, 256), dtype=np.uint8)


@pytest.fixture
def sample_image_dir(temp_dir, sample_image):
    """Create a directory with sample images."""
    from PIL import Image

    images_dir = temp_dir / "images"
    images_dir.mkdir()

    for i in range(5):
        img = Image.fromarray(sample_image)
        img.save(images_dir / f"image_{i:03d}.png")

    return images_dir


@pytest.fixture
def sample_mask_dir(temp_dir, sample_mask):
    """Create a directory with sample masks."""
    from PIL import Image

    masks_dir = temp_dir / "masks"
    masks_dir.mkdir()

    for i in range(5):
        mask = Image.fromarray(sample_mask)
        mask.save(masks_dir / f"image_{i:03d}.png")

    return masks_dir


@pytest.fixture
def sample_dataset_dir(temp_dir, sample_image, sample_mask):
    """Create a complete sample dataset with train/val splits."""
    from PIL import Image

    for split in ["train", "val"]:
        images_dir = temp_dir / split / "images"
        masks_dir = temp_dir / split / "masks"
        images_dir.mkdir(parents=True)
        masks_dir.mkdir(parents=True)

        num_samples = 10 if split == "train" else 3
        for i in range(num_samples):
            img = Image.fromarray(sample_image)
            img.save(images_dir / f"image_{i:03d}.png")

            mask = Image.fromarray(sample_mask)
            mask.save(masks_dir / f"image_{i:03d}.png")

    return temp_dir
