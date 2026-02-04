# Altair Documentation

Altair is a flexible, config-driven segmentation framework for deep learning experiments.

## Features

- **Config-driven experiments**: Define everything in YAML, run with a single command
- **Swappable components**: Easily swap models, losses, optimizers via configuration
- **Multiple architectures**: UNet, UNet++, encoder-decoder with timm backbones
- **Flexible output**: Binary segmentation, multi-class, and regression tasks
- **Experiment tracking**: Built-in MLflow integration
- **Simple API**: `import altair as alt` then `alt.train()`, `alt.evaluate()`, `alt.predict()`

## Quick Start

```python
import altair as alt

# Train a model
run = alt.train("configs/unet_resnet50.yaml")

# Evaluate on test data
results = alt.evaluate(run.id, data="path/to/test")

# Run inference
predictions = alt.predict(run.id, images="path/to/images")
```

## Installation

```bash
pip install altair-seg
```

Or install from source:

```bash
git clone https://github.com/rkarmaka/altair.git
cd altair
pip install -e ".[dev]"
```

## Documentation Sections

- [Getting Started](getting-started.md) - Installation and first experiment
- [Configuration](configuration.md) - Complete config reference
- [Models](models.md) - Available architectures
- [Metrics](metrics.md) - Evaluation metrics (IoU, Dice, etc.)
- [Visualization](visualization.md) - Visualization and sample export
- [Export](export.md) - ONNX and TorchScript export for deployment
- [Data Formats](data-formats.md) - PNG and COCO dataset support
- [API Reference](api-reference.md) - Python API documentation
- [Roadmap](roadmap.md) - Planned features and future work

## Example Notebooks

- [01 - Training](../examples/01_training.ipynb) - Train your first model
- [02 - Evaluation](../examples/02_evaluation.ipynb) - Evaluate and analyze results
- [03 - Inference](../examples/03_inference.ipynb) - Run predictions on images
- [04 - Export](../examples/04_export.ipynb) - Export models for deployment
- [05 - Custom Config](../examples/05_custom_config.ipynb) - Advanced configuration
