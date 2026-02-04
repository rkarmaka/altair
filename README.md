# Altair

A flexible, config-driven segmentation framework for deep learning experiments.

## Features

- **Config-driven**: Define experiments in YAML, run with a single command
- **Swappable components**: Easily swap models, losses, optimizers via configuration
- **Multiple architectures**: UNet, UNet++, and encoder-decoder models with timm backbones
- **Flexible output**: Binary segmentation, multi-class, and regression tasks
- **Experiment tracking**: Built-in MLflow integration
- **Simple API**: `import altair as alt` then `alt.train()`, `alt.evaluate()`, `alt.predict()`

## Installation

```bash
# From source
git clone https://github.com/rkarmaka/altair.git
cd altair
pip install -e ".[dev]"
```

## Quick Start

### Python API

```python
import altair as alt

# Train a model
run = alt.train("configs/examples/unet_binary.yaml")

# Evaluate on test data
results = alt.evaluate(run.id, data="path/to/test")
print(results.metrics)

# Run inference
predictions = alt.predict(run.id, images="path/to/images")
for pred in predictions:
    pred.save("outputs/")
```

### Command Line

```bash
# Train
altair train --config configs/examples/unet_binary.yaml

# Evaluate
altair evaluate --run exp_abc123 --data path/to/test

# Predict
altair predict --run exp_abc123 --images path/to/images --output predictions/

# List runs
altair list --project tutorials
```

## Configuration

Altair uses YAML configuration files. See `configs/examples/` for examples.

```yaml
experiment:
  name: "my-experiment"
  project: "segmentation"

model:
  architecture: "unet"
  task: "binary"  # binary, multiclass, regression
  num_classes: 1
  dropout: 0.2

data:
  format: "png"  # png or coco
  train:
    images: "data/train/images"
    masks: "data/train/masks"
  val:
    images: "data/val/images"
    masks: "data/val/masks"
  batch_size: 16

training:
  epochs: 100
  optimizer:
    name: "adamw"
    lr: 0.001
  loss: "bce_dice"
```

## Available Models

| Model | Description |
|-------|-------------|
| `unet` | Classic U-Net architecture |
| `unet++` | Nested U-Net with dense skip connections |
| `unet_encoder` | U-Net with pretrained timm encoder (recommended) |

### Pretrained Encoders

When using `unet_encoder`, you can use any timm backbone:

```yaml
model:
  architecture: "unet_encoder"
  encoder: "resnet50"        # or efficientnet_b4, convnext_tiny, etc.
  encoder_weights: "imagenet"
  num_classes: 10
```

Common encoders: `resnet18`, `resnet50`, `efficientnet_b0`, `efficientnet_b4`, `convnext_tiny`, `mobilenetv3_large_100`

## Data Formats

### PNG Format
```
data/
├── train/
│   ├── images/
│   └── masks/
└── val/
    ├── images/
    └── masks/
```

### COCO Format
```
data/
├── train/
├── val/
└── annotations/
    ├── train.json
    └── val.json
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=altair

# Format code
black altair tests
ruff check altair tests
```

## Project Structure

```
altair/
├── altair/
│   ├── core/           # Config, registry, run management
│   ├── models/         # Model architectures
│   ├── data/           # Datasets and transforms
│   ├── engine/         # Training, evaluation, prediction
│   ├── tracking/       # Experiment tracking (MLflow)
│   └── cli.py          # Command-line interface
├── configs/            # Example configurations
├── docs/               # Documentation
└── tests/              # Test suite
```

## License

MIT License
