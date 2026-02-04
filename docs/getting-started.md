# Getting Started

This guide will help you set up Altair and run your first segmentation experiment.

## Installation

### From PyPI (when available)

```bash
pip install altair-seg
```

### From Source

```bash
git clone https://github.com/rkarmaka/altair.git
cd altair
pip install -e ".[dev]"
```

### Verify Installation

```python
import altair as alt
print(alt.__version__)
```

## Your First Experiment

### 1. Prepare Your Data

Altair supports two data formats:

**PNG Format** (simple folder structure):
```
data/
├── train/
│   ├── images/
│   │   ├── 001.png
│   │   └── 002.png
│   └── masks/
│       ├── 001.png
│       └── 002.png
└── val/
    ├── images/
    └── masks/
```

**COCO Format** (JSON annotations):
```
data/
├── train/
│   ├── 001.png
│   └── 002.png
├── val/
└── annotations/
    ├── train.json
    └── val.json
```

### 2. Create a Configuration File

Create `configs/my_experiment.yaml`:

```yaml
experiment:
  name: "my-first-segmentation"
  project: "tutorials"

model:
  architecture: "unet"
  task: "binary"  # or "multiclass", "regression"
  num_classes: 1

data:
  format: "png"
  train:
    images: "data/train/images"
    masks: "data/train/masks"
  val:
    images: "data/val/images"
    masks: "data/val/masks"
  batch_size: 16

training:
  epochs: 50
  optimizer:
    name: "adamw"
    lr: 0.001
```

### 3. Train Your Model

**Using Python:**

```python
import altair as alt

run = alt.train("configs/my_experiment.yaml")
print(f"Run ID: {run.id}")
print(f"Best checkpoint: {run.best_checkpoint}")
```

**Using CLI:**

```bash
altair train --config configs/my_experiment.yaml
```

### 4. Monitor Training

Training logs are saved to `experiments/<run_id>/logs/`.

With MLflow (default):
```bash
mlflow ui
# Open http://localhost:5000
```

### 5. Evaluate Your Model

```python
import altair as alt

results = alt.evaluate("my-first-segmentation_20240115_120000", data="data/test")
print(results.metrics)
# {'mIoU': 0.82, 'dice': 0.85, ...}

results.to_csv("results.csv")
```

### 6. Run Inference

```python
import altair as alt

predictions = alt.predict(
    "my-first-segmentation_20240115_120000",
    images="data/test/images"
)

for pred in predictions:
    print(f"{pred.image_path}: {pred.mask.shape}")
    pred.save("outputs/")
```

## Next Steps

- [Configuration Reference](configuration.md) - All configuration options
- [Models](models.md) - Available architectures (UNet, UNet++, etc.)
- [Data Formats](data-formats.md) - Detailed data preparation guide
