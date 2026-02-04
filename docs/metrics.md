# Metrics

Altair provides comprehensive metrics for evaluating segmentation models.

## Overview

Metrics are computed automatically during:
- **Training**: Validation metrics computed each epoch
- **Evaluation**: Full metrics with `alt.evaluate()`

## Standard Metrics

### Multi-class Segmentation

| Metric | Description |
|--------|-------------|
| `mIoU` | Mean Intersection over Union across all classes |
| `mDice` | Mean Dice coefficient (F1 score) across all classes |
| `mPrecision` | Mean precision across all classes |
| `mRecall` | Mean recall across all classes |
| `mF1` | Mean F1 score across all classes |
| `pixel_accuracy` | Overall pixel accuracy |

Per-class versions are also computed (e.g., `IoU/class_name`).

### Binary Segmentation

| Metric | Description |
|--------|-------------|
| `IoU` | Intersection over Union (Jaccard index) |
| `Dice` | Dice coefficient (F1 score) |
| `Precision` | True positives / (True positives + False positives) |
| `Recall` | True positives / (True positives + False negatives) |
| `Specificity` | True negatives / (True negatives + False positives) |
| `Accuracy` | Overall accuracy |

### Regression

| Metric | Description |
|--------|-------------|
| `MSE` | Mean Squared Error |
| `MAE` | Mean Absolute Error |
| `RMSE` | Root Mean Squared Error |
| `R2` | R-squared (coefficient of determination) |

## Advanced Metrics

When the `segmentation-evaluation` package is installed, additional metrics are computed:

| Metric | Description |
|--------|-------------|
| `soft_pq` | Soft Panoptic Quality |
| `mAP` | Mean Average Precision |

Install with:
```bash
pip install git+https://github.com/rkarmaka/segmentation-evaluation.git
```

## Configuration

Metrics configuration is set in the `evaluation` section of your config:

```yaml
evaluation:
  thresh: 0.5           # Threshold for binary predictions
  iou_high: 0.5         # Upper IoU bound for soft PQ
  iou_low: 0.05         # Lower IoU bound for soft PQ
  soft_pq_method: "sqrt"  # Method for soft PQ ("sqrt" or "linear")
```

## Usage

### During Training

Metrics are computed automatically during validation:

```python
import altair as alt

run = alt.train("config.yaml")
print(run.metrics)
# {'val/loss': 0.15, 'val/mIoU': 0.82, 'val/mDice': 0.85, ...}
```

Training logs show key metrics:
```
Epoch 10/100 - train_loss: 0.2341, val_loss: 0.1523, mIoU: 0.8234, lr: 0.0001
```

### Evaluation

Full evaluation with `alt.evaluate()`:

```python
import altair as alt

results = alt.evaluate("run_id", data="path/to/test")

# Aggregate metrics
print(results.metrics)
# {'mIoU': 0.82, 'mDice': 0.85, 'mPrecision': 0.88, ...}

# Per-class metrics
print(results.per_class_metrics)
# {'IoU': {0: 0.95, 1: 0.78, 2: 0.73}, ...}

# Per-sample metrics
print(results.per_sample_metrics[0])
# {'mIoU': 0.84, 'pixel_accuracy': 0.92, 'image_path': '...'}

# Save results
results.to_csv("results.csv")
results.to_json("results.json")
```

### Standalone Metrics

Use metrics directly:

```python
from altair.engine.metrics import SegmentationMetrics, BinaryMetrics

# Multi-class
metrics = SegmentationMetrics(num_classes=10)
metrics.update(predictions, targets)
results = metrics.compute()
print(results.metrics)

# Binary
metrics = BinaryMetrics(threshold=0.5)
metrics.update(predictions, targets)
results = metrics.compute()
print(results.metrics)
```

## Metric Details

### IoU (Jaccard Index)

```
IoU = TP / (TP + FP + FN)
```

Measures overlap between prediction and ground truth. Range: [0, 1].

### Dice Coefficient

```
Dice = 2 * TP / (2 * TP + FP + FN)
```

Also known as F1 score. More sensitive to small objects than IoU.

### Precision & Recall

```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
```

Precision measures false positive rate; recall measures false negative rate.

## Best Practices

1. **Use mIoU for multi-class**: It's the standard metric for semantic segmentation
2. **Use Dice for medical imaging**: Common in medical applications, handles class imbalance better
3. **Monitor validation loss too**: Metrics can plateau while loss still improves
4. **Check per-class metrics**: Identify which classes need improvement
5. **Use soft_pq for instance segmentation**: Handles over/under-segmentation

## Monitoring During Training

Set the checkpoint monitor based on your metric:

```yaml
checkpointing:
  monitor: "val/mIoU"  # For multi-class
  mode: "max"          # Higher is better

# Or for binary
checkpointing:
  monitor: "val/Dice"
  mode: "max"
```
