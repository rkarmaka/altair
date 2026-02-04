# Configuration Reference

Altair uses YAML configuration files to define experiments. This page documents all available options.

## Complete Example

```yaml
experiment:
  name: "unet-resnet50-cityscapes"
  project: "segmentation"
  output_dir: "experiments/"
  seed: 42

model:
  architecture: "unet_resnet"
  task: "multiclass"
  num_classes: 19

  # Encoder
  encoder: "resnet50"
  encoder_weights: "imagenet"
  encoder_depth: 5

  # Decoder
  decoder_channels: [256, 128, 64, 32, 16]
  decoder_attention: "scse"

  # Regularization
  dropout: 0.2
  drop_path: 0.0

  # Architecture
  norm_layer: "batchnorm"
  activation: "relu"

data:
  format: "coco"
  train:
    images: "data/train"
    annotations: "data/annotations/train.json"
  val:
    images: "data/val"
    annotations: "data/annotations/val.json"
  test:
    images: "data/test"
    annotations: "data/annotations/test.json"
  batch_size: 16
  num_workers: 4
  pin_memory: true

training:
  epochs: 100

  optimizer:
    name: "adamw"
    lr: 0.0001
    weight_decay: 0.01
    betas: [0.9, 0.999]

  scheduler:
    name: "cosine"
    warmup_epochs: 5
    min_lr: 0.000001

  loss: "ce_dice"
  loss_weights:
    ce: 1.0
    dice: 1.0

  amp: true
  gradient_clip: 1.0
  accumulation_steps: 1

augmentations:
  train:
    - name: "resize"
      height: 512
      width: 512
    - name: "horizontal_flip"
      p: 0.5
    - name: "vertical_flip"
      p: 0.5
    - name: "rotate"
      limit: 30
      p: 0.5
    - name: "color_jitter"
      brightness: 0.2
      contrast: 0.2
      p: 0.3
    - name: "normalize"
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
  val:
    - name: "resize"
      height: 512
      width: 512
    - name: "normalize"
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]

evaluation:
  thresh: 0.5
  iou_high: 0.5
  iou_low: 0.05
  soft_pq_method: "sqrt"

tracking:
  backend: "mlflow"
  uri: "mlruns"
  log_every_n_steps: 50
  log_images: true
  log_images_every_n_epochs: 5

checkpointing:
  save_best: true
  save_last: true
  save_every_n_epochs: null
  monitor: "val/mIoU"
  mode: "max"
```

## Section Reference

### experiment

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | **required** | Unique experiment name |
| `project` | string | `"default"` | Project for grouping experiments |
| `output_dir` | path | `"experiments/"` | Output directory |
| `seed` | int \| null | `42` | Random seed (null for random) |

### model

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `architecture` | string | **required** | Model architecture: `"unet"`, `"unet++"`, `"unet_resnet"` |
| `task` | string | `"multiclass"` | Task type: `"binary"`, `"multiclass"`, `"regression"` |
| `num_classes` | int | `1` | Number of output classes |
| `encoder` | string | `null` | Encoder backbone (for encoder-decoder models) |
| `encoder_weights` | string | `"imagenet"` | Pretrained weights: `"imagenet"`, `null`, or path |
| `encoder_depth` | int | `5` | Number of encoder stages (1-7) |
| `decoder_channels` | list[int] | `[256,128,64,32,16]` | Decoder channel sizes |
| `decoder_attention` | string | `null` | Attention: `null` or `"scse"` |
| `dropout` | float | `0.0` | Dropout probability (0-1) |
| `drop_path` | float | `0.0` | Stochastic depth rate (0-1) |
| `norm_layer` | string | `"batchnorm"` | Normalization: `"batchnorm"`, `"layernorm"`, `"groupnorm"` |
| `activation` | string | `"relu"` | Activation: `"relu"`, `"gelu"`, `"silu"` |

### data

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `format` | string | `"png"` | Data format: `"png"` or `"coco"` |
| `train` | object | **required** | Training data paths |
| `val` | object | **required** | Validation data paths |
| `test` | object | `null` | Test data paths (optional) |
| `batch_size` | int | `16` | Batch size |
| `num_workers` | int | `4` | DataLoader workers |
| `pin_memory` | bool | `true` | Pin memory for GPU |

**For PNG format:**
```yaml
train:
  images: "path/to/images"
  masks: "path/to/masks"
```

**For COCO format:**
```yaml
train:
  images: "path/to/images"
  annotations: "path/to/annotations.json"
```

### training

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `epochs` | int | `100` | Number of epochs |
| `optimizer` | object | see below | Optimizer config |
| `scheduler` | object | see below | LR scheduler config |
| `loss` | string | `"ce_dice"` | Loss function |
| `loss_weights` | dict | `{ce:1, dice:1}` | Loss weights |
| `amp` | bool | `true` | Mixed precision training |
| `gradient_clip` | float | `1.0` | Gradient clipping (null to disable) |
| `accumulation_steps` | int | `1` | Gradient accumulation |

**optimizer:**
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | `"adamw"` | `"adam"`, `"adamw"`, `"sgd"` |
| `lr` | float | `0.0001` | Learning rate |
| `weight_decay` | float | `0.01` | Weight decay |
| `betas` | list | `[0.9, 0.999]` | Adam betas |
| `momentum` | float | `0.9` | SGD momentum |

**scheduler:**
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | `"cosine"` | `"cosine"`, `"step"`, `"plateau"`, `"none"` |
| `warmup_epochs` | int | `5` | Warmup epochs |
| `min_lr` | float | `1e-6` | Minimum LR (cosine) |
| `step_size` | int | `30` | Step size (step) |
| `gamma` | float | `0.1` | Decay factor |
| `patience` | int | `10` | Patience (plateau) |

### augmentations

List of augmentation transforms. Each item has:
- `name`: Augmentation name (maps to albumentations)
- `p`: Probability (default 1.0)
- Additional parameters specific to each augmentation

Common augmentations:
- `resize`: `height`, `width`
- `horizontal_flip`: `p`
- `vertical_flip`: `p`
- `rotate`: `limit`, `p`
- `color_jitter`: `brightness`, `contrast`, `saturation`, `hue`, `p`
- `normalize`: `mean`, `std`

### tracking

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `backend` | string | `"mlflow"` | `"mlflow"`, `"wandb"`, `"none"` |
| `uri` | string | `"mlruns"` | Tracking server URI |
| `log_every_n_steps` | int | `50` | Logging frequency |
| `log_images` | bool | `true` | Log sample images |
| `log_images_every_n_epochs` | int | `5` | Image logging frequency |

### checkpointing

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `save_best` | bool | `true` | Save best model |
| `save_last` | bool | `true` | Save last model |
| `save_every_n_epochs` | int | `null` | Periodic saving |
| `monitor` | string | `"val/mIoU"` | Metric to monitor |
| `mode` | string | `"max"` | `"max"` or `"min"` |

### evaluation

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `thresh` | float | `0.5` | Matching threshold |
| `iou_high` | float | `0.5` | Upper IoU bound (soft PQ) |
| `iou_low` | float | `0.05` | Lower IoU bound (soft PQ) |
| `soft_pq_method` | string | `"sqrt"` | `"sqrt"` or `"linear"` |

## Config Overrides

Override config values from command line or code:

```bash
altair train --config config.yaml --set training.lr=0.001 --set model.dropout=0.3
```

```python
run = alt.train("config.yaml", overrides={
    "training.optimizer.lr": 0.001,
    "model.dropout": 0.3
})
```
