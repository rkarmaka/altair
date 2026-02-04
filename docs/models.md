# Models

Altair provides several segmentation model architectures, from simple UNet to encoder-decoder models with pretrained backbones.

## Available Architectures

| Architecture | Description | Use Case |
|-------------|-------------|----------|
| `unet` | Classic U-Net | Simple tasks, learning, small datasets |
| `unet++` | Nested U-Net with dense skip connections | Better feature reuse |
| `unet_encoder` | U-Net decoder with pretrained encoder | Production, transfer learning |

## UNet

Classic U-Net architecture with configurable depth and channels.

```yaml
model:
  architecture: "unet"
  task: "binary"
  num_classes: 1
  encoder_depth: 5
  decoder_channels: [256, 128, 64, 32, 16]
  dropout: 0.2
```

## UNet++

Nested U-Net with dense skip connections for better feature propagation.

```yaml
model:
  architecture: "unet++"
  task: "multiclass"
  num_classes: 10
  encoder_depth: 4
  decoder_channels: [64, 128, 256, 512]
  dropout: 0.2
```

## UNet with Pretrained Encoder

**Recommended for production use.** Uses pretrained encoders from timm for better performance.

```yaml
model:
  architecture: "unet_encoder"  # or "unet_resnet"
  task: "multiclass"
  num_classes: 19

  # Encoder
  encoder: "resnet50"
  encoder_weights: "imagenet"
  encoder_depth: 5

  # Decoder
  decoder_channels: [256, 128, 64, 32, 16]
  decoder_attention: "scse"
  dropout: 0.2
```

### Available Encoders

Any timm model can be used as an encoder. Common choices:

**ResNet Family** (good balance of speed and accuracy):
- `resnet18` - Smallest, fastest
- `resnet34` - Good for medium tasks
- `resnet50` - Recommended default
- `resnet101` - Higher capacity

**EfficientNet** (efficient for deployment):
- `efficientnet_b0` - Smallest
- `efficientnet_b4` - Good balance
- `efficientnet_b7` - Highest accuracy

**ConvNeXt** (modern architecture):
- `convnext_tiny` - Small
- `convnext_small` - Medium
- `convnext_base` - Large

**MobileNet** (edge deployment):
- `mobilenetv3_large_100` - Larger variant
- `mobilenetv3_small_100` - Smaller variant

### Convenience Aliases

Pre-configured model shortcuts:

```python
import altair as alt
from altair.core.registry import MODELS

# These are all registered
model = MODELS.build("unet_resnet50", num_classes=10)
model = MODELS.build("unet_efficientnet_b0", num_classes=10)
model = MODELS.build("unet_convnext_tiny", num_classes=10)
model = MODELS.build("unet_mobilenet", num_classes=10)
```

## Encoder Configuration

### Pretrained Weights

```yaml
# Use ImageNet pretrained weights (recommended)
encoder_weights: "imagenet"

# Random initialization
encoder_weights: null

# Custom weights (path)
encoder_weights: "/path/to/weights.pth"
```

### Encoder Depth

Controls how many feature levels to use (1-5):

```yaml
encoder_depth: 5  # Full encoder
encoder_depth: 4  # Faster, slightly less accurate
encoder_depth: 3  # Much faster, for simple tasks
```

## Decoder Configuration

### Decoder Channels

Controls decoder capacity:

```yaml
# Default (recommended for ResNet)
decoder_channels: [256, 128, 64, 32, 16]

# Smaller (faster inference)
decoder_channels: [128, 64, 32, 16, 8]

# Larger (more capacity)
decoder_channels: [512, 256, 128, 64, 32]
```

### Attention

Optional squeeze-and-excitation attention in decoder:

```yaml
# No attention (faster)
decoder_attention: null

# SCSE attention (better accuracy)
decoder_attention: "scse"
```

## Task Types

### Binary Segmentation

Single class + background:

```yaml
model:
  task: "binary"
  num_classes: 1  # Single output channel
```

Output: Sigmoid activation, values in [0, 1]

### Multi-class Segmentation

Multiple classes:

```yaml
model:
  task: "multiclass"
  num_classes: 19  # Number of classes
```

Output: Softmax activation, class probabilities

### Regression

Continuous output (e.g., depth estimation):

```yaml
model:
  task: "regression"
  num_classes: 1  # Or more for multi-channel
```

Output: No activation, raw values

## Regularization

### Dropout

Applied in decoder blocks:

```yaml
dropout: 0.2  # 20% dropout
dropout: 0.0  # No dropout (default)
```

### Freezing Encoder

For fine-tuning with limited data:

```python
from altair.models import build_model

model = build_model(config)
model.freeze_encoder()  # Only train decoder
# ... train for a few epochs ...
model.unfreeze_encoder()  # Then train full model
```

## Example Configurations

### Medical Imaging (Binary)

```yaml
model:
  architecture: "unet_encoder"
  task: "binary"
  num_classes: 1
  encoder: "efficientnet_b4"
  encoder_weights: "imagenet"
  decoder_attention: "scse"
  dropout: 0.3
```

### Semantic Segmentation (Multi-class)

```yaml
model:
  architecture: "unet_encoder"
  task: "multiclass"
  num_classes: 19
  encoder: "resnet50"
  encoder_weights: "imagenet"
  decoder_channels: [256, 128, 64, 32, 16]
  dropout: 0.2
```

### Edge Deployment

```yaml
model:
  architecture: "unet_encoder"
  task: "binary"
  num_classes: 1
  encoder: "mobilenetv3_large_100"
  encoder_weights: "imagenet"
  encoder_depth: 4
  decoder_channels: [64, 32, 16, 8]
  dropout: 0.0
```
