# Roadmap

This document outlines planned features and improvements for Altair.

## Planned Features

### Distributed Training (Priority: High)

Multi-GPU and multi-node training support using [Hugging Face Accelerate](https://huggingface.co/docs/accelerate).

**Planned Implementation:**

```python
# Future API
import altair as alt

# Automatic distributed training
run = alt.train("config.yaml", distributed=True)

# Or via CLI
# accelerate launch altair train --config config.yaml
```

**Key Features:**
- Automatic device placement
- Mixed precision training (FP16/BF16)
- Gradient accumulation
- Multi-GPU data parallelism
- DeepSpeed integration for large models

**Implementation Notes:**
1. Wrap `Trainer` with Accelerate's `Accelerator`
2. Use `accelerate.prepare()` for model, optimizer, dataloaders
3. Handle distributed metrics aggregation
4. Support `accelerate config` for easy setup

**Reference Implementation:**
```python
from accelerate import Accelerator

class DistributedTrainer(Trainer):
    def __init__(self, config, **kwargs):
        self.accelerator = Accelerator(
            mixed_precision="fp16",
            gradient_accumulation_steps=config.training.gradient_accumulation,
        )
        super().__init__(config, **kwargs)

    def _setup(self):
        # Prepare for distributed training
        self.model, self.optimizer, self.train_loader, self.val_loader = \
            self.accelerator.prepare(
                self.model, self.optimizer,
                self.train_loader, self.val_loader
            )

    def _backward(self, loss):
        self.accelerator.backward(loss)
```

---

### Additional Architectures (Priority: Medium)

#### DeepLabV3+
- Atrous Spatial Pyramid Pooling (ASPP)
- Encoder-decoder with atrous convolution
- Good for multi-scale feature extraction

#### PSPNet (Pyramid Scene Parsing)
- Pyramid pooling module
- Global context aggregation
- Effective for scene parsing

#### SegFormer
- Transformer-based architecture
- Hierarchical feature extraction
- Mix-FFN for positional encoding

---

### Test-Time Augmentation (Priority: Medium)

Apply augmentations during inference and aggregate predictions:

```python
# Future API
predictions = alt.predict(
    "run_id",
    images="path/to/images",
    tta=True,
    tta_transforms=["hflip", "vflip", "rotate90"],
)
```

**Implementation Notes:**
- Horizontal/vertical flips
- 90-degree rotations
- Scale variations
- Soft voting or averaging for final prediction

---

### Sliding Window Inference (Priority: Medium)

For large images that don't fit in memory:

```python
# Future API
predictions = alt.predict(
    "run_id",
    images="large_image.tif",
    sliding_window=True,
    window_size=(512, 512),
    overlap=0.25,
)
```

**Implementation Notes:**
- Configurable window size and overlap
- Weighted blending at boundaries
- Memory-efficient processing
- Support for very large images (satellite, medical)

---

### Learning Rate Schedulers (Priority: Low)

Additional scheduler options:

- **OneCycleLR**: Fast convergence with super-convergence
- **Polynomial Decay**: Smooth decay to zero
- **Warmup + Cosine**: Linear warmup followed by cosine annealing
- **ReduceLROnPlateau**: Automatic reduction on metric plateau

---

### Augmentation Presets (Priority: Low)

Pre-configured augmentation pipelines:

```yaml
# Future config
augmentations:
  preset: "heavy"  # Options: light, medium, heavy, medical, satellite
```

**Presets:**
- `light`: Basic flips, small rotations
- `medium`: + color jitter, elastic transforms
- `heavy`: + cutout, grid distortion, advanced color
- `medical`: Specialized for medical imaging
- `satellite`: Specialized for remote sensing

---

### Self-Supervised Pretraining (Priority: Low)

Support for self-supervised learning methods:

- **MAE** (Masked Autoencoder)
- **SimCLR** contrastive learning
- **DINO** self-distillation

---

## Contributing

We welcome contributions! If you'd like to work on any of these features:

1. Open an issue to discuss the implementation approach
2. Fork the repository and create a feature branch
3. Submit a pull request with tests and documentation

See [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed guidelines.
