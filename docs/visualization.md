# Visualization

Altair provides comprehensive visualization utilities for inspecting segmentation results.

## Overview

The visualization module offers:
- **Overlay creation**: Blend segmentation masks onto images
- **Comparison views**: Side-by-side ground truth vs prediction
- **Error maps**: Highlight correct and incorrect predictions
- **Sample export**: Automated export with summaries and grids
- **Color palettes**: Default, medical, and cityscapes palettes

## Quick Start

```python
from altair.utils import create_overlay, visualize_prediction, SampleExporter

# Create overlay
overlay = create_overlay(image, mask, alpha=0.5)

# Visualize with matplotlib
visualize_prediction(
    image,
    ground_truth=gt_mask,
    prediction=pred_mask,
    save_path="result.png"
)

# Export multiple samples
exporter = SampleExporter("outputs/samples", max_samples=20)
for img, mask, pred, metrics in results:
    exporter.add_sample(img, pred, ground_truth=mask, metrics=metrics)
exporter.save_summary()
exporter.save_grid()
```

## Color Palettes

Altair includes three built-in color palettes:

```python
from altair.utils import DEFAULT_PALETTE, MEDICAL_PALETTE, CITYSCAPES_PALETTE, get_palette

# Get palette by name
palette = get_palette("default")      # General purpose
palette = get_palette("medical")      # Medical imaging
palette = get_palette("cityscapes")   # Urban scenes

# Auto-extend for many classes
palette = get_palette("default", num_classes=50)
```

### Default Palette
20 colors for general segmentation tasks:
- Background (black), Red, Green, Blue, Yellow, Magenta, Cyan, etc.

### Medical Palette
Colors optimized for medical imaging:
- Background, Lesion (Red), Healthy (Green), Vessel (Blue), Necrosis, Edema, Enhancement

### Cityscapes Palette
Urban scene segmentation colors following the Cityscapes dataset conventions.

## Core Functions

### mask_to_rgb

Convert a class mask to an RGB image:

```python
from altair.utils import mask_to_rgb

# Basic usage
rgb_mask = mask_to_rgb(mask)

# With custom palette
rgb_mask = mask_to_rgb(mask, palette=[[0,0,0], [255,0,0], [0,255,0]])
```

### create_overlay

Blend a segmentation mask onto an image:

```python
from altair.utils import create_overlay

# Basic overlay (50% transparency)
overlay = create_overlay(image, mask)

# Adjust transparency
overlay = create_overlay(image, mask, alpha=0.3)  # More image visible
overlay = create_overlay(image, mask, alpha=0.7)  # More mask visible

# Ignore background class
overlay = create_overlay(image, mask, ignore_index=0)

# Custom colors
overlay = create_overlay(image, mask, palette=my_palette)
```

### create_comparison

Create side-by-side comparison of ground truth and prediction:

```python
from altair.utils import create_comparison

# Creates [Image | GT Overlay | Pred Overlay]
comparison = create_comparison(image, ground_truth, prediction)

# Save to file
from PIL import Image
Image.fromarray(comparison).save("comparison.png")
```

### create_error_map

Visualize prediction errors:

```python
from altair.utils import create_error_map

# Green = correct, Red = error
error_map = create_error_map(ground_truth, prediction)

# Custom colors
error_map = create_error_map(
    ground_truth, prediction,
    correct_color=(0, 255, 0),   # Green
    error_color=(255, 0, 0),     # Red
)

# Ignore certain classes
error_map = create_error_map(ground_truth, prediction, ignore_index=255)
```

### visualize_prediction

Create a matplotlib figure with multiple views:

```python
from altair.utils import visualize_prediction

# Basic visualization (saved to file)
visualize_prediction(
    image,
    ground_truth=gt_mask,
    prediction=pred_mask,
    save_path="result.png"
)

# With error map
visualize_prediction(
    image,
    ground_truth=gt_mask,
    prediction=pred_mask,
    show_error_map=True,
    save_path="result_with_errors.png"
)

# Get as numpy array (for further processing)
fig_array = visualize_prediction(image, prediction=pred_mask)
```

### save_prediction

Save all prediction outputs to files:

```python
from altair.utils import save_prediction

saved_files = save_prediction(
    image=image,
    prediction=pred_mask,
    output_dir="outputs/",
    filename="sample_001",
    ground_truth=gt_mask,  # Optional
    save_mask=True,        # Save raw mask
    save_overlay=True,     # Save overlay image
    save_comparison=True,  # Save comparison (needs GT)
)

print(saved_files)
# {'mask': Path('outputs/sample_001_mask.png'),
#  'overlay': Path('outputs/sample_001_overlay.png'),
#  'comparison': Path('outputs/sample_001_comparison.png')}
```

## Sample Exporter

For batch export of prediction samples with automated organization:

```python
from altair.utils import SampleExporter

# Create exporter
exporter = SampleExporter(
    output_dir="outputs/samples",
    palette=None,           # Uses default
    alpha=0.5,              # Overlay transparency
    max_samples=20,         # Limit samples
)

# Add samples during evaluation
for batch in dataloader:
    images, masks = batch["image"], batch["mask"]
    predictions = model(images)

    for i in range(len(images)):
        exporter.add_sample(
            image=images[i],
            prediction=predictions[i],
            ground_truth=masks[i],
            metrics={"IoU": 0.85, "Dice": 0.91},
            image_path=batch["path"][i],
        )

# Save summary JSON
exporter.save_summary()

# Create and save grid visualization
exporter.save_grid(cols=4, cell_size=(256, 256))
```

### Output Structure

```
outputs/samples/
├── sample_0001_mask.png       # Raw mask
├── sample_0001_overlay.png    # Overlay visualization
├── sample_0001_comparison.png # GT vs Pred comparison
├── sample_0002_mask.png
├── ...
├── samples_summary.json       # Summary with metrics
└── samples_grid.png           # Grid of all overlays
```

### Summary JSON Format

```json
{
  "num_samples": 20,
  "samples": [
    {
      "filename": "sample_0001",
      "files": {
        "mask": "outputs/samples/sample_0001_mask.png",
        "overlay": "outputs/samples/sample_0001_overlay.png",
        "comparison": "outputs/samples/sample_0001_comparison.png"
      },
      "metrics": {"IoU": 0.85, "Dice": 0.91},
      "image_path": "/data/images/001.png"
    },
    ...
  ]
}
```

## Integration with Prediction API

The prediction API includes built-in visualization support:

```python
import altair as alt

# Run prediction with visualization
predictor = alt.load("run_id")
results = predictor.predict(
    "path/to/images",
    output_dir="outputs/",
    save_overlay=True,
)

# Or save manually
for pred in results:
    pred.save_overlay("outputs/", alpha=0.5)
    pred.visualize(save_path=f"outputs/{pred.image_path.stem}.png")
```

## Integration with Evaluation

Export samples during evaluation:

```python
import altair as alt

# Evaluate with sample export
results = alt.evaluate(
    "run_id",
    export_samples=True,
    export_dir="outputs/eval_samples",
    n_export_samples=20,
)

# Or manually from results
results.export_samples(
    output_dir="outputs/eval_samples",
    n_samples=10,
    alpha=0.5,
)
```

## Best Practices

1. **Use appropriate palettes**: Medical for medical imaging, cityscapes for urban scenes
2. **Adjust alpha**: Lower alpha (0.3) to see image details, higher (0.7) to emphasize masks
3. **Include ground truth**: Always show GT for training/validation samples to spot issues
4. **Use error maps**: Quickly identify problematic regions
5. **Limit sample export**: Don't export thousands - 20-50 representative samples is usually enough
6. **Include metrics**: Export per-sample metrics to correlate visual quality with scores
