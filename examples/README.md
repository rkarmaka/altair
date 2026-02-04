# Altair Examples

This directory contains Jupyter notebooks demonstrating how to use Altair for image segmentation tasks.

## Notebooks

| Notebook | Description |
|----------|-------------|
| [01_training.ipynb](01_training.ipynb) | Train your first segmentation model |
| [02_evaluation.ipynb](02_evaluation.ipynb) | Evaluate models and analyze results |
| [03_inference.ipynb](03_inference.ipynb) | Run predictions on new images |
| [04_export.ipynb](04_export.ipynb) | Export models for deployment (ONNX, TorchScript) |
| [05_custom_config.ipynb](05_custom_config.ipynb) | Advanced configuration options |

## Quick Start

1. Install Altair:
   ```bash
   pip install altair-seg
   ```

2. Prepare your data in PNG format:
   ```
   data/
     train/
       images/
         001.png
         002.png
       masks/
         001.png
         002.png
     val/
       images/
       masks/
   ```

3. Create a config file (`config.yaml`):
   ```yaml
   experiment:
     name: "my_experiment"

   model:
     architecture: "unet"
     encoder: "resnet34"
     num_classes: 2
     task: "binary"

   data:
     train_images: "data/train/images"
     train_masks: "data/train/masks"
     val_images: "data/val/images"
     val_masks: "data/val/masks"
     batch_size: 8

   training:
     epochs: 50
     lr: 0.001
     loss: "dice"
   ```

4. Train:
   ```python
   import altair as alt
   run = alt.train("config.yaml")
   ```

5. Evaluate:
   ```python
   results = alt.evaluate(run.id)
   print(results.metrics)
   ```

6. Predict:
   ```python
   predictions = alt.predict(run.id, images="path/to/images")
   predictions.save_all("outputs/")
   ```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- See `pyproject.toml` for full dependencies

## Running Notebooks

```bash
# Install Jupyter
pip install jupyter

# Start Jupyter
jupyter notebook

# Or use JupyterLab
pip install jupyterlab
jupyter lab
```
