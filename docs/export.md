# Model Export

Altair supports exporting trained models to deployment formats for inference in production environments.

## Supported Formats

| Format | File Extension | Use Case |
|--------|---------------|----------|
| **ONNX** | `.onnx` | Cross-platform deployment, TensorRT, OpenVINO |
| **TorchScript** | `.pt` | PyTorch ecosystem, LibTorch C++ |

## Quick Start

### Python API

```python
import altair as alt

# Export to ONNX
path = alt.export("run_id", "model.onnx", format="onnx")

# Export to TorchScript
path = alt.export("run_id", "model.pt", format="torchscript")

# With custom input shape
path = alt.export(
    "run_id",
    "model.onnx",
    input_shape=(1, 3, 1024, 1024),
)
```

### Command Line

```bash
# Export to ONNX
altair export --run exp_abc123 --output model.onnx

# Export to TorchScript
altair export --run exp_abc123 --output model.pt --format torchscript

# Custom input shape
altair export --run exp_abc123 --output model.onnx --input-shape 1,3,1024,1024

# Fixed input size (no dynamic axes)
altair export --run exp_abc123 --output model.onnx --no-dynamic
```

## ONNX Export

ONNX (Open Neural Network Exchange) provides cross-platform model deployment.

### Basic Export

```python
import altair as alt

# Simple export
path = alt.export("run_id", "model.onnx")

# With options
path = alt.export(
    "run_id",
    "model.onnx",
    format="onnx",
    input_shape=(1, 3, 512, 512),
    opset_version=17,
    dynamic_axes=True,
    simplify=True,
    validate=True,
)
```

### Using the Exporter Class

For more control, use the `ModelExporter` class directly:

```python
from altair.export import ModelExporter

# Load your model
model = ...

# Create exporter
exporter = ModelExporter(model, input_shape=(1, 3, 512, 512))

# Export with full control
result = exporter.to_onnx(
    "model.onnx",
    opset_version=17,
    dynamic_axes={
        "input": {0: "batch", 2: "height", 3: "width"},
        "output": {0: "batch", 2: "height", 3: "width"},
    },
    simplify=True,
    validate=True,
    input_names=["image"],
    output_names=["segmentation"],
)

print(f"Exported to {result.path} ({result.file_size_mb:.2f}MB)")
```

### Dynamic Axes

By default, ONNX exports use dynamic axes for batch size and spatial dimensions, allowing inference with different input sizes:

```python
# Dynamic axes (default)
path = alt.export("run_id", "model.onnx", dynamic_axes=True)

# Fixed input size
path = alt.export("run_id", "model.onnx", dynamic_axes=False)
```

### ONNX Simplification

The `simplify` option uses [onnxsim](https://github.com/daquexian/onnx-simplifier) to optimize the graph:

```python
# With simplification (default)
path = alt.export("run_id", "model.onnx", simplify=True)

# Without simplification
path = alt.export("run_id", "model.onnx", simplify=False)
```

Install onnxsim for simplification:
```bash
pip install onnxsim
```

### ONNX Opset Version

Different opset versions support different operations:

```python
# Latest stable (recommended)
path = alt.export("run_id", "model.onnx", opset_version=17)

# Older version for compatibility
path = alt.export("run_id", "model.onnx", opset_version=14)
```

## TorchScript Export

TorchScript enables deployment in the PyTorch ecosystem without Python.

### Basic Export

```python
import altair as alt

# Simple export
path = alt.export("run_id", "model.pt", format="torchscript")
```

### Export Methods

TorchScript supports two export methods:

#### Tracing (Default)

Records operations from a forward pass with example input:

```python
from altair.export import ModelExporter

exporter = ModelExporter(model, input_shape=(1, 3, 512, 512))
result = exporter.to_torchscript("model.pt", method="trace")
```

**Pros**: Works with most models, captures actual execution path
**Cons**: May miss dynamic control flow

#### Scripting

Analyzes the model code directly:

```python
result = exporter.to_torchscript("model.pt", method="script")
```

**Pros**: Preserves dynamic control flow
**Cons**: May fail on complex Python features

### Optimization

TorchScript models can be optimized for inference:

```python
result = exporter.to_torchscript("model.pt", optimize=True)
```

This applies `torch.jit.optimize_for_inference()` for faster execution.

## Validation

Exported models are validated by default to ensure correctness.

### Automatic Validation

```python
# Validation enabled (default)
path = alt.export("run_id", "model.onnx", validate=True)

# Skip validation
path = alt.export("run_id", "model.onnx", validate=False)
```

Validation compares outputs between the original PyTorch model and the exported model.

### Manual Validation

```python
from altair.export import validate_onnx, validate_torchscript

# Validate ONNX
results = validate_onnx("model.onnx", input_shape=(1, 3, 512, 512))
print(f"Valid: {results['valid']}")
print(f"Inference OK: {results['inference_ok']}")
print(f"Output shape: {results['output_shape']}")

# Validate TorchScript
results = validate_torchscript("model.pt", input_shape=(1, 3, 512, 512))
print(f"Valid: {results['valid']}")
```

## Inference Sessions

Altair provides wrapper classes for easy inference with exported models.

### ONNX Inference

```python
from altair.export.exporter import ONNXInferenceSession
import numpy as np

# Create session
session = ONNXInferenceSession("model.onnx")

# Run inference
input_array = np.random.randn(1, 3, 512, 512).astype(np.float32)
output = session(input_array)

print(f"Input shape: {session.input_shape}")
print(f"Output shape: {session.output_shape}")
```

### TorchScript Inference

```python
from altair.export.exporter import TorchScriptInferenceSession
import torch

# Create session
session = TorchScriptInferenceSession("model.pt", device="cuda")

# Run inference
input_tensor = torch.randn(1, 3, 512, 512)
output = session(input_tensor)
```

## Export All Formats

Export to multiple formats at once:

```python
from altair.export import ModelExporter

exporter = ModelExporter(model, input_shape=(1, 3, 512, 512))

results = exporter.export_all(
    output_dir="exported_models/",
    name="segmentation_model",
    formats=["onnx", "torchscript"],
)

for fmt, result in results.items():
    print(f"{fmt}: {result.path} ({result.file_size_mb:.2f}MB)")
```

## FP16 Export

Export models in half precision (FP16) for faster inference:

```python
from altair.export import ModelExporter

exporter = ModelExporter(model, input_shape=(1, 3, 512, 512))

# Convert to FP16
exporter.to_half()

# Export
result = exporter.to_onnx("model_fp16.onnx")
```

Note: FP16 may reduce accuracy slightly but provides faster inference on GPUs with FP16 support.

## Deployment Examples

### TensorRT (via ONNX)

```python
# Export to ONNX
alt.export("run_id", "model.onnx", opset_version=17)

# Then use TensorRT
# trtexec --onnx=model.onnx --saveEngine=model.trt
```

### OpenVINO (via ONNX)

```python
# Export to ONNX
alt.export("run_id", "model.onnx")

# Convert with OpenVINO Model Optimizer
# mo --input_model model.onnx --output_dir openvino_model
```

### LibTorch C++ (via TorchScript)

```python
# Export to TorchScript
alt.export("run_id", "model.pt", format="torchscript")
```

```cpp
// C++ inference
#include <torch/script.h>

torch::jit::script::Module model = torch::jit::load("model.pt");
torch::Tensor input = torch::randn({1, 3, 512, 512});
torch::Tensor output = model.forward({input}).toTensor();
```

## Troubleshooting

### ONNX Export Fails

1. **Unsupported operations**: Some PyTorch operations aren't supported in ONNX. Try a different opset version or simplify the model.

2. **Dynamic shapes**: If export fails with dynamic axes, try fixed input sizes.

3. **Custom operators**: Register custom ONNX operators if needed.

### TorchScript Export Fails

1. **Script method fails**: Try `method="trace"` instead.

2. **Control flow issues**: Tracing may not capture all branches. Use scripting for dynamic control flow.

3. **Python features**: Some Python features (e.g., generators, some comprehensions) aren't supported in TorchScript.

### Validation Fails

1. **Numerical differences**: Small differences (< 1e-3) are normal due to floating-point precision.

2. **Large differences**: Check for operations that behave differently in inference mode (e.g., dropout, batch norm).

## Best Practices

1. **Always validate**: Use `validate=True` to catch export issues early.

2. **Test inference**: Run inference on representative data after export.

3. **Use dynamic axes**: Unless you have fixed input sizes, dynamic axes provide flexibility.

4. **Benchmark**: Compare inference speed between formats for your deployment target.

5. **Version control**: Keep track of opset versions and export settings for reproducibility.
