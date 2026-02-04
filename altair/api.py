"""
Top-level API for Altair.

This module provides the main entry points for training, evaluation, prediction,
and experiment management. These functions are re-exported at the package level
for convenient access.

Example:
    >>> import altair as alt
    >>>
    >>> # Train a model
    >>> run = alt.train("configs/unet_resnet50.yaml")
    >>>
    >>> # Evaluate
    >>> results = alt.evaluate(run.id, data="path/to/test")
    >>>
    >>> # Predict
    >>> masks = alt.predict(run.id, images="path/to/images")
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from altair.core.config import Config
from altair.core.run import Run, RunStatus

if TYPE_CHECKING:
    from altair.engine.evaluator import EvaluationResults
    from altair.engine.predictor import PredictionResults


def train(
    config: str | Path | dict[str, Any] | Config,
    *,
    overrides: dict[str, Any] | None = None,
    resume: str | Path | None = None,
) -> Run:
    """
    Train a segmentation model.

    Args:
        config: Path to YAML config, dictionary, or Config object.
        overrides: Optional dictionary of config overrides.
        resume: Path to checkpoint or run ID to resume from.

    Returns:
        A Run object representing the training run.

    Example:
        >>> run = alt.train("configs/unet.yaml")
        >>> run = alt.train("configs/unet.yaml", overrides={"training.lr": 0.001})
        >>> run = alt.train(config_dict)

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValidationError: If config is invalid.
    """
    # Import here to avoid circular imports and allow lazy loading
    from altair.engine.trainer import Trainer

    # Load configuration
    if isinstance(config, (str, Path)):
        cfg = Config.from_yaml(config)
    elif isinstance(config, dict):
        cfg = Config.from_dict(config)
    elif isinstance(config, Config):
        cfg = config
    else:
        raise TypeError(f"config must be str, Path, dict, or Config, got {type(config)}")

    # Apply overrides
    if overrides:
        cfg = _apply_overrides(cfg, overrides)

    # Create trainer and run
    trainer = Trainer(cfg, resume=resume)
    run = trainer.fit()

    return run


def evaluate(
    run_id_or_checkpoint: str | Path,
    *,
    data: str | Path | None = None,
    batch_size: int | None = None,
    device: str | None = None,
) -> "EvaluationResults":
    """
    Evaluate a trained model.

    Args:
        run_id_or_checkpoint: Run ID, run directory path, or checkpoint path.
        data: Path to evaluation data. If None, uses validation data from config.
        batch_size: Override batch size for evaluation.
        device: Device to use for evaluation (e.g., 'cuda', 'cpu').

    Returns:
        EvaluationResults object with metrics and predictions.

    Example:
        >>> results = alt.evaluate("exp_abc123", data="path/to/test")
        >>> print(results.metrics)
        {'mIoU': 0.82, 'dice': 0.85}
        >>> results.to_csv("results.csv")
    """
    from altair.engine.evaluator import Evaluator

    # Load run or create from checkpoint
    run, checkpoint = _resolve_run_and_checkpoint(run_id_or_checkpoint)

    evaluator = Evaluator(
        run=run,
        checkpoint=checkpoint,
        data_path=Path(data) if data else None,
        batch_size=batch_size,
        device=device,
    )

    return evaluator.evaluate()


def predict(
    run_id_or_checkpoint: str | Path,
    images: str | Path | list[str | Path],
    *,
    batch_size: int = 1,
    device: str | None = None,
    output_dir: str | Path | None = None,
) -> "PredictionResults":
    """
    Run inference on images.

    Args:
        run_id_or_checkpoint: Run ID, run directory path, or checkpoint path.
        images: Path to image file, directory, or list of paths.
        batch_size: Batch size for inference.
        device: Device to use (e.g., 'cuda', 'cpu').
        output_dir: Optional directory to save predictions.

    Returns:
        PredictionResults object with masks and metadata.

    Example:
        >>> predictions = alt.predict("exp_abc123", images="path/to/images")
        >>> for pred in predictions:
        ...     pred.save("outputs/")
    """
    from altair.engine.predictor import Predictor

    # Load run or create from checkpoint
    run, checkpoint = _resolve_run_and_checkpoint(run_id_or_checkpoint)

    predictor = Predictor(
        run=run,
        checkpoint=checkpoint,
        device=device,
    )

    return predictor.predict(
        images=images,
        batch_size=batch_size,
        output_dir=Path(output_dir) if output_dir else None,
    )


def load(run_id_or_path: str | Path) -> Run:
    """
    Load an existing run.

    Args:
        run_id_or_path: Run ID or path to run directory.

    Returns:
        The loaded Run object.

    Example:
        >>> run = alt.load("exp_abc123")
        >>> print(run.metrics)
        >>> print(run.best_checkpoint)
    """
    path = _resolve_run_path(run_id_or_path)
    return Run.load(path)


def list_runs(
    project: str | None = None,
    output_dir: str | Path = "experiments",
    status: str | RunStatus | None = None,
) -> list[Run]:
    """
    List all runs, optionally filtered by project or status.

    Args:
        project: Filter by project name.
        output_dir: Directory containing run directories.
        status: Filter by run status.

    Returns:
        List of Run objects matching the filters.

    Example:
        >>> runs = alt.list_runs(project="segmentation")
        >>> for run in runs:
        ...     print(f"{run.id}: {run.status}")
    """
    output_dir = Path(output_dir)
    runs = []

    if not output_dir.exists():
        return runs

    for run_dir in output_dir.iterdir():
        if not run_dir.is_dir():
            continue

        run_file = run_dir / "run.json"
        if not run_file.exists():
            continue

        try:
            run = Run.load(run_dir)

            # Apply filters
            if project is not None and run.project != project:
                continue
            if status is not None:
                if isinstance(status, str):
                    status = RunStatus(status)
                if run.status != status:
                    continue

            runs.append(run)
        except Exception:
            # Skip invalid run directories
            continue

    # Sort by creation time, newest first
    runs.sort(key=lambda r: r.created_at, reverse=True)
    return runs


def export(
    run_id_or_checkpoint: str | Path,
    output: str | Path,
    *,
    format: str = "onnx",
    input_shape: tuple[int, int, int, int] | None = None,
    dynamic_axes: bool = True,
    opset_version: int = 17,
    simplify: bool = True,
    validate: bool = True,
    device: str | None = None,
) -> Path:
    """
    Export a trained model to a deployment format.

    Args:
        run_id_or_checkpoint: Run ID, run directory path, or checkpoint path.
        output: Output file path.
        format: Export format ('onnx', 'torchscript').
        input_shape: Input tensor shape (batch, channels, height, width).
            If None, inferred from config (default: 1, 3, 512, 512).
        dynamic_axes: Whether to use dynamic axes for batch and spatial dims.
        opset_version: ONNX opset version (default: 17).
        simplify: Whether to simplify ONNX model (requires onnxsim).
        validate: Whether to validate the exported model.
        device: Device to use for export.

    Returns:
        Path to the exported model file.

    Example:
        >>> path = alt.export("exp_abc123", "model.onnx")
        >>> path = alt.export("exp_abc123", "model.pt", format="torchscript")
        >>> path = alt.export("exp_abc123", "model.onnx", input_shape=(1, 3, 1024, 1024))
    """
    import torch

    from altair.export import ModelExporter
    from altair.models import build_model

    run, checkpoint = _resolve_run_and_checkpoint(run_id_or_checkpoint)

    # Determine input shape from config if not specified
    if input_shape is None:
        aug_config = run.config.get("augmentations", {}).get("val", [])
        height, width = 512, 512  # defaults
        for aug in aug_config:
            if aug.get("name") == "resize":
                height = aug.get("height", 512)
                width = aug.get("width", 512)
                break
        input_shape = (1, 3, height, width)

    # Build and load model
    model = build_model(run.config["model"])
    checkpoint_data = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint_data["model_state_dict"])

    # Create exporter
    exporter = ModelExporter(model, input_shape=input_shape, device=device)

    output = Path(output)

    # Set up dynamic axes if requested
    dyn_axes = None
    if dynamic_axes:
        dyn_axes = {
            "input": {0: "batch", 2: "height", 3: "width"},
            "output": {0: "batch", 2: "height", 3: "width"},
        }

    # Export based on format
    if format.lower() == "onnx":
        result = exporter.to_onnx(
            output,
            opset_version=opset_version,
            dynamic_axes=dyn_axes,
            simplify=simplify,
            validate=validate,
        )
    elif format.lower() in ("torchscript", "pt", "jit"):
        result = exporter.to_torchscript(
            output,
            optimize=True,
            validate=validate,
        )
    else:
        raise ValueError(f"Unknown format: {format}. Use 'onnx' or 'torchscript'.")

    return result.path


# =============================================================================
# Helper functions
# =============================================================================


def _resolve_run_path(run_id_or_path: str | Path) -> Path:
    """
    Resolve a run ID or path to the actual run directory.

    Args:
        run_id_or_path: Either a run ID or path to run directory.

    Returns:
        Path to the run directory.

    Raises:
        FileNotFoundError: If run cannot be found.
    """
    path = Path(run_id_or_path)

    # If it's already a valid path, return it
    if path.exists() and (path / "run.json").exists():
        return path

    # Search in default experiments directory
    experiments_dir = Path("experiments")
    if experiments_dir.exists():
        for run_dir in experiments_dir.iterdir():
            if run_dir.name == str(run_id_or_path) or run_dir.name.startswith(
                str(run_id_or_path)
            ):
                if (run_dir / "run.json").exists():
                    return run_dir

    raise FileNotFoundError(f"Run not found: {run_id_or_path}")


def _resolve_run_and_checkpoint(run_id_or_checkpoint: str | Path) -> tuple[Run, Path]:
    """
    Resolve run and checkpoint from various input types.

    Args:
        run_id_or_checkpoint: Run ID, run path, or checkpoint path.

    Returns:
        Tuple of (Run, checkpoint_path).
    """
    path = Path(run_id_or_checkpoint)

    # If it's a checkpoint file
    if path.suffix in (".pt", ".pth", ".ckpt"):
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        # Try to find the run directory (parent of checkpoints/)
        if path.parent.name == "checkpoints":
            run_dir = path.parent.parent
            if (run_dir / "run.json").exists():
                return Run.load(run_dir), path

        # Create a minimal run for standalone checkpoint
        raise ValueError(
            f"Cannot determine run for checkpoint: {path}. "
            "Please provide a run ID or run directory instead."
        )

    # It's a run ID or run directory
    run_path = _resolve_run_path(run_id_or_checkpoint)
    run = Run.load(run_path)

    # Use best checkpoint by default
    checkpoint = run.best_checkpoint or run.last_checkpoint
    if checkpoint is None:
        raise FileNotFoundError(f"No checkpoint found for run: {run.id}")

    return run, checkpoint


def _apply_overrides(config: Config, overrides: dict[str, Any]) -> Config:
    """
    Apply dot-notation overrides to a config.

    Args:
        config: Original configuration.
        overrides: Dictionary with dot-notation keys (e.g., "training.lr": 0.001).

    Returns:
        New Config with overrides applied.
    """
    data = config.to_dict()

    for key, value in overrides.items():
        parts = key.split(".")
        target = data
        for part in parts[:-1]:
            target = target[part]
        target[parts[-1]] = value

    return Config.from_dict(data)
