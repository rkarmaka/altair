"""
Command-line interface for Altair.

Provides commands for training, evaluation, and inference.

Usage:
    altair train --config configs/unet.yaml
    altair evaluate --run exp_abc123 --data path/to/test
    altair predict --run exp_abc123 --images path/to/images
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from altair.utils.console import (
    console,
    print_banner,
    print_error,
    print_evaluation_results,
    print_export_result,
    print_header,
    print_info,
    print_runs_table,
    print_success,
    print_warning,
    status_spinner,
)

logger = logging.getLogger(__name__)


def _configure_logging(verbose: int = 0, quiet: bool = False) -> None:
    """
    Configure logging level based on verbosity flags and environment.

    Args:
        verbose: Verbosity level (0=default, 1=INFO, 2+=DEBUG).
        quiet: If True, set to ERROR level (overrides verbose).
    """
    # Check environment variable first
    if os.environ.get("ALTAIR_DEBUG", "").lower() in ("1", "true", "yes"):
        level = logging.DEBUG
    elif quiet:
        level = logging.ERROR
    elif verbose >= 2:
        level = logging.DEBUG
    elif verbose == 1:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )


def main():
    """Main entry point for the CLI."""
    # Print banner
    print_banner()

    parser = argparse.ArgumentParser(
        prog="altair",
        description="Altair: A flexible segmentation framework",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="count",
        default=0,
        help="Increase verbosity (-v for INFO, -vv for DEBUG)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress most output (only show errors)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a segmentation model")
    train_parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to configuration file",
    )
    train_parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override config values (e.g., --set training.lr=0.001)",
    )
    train_parser.add_argument(
        "--resume",
        type=str,
        help="Path to checkpoint or run ID to resume from",
    )

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument(
        "--run", "-r",
        type=str,
        required=True,
        help="Run ID or path to run directory",
    )
    eval_parser.add_argument(
        "--data", "-d",
        type=str,
        help="Path to evaluation data (default: validation data from config)",
    )
    eval_parser.add_argument(
        "--output", "-o",
        type=str,
        help="Path to save evaluation results",
    )
    eval_parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for evaluation",
    )

    # Predict command
    pred_parser = subparsers.add_parser("predict", help="Run inference on images")
    pred_parser.add_argument(
        "--run", "-r",
        type=str,
        required=True,
        help="Run ID or path to run directory",
    )
    pred_parser.add_argument(
        "--images", "-i",
        type=str,
        required=True,
        help="Path to image file or directory",
    )
    pred_parser.add_argument(
        "--output", "-o",
        type=str,
        default="predictions",
        help="Directory to save predictions",
    )
    pred_parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference",
    )

    # List command
    list_parser = subparsers.add_parser("list", help="List experiment runs")
    list_parser.add_argument(
        "--project", "-p",
        type=str,
        help="Filter by project name",
    )
    list_parser.add_argument(
        "--status", "-s",
        type=str,
        choices=["pending", "running", "completed", "failed"],
        help="Filter by status",
    )

    # Export command
    export_parser = subparsers.add_parser("export", help="Export model for deployment")
    export_parser.add_argument(
        "--run", "-r",
        type=str,
        required=True,
        help="Run ID or path to run directory",
    )
    export_parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output file path (e.g., model.onnx or model.pt)",
    )
    export_parser.add_argument(
        "--format", "-f",
        type=str,
        choices=["onnx", "torchscript"],
        default="onnx",
        help="Export format (default: onnx)",
    )
    export_parser.add_argument(
        "--input-shape",
        type=str,
        help="Input shape as N,C,H,W (e.g., 1,3,512,512)",
    )
    export_parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)",
    )
    export_parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Don't simplify ONNX model",
    )
    export_parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Don't validate exported model",
    )
    export_parser.add_argument(
        "--no-dynamic",
        action="store_true",
        help="Don't use dynamic axes (fixed input size)",
    )

    # Parse arguments
    args = parser.parse_args()

    # Configure logging based on verbosity flags
    _configure_logging(verbose=args.verbose, quiet=args.quiet)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Execute command
    if args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "export":
        cmd_export(args)


def cmd_train(args):
    """Execute train command."""
    import altair as alt
    from altair.utils.console import print_config_panel, print_training_complete, format_duration
    import time

    print_header("Training")

    # Parse overrides
    overrides = {}
    for override in args.set:
        key, value = override.split("=", 1)
        # Try to parse as number
        try:
            value = float(value)
            if value.is_integer():
                value = int(value)
        except ValueError:
            pass
        overrides[key] = value

    print_info(f"Loading config: [path]{args.config}[/path]")

    if overrides:
        print_info(f"Applying {len(overrides)} override(s)")

    start_time = time.time()

    # Run training
    run = alt.train(
        args.config,
        overrides=overrides if overrides else None,
        resume=args.resume,
    )

    duration = format_duration(time.time() - start_time)

    # Get best metric
    best_metric = 0.0
    if run.metrics:
        for key in ["mIoU", "Dice", "IoU", "val/mIoU", "val/Dice"]:
            if key in run.metrics:
                best_metric = run.metrics[key]
                break

    print_training_complete(
        run_id=run.id,
        best_metric=best_metric,
        best_checkpoint=str(run.best_checkpoint) if run.best_checkpoint else "N/A",
        duration=duration,
    )


def cmd_evaluate(args):
    """Execute evaluate command."""
    import altair as alt
    from altair.utils.console import format_duration
    import time

    print_header("Evaluation")

    print_info(f"Loading run: [cyan]{args.run}[/cyan]")

    if args.data:
        print_info(f"Data path: [path]{args.data}[/path]")

    start_time = time.time()

    with status_spinner("Running evaluation..."):
        results = alt.evaluate(
            args.run,
            data=args.data,
            batch_size=args.batch_size,
        )

    duration = format_duration(time.time() - start_time)

    # Print results
    print_evaluation_results(
        metrics=results.metrics,
        num_samples=len(results.per_sample_metrics),
        duration=duration,
    )

    # Save results
    if args.output:
        output_path = Path(args.output)
        if output_path.suffix == ".csv":
            results.to_csv(output_path)
        else:
            results.to_json(output_path)
        print_success(f"Results saved to: [path]{output_path}[/path]")


def cmd_predict(args):
    """Execute predict command."""
    import altair as alt
    from altair.utils.console import create_inference_progress

    print_header("Inference")

    print_info(f"Loading run: [cyan]{args.run}[/cyan]")
    print_info(f"Input: [path]{args.images}[/path]")
    print_info(f"Output: [path]{args.output}[/path]")

    console.print()

    results = alt.predict(
        args.run,
        images=args.images,
        batch_size=args.batch_size,
        output_dir=args.output,
    )

    console.print()
    print_success(f"Predicted [bold]{len(results)}[/bold] images")
    print_success(f"Saved to: [path]{args.output}[/path]")


def cmd_list(args):
    """Execute list command."""
    import altair as alt

    print_header("Experiment Runs")

    with status_spinner("Loading runs..."):
        runs = alt.list_runs(project=args.project, status=args.status)

    print_runs_table(runs)


def cmd_export(args):
    """Execute export command."""
    import altair as alt

    print_header("Model Export")

    # Parse input shape if provided
    input_shape = None
    if args.input_shape:
        try:
            parts = [int(x) for x in args.input_shape.split(",")]
            if len(parts) != 4:
                raise ValueError("Input shape must have 4 values (N,C,H,W)")
            input_shape = tuple(parts)
        except ValueError as e:
            print_error(f"Invalid input shape: {e}")
            sys.exit(1)

    print_info(f"Loading run: [cyan]{args.run}[/cyan]")
    print_info(f"Format: [bold]{args.format.upper()}[/bold]")
    print_info(f"Output: [path]{args.output}[/path]")

    console.print()

    try:
        with status_spinner("Exporting model..."):
            path = alt.export(
                args.run,
                args.output,
                format=args.format,
                input_shape=input_shape,
                dynamic_axes=not args.no_dynamic,
                opset_version=args.opset,
                simplify=not args.no_simplify,
                validate=not args.no_validate,
            )

        # Get file size
        size_mb = path.stat().st_size / (1024 * 1024)

        # Determine actual input shape used
        actual_shape = input_shape or (1, 3, 512, 512)

        print_export_result(
            format=args.format,
            path=str(path),
            size_mb=size_mb,
            input_shape=actual_shape,
        )

    except Exception as e:
        print_error(f"Export failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
