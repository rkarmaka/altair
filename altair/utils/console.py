"""
Rich console utilities for beautiful terminal output.

Provides consistent styling and formatting across the Altair CLI.

Note: This module provides VISUAL output for the user. Logging is still
handled separately via Python's logging module for debugging and file logs.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any, Iterator

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# Module logger for internal logging
logger = logging.getLogger(__name__)

# Custom theme for Altair
ALTAIR_THEME = Theme(
    {
        "info": "cyan",
        "success": "green",
        "warning": "yellow",
        "error": "red bold",
        "highlight": "magenta",
        "metric": "blue",
        "metric_value": "bold cyan",
        "path": "dim cyan",
        "header": "bold magenta",
    }
)

# Global console instance
console = Console(theme=ALTAIR_THEME)


def print_banner() -> None:
    """Print the Altair banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║     █████╗ ██╗  ████████╗ █████╗ ██╗██████╗              ║
    ║    ██╔══██╗██║  ╚══██╔══╝██╔══██╗██║██╔══██╗             ║
    ║    ███████║██║     ██║   ███████║██║██████╔╝             ║
    ║    ██╔══██║██║     ██║   ██╔══██║██║██╔══██╗             ║
    ║    ██║  ██║███████╗██║   ██║  ██║██║██║  ██║             ║
    ║    ╚═╝  ╚═╝╚══════╝╚═╝   ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝             ║
    ║                                                           ║
    ║          Config-Driven Segmentation Framework             ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    console.print(banner, style="bold cyan")


def print_header(text: str, style: str = "header") -> None:
    """Print a styled header."""
    console.print()
    console.print(f"━━━ {text} ━━━", style=style)
    console.print()


def print_info(message: str) -> None:
    """Print an info message."""
    console.print(f"[info]ℹ[/info]  {message}")


def print_success(message: str) -> None:
    """Print a success message."""
    console.print(f"[success]✓[/success]  {message}")


def print_warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"[warning]⚠[/warning]  {message}")


def print_error(message: str) -> None:
    """Print an error message."""
    console.print(f"[error]✗[/error]  {message}")


def print_metric(name: str, value: float | str, precision: int = 4) -> None:
    """Print a metric with formatting."""
    if isinstance(value, float):
        value_str = f"{value:.{precision}f}"
    else:
        value_str = str(value)
    console.print(f"  [metric]{name}:[/metric] [metric_value]{value_str}[/metric_value]")


def print_metrics_table(
    metrics: dict[str, float],
    title: str = "Metrics",
    precision: int = 4,
) -> None:
    """Print metrics in a nice table."""
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    for name, value in sorted(metrics.items()):
        if "/" not in name:  # Skip nested metrics in main table
            table.add_row(name, f"{value:.{precision}f}")

    console.print(table)


def print_config_panel(config: dict[str, Any], title: str = "Configuration") -> None:
    """Print configuration in a panel."""
    lines = []
    for section, values in config.items():
        if isinstance(values, dict):
            lines.append(f"[bold cyan]{section}:[/bold cyan]")
            for key, val in values.items():
                if isinstance(val, dict):
                    lines.append(f"  [dim]{key}:[/dim] ...")
                elif isinstance(val, list):
                    lines.append(f"  [dim]{key}:[/dim] [{len(val)} items]")
                else:
                    lines.append(f"  [dim]{key}:[/dim] {val}")
        else:
            lines.append(f"[bold cyan]{section}:[/bold cyan] {values}")

    panel = Panel("\n".join(lines), title=title, border_style="blue")
    console.print(panel)


def print_model_summary(
    architecture: str,
    encoder: str,
    num_classes: int,
    task: str,
    num_params: int | None = None,
) -> None:
    """Print model summary in a panel."""
    lines = [
        f"[cyan]Architecture:[/cyan] {architecture}",
        f"[cyan]Encoder:[/cyan] {encoder}",
        f"[cyan]Classes:[/cyan] {num_classes}",
        f"[cyan]Task:[/cyan] {task}",
    ]
    if num_params is not None:
        lines.append(f"[cyan]Parameters:[/cyan] {num_params:,}")

    panel = Panel("\n".join(lines), title="🧠 Model", border_style="green")
    console.print(panel)


def print_training_start(
    run_id: str,
    epochs: int,
    device: str,
    output_dir: str,
) -> None:
    """Print training start information."""
    lines = [
        f"[cyan]Run ID:[/cyan] {run_id}",
        f"[cyan]Epochs:[/cyan] {epochs}",
        f"[cyan]Device:[/cyan] {device}",
        f"[cyan]Output:[/cyan] {output_dir}",
    ]
    panel = Panel("\n".join(lines), title="🚀 Starting Training", border_style="green")
    console.print(panel)


def print_training_complete(
    run_id: str,
    best_metric: float,
    best_checkpoint: str,
    duration: str,
) -> None:
    """Print training completion information."""
    lines = [
        f"[green]✓[/green] Training completed successfully!",
        "",
        f"[cyan]Run ID:[/cyan] {run_id}",
        f"[cyan]Best Metric:[/cyan] {best_metric:.4f}",
        f"[cyan]Best Checkpoint:[/cyan] {best_checkpoint}",
        f"[cyan]Duration:[/cyan] {duration}",
    ]
    panel = Panel("\n".join(lines), title="🎉 Training Complete", border_style="green")
    console.print(panel)


def print_evaluation_results(
    metrics: dict[str, float],
    num_samples: int,
    duration: str | None = None,
) -> None:
    """Print evaluation results."""
    # Main metrics table
    table = Table(title="📊 Evaluation Results", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    for name, value in sorted(metrics.items()):
        if "/" not in name:
            table.add_row(name, f"{value:.4f}")

    console.print(table)

    # Summary line
    summary = f"Evaluated [bold]{num_samples}[/bold] samples"
    if duration:
        summary += f" in [bold]{duration}[/bold]"
    console.print(f"\n{summary}")


def print_export_result(
    format: str,
    path: str,
    size_mb: float,
    input_shape: tuple,
) -> None:
    """Print export result."""
    lines = [
        f"[green]✓[/green] Model exported successfully!",
        "",
        f"[cyan]Format:[/cyan] {format.upper()}",
        f"[cyan]Path:[/cyan] {path}",
        f"[cyan]Size:[/cyan] {size_mb:.2f} MB",
        f"[cyan]Input Shape:[/cyan] {input_shape}",
    ]
    panel = Panel("\n".join(lines), title="📦 Export Complete", border_style="green")
    console.print(panel)


def print_runs_table(runs: list) -> None:
    """Print runs in a table."""
    if not runs:
        console.print("[dim]No runs found[/dim]")
        return

    table = Table(title="📋 Experiment Runs", show_header=True, header_style="bold magenta")
    table.add_column("Status", justify="center", width=3)
    table.add_column("Run ID", style="cyan")
    table.add_column("Project", style="dim")
    table.add_column("Created", style="dim")
    table.add_column("Metrics", style="green")

    status_icons = {
        "pending": "⏳",
        "running": "🔄",
        "completed": "✅",
        "failed": "❌",
        "interrupted": "⚠️",
    }

    for run in runs:
        status = status_icons.get(run.status.value, "?")
        created = run.created_at.strftime("%Y-%m-%d %H:%M")

        # Get key metric if available
        metrics_str = ""
        if run.metrics:
            for key in ["mIoU", "Dice", "IoU"]:
                if key in run.metrics:
                    metrics_str = f"{key}: {run.metrics[key]:.3f}"
                    break

        table.add_row(status, run.id, run.project, created, metrics_str)

    console.print(table)


@contextmanager
def status_spinner(message: str) -> Iterator[None]:
    """Context manager for a status spinner."""
    with console.status(f"[bold cyan]{message}[/bold cyan]", spinner="dots"):
        yield


def create_training_progress() -> Progress:
    """Create a progress bar for training."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
        expand=False,
    )


def create_epoch_progress() -> Progress:
    """Create a progress bar for epochs."""
    return Progress(
        TextColumn("[bold cyan]Epoch {task.completed}/{task.total}"),
        BarColumn(bar_width=30, complete_style="green", finished_style="green"),
        TaskProgressColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
        console=console,
        expand=False,
    )


def create_batch_progress() -> Progress:
    """Create a progress bar for batches."""
    return Progress(
        TextColumn("  [dim]Batch"),
        BarColumn(bar_width=20),
        MofNCompleteColumn(),
        TextColumn("•"),
        TextColumn("[dim]{task.fields[loss]:.4f}[/dim]", justify="right"),
        console=console,
        expand=False,
    )


def create_inference_progress() -> Progress:
    """Create a progress bar for inference."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        console=console,
        expand=False,
    )


def format_duration(seconds: float) -> str:
    """Format duration in human-readable format."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def print_epoch_summary(
    epoch: int,
    total_epochs: int,
    train_loss: float,
    val_loss: float,
    metrics: dict[str, float],
    lr: float,
    best: bool = False,
) -> None:
    """Print epoch summary."""
    # Build metrics string
    metrics_parts = []
    for name, value in metrics.items():
        if "/" not in name and name not in ["loss"]:
            metrics_parts.append(f"{name}: {value:.4f}")
    metrics_str = " | ".join(metrics_parts[:3])  # Limit to 3 metrics

    # Status indicator
    status = "[green]★[/green] " if best else "  "

    console.print(
        f"{status}Epoch [bold]{epoch:3d}[/bold]/{total_epochs} │ "
        f"train_loss: [yellow]{train_loss:.4f}[/yellow] │ "
        f"val_loss: [yellow]{val_loss:.4f}[/yellow] │ "
        f"{metrics_str} │ "
        f"lr: [dim]{lr:.2e}[/dim]"
    )
