"""
Utility functions for Altair.

Provides visualization, file handling, console output, and other helper functions.
"""

from altair.utils.visualization import (
    DEFAULT_PALETTE,
    MEDICAL_PALETTE,
    CITYSCAPES_PALETTE,
    SampleExporter,
    create_comparison,
    create_error_map,
    create_overlay,
    get_palette,
    mask_to_rgb,
    save_prediction,
    visualize_prediction,
)

from altair.utils.console import (
    console,
    print_banner,
    print_header,
    print_info,
    print_success,
    print_warning,
    print_error,
    print_metric,
    print_metrics_table,
)

__all__ = [
    # Palettes
    "DEFAULT_PALETTE",
    "MEDICAL_PALETTE",
    "CITYSCAPES_PALETTE",
    "get_palette",
    # Visualization
    "mask_to_rgb",
    "create_overlay",
    "create_comparison",
    "create_error_map",
    "visualize_prediction",
    "save_prediction",
    # Export
    "SampleExporter",
    # Console
    "console",
    "print_banner",
    "print_header",
    "print_info",
    "print_success",
    "print_warning",
    "print_error",
    "print_metric",
    "print_metrics_table",
]
