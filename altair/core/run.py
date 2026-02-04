"""
Run management for Altair experiments.

A Run represents a single experiment execution, providing access to
configuration, metrics, checkpoints, and status.

Example:
    >>> run = alt.train("configs/unet.yaml")
    >>> print(run.id)
    'exp_2024_abc123'
    >>> print(run.status)
    'completed'
    >>> print(run.metrics)
    {'val/mIoU': 0.82, 'val/loss': 0.15}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


class RunStatus(str, Enum):
    """Status of an experiment run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


@dataclass
class Run:
    """
    Represents an experiment run.

    A Run encapsulates all information about an experiment execution,
    including configuration, metrics, checkpoints, and artifacts.

    Attributes:
        id: Unique identifier for this run.
        name: Human-readable experiment name.
        project: Project this run belongs to.
        status: Current status of the run.
        config: Full experiment configuration.
        metrics: Dictionary of metric values.
        output_dir: Directory containing run outputs.
        created_at: Timestamp when run was created.
        completed_at: Timestamp when run completed (if applicable).

    Example:
        >>> run = Run.load("experiments/exp_abc123")
        >>> print(run.best_checkpoint)
        Path('experiments/exp_abc123/checkpoints/best.pt')
    """

    id: str
    name: str
    project: str
    status: RunStatus
    config: dict[str, Any]
    output_dir: Path
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    _metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def checkpoints_dir(self) -> Path:
        """Path to the checkpoints directory."""
        return self.output_dir / "checkpoints"

    @property
    def best_checkpoint(self) -> Path | None:
        """Path to the best model checkpoint, if it exists."""
        path = self.checkpoints_dir / "best.pt"
        return path if path.exists() else None

    @property
    def last_checkpoint(self) -> Path | None:
        """Path to the last model checkpoint, if it exists."""
        path = self.checkpoints_dir / "last.pt"
        return path if path.exists() else None

    @property
    def logs_dir(self) -> Path:
        """Path to the logs directory."""
        return self.output_dir / "logs"

    @property
    def config_path(self) -> Path:
        """Path to the saved configuration file."""
        return self.output_dir / "config.yaml"

    @property
    def is_running(self) -> bool:
        """Check if the run is currently in progress."""
        return self.status == RunStatus.RUNNING

    @property
    def is_completed(self) -> bool:
        """Check if the run completed successfully."""
        return self.status == RunStatus.COMPLETED

    @property
    def is_failed(self) -> bool:
        """Check if the run failed."""
        return self.status in (RunStatus.FAILED, RunStatus.INTERRUPTED)

    def get_metric(self, name: str, default: float | None = None) -> float | None:
        """
        Get a specific metric value.

        Args:
            name: Metric name (e.g., 'val/mIoU').
            default: Default value if metric not found.

        Returns:
            The metric value or default.
        """
        return self.metrics.get(name, default)

    def save(self) -> None:
        """
        Save run metadata to disk.

        Saves the run state to a JSON file in the output directory,
        allowing it to be loaded later.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Save run metadata
        metadata = {
            "id": self.id,
            "name": self.name,
            "project": self.project,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metrics": self.metrics,
            "metadata": self._metadata,
        }

        with open(self.output_dir / "run.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Save config
        with open(self.config_path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, path: str | Path) -> "Run":
        """
        Load a run from disk.

        Args:
            path: Path to the run directory.

        Returns:
            A Run instance with loaded state.

        Raises:
            FileNotFoundError: If the run directory or metadata doesn't exist.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Run directory not found: {path}")

        run_file = path / "run.json"
        if not run_file.exists():
            raise FileNotFoundError(f"Run metadata not found: {run_file}")

        with open(run_file) as f:
            metadata = json.load(f)

        # Load config
        config_path = path / "config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        return cls(
            id=metadata["id"],
            name=metadata["name"],
            project=metadata["project"],
            status=RunStatus(metadata["status"]),
            config=config,
            output_dir=path,
            created_at=datetime.fromisoformat(metadata["created_at"]),
            completed_at=(
                datetime.fromisoformat(metadata["completed_at"])
                if metadata.get("completed_at")
                else None
            ),
            metrics=metadata.get("metrics", {}),
            _metadata=metadata.get("metadata", {}),
        )

    @classmethod
    def create(
        cls,
        name: str,
        project: str,
        config: dict[str, Any],
        output_dir: Path,
    ) -> "Run":
        """
        Create a new run.

        Args:
            name: Experiment name.
            project: Project name.
            config: Experiment configuration.
            output_dir: Base output directory.

        Returns:
            A new Run instance with a generated ID.
        """
        # Generate unique run ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{name}_{timestamp}"

        # Create run directory
        run_dir = output_dir / run_id

        run = cls(
            id=run_id,
            name=name,
            project=project,
            status=RunStatus.PENDING,
            config=config,
            output_dir=run_dir,
        )

        run.save()
        return run

    def update_status(self, status: RunStatus) -> None:
        """
        Update the run status and save.

        Args:
            status: New status for the run.
        """
        self.status = status
        if status == RunStatus.COMPLETED:
            self.completed_at = datetime.now()
        self.save()

    def update_metrics(self, metrics: dict[str, float]) -> None:
        """
        Update metrics and save.

        Args:
            metrics: Dictionary of metric values to update.
        """
        self.metrics.update(metrics)
        self.save()

    def __repr__(self) -> str:
        """Return string representation of the run."""
        return (
            f"Run(id='{self.id}', name='{self.name}', "
            f"status={self.status.value}, metrics={len(self.metrics)} items)"
        )
