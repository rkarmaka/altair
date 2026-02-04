"""Tests for the Run class."""

from datetime import datetime

import pytest

from altair.core.run import Run, RunStatus


class TestRunStatus:
    """Test cases for RunStatus enum."""

    def test_status_values(self):
        """Test all status values exist."""
        assert RunStatus.PENDING.value == "pending"
        assert RunStatus.RUNNING.value == "running"
        assert RunStatus.COMPLETED.value == "completed"
        assert RunStatus.FAILED.value == "failed"
        assert RunStatus.INTERRUPTED.value == "interrupted"


class TestRun:
    """Test cases for the Run class."""

    def test_create_run(self, temp_dir, sample_config_dict):
        """Test creating a new run."""
        run = Run.create(
            name="test-run",
            project="test-project",
            config=sample_config_dict,
            output_dir=temp_dir,
        )

        assert run.name == "test-run"
        assert run.project == "test-project"
        assert run.status == RunStatus.PENDING
        assert "test-run" in run.id

    def test_run_creates_directories(self, temp_dir, sample_config_dict):
        """Test that creating a run creates necessary directories."""
        run = Run.create(
            name="test-run",
            project="test-project",
            config=sample_config_dict,
            output_dir=temp_dir,
        )

        assert run.output_dir.exists()
        assert run.checkpoints_dir.exists()
        assert run.logs_dir.exists()

    def test_run_saves_metadata(self, temp_dir, sample_config_dict):
        """Test that run metadata is saved."""
        run = Run.create(
            name="test-run",
            project="test-project",
            config=sample_config_dict,
            output_dir=temp_dir,
        )

        assert (run.output_dir / "run.json").exists()
        assert run.config_path.exists()

    def test_load_run(self, temp_dir, sample_config_dict):
        """Test loading a run from disk."""
        # Create and save a run
        original = Run.create(
            name="test-run",
            project="test-project",
            config=sample_config_dict,
            output_dir=temp_dir,
        )
        original.update_metrics({"val/mIoU": 0.85, "val/loss": 0.15})

        # Load and verify
        loaded = Run.load(original.output_dir)

        assert loaded.id == original.id
        assert loaded.name == original.name
        assert loaded.project == original.project
        assert loaded.metrics["val/mIoU"] == 0.85

    def test_update_status(self, temp_dir, sample_config_dict):
        """Test updating run status."""
        run = Run.create(
            name="test-run",
            project="test-project",
            config=sample_config_dict,
            output_dir=temp_dir,
        )

        assert run.status == RunStatus.PENDING

        run.update_status(RunStatus.RUNNING)
        assert run.status == RunStatus.RUNNING
        assert run.is_running

        run.update_status(RunStatus.COMPLETED)
        assert run.status == RunStatus.COMPLETED
        assert run.is_completed
        assert run.completed_at is not None

    def test_update_metrics(self, temp_dir, sample_config_dict):
        """Test updating run metrics."""
        run = Run.create(
            name="test-run",
            project="test-project",
            config=sample_config_dict,
            output_dir=temp_dir,
        )

        run.update_metrics({"train/loss": 0.5})
        assert run.metrics["train/loss"] == 0.5

        run.update_metrics({"val/loss": 0.3, "val/mIoU": 0.8})
        assert run.metrics["val/loss"] == 0.3
        assert run.metrics["val/mIoU"] == 0.8
        assert run.metrics["train/loss"] == 0.5  # Previous metrics preserved

    def test_get_metric(self, temp_dir, sample_config_dict):
        """Test getting a specific metric."""
        run = Run.create(
            name="test-run",
            project="test-project",
            config=sample_config_dict,
            output_dir=temp_dir,
        )
        run.update_metrics({"val/mIoU": 0.85})

        assert run.get_metric("val/mIoU") == 0.85
        assert run.get_metric("nonexistent") is None
        assert run.get_metric("nonexistent", default=0.0) == 0.0

    def test_checkpoint_properties(self, temp_dir, sample_config_dict):
        """Test checkpoint path properties."""
        run = Run.create(
            name="test-run",
            project="test-project",
            config=sample_config_dict,
            output_dir=temp_dir,
        )

        # No checkpoints initially
        assert run.best_checkpoint is None
        assert run.last_checkpoint is None

        # Create checkpoint files
        (run.checkpoints_dir / "best.pt").touch()
        (run.checkpoints_dir / "last.pt").touch()

        assert run.best_checkpoint is not None
        assert run.last_checkpoint is not None
        assert run.best_checkpoint.name == "best.pt"

    def test_status_properties(self, temp_dir, sample_config_dict):
        """Test status helper properties."""
        run = Run.create(
            name="test-run",
            project="test-project",
            config=sample_config_dict,
            output_dir=temp_dir,
        )

        assert not run.is_running
        assert not run.is_completed
        assert not run.is_failed

        run.status = RunStatus.RUNNING
        assert run.is_running

        run.status = RunStatus.COMPLETED
        assert run.is_completed

        run.status = RunStatus.FAILED
        assert run.is_failed

    def test_load_nonexistent_raises_error(self, temp_dir):
        """Test that loading nonexistent run raises error."""
        with pytest.raises(FileNotFoundError):
            Run.load(temp_dir / "nonexistent")

    def test_run_repr(self, temp_dir, sample_config_dict):
        """Test string representation of run."""
        run = Run.create(
            name="test-run",
            project="test-project",
            config=sample_config_dict,
            output_dir=temp_dir,
        )

        repr_str = repr(run)
        assert "test-run" in repr_str
        assert "pending" in repr_str

    def test_run_id_format(self, temp_dir, sample_config_dict):
        """Test that run ID has expected format."""
        run = Run.create(
            name="my-experiment",
            project="test",
            config=sample_config_dict,
            output_dir=temp_dir,
        )

        # ID should contain name and timestamp
        assert "my-experiment" in run.id
        # Should have timestamp-like pattern
        assert "_" in run.id

    def test_created_at_timestamp(self, temp_dir, sample_config_dict):
        """Test that created_at is set correctly."""
        before = datetime.now()
        run = Run.create(
            name="test-run",
            project="test",
            config=sample_config_dict,
            output_dir=temp_dir,
        )
        after = datetime.now()

        assert before <= run.created_at <= after

    def test_completed_at_set_on_completion(self, temp_dir, sample_config_dict):
        """Test that completed_at is set when status becomes COMPLETED."""
        run = Run.create(
            name="test-run",
            project="test",
            config=sample_config_dict,
            output_dir=temp_dir,
        )

        assert run.completed_at is None

        run.update_status(RunStatus.COMPLETED)

        assert run.completed_at is not None
        assert run.completed_at >= run.created_at


class TestRunPersistence:
    """Test cases for run save/load functionality."""

    def test_round_trip_preserves_data(self, temp_dir, sample_config_dict):
        """Test that save and load preserves all data."""
        original = Run.create(
            name="test-run",
            project="test-project",
            config=sample_config_dict,
            output_dir=temp_dir,
        )
        original.update_status(RunStatus.RUNNING)
        original.update_metrics({"epoch": 50, "val/mIoU": 0.75})
        original.update_status(RunStatus.COMPLETED)

        # Load from disk
        loaded = Run.load(original.output_dir)

        assert loaded.id == original.id
        assert loaded.name == original.name
        assert loaded.project == original.project
        assert loaded.status == original.status
        assert loaded.metrics == original.metrics
        assert loaded.created_at == original.created_at
        # Note: datetime comparison might have microsecond differences
        assert loaded.completed_at is not None

    def test_config_preserved_on_load(self, temp_dir, sample_config_dict):
        """Test that config is preserved through save/load."""
        original = Run.create(
            name="test-run",
            project="test",
            config=sample_config_dict,
            output_dir=temp_dir,
        )

        loaded = Run.load(original.output_dir)

        assert loaded.config == sample_config_dict
