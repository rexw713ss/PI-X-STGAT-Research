"""Canonical locations for generated experiment artifacts."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
FIGURES_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"


def _artifact_path(directory: Path, filename: str | Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    return directory / Path(filename).name


def figure_path(filename: str | Path) -> Path:
    """Return a path in the repository's figure directory."""
    return _artifact_path(FIGURES_DIR, filename)


def result_path(filename: str | Path) -> Path:
    """Return a path in the repository's result directory."""
    return _artifact_path(RESULTS_DIR, filename)


def checkpoint_path(filename: str | Path) -> Path:
    """Return a path in the repository's checkpoint directory."""
    return _artifact_path(CHECKPOINTS_DIR, filename)
