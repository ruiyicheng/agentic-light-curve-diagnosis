"""Filesystem interface for artifact management.

Provides abstraction for storing and retrieving analysis artifacts.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterator

from lightcurve_agent.config import get_settings


class ArtifactStore:
    """Manages storage and retrieval of analysis artifacts."""

    def __init__(self, base_dir: Path | None = None):
        """Initialize artifact store.

        Args:
            base_dir: Base directory for artifacts. Uses settings if None.
        """
        if base_dir is None:
            from lightcurve_agent.config import get_settings
            base_dir = get_settings().artifacts_dir
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def get_path(self, name: str, subdir: str | None = None) -> Path:
        """Get path for an artifact.

        Args:
            name: Filename
            subdir: Optional subdirectory

        Returns:
            Path to artifact location
        """
        if subdir:
            path = self.base_dir / subdir / name
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            path = self.base_dir / name
        return path

    def save(self, source: Path, name: str | None = None, subdir: str | None = None) -> Path:
        """Save a file to the artifact store.

        Args:
            source: Source file path
            name: Target name. Uses source name if None.
            subdir: Optional subdirectory

        Returns:
            Path to saved artifact
        """
        target_name = name or source.name
        target = self.get_path(target_name, subdir)
        shutil.copy2(source, target)
        return target

    def list(self, pattern: str = "*", subdir: str | None = None) -> Iterator[Path]:
        """List artifacts matching pattern.

        Args:
            pattern: Glob pattern
            subdir: Optional subdirectory

        Yields:
            Paths to matching artifacts
        """
        search_dir = self.base_dir / subdir if subdir else self.base_dir
        yield from search_dir.glob(pattern)

    def clear(self, confirm: bool = False) -> None:
        """Clear all artifacts.

        Args:
            confirm: Must be True to actually delete
        """
        if not confirm:
            raise ValueError("Set confirm=True to clear artifacts")

        for item in self.base_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

    def exists(self, name: str, subdir: str | None = None) -> bool:
        """Check if an artifact exists.

        Args:
            name: Filename
            subdir: Optional subdirectory

        Returns:
            True if exists
        """
        return self.get_path(name, subdir).exists()


class DataStore:
    """Manages access to input data files."""

    def __init__(self, base_dir: Path | None = None):
        """Initialize data store.

        Args:
            base_dir: Base directory for data. Uses settings if None.
        """
        if base_dir is None:
            from lightcurve_agent.config import get_settings
            base_dir = get_settings().data_dir
        self.base_dir = Path(base_dir)

    def get_path(self, name: str) -> Path:
        """Get path to a data file.

        Args:
            name: Filename

        Returns:
            Path to data file

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        path = self.base_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        return path

    def list(self, pattern: str = "*.csv") -> Iterator[Path]:
        """List data files matching pattern.

        Args:
            pattern: Glob pattern

        Yields:
            Paths to matching data files
        """
        yield from self.base_dir.glob(pattern)
