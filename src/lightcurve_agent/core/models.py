"""Core data models for light curve analysis.

These are pure data structures with no dependencies on external services.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LightCurve:
    """Immutable light curve data container."""

    time: pd.Series
    magnitude: pd.Series
    error: pd.Series | None = None
    time_col: str = "time"
    mag_col: str = "mag"
    err_col: str = "mag_err"
    scale: Literal["mag", "flux"] = "mag"

    def __post_init__(self) -> None:
        """Validate data consistency."""
        n = len(self.time)
        if len(self.magnitude) != n:
            raise ValueError("time and magnitude must have same length")
        if self.error is not None and len(self.error) != n:
            raise ValueError("error series must match time length")

    def masked(self, mask: np.ndarray) -> Self:
        """Return a new LightCurve with masked data."""
        return self.__class__(
            time=self.time[mask],
            magnitude=self.magnitude[mask],
            error=self.error[mask] if self.error is not None else None,
            time_col=self.time_col,
            mag_col=self.mag_col,
            err_col=self.err_col,
            scale=self.scale,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame."""
        df = pd.DataFrame({
            self.time_col: self.time,
            self.mag_col: self.magnitude,
        })
        if self.error is not None:
            df[self.err_col] = self.error
        return df


@dataclass(frozen=True)
class Periodogram:
    """Periodogram results container."""

    grid: pd.Series  # Period or frequency values
    power: pd.Series
    grid_kind: Literal["period", "frequency"] = "period"
    best_value: float = 0.0
    best_power: float = 0.0

    def __post_init__(self) -> None:
        """Validate data."""
        if len(self.grid) != len(self.power):
            raise ValueError("grid and power must have same length")


@dataclass(frozen=True)
class PhaseFoldedResult:
    """Container for phase-folded light curve results."""

    best_period: float
    best_power: float
    periodogram_path: Path
    phase_folded_P_path: Path
    phase_folded_P2_path: Path  # P/2
    phase_folded_2P_path: Path

    def all_paths(self) -> list[Path]:
        """Return all generated plot paths."""
        return [
            self.periodogram_path,
            self.phase_folded_P_path,
            self.phase_folded_P2_path,
            self.phase_folded_2P_path,
        ]


@dataclass(frozen=True)
class PlotResult:
    """Result of plotting operation."""

    image_path: Path
    n_points: int
    columns_used: dict


@dataclass(frozen=True)
class DiagnosisResult:
    """Structured diagnosis from VLM analysis."""

    classification: str
    probability: Literal["very high", "high", "middle", "low", "very low"]
    reasoning: str
    recommendations: list[str]
    candidates: list[str]
