"""Light curve data loading and validation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
import numpy as np

from lightcurve_agent.core.models import LightCurve


def load_light_curve(
    csv_path: str | Path,
    *,
    time_col: str = "time",
    mag_col: str = "mag",
    err_col: str | None = "mag_err",
    scale: Literal["mag", "flux"] = "mag",
    validate: bool = True,
) -> LightCurve:
    """Load a light curve from a CSV file.

    Args:
        csv_path: Path to the CSV file
        time_col: Name of the time column
        mag_col: Name of the magnitude/flux column
        err_col: Name of the error column (optional)
        scale: Whether data is in magnitude or flux scale
        validate: Whether to validate data after loading

    Returns:
        LightCurve instance

    Raises:
        FileNotFoundError: If CSV doesn't exist
        ValueError: If required columns missing or data invalid
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Light curve file not found: {path}")

    df = pd.read_csv(path)

    # Check required columns
    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found. Columns: {list(df.columns)}")
    if mag_col not in df.columns:
        raise ValueError(f"Magnitude column '{mag_col}' not found. Columns: {list(df.columns)}")

    # Extract error column if specified
    error = None
    if err_col and err_col in df.columns:
        error = df[err_col]
    elif err_col and err_col not in df.columns:
        # Try common alternatives
        alternatives = ["err", "error", "sigma", "magerr", "mag_err"]
        for alt in alternatives:
            if alt in df.columns:
                error = df[alt]
                err_col = alt
                break

    if validate:
        # Remove non-finite values
        mask = np.isfinite(df[time_col]) & np.isfinite(df[mag_col])
        if error is not None:
            mask &= np.isfinite(error) & (error >= 0)

        if not mask.all():
            df = df[mask]

    return LightCurve(
        time=df[time_col],
        magnitude=df[mag_col],
        error=error,
        time_col=time_col,
        mag_col=mag_col,
        err_col=err_col or "",
        scale=scale,
    )


def detect_columns(df: pd.DataFrame) -> tuple[str, str, str | None]:
    """Auto-detect column names from common conventions.

    Args:
        df: DataFrame to inspect

    Returns:
        Tuple of (time_col, mag_col, err_col)

    Raises:
        ValueError: If columns cannot be determined
    """
    cols_lower = {c.lower(): c for c in df.columns}

    # Time column detection
    time_candidates = ["time", "jd", "mjd", "hjd", "bjd", "date"]
    time_col = None
    for cand in time_candidates:
        if cand in cols_lower:
            time_col = cols_lower[cand]
            break

    # Magnitude/flux column detection
    mag_candidates = ["mag", "magnitude", "flux", "brightness", "intensity"]
    mag_col = None
    for cand in mag_candidates:
        if cand in cols_lower:
            mag_col = cols_lower[cand]
            break

    # Error column detection
    err_candidates = ["mag_err", "error", "err", "sigma", "magerr", "flux_err"]
    err_col = None
    for cand in err_candidates:
        if cand in cols_lower:
            err_col = cols_lower[cand]
            break

    if time_col is None or mag_col is None:
        raise ValueError(
            f"Could not auto-detect columns. Available: {list(df.columns)}. "
            "Please specify column names explicitly."
        )

    return time_col, mag_col, err_col


def auto_load_light_curve(
    csv_path: str | Path,
    scale: Literal["mag", "flux"] = "mag",
) -> LightCurve:
    """Load light curve with automatic column detection.

    Args:
        csv_path: Path to CSV file
        scale: Data scale type

    Returns:
        LightCurve instance
    """
    path = Path(csv_path)
    df = pd.read_csv(path)
    time_col, mag_col, err_col = detect_columns(df)
    return load_light_curve(path, time_col=time_col, mag_col=mag_col, err_col=err_col, scale=scale)
