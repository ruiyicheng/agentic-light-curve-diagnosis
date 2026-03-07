"""Plotting tool for light curves."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from langchain_core.tools import tool

from lightcurve_agent.config import get_settings
from lightcurve_agent.core.data import load_light_curve
from lightcurve_agent.core.plotting.base import PlotConfig, plot_light_curve


@tool
def plot_light_curve_csv(
    csv_path: str,
    time_col: str = "time",
    y_col: str = "mag",
    yerr_col: str | None = "mag_err",
    y_axis_type: Literal["mag", "flux"] = "mag",
    out_path: str | None = None,
) -> dict:
    """Load a light curve CSV and save a matplotlib plot as a PNG.

    Args:
        csv_path: Path to the CSV file
        time_col: Name of the time column
        y_col: Name of the magnitude/flux column
        yerr_col: Name of the error column (optional)
        y_axis_type: "mag" or "flux" - controls y-axis inversion
        out_path: Output path. If None, uses artifacts directory.

    Returns:
        Dict with keys: image_path, n_points, columns_used
    """
    try:
        settings = get_settings()

        # Load data
        lc = load_light_curve(
            csv_path,
            time_col=time_col,
            mag_col=y_col,
            err_col=yerr_col,
            scale=y_axis_type,
        )

        # Determine output path
        if out_path is None:
            stem = Path(csv_path).stem
            out_path = settings.artifacts_dir / f"{stem}_lightcurve.png"

        # Create plot config
        config = PlotConfig(
            title="Full Light Curve",
            x_label=f"{time_col} [d]",
            y_label=y_col,
            invert_y=y_axis_type == "mag",
        )

        # Plot
        image_path = plot_light_curve(lc, config, out_path)

        return {
            "image_path": str(image_path),
            "n_points": len(lc.time),
            "columns_used": {
                "time_col": time_col,
                "y_col": y_col,
                "yerr_col": yerr_col,
            },
        }

    except Exception as e:
        return {"error": str(e)}
