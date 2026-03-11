"""Plotting tool for light curves."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from langchain_core.tools import tool

from lightcurve_agent.config import get_settings
from lightcurve_agent.core.data import inspect_light_curve_file, load_light_curve
from lightcurve_agent.core.plotting.base import PlotConfig, plot_light_curve


@tool
def plot_light_curve_csv(
    csv_path: str,
    time_col: str | None = None,
    y_col: str | None = None,
    yerr_col: str | None = None,
    y_axis_type: Literal["mag", "flux", "auto"] = "auto",
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
        profile = inspect_light_curve_file(csv_path, scale=y_axis_type)

        # Load data
        lc = load_light_curve(
            csv_path,
            time_col=time_col or profile.time_col,
            mag_col=y_col or profile.mag_col,
            err_col=yerr_col,
            band_col=profile.band_col,
            scale=y_axis_type,
        )

        # Determine output path
        if out_path is None:
            stem = Path(csv_path).stem
            out_path = settings.artifacts_dir / f"{stem}_lightcurve.png"

        # Create plot config
        config = PlotConfig(
            title="Full Light Curve",
            x_label=f"{lc.time_col} [d]",
            y_label=lc.mag_col,
            invert_y=lc.scale == "mag",
        )

        # Plot
        image_path = plot_light_curve(lc, config, out_path)

        return {
            "image_path": str(image_path),
            "n_points": len(lc.time),
            "columns_used": {
                "time_col": lc.time_col,
                "y_col": lc.mag_col,
                "yerr_col": lc.err_col or None,
                "band_col": lc.band_col or None,
            },
            "band_labels": lc.band_labels,
            "source_context": {
                "source_name": lc.source_name,
                "ra_deg": lc.ra_deg,
                "dec_deg": lc.dec_deg,
            },
        }

    except Exception as e:
        return {"error": str(e)}
