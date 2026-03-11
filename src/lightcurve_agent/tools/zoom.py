"""Zoom-in analysis tools for detailed local feature inspection."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from langchain_core.tools import tool

from lightcurve_agent.config import get_settings
from lightcurve_agent.core.data import inspect_light_curve_file, load_light_curve
from lightcurve_agent.core.models import LightCurve
from lightcurve_agent.core.plotting.base import PlotConfig, plot_light_curve
from lightcurve_agent.core.plotting.grids import create_zoom_grid
from lightcurve_agent.interfaces.vlm import get_vlm_provider


@tool
def obtain_zoom_in(
    csv_path: str,
    time_ranges: list[tuple[float, float]],
    time_col: str | None = None,
    y_col: str | None = None,
    yerr_col: str | None = None,
    y_axis_type: Literal["mag", "flux", "auto"] = "auto",
) -> str:
    """Create zoomed-in plots for specific time ranges.

    Args:
        csv_path: Path to the CSV file
        time_ranges: List of 3 [start, end] time ranges in days
        time_col: Name of the time column
        y_col: Name of the magnitude/flux column
        yerr_col: Name of the error column (optional)
        y_axis_type: "mag" or "flux"

    Returns:
        Path to combined 2x2 grid image
    """
    try:
        if len(time_ranges) != 3:
            raise ValueError("Exactly 3 time ranges required for 2x2 grid")

        settings = get_settings()
        out_dir = settings.artifacts_dir / "zooms"
        out_dir.mkdir(parents=True, exist_ok=True)
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

        t = np.asarray(lc.time)
        y = np.asarray(lc.magnitude)
        yerr = np.asarray(lc.error) if lc.error is not None else None

        # Generate full plot
        full_path = out_dir / "Zoom_in_full_view.png"
        full_config = PlotConfig(
            title="Full Light Curve",
            x_label=f"{lc.time_col} [d]",
            y_label=lc.mag_col,
            invert_y=lc.scale == "mag",
        )
        plot_light_curve(lc, full_config, full_path)

        # Generate zoom plots
        zoom_paths = []
        for i, (start, end) in enumerate(time_ranges):
            mask = (t >= start) & (t <= end)
            if not np.any(mask):
                raise ValueError(f"No data in time range [{start}, {end}]")

            t_z = pd.Series(t[mask])
            y_z = pd.Series(y[mask])
            yerr_z = pd.Series(yerr[mask]) if yerr is not None else None

            zoom_lc = LightCurve(
                time=t_z,
                magnitude=y_z,
                error=yerr_z,
                band=lc.band.loc[mask].reset_index(drop=True) if lc.band is not None else None,
                time_col=f"{lc.time_col} [d]",
                mag_col=lc.mag_col,
                err_col=lc.err_col,
                band_col=lc.band_col,
                scale=lc.scale,
                source_name=lc.source_name,
                ra_deg=lc.ra_deg,
                dec_deg=lc.dec_deg,
                metadata=lc.metadata.copy(),
            )

            z_path = out_dir / f"zoom_{i}.png"
            z_config = PlotConfig(
                title=f"Zoom: {start} - {end} d",
                x_label=f"{lc.time_col} [d]",
                y_label=lc.mag_col,
                invert_y=lc.scale == "mag",
            )
            plot_light_curve(zoom_lc, z_config, z_path)
            zoom_paths.append(z_path)

        # Create combined grid
        combined_path = out_dir / "zoomed_22_grid.png"
        create_zoom_grid(full_path, zoom_paths, combined_path)

        return str(combined_path)

    except Exception as e:
        return f"error: {str(e)}"


@tool
def analyze_zoomed_plots(
    combined_image_path: str,
    candidate_type: list[str] | None = None,
) -> str:
    """Analyze zoomed-in plots for local morphological features.

    Args:
        combined_image_path: Path to the 2x2 zoom grid image
        candidate_type: List of candidate types to focus on

    Returns:
        VLM analysis text
    """
    try:
        default_prompt = """
Analyze this 2x2 light curve grid.
Top-Left: Full survey. Top-Right, Bottom-Left, Bottom-Right: Zoom-ins.

Evaluate:
1. **Sampling**: Are zoomed features well-sampled? Variation exceeding
   measurement uncertainty suggests under-sampling. If undersampled,
   indicate need for phase-folding analysis.

2. **Local Morphology** (if well-sampled): Describe shapes in zooms
   (ingress/egress symmetry, flare decay, occultation, stochastic noise).

3. **Consistency** (if well-sampled): Are zoom features representative
   of the whole or unique events?

4. **Data Integrity**: Local outliers or instrumental jumps.

5. **Conclusion** (if well-sampled): How do details refine classification?

6. **Further zoom-in** (if well-sampled): Determine if smaller time ranges
   need inspection. If so, specify 3 time ranges [t_start, t_end] in days.
   Only if data points are too crowded.
"""

        if candidate_type:
            default_prompt += f"\n\nPay special attention to: {', '.join(candidate_type)}."

        print(f"Analyzing zoomed plots: {combined_image_path}")

        provider = get_vlm_provider("openai")
        result = provider.analyze_image(combined_image_path, default_prompt)

        print("Analysis complete")
        return result

    except Exception as e:
        return f"error: {str(e)}"
