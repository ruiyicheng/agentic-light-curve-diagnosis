"""Periodogram analysis tools: GLS, BLS, and phase-folded analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from langchain_core.tools import tool

from lightcurve_agent.config import get_settings
from lightcurve_agent.core.data import inspect_light_curve_file, load_light_curve, normalize_multiband_light_curve
from lightcurve_agent.core.analysis.gls import compute_gls_periodogram, phase_fold
from lightcurve_agent.core.analysis.bls import compute_bls_periodogram
from lightcurve_agent.core.plotting.base import PlotConfig, plot_light_curve, plot_periodogram
from lightcurve_agent.core.plotting.grids import create_grid
from lightcurve_agent.core.models import LightCurve
from lightcurve_agent.interfaces.vlm import get_vlm_provider


@tool
def obtain_GLS(
    csv_path: str,
    time_col: str | None = None,
    y_col: str | None = None,
    yerr_col: str | None = None,
    y_axis_type: Literal["mag", "flux", "auto"] = "auto",
) -> dict:
    """Compute Generalized Lomb-Scargle periodogram for a light curve.

    Args:
        csv_path: Path to the CSV file
        time_col: Name of the time column
        y_col: Name of the magnitude/flux column
        yerr_col: Name of the error column (optional)
        y_axis_type: "mag" or "flux"

    Returns:
        Dict with keys:
            - best_period: float
            - best_power: float
            - periodogram_path: str
            - phase_folded_P_path: str
            - phase_folded_P/2_path: str
            - phase_folded_2*P_path: str
    """
    try:
        settings = get_settings()
        out_dir = settings.artifacts_dir

        print(f"Computing GLS for {csv_path}")
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
        analysis_lc = normalize_multiband_light_curve(lc)

        # Compute periodogram
        periodogram = compute_gls_periodogram(analysis_lc)

        best_period = periodogram.best_value
        best_power = periodogram.best_power

        print(f"Best period: {best_period:.5f} days with power {best_power:.5f} [{periodogram.backend}]")

        # Save periodogram plot
        periodogram_path = out_dir / "gls_periodogram.png"
        pconfig = PlotConfig(
            title="GLS Periodogram",
            x_label="Period [d]",
            y_label="GLS Power",
            x_scale="log",
            y_scale="log",
            plot_style="-",
        )
        plot_periodogram(periodogram, pconfig, periodogram_path)

        # Generate phase-folded plots at P, P/2, and 2P
        results = {
            "P": (best_period, "GLS_phase_folded_P.png"),
            "P/2": (best_period / 2, "GLS_phase_folded_P2.png"),
            "2P": (best_period * 2, "GLS_phase_folded_2P.png"),
        }

        output_paths = {}
        for label, (period_val, filename) in results.items():
            t_folded = phase_fold(analysis_lc, period_val)

            folded_lc = LightCurve(
                time=pd.Series(t_folded),
                magnitude=analysis_lc.magnitude.reset_index(drop=True),
                error=analysis_lc.error.reset_index(drop=True) if analysis_lc.error is not None else None,
                band=analysis_lc.band.reset_index(drop=True) if analysis_lc.band is not None else None,
                time_col=f"Phase (P={period_val:.5f} d)",
                mag_col=analysis_lc.mag_col,
                err_col=analysis_lc.err_col,
                band_col=analysis_lc.band_col,
                scale=analysis_lc.scale,
                source_name=analysis_lc.source_name,
                ra_deg=analysis_lc.ra_deg,
                dec_deg=analysis_lc.dec_deg,
                metadata=analysis_lc.metadata.copy(),
            )

            path = out_dir / filename
            config = PlotConfig(
                title=f"Phase Folded at {label}",
                x_label=f"Phase (P={period_val:.5f} d)",
                y_label=analysis_lc.mag_col,
                require_binning=True,
                invert_y=analysis_lc.scale == "mag",
            )
            plot_light_curve(folded_lc, config, path)
            output_paths[label] = str(path)

        return {
            "best_period": best_period,
            "best_power": best_power,
            "backend": periodogram.backend,
            "periodogram_path": str(periodogram_path),
            "phase_folded_P_path": output_paths["P"],
            "phase_folded_P/2_path": output_paths["P/2"],
            "phase_folded_2*P_path": output_paths["2P"],
            "band_labels": lc.band_labels,
            "columns_used": {
                "time_col": lc.time_col,
                "y_col": lc.mag_col,
                "yerr_col": lc.err_col or None,
                "band_col": lc.band_col or None,
            },
        }

    except Exception as e:
        return {"error": str(e)}


@tool
def obtain_BLS(
    csv_path: str,
    time_col: str | None = None,
    y_col: str | None = None,
    yerr_col: str | None = None,
    y_axis_type: Literal["mag", "flux", "auto"] = "auto",
) -> dict:
    """Compute Box Least Squares periodogram for transit detection.

    Args:
        csv_path: Path to the CSV file
        time_col: Name of the time column
        y_col: Name of the magnitude/flux column
        yerr_col: Name of the error column (optional)
        y_axis_type: "mag" or "flux"

    Returns:
        Dict with keys:
            - best_period: float
            - best_power: float
            - periodogram_path: str
            - phase_folded_P_path: str
            - phase_folded_P/2_path: str
            - phase_folded_2*P_path: str
    """
    try:
        settings = get_settings()
        out_dir = settings.artifacts_dir

        print(f"Computing BLS for {csv_path}")
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
        analysis_lc = normalize_multiband_light_curve(lc)

        # Compute BLS periodogram
        periodogram = compute_bls_periodogram(analysis_lc, duration=settings.bls_default_duration)

        best_period = periodogram.best_value
        best_power = periodogram.best_power

        print(f"Best BLS period: {best_period:.5f} days with power {best_power:.5f} [{periodogram.backend}]")

        # Save periodogram plot
        periodogram_path = out_dir / "bls_periodogram.png"
        pconfig = PlotConfig(
            title="BLS Periodogram",
            x_label="Period [d]",
            y_label="BLS Power",
            x_scale="log",
            y_scale="linear",
            plot_style="-",
        )
        plot_periodogram(periodogram, pconfig, periodogram_path)

        # Generate phase-folded plots
        results = {
            "P": (best_period, "BLS_phase_folded_P.png"),
            "P/2": (best_period / 2, "BLS_phase_folded_P2.png"),
            "2P": (best_period * 2, "BLS_phase_folded_2P.png"),
        }

        output_paths = {}
        for label, (period_val, filename) in results.items():
            # Phase fold: normalize to [0, 1]
            t_folded = pd.Series((np.asarray(analysis_lc.time) / period_val) % 1.0)

            folded_lc = LightCurve(
                time=t_folded,
                magnitude=analysis_lc.magnitude.reset_index(drop=True),
                error=analysis_lc.error.reset_index(drop=True) if analysis_lc.error is not None else None,
                band=analysis_lc.band.reset_index(drop=True) if analysis_lc.band is not None else None,
                time_col=f"Phase (P={period_val:.5f} d)",
                mag_col=analysis_lc.mag_col,
                err_col=analysis_lc.err_col,
                band_col=analysis_lc.band_col,
                scale=analysis_lc.scale,
                source_name=analysis_lc.source_name,
                ra_deg=analysis_lc.ra_deg,
                dec_deg=analysis_lc.dec_deg,
                metadata=analysis_lc.metadata.copy(),
            )

            path = out_dir / filename
            config = PlotConfig(
                title=f"Phase Folded at {label}",
                x_label=f"Phase (P={period_val:.5f} d)",
                y_label=analysis_lc.mag_col,
                require_binning=True,
                invert_y=analysis_lc.scale == "mag",
            )
            plot_light_curve(folded_lc, config, path)
            output_paths[label] = str(path)

        return {
            "best_period": best_period,
            "best_power": best_power,
            "backend": periodogram.backend,
            "periodogram_path": str(periodogram_path),
            "phase_folded_P_path": output_paths["P"],
            "phase_folded_P/2_path": output_paths["P/2"],
            "phase_folded_2*P_path": output_paths["2P"],
            "band_labels": lc.band_labels,
            "columns_used": {
                "time_col": lc.time_col,
                "y_col": lc.mag_col,
                "yerr_col": lc.err_col or None,
                "band_col": lc.band_col or None,
            },
        }

    except Exception as e:
        return {"error": str(e)}


@tool
def analysis_LS(
    phase_folded_result: dict,
    candidate_type: list[str] | None = None,
    prompt: str | None = None,
) -> str:
    """Analyze phase-folded periodogram results using VLM.

    Args:
        phase_folded_result: Dict from obtain_GLS or obtain_BLS with paths
        candidate_type: List of candidate variability types to focus on
        prompt: Custom analysis prompt. Uses default if None.

    Returns:
        VLM analysis text
    """
    try:
        print("Analyzing phase-folded results with VLM...")

        if prompt is None:
            prompt = """
Analyze this 2x2 grid of light curve analysis plots.
Top-Left: Periodogram. Top-Right: Phase-folded at P.
Bottom-Left: Phase-folded at P/2. Bottom-Right: Phase-folded at 2P.

Is the phase well-folded for P, 2P, or P/2 (significant periodic feature)?
Trust the GLS/BLS result if the binned points show a clear, coherent trend in any one of the P, 2P, or P/2 panels.
If so, provide diagnosis with:
1. Probability (Very High, High, Medium, Low, Very Low) for each class
2. Reasoning with specific evidence from plots
3. Requirements for confirmation (observations needed)
4. Scientific interest level

Report which period (P, 2P, or P/2) should be the real period.
If no significant periodic feature, propose candidates or claim non-variable.

Amplitude Note: Estimate peak-to-peak amplitude from best-folded plot.
Period Note: Consider that true period can be 2P even if P has significant power.
"""

        if candidate_type:
            prompt += f"\n\nSpecial attention to these candidates: {', '.join(candidate_type)}."

        # Create combined grid
        periodogram_path = phase_folded_result.get("periodogram_path")
        paths = {
            "periodogram_path": Path(periodogram_path),
            "phase_folded_P_path": Path(phase_folded_result["phase_folded_P_path"]),
            "phase_folded_P2_path": Path(phase_folded_result["phase_folded_P/2_path"]),
            "phase_folded_2P_path": Path(phase_folded_result["phase_folded_2*P_path"]),
        }

        combined_path = Path(periodogram_path).parent / f"{Path(periodogram_path).stem}_combined_analysis.png"
        create_grid(paths, combined_path)

        # Analyze
        provider = get_vlm_provider("openai")
        result = provider.analyze_image(combined_path, prompt)

        print("Analysis complete")
        return result

    except Exception as e:
        return f"error: {str(e)}"
