"""Core plotting functions for light curves.

All matplotlib imports are isolated here to allow backend configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

# Configure matplotlib backend before importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lightcurve_agent.config import get_settings
from lightcurve_agent.core.models import LightCurve, Periodogram


@dataclass
class PlotConfig:
    """Configuration for a plot."""

    title: str = "Light Curve"
    x_label: str = "Time [d]"
    y_label: str = "Magnitude"
    x_scale: Literal["linear", "log"] | None = None
    y_scale: Literal["linear", "log"] | None = None
    plot_style: str = "."
    dpi: int | None = None
    invert_y: bool = False
    require_binning: bool = False
    bin_number: int = 50
    show_grid: bool = True


def plot_light_curve(
    lc: LightCurve,
    config: PlotConfig | None = None,
    out_path: str | Path | None = None,
) -> Path:
    """Plot a light curve.

    Args:
        lc: LightCurve instance
        config: Plot configuration. If None, uses defaults.
        out_path: Output path. If None, uses artifacts directory.

    Returns:
        Path to saved plot
    """
    settings = get_settings()
    cfg = config or PlotConfig()
    dpi = cfg.dpi or settings.default_dpi

    if out_path is None:
        out_path = settings.artifacts_dir / "lightcurve.png"
    out_path = Path(out_path)

    fig, ax = plt.subplots(figsize=(10, 6))

    t = np.asarray(lc.time)
    y = np.asarray(lc.magnitude)
    yerr = np.asarray(lc.error) if lc.error is not None else None
    band = np.asarray(lc.band.astype(str)) if lc.band is not None else None

    # Plot data
    if band is not None and lc.band_labels:
        colors = plt.cm.tab10(np.linspace(0, 1, max(len(lc.band_labels), 1)))
        for color, label in zip(colors, lc.band_labels, strict=False):
            mask = band == label
            if yerr is not None:
                ax.errorbar(
                    t[mask],
                    y[mask],
                    yerr=yerr[mask],
                    fmt=cfg.plot_style,
                    capsize=0,
                    elinewidth=0.5,
                    color=color,
                    label=label,
                )
            else:
                ax.plot(t[mask], y[mask], cfg.plot_style, markersize=3, color=color, label=label)
        ax.legend(title=lc.band_col or "band")
    elif yerr is not None:
        ax.errorbar(t, y, yerr=yerr, fmt=cfg.plot_style, capsize=0, elinewidth=0.5, label="data")
    else:
        ax.plot(t, y, cfg.plot_style, markersize=3, label="data")

    # Binning
    if cfg.require_binning:
        bins = np.linspace(t.min(), t.max(), cfg.bin_number + 1)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        digitized = np.digitize(t, bins)

        binned_y = []
        binned_err = []

        for i in range(1, len(bins)):
            mask = digitized == i
            if np.any(mask):
                y_slice = y[mask]
                if yerr is not None:
                    weights = 1.0 / (yerr[mask] ** 2)
                    w_mean = np.sum(weights * y_slice) / np.sum(weights)
                    w_err = np.sqrt(1.0 / np.sum(weights))
                    binned_y.append(w_mean)
                    binned_err.append(w_err)
                else:
                    binned_y.append(y_slice.mean())
                    binned_err.append(y_slice.std() if len(y_slice) > 1 else 0)
            else:
                binned_y.append(np.nan)
                binned_err.append(np.nan)

        binned_y = np.array(binned_y)
        binned_err = np.array(binned_err)

        ax.errorbar(
            bin_centers,
            binned_y,
            yerr=binned_err,
            fmt="r.",
            linewidth=1.5,
            label="binned",
            capsize=2,
        )
        ax.legend()

    # Y-axis limits (prevent excessive whitespace)
    if cfg.y_scale != "log" and yerr is not None:
        y_range = abs(y.max() - y.min())
        ax.set_ylim(y.min() - 0.1 * y_range, y.max() + 0.1 * y_range)

    # Labels and scales
    ax.set_xlabel(cfg.x_label)
    ax.set_ylabel(cfg.y_label)
    ax.set_title(cfg.title)

    if cfg.x_scale:
        ax.set_xscale(cfg.x_scale)
    if cfg.y_scale:
        ax.set_yscale(cfg.y_scale)

    if cfg.invert_y:
        ax.invert_yaxis()

    if cfg.show_grid:
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

    return out_path


def plot_periodogram(
    periodogram: Periodogram,
    config: PlotConfig | None = None,
    out_path: str | Path | None = None,
) -> Path:
    """Plot a periodogram.

    Args:
        periodogram: Periodogram instance
        config: Plot configuration
        out_path: Output path

    Returns:
        Path to saved plot
    """
    settings = get_settings()
    cfg = config or PlotConfig(
        title="Periodogram",
        x_label=f"{periodogram.grid_kind.capitalize()} [d]" if periodogram.grid_kind == "period" else "Frequency [1/d]",
        y_label="Power",
    )
    dpi = cfg.dpi or settings.default_dpi

    if out_path is None:
        out_path = settings.artifacts_dir / "periodogram.png"
    out_path = Path(out_path)

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.asarray(periodogram.grid)
    y = np.asarray(periodogram.power)

    ax.plot(x, y, cfg.plot_style or "-", linewidth=1)

    # Mark best period
    if periodogram.best_value > 0:
        ax.axvline(periodogram.best_value, color="r", linestyle="--", alpha=0.5, label=f"Best: {periodogram.best_value:.4f}")
        ax.legend()

    ax.set_xlabel(cfg.x_label)
    ax.set_ylabel(cfg.y_label)
    ax.set_title(cfg.title)

    if cfg.x_scale:
        ax.set_xscale(cfg.x_scale)
    if cfg.y_scale:
        ax.set_yscale(cfg.y_scale)

    if cfg.show_grid:
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

    return out_path


def plot_phase_folded(
    lc: LightCurve,
    period: float,
    config: PlotConfig | None = None,
    out_path: str | Path | None = None,
) -> Path:
    """Plot phase-folded light curve.

    Args:
        lc: LightCurve instance
        period: Period to fold at
        config: Plot configuration
        out_path: Output path

    Returns:
        Path to saved plot
    """
    from lightcurve_agent.core.analysis.gls import phase_fold

    settings = get_settings()
    cfg = config or PlotConfig(
        title=f"Phase Folded (P={period:.5f} d)",
        x_label=f"Phase (P={period:.5f} d)",
        y_label=lc.mag_col,
        require_binning=True,
    )
    cfg.invert_y = lc.scale == "mag"

    t_folded = phase_fold(lc, period)

    # Create temporary light curve with folded time
    folded_lc = LightCurve(
        time=pd.Series(t_folded),
        magnitude=lc.magnitude,
        error=lc.error,
        band=lc.band,
        time_col=cfg.x_label,
        mag_col=lc.mag_col,
        err_col=lc.err_col,
        band_col=lc.band_col,
        scale=lc.scale,
        source_name=lc.source_name,
        ra_deg=lc.ra_deg,
        dec_deg=lc.dec_deg,
        metadata=lc.metadata.copy(),
    )

    return plot_light_curve(folded_lc, cfg, out_path)
