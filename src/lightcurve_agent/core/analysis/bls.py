"""Box Least Squares periodogram computation.

Wraps astropy's BLS implementation with our data models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.timeseries import BoxLeastSquares

from lightcurve_agent.core.models import LightCurve, Periodogram


def compute_bls_periodogram(
    lc: LightCurve,
    duration: float = 0.2,
    period_grid: np.ndarray | None = None,
) -> Periodogram:
    """Compute Box Least Squares periodogram.

    Args:
        lc: LightCurve instance (will be converted to flux)
        duration: Typical transit duration in days
        period_grid: Period grid to search. If None, uses default.

    Returns:
        Periodogram with best period and power
    """
    # Convert magnitude to flux for BLS
    # BLS expects dips (transits), so we invert magnitude
    y_median = np.median(lc.magnitude)
    y_flux = 10 ** (-0.4 * (lc.magnitude - y_median))

    if lc.error is not None:
        # Error propagation for flux
        dy_flux = y_flux * 0.4 * np.log(10) * lc.error
    else:
        dy_flux = None

    t = np.asarray(lc.time)
    y = np.asarray(y_flux)

    model = BoxLeastSquares(t * u.day, y, dy=dy_flux)

    # Default period grid if not provided
    if period_grid is None:
        from lightcurve_agent.config import get_settings

        settings = get_settings()
        period_grid = np.exp(
            np.linspace(
                np.log(max(settings.default_period_min_days, 0.3)),
                np.log(settings.default_period_max_days),
                100_000,
            )
        )

    results = model.autopower(duration * u.day)

    best_idx = np.argmax(results.power)
    best_period = float(results.period[best_idx].value)
    best_power = float(results.power[best_idx])

    return Periodogram(
        grid=pd.Series(results.period.value, name="period"),
        power=pd.Series(results.power, name="bls_power"),
        grid_kind="period",
        best_value=best_period,
        best_power=best_power,
    )
