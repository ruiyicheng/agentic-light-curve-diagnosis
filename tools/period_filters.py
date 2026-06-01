"""Shared period-grid filters for light-curve period searches."""

from __future__ import annotations

import numpy as np


DEFAULT_DAILY_ALIAS_TOLERANCE_DAY = 0.01
DAILY_ALIAS_PERIOD_WINDOWS_DAY = (
    (1.0, DEFAULT_DAILY_ALIAS_TOLERANCE_DAY),
    (0.5, 0.005),
    (1.0 / 3.0, 1.0 / 300.0),
    (2.0, DEFAULT_DAILY_ALIAS_TOLERANCE_DAY),
    (3.0, DEFAULT_DAILY_ALIAS_TOLERANCE_DAY),
)
DAILY_ALIAS_PERIODS_DAY = tuple(
    period_day for period_day, _ in DAILY_ALIAS_PERIOD_WINDOWS_DAY
)


def daily_alias_period_mask(
    periods_day: np.ndarray,
    *,
    tolerance_day: float | None = None,
) -> np.ndarray:
    """Return True for periods that should be ignored as daily aliases."""
    if tolerance_day is not None and tolerance_day < 0:
        raise ValueError("tolerance_day must be non-negative")

    periods = np.asarray(periods_day, dtype=np.float64)
    mask = np.zeros(periods.shape, dtype=bool)
    finite_mask = np.isfinite(periods)
    period_windows = (
        [(period_day, tolerance_day) for period_day in DAILY_ALIAS_PERIODS_DAY]
        if tolerance_day is not None
        else DAILY_ALIAS_PERIOD_WINDOWS_DAY
    )
    for alias_period_day, alias_tolerance_day in period_windows:
        epsilon = (
            np.finfo(np.float64).eps
            * max(abs(alias_period_day), abs(alias_tolerance_day), 1.0)
            * 16.0
        )
        mask |= finite_mask & (
            np.abs(periods - alias_period_day) <= alias_tolerance_day + epsilon
        )
    return mask


def mask_daily_alias_powers(
    periods_day: np.ndarray,
    powers: np.ndarray,
    *,
    tolerance_day: float | None = None,
) -> np.ndarray:
    """Return a copy of powers with daily-alias periods set to NaN."""
    masked_powers = np.asarray(powers, dtype=np.float64).copy()
    masked_powers[daily_alias_period_mask(periods_day, tolerance_day=tolerance_day)] = np.nan
    return masked_powers
