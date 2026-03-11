"""Box Least Squares periodogram computation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import astropy.units as u
from astropy.timeseries import BoxLeastSquares

from lightcurve_agent.config import get_settings
from lightcurve_agent.core.models import LightCurve, Periodogram


def _default_period_grid() -> np.ndarray:
    settings = get_settings()
    return np.exp(
        np.linspace(
            np.log(max(settings.default_period_min_days, 0.3)),
            np.log(settings.default_period_max_days),
            100_000,
        )
    )


def _light_curve_to_flux(lc: LightCurve) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.asarray(lc.time, dtype=np.float64)

    if lc.scale == "flux":
        y_flux = np.asarray(lc.magnitude, dtype=np.float64)
        if lc.error is not None:
            dy_flux = np.asarray(lc.error, dtype=np.float64)
        else:
            dy_flux = np.ones_like(y_flux, dtype=np.float64)
    else:
        y_mag = np.asarray(lc.magnitude, dtype=np.float64)
        y_median = np.nanmedian(y_mag)
        y_flux = 10 ** (-0.4 * (y_mag - y_median))
        if lc.error is not None:
            dy_flux = y_flux * 0.4 * np.log(10) * np.asarray(lc.error, dtype=np.float64)
        else:
            dy_flux = np.ones_like(y_flux, dtype=np.float64)

    mask = np.isfinite(t) & np.isfinite(y_flux) & np.isfinite(dy_flux) & (dy_flux > 0)
    if not np.any(mask):
        raise ValueError("No finite samples available for BLS")

    return t[mask], y_flux[mask], dy_flux[mask]


def _compute_bls_periodogram_torch(
    *,
    t: np.ndarray,
    y_flux: np.ndarray,
    dy_flux: np.ndarray,
    period_grid: np.ndarray,
    duration: float,
    phase_bins: int = 256,
    chunk_size: int = 1024,
) -> Periodogram | None:
    try:
        import torch
    except ModuleNotFoundError:
        return None

    if not torch.cuda.is_available():
        return None

    device = torch.device("cuda")
    tensor_dtype = torch.float32
    eps = 1e-6

    t_tensor = torch.as_tensor(t, dtype=tensor_dtype, device=device)
    y_tensor = torch.as_tensor(y_flux, dtype=tensor_dtype, device=device)
    dy_tensor = torch.as_tensor(dy_flux, dtype=tensor_dtype, device=device)
    weights = 1.0 / torch.clamp(dy_tensor.square(), min=eps)
    centered = y_tensor - torch.sum(weights * y_tensor) / torch.sum(weights)

    powers = np.empty(period_grid.shape[0], dtype=np.float32)
    weighted_values = weights * centered
    n_points = t_tensor.shape[0]

    for k0 in range(0, period_grid.shape[0], chunk_size):
        periods = torch.as_tensor(period_grid[k0 : k0 + chunk_size], dtype=tensor_dtype, device=device)
        if periods.numel() == 0:
            continue

        phase = torch.remainder(t_tensor[None, :] / periods[:, None], 1.0)
        bin_index = torch.clamp((phase * phase_bins).to(torch.long), 0, phase_bins - 1)

        row_offsets = (
            torch.arange(periods.shape[0], device=device, dtype=torch.long)[:, None] * phase_bins
        )
        flat_index = (row_offsets + bin_index).reshape(-1)

        sum_y_flat = torch.zeros(periods.shape[0] * phase_bins, dtype=tensor_dtype, device=device)
        sum_w_flat = torch.zeros_like(sum_y_flat)

        sum_y_flat.scatter_add_(
            0,
            flat_index,
            weighted_values.expand(periods.shape[0], n_points).reshape(-1),
        )
        sum_w_flat.scatter_add_(
            0,
            flat_index,
            weights.expand(periods.shape[0], n_points).reshape(-1),
        )

        sum_y = sum_y_flat.view(periods.shape[0], phase_bins)
        sum_w = sum_w_flat.view(periods.shape[0], phase_bins)

        duplicated_y = torch.cat([sum_y, sum_y], dim=1)
        duplicated_w = torch.cat([sum_w, sum_w], dim=1)
        cumulative_y = torch.nn.functional.pad(torch.cumsum(duplicated_y, dim=1), (1, 0))
        cumulative_w = torch.nn.functional.pad(torch.cumsum(duplicated_w, dim=1), (1, 0))

        window_bins = torch.clamp(
            torch.round((duration / periods) * phase_bins).to(torch.long),
            min=1,
            max=max(phase_bins // 2, 1),
        )
        start = torch.arange(phase_bins, device=device, dtype=torch.long)[None, :]
        end = start + window_bins[:, None]

        window_y = cumulative_y.gather(1, end) - cumulative_y.gather(1, start)
        window_w = cumulative_w.gather(1, end) - cumulative_w.gather(1, start)

        depth = torch.clamp(-(window_y / torch.clamp(window_w, min=eps)), min=0.0)
        snr = depth * torch.sqrt(torch.clamp(window_w, min=eps))
        best_power = torch.max(snr.square(), dim=1).values
        powers[k0 : k0 + periods.shape[0]] = best_power.detach().cpu().numpy()

    best_index = int(np.argmax(powers))
    return Periodogram(
        grid=pd.Series(period_grid, name="period"),
        power=pd.Series(powers, name="bls_power"),
        grid_kind="period",
        best_value=float(period_grid[best_index]),
        best_power=float(powers[best_index]),
        backend=f"torch-{device.type}",
    )


def _compute_bls_periodogram_astropy(
    *,
    t: np.ndarray,
    y_flux: np.ndarray,
    dy_flux: np.ndarray,
    duration: float,
    period_grid: np.ndarray | None,
) -> Periodogram:
    model = BoxLeastSquares(t * u.day, y_flux, dy=dy_flux)
    results = model.autopower(duration * u.day) if period_grid is None else model.power(period_grid * u.day, duration * u.day)

    best_idx = int(np.argmax(results.power))
    return Periodogram(
        grid=pd.Series(np.asarray(results.period.value), name="period"),
        power=pd.Series(np.asarray(results.power), name="bls_power"),
        grid_kind="period",
        best_value=float(results.period[best_idx].value),
        best_power=float(results.power[best_idx]),
        backend="astropy-cpu",
    )


def compute_bls_periodogram(
    lc: LightCurve,
    duration: float = 0.2,
    period_grid: np.ndarray | None = None,
) -> Periodogram:
    """Compute a BLS periodogram with CUDA acceleration when available."""
    if period_grid is None:
        period_grid = _default_period_grid()

    t, y_flux, dy_flux = _light_curve_to_flux(lc)

    torch_result = _compute_bls_periodogram_torch(
        t=t,
        y_flux=y_flux,
        dy_flux=dy_flux,
        period_grid=np.asarray(period_grid, dtype=np.float64),
        duration=duration,
    )
    if torch_result is not None:
        return torch_result

    return _compute_bls_periodogram_astropy(
        t=t,
        y_flux=y_flux,
        dy_flux=dy_flux,
        duration=duration,
        period_grid=np.asarray(period_grid, dtype=np.float64),
    )
