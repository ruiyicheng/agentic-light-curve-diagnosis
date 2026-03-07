"""Generalized Lomb-Scargle periodogram computation.

Pure numerical implementation with no external service dependencies.
"""

from __future__ import annotations

import math
from multiprocessing import shared_memory
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from lightcurve_agent.config import get_settings
from lightcurve_agent.core.models import LightCurve, Periodogram


def _gls_chunk_worker(payload: tuple) -> tuple[int, np.ndarray]:
    """Compute GLS power for one chunk of periods/frequencies.

    This is a standalone function for multiprocessing compatibility.
    """
    (
        k0,
        grid_chunk,
        grid_kind,
        shm_t_name,
        t_shape,
        t_dtype_str,
        shm_w_name,
        w_shape,
        w_dtype_str,
        shm_yw_name,
        yw_shape,
        yw_dtype_str,
        W,
        Y,
        var_y,
        eps,
    ) = payload

    shm_t = shared_memory.SharedMemory(name=shm_t_name)
    shm_w = shared_memory.SharedMemory(name=shm_w_name)
    shm_yw = shared_memory.SharedMemory(name=shm_yw_name)

    try:
        t = np.ndarray(t_shape, dtype=np.dtype(t_dtype_str), buffer=shm_t.buf)
        w = np.ndarray(w_shape, dtype=np.dtype(w_dtype_str), buffer=shm_w.buf)
        yw = np.ndarray(yw_shape, dtype=np.dtype(yw_dtype_str), buffer=shm_yw.buf)

        # Compute omega
        two_pi = 2.0 * math.pi
        if grid_kind == "period":
            omega = two_pi / np.maximum(grid_chunk, eps)
        else:
            omega = two_pi * np.maximum(grid_chunk, 0.0)

        # Phase matrix [N, kc]
        phase = t[:, None] * omega[None, :]
        c = np.cos(phase)
        s = np.sin(phase)

        # Weighted sums over time axis
        C = w @ c
        S = w @ s
        CC = w @ (c * c)
        SS = w @ (s * s)
        CS = w @ (c * s)
        YC = yw @ c
        YS = yw @ s

        # Centered cross-terms
        YC0 = YC - (Y * C) / max(W, eps)
        YS0 = YS - (Y * S) / max(W, eps)

        # Denominator
        D = np.maximum(CC * SS - CS * CS, eps)

        # GLS power
        num = SS * (YC0 * YC0) + CC * (YS0 * YS0) - 2.0 * CS * YC0 * YS0
        power = num / (D * max(var_y, eps))

        # Sanitize
        power = np.nan_to_num(power, nan=0.0, posinf=0.0, neginf=0.0)
        power = np.maximum(power, 0.0)

        return k0, power

    finally:
        shm_t.close()
        shm_w.close()
        shm_yw.close()


def compute_gls_periodogram(
    lc: LightCurve,
    grid: np.ndarray | None = None,
    grid_kind: Literal["period", "frequency"] = "period",
    eps: float = 1e-12,
    chunk_size: int | None = None,
    processes: int | None = None,
    dtype: np.dtype = np.float64,
    mp_context: Literal["spawn", "fork", "forkserver"] = "spawn",
) -> Periodogram:
    """Compute Generalized Lomb-Scargle periodogram.

    Args:
        lc: LightCurve instance
        grid: Period/frequency grid. If None, uses default log-spaced grid.
        grid_kind: "period" or "frequency"
        eps: Small value for numerical stability
        chunk_size: Grid points per worker task. Default from settings.
        processes: Number of worker processes. Default from settings.
        dtype: NumPy dtype for computation
        mp_context: Multiprocessing context

    Returns:
        Periodogram with best period and power identified
    """
    import multiprocessing as mp

    settings = get_settings()
    if chunk_size is None:
        chunk_size = settings.default_chunk_size

    # Prepare data
    t = np.asarray(lc.time, dtype=dtype).reshape(-1)
    y = np.asarray(lc.magnitude, dtype=dtype).reshape(-1)

    if lc.error is not None:
        yerr = np.asarray(lc.error, dtype=dtype).reshape(-1)
    else:
        yerr = np.ones_like(y)

    # Validate shapes
    if t.shape != y.shape or t.shape != yerr.shape:
        raise ValueError(f"Shape mismatch: t={t.shape}, y={y.shape}, yerr={yerr.shape}")

    N = t.shape[0]

    # Filter valid data
    mask = np.isfinite(t) & np.isfinite(y) & np.isfinite(yerr) & (yerr > 0)
    if not mask.all():
        t = t[mask]
        y = y[mask]
        yerr = yerr[mask]
        N = len(t)

    # Weights
    w = np.zeros(N, dtype=dtype)
    w[mask] = 1.0 / (yerr[mask] ** 2)

    # Precompute scalars
    W = float(np.sum(w))
    if not np.isfinite(W) or W <= 0:
        raise ValueError("All weights are zero/invalid")

    yw = w * y
    Y = float(np.sum(yw))
    YY = float(np.sum(w * y * y))
    var_y = float(max(YY - (Y * Y) / max(W, eps), eps))

    # Default grid
    if grid is None:
        grid = np.exp(
            np.linspace(
                np.log(settings.default_period_min_days),
                np.log(settings.default_period_max_days),
                settings.default_period_grid_size,
            )
        )

    grid_arr = np.asarray(grid, dtype=dtype).reshape(-1)
    K = grid_arr.shape[0]

    if K == 0:
        raise ValueError("Grid must be non-empty")

    power = np.empty(K, dtype=dtype)

    # Setup shared memory
    def _to_shm(a: np.ndarray) -> tuple:
        shm = shared_memory.SharedMemory(create=True, size=a.nbytes)
        view = np.ndarray(a.shape, dtype=a.dtype, buffer=shm.buf)
        view[...] = a
        return shm, a.shape, a.dtype.str

    shm_t, t_shape, t_dtype_str = _to_shm(t)
    shm_w, w_shape, w_dtype_str = _to_shm(w)
    shm_yw, yw_shape, yw_dtype_str = _to_shm(yw)

    ctx = mp.get_context(mp_context)
    if processes is None:
        processes = ctx.cpu_count()

    try:
        tasks = []
        for k0 in range(0, K, chunk_size):
            k1 = min(k0 + chunk_size, K)
            tasks.append(
                (
                    k0,
                    grid_arr[k0:k1],
                    grid_kind,
                    shm_t.name,
                    t_shape,
                    t_dtype_str,
                    shm_w.name,
                    w_shape,
                    w_dtype_str,
                    shm_yw.name,
                    yw_shape,
                    yw_dtype_str,
                    W,
                    Y,
                    var_y,
                    eps,
                )
            )

        with ctx.Pool(processes=processes) as pool:
            for k0, pchunk in pool.imap_unordered(_gls_chunk_worker, tasks, chunksize=1):
                power[k0 : k0 + pchunk.shape[0]] = pchunk

    finally:
        shm_t.close()
        shm_t.unlink()
        shm_w.close()
        shm_w.unlink()
        shm_yw.close()
        shm_yw.unlink()

    # Find best period
    best_pos = int(np.argmax(power))
    best_grid_value = float(grid_arr[best_pos])
    best_power = float(power[best_pos])

    idx = pd.Index(np.asarray(grid_arr), name=grid_kind)
    power_s = pd.Series(power, index=idx, name="gls_power")

    return Periodogram(
        grid=power_s.index.to_series(),
        power=power_s,
        grid_kind=grid_kind,
        best_value=best_grid_value,
        best_power=best_power,
    )


def phase_fold(
    lc: LightCurve,
    period: float,
    n_phases: int = 1,
) -> pd.Series:
    """Phase fold a light curve.

    Args:
        lc: LightCurve instance
        period: Period to fold at
        n_phases: Number of periods to fold into (1 = 0 to period)

    Returns:
        Phase-folded time series
    """
    t = np.asarray(lc.time)
    effective_period = period / n_phases
    return pd.Series(t - effective_period * np.floor(t / effective_period))
