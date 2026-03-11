"""Generalized Lomb-Scargle periodogram computation."""

from __future__ import annotations

import math
from multiprocessing import shared_memory
from typing import Literal

import numpy as np
import pandas as pd

from lightcurve_agent.config import get_settings
from lightcurve_agent.core.models import LightCurve, Periodogram


def _gls_chunk_worker(payload: tuple) -> tuple[int, np.ndarray]:
    """Compute GLS power for one chunk of periods or frequencies."""
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

        two_pi = 2.0 * math.pi
        if grid_kind == "period":
            omega = two_pi / np.maximum(grid_chunk, eps)
        else:
            omega = two_pi * np.maximum(grid_chunk, 0.0)

        phase = t[:, None] * omega[None, :]
        c = np.cos(phase)
        s = np.sin(phase)

        C = w @ c
        S = w @ s
        CC = w @ (c * c)
        SS = w @ (s * s)
        CS = w @ (c * s)
        YC = yw @ c
        YS = yw @ s

        YC0 = YC - (Y * C) / max(W, eps)
        YS0 = YS - (Y * S) / max(W, eps)

        D = np.maximum(CC * SS - CS * CS, eps)
        num = SS * (YC0 * YC0) + CC * (YS0 * YS0) - 2.0 * CS * YC0 * YS0
        power = num / (D * max(var_y, eps))
        power = np.nan_to_num(power, nan=0.0, posinf=0.0, neginf=0.0)
        return k0, np.maximum(power, 0.0)
    finally:
        shm_t.close()
        shm_w.close()
        shm_yw.close()


def _default_grid() -> np.ndarray:
    settings = get_settings()
    return np.exp(
        np.linspace(
            np.log(settings.default_period_min_days),
            np.log(settings.default_period_max_days),
            settings.default_period_grid_size,
        )
    )


def _maybe_torch_cuda_periodogram(
    *,
    t: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    grid_arr: np.ndarray,
    grid_kind: Literal["period", "frequency"],
    eps: float,
    chunk_size: int,
) -> Periodogram | None:
    try:
        import torch
    except ModuleNotFoundError:
        return None

    if not torch.cuda.is_available():
        return None

    device = torch.device("cuda")
    tensor_dtype = torch.float32

    t_tensor = torch.as_tensor(t, dtype=tensor_dtype, device=device)
    y_tensor = torch.as_tensor(y, dtype=tensor_dtype, device=device)
    yerr_tensor = torch.as_tensor(yerr, dtype=tensor_dtype, device=device)
    grid_tensor = torch.as_tensor(grid_arr, dtype=tensor_dtype, device=device)

    w = 1.0 / torch.clamp(yerr_tensor * yerr_tensor, min=eps)
    yw = w * y_tensor
    W = torch.sum(w)
    Y = torch.sum(yw)
    YY = torch.sum(w * y_tensor * y_tensor)
    var_y = torch.clamp(YY - (Y * Y) / torch.clamp(W, min=eps), min=eps)

    power = np.empty(grid_arr.shape[0], dtype=np.float32)
    two_pi = 2.0 * math.pi

    for k0 in range(0, grid_arr.shape[0], chunk_size):
        chunk = grid_tensor[k0 : k0 + chunk_size]
        omega = two_pi / torch.clamp(chunk, min=eps) if grid_kind == "period" else two_pi * chunk

        phase = torch.outer(t_tensor, omega)
        c = torch.cos(phase)
        s = torch.sin(phase)

        C = torch.matmul(w, c)
        S = torch.matmul(w, s)
        CC = torch.matmul(w, c * c)
        SS = torch.matmul(w, s * s)
        CS = torch.matmul(w, c * s)
        YC = torch.matmul(yw, c)
        YS = torch.matmul(yw, s)

        YC0 = YC - (Y * C) / torch.clamp(W, min=eps)
        YS0 = YS - (Y * S) / torch.clamp(W, min=eps)
        D = torch.clamp(CC * SS - CS * CS, min=eps)

        num = SS * YC0.square() + CC * YS0.square() - 2.0 * CS * YC0 * YS0
        pchunk = torch.nan_to_num(num / (D * var_y), nan=0.0, posinf=0.0, neginf=0.0)
        power[k0 : k0 + len(chunk)] = torch.clamp(pchunk, min=0.0).detach().cpu().numpy()

    best_pos = int(np.argmax(power))
    grid_series = pd.Series(grid_arr, name=grid_kind)
    power_series = pd.Series(power, name="gls_power")
    return Periodogram(
        grid=grid_series,
        power=power_series,
        grid_kind=grid_kind,
        best_value=float(grid_arr[best_pos]),
        best_power=float(power[best_pos]),
        backend=f"torch-{device.type}",
    )


def _numpy_shared_periodogram(
    *,
    t: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    grid_arr: np.ndarray,
    grid_kind: Literal["period", "frequency"],
    eps: float,
    chunk_size: int,
    processes: int | None,
    mp_context: Literal["spawn", "fork", "forkserver"],
) -> Periodogram:
    import multiprocessing as mp

    w = 1.0 / np.maximum(yerr * yerr, eps)
    yw = w * y
    W = float(np.sum(w))
    Y = float(np.sum(yw))
    YY = float(np.sum(w * y * y))
    var_y = float(max(YY - (Y * Y) / max(W, eps), eps))

    def _to_shm(array: np.ndarray) -> tuple:
        shm = shared_memory.SharedMemory(create=True, size=array.nbytes)
        view = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
        view[...] = array
        return shm, array.shape, array.dtype.str

    shm_t, t_shape, t_dtype_str = _to_shm(t)
    shm_w, w_shape, w_dtype_str = _to_shm(w)
    shm_yw, yw_shape, yw_dtype_str = _to_shm(yw)

    ctx = mp.get_context(mp_context)
    processes = processes or ctx.cpu_count()
    power = np.empty(grid_arr.shape[0], dtype=np.float64)

    try:
        tasks = []
        for k0 in range(0, grid_arr.shape[0], chunk_size):
            k1 = min(k0 + chunk_size, grid_arr.shape[0])
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
                power[k0 : k0 + len(pchunk)] = pchunk
    finally:
        shm_t.close()
        shm_t.unlink()
        shm_w.close()
        shm_w.unlink()
        shm_yw.close()
        shm_yw.unlink()

    best_pos = int(np.argmax(power))
    grid_series = pd.Series(grid_arr, name=grid_kind)
    power_series = pd.Series(power, name="gls_power")
    return Periodogram(
        grid=grid_series,
        power=power_series,
        grid_kind=grid_kind,
        best_value=float(grid_arr[best_pos]),
        best_power=float(power[best_pos]),
        backend="numpy-multiprocessing",
    )


def _numpy_chunked_periodogram(
    *,
    t: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    grid_arr: np.ndarray,
    grid_kind: Literal["period", "frequency"],
    eps: float,
    chunk_size: int,
) -> Periodogram:
    """Compute GLS without multiprocessing for restricted environments."""
    w = 1.0 / np.maximum(yerr * yerr, eps)
    yw = w * y
    W = float(np.sum(w))
    Y = float(np.sum(yw))
    YY = float(np.sum(w * y * y))
    var_y = float(max(YY - (Y * Y) / max(W, eps), eps))

    power = np.empty(grid_arr.shape[0], dtype=np.float64)
    two_pi = 2.0 * math.pi

    for k0 in range(0, grid_arr.shape[0], chunk_size):
        chunk = grid_arr[k0 : k0 + chunk_size]
        omega = two_pi / np.maximum(chunk, eps) if grid_kind == "period" else two_pi * np.maximum(chunk, 0.0)
        phase = t[:, None] * omega[None, :]
        c = np.cos(phase)
        s = np.sin(phase)

        C = w @ c
        S = w @ s
        CC = w @ (c * c)
        SS = w @ (s * s)
        CS = w @ (c * s)
        YC = yw @ c
        YS = yw @ s

        YC0 = YC - (Y * C) / max(W, eps)
        YS0 = YS - (Y * S) / max(W, eps)
        D = np.maximum(CC * SS - CS * CS, eps)
        num = SS * (YC0 * YC0) + CC * (YS0 * YS0) - 2.0 * CS * YC0 * YS0
        pchunk = num / (D * max(var_y, eps))
        power[k0 : k0 + len(chunk)] = np.maximum(np.nan_to_num(pchunk, nan=0.0, posinf=0.0, neginf=0.0), 0.0)

    best_pos = int(np.argmax(power))
    return Periodogram(
        grid=pd.Series(grid_arr, name=grid_kind),
        power=pd.Series(power, name="gls_power"),
        grid_kind=grid_kind,
        best_value=float(grid_arr[best_pos]),
        best_power=float(power[best_pos]),
        backend="numpy-chunked",
    )


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
    """Compute a GLS periodogram with CUDA acceleration when available."""
    settings = get_settings()
    chunk_size = chunk_size or settings.default_chunk_size

    t = np.asarray(lc.time, dtype=dtype).reshape(-1)
    y = np.asarray(lc.magnitude, dtype=dtype).reshape(-1)
    if lc.error is not None:
        yerr = np.asarray(lc.error, dtype=dtype).reshape(-1)
    else:
        yerr = np.ones_like(y, dtype=dtype)

    if t.shape != y.shape or t.shape != yerr.shape:
        raise ValueError(f"Shape mismatch: t={t.shape}, y={y.shape}, yerr={yerr.shape}")

    mask = np.isfinite(t) & np.isfinite(y) & np.isfinite(yerr) & (yerr > 0)
    if not np.any(mask):
        raise ValueError("No finite samples available for GLS")

    t = t[mask]
    y = y[mask]
    yerr = yerr[mask]

    grid_arr = np.asarray(grid if grid is not None else _default_grid(), dtype=dtype).reshape(-1)
    if grid_arr.size == 0:
        raise ValueError("Grid must be non-empty")

    torch_result = _maybe_torch_cuda_periodogram(
        t=t,
        y=y,
        yerr=yerr,
        grid_arr=grid_arr,
        grid_kind=grid_kind,
        eps=eps,
        chunk_size=chunk_size,
    )
    if torch_result is not None:
        return torch_result

    try:
        return _numpy_shared_periodogram(
            t=t,
            y=y,
            yerr=yerr,
            grid_arr=grid_arr,
            grid_kind=grid_kind,
            eps=eps,
            chunk_size=chunk_size,
            processes=processes,
            mp_context=mp_context,
        )
    except PermissionError:
        return _numpy_chunked_periodogram(
            t=t,
            y=y,
            yerr=yerr,
            grid_arr=grid_arr,
            grid_kind=grid_kind,
            eps=eps,
            chunk_size=chunk_size,
        )


def phase_fold(
    lc: LightCurve,
    period: float,
    n_phases: int = 1,
) -> pd.Series:
    """Phase fold a light curve."""
    t = np.asarray(lc.time)
    effective_period = period / n_phases
    return pd.Series(t - effective_period * np.floor(t / effective_period))
