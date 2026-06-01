"""Generate box least-squares diagnostic plots for transit-like light curves."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "assets" / "light_curve"
DEFAULT_MIN_PERIOD_DAY = 60.0 / (60.0 * 24.0)
DEFAULT_MAX_PERIOD_DAY = 2000.0
DEFAULT_GRID_SIZE = 50_000
DEFAULT_MIN_DURATION_DAY = 5.0 / (60.0 * 24.0)
DEFAULT_MAX_DURATION_DAY = 1.0
DEFAULT_DURATION_GRID_SIZE = 8
DEFAULT_PHASE_GRID_SIZE = 64
DEFAULT_MAX_DURATION_FRACTION = 0.2
DEFAULT_PHASE_CHUNK_SIZE = 32
DEFAULT_MIN_TRANSIT_POINTS = 2

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.plot_whole_light_curve import (  # noqa: E402
    _axis_label,
    _coerce_schema,
    _collect_series,
    _column_indices,
    _limits,
    _read_csv_rows,
)
from tools.period_filters import (  # noqa: E402
    DAILY_ALIAS_PERIOD_WINDOWS_DAY,
    mask_daily_alias_powers,
)


__all__ = [
    "BLSPlotError",
    "box_least_squares",
    "plot_BLS",
    "plot_bls",
]


TIME_UNIT_TO_DAY = {
    "day": 1.0,
    "hour": 1.0 / 24.0,
    "minute": 1.0 / (24.0 * 60.0),
    "second": 1.0 / (24.0 * 60.0 * 60.0),
}


class BLSPlotError(ValueError):
    """Raised when a BLS diagnostic plot cannot be generated."""


def _time_factor_to_day(time_unit: str) -> float:
    try:
        return TIME_UNIT_TO_DAY[time_unit]
    except KeyError as exc:
        raise BLSPlotError(f"Unsupported time unit for BLS: {time_unit!r}") from exc


def _prepare_series(
    csv_path: Path,
    header_parse_json_results: dict[str, Any] | str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    schema = _coerce_schema(header_parse_json_results)
    columns, rows = _read_csv_rows(csv_path)
    index_by_column = _column_indices(columns, schema)
    raw_payloads = _collect_series(rows, schema, index_by_column)
    time_factor = _time_factor_to_day(schema["time_unit"])

    payloads: list[dict[str, Any]] = []
    for payload in raw_payloads:
        time = np.asarray(payload["time"], dtype=float)
        value = np.asarray(payload["value"], dtype=float)
        error = (
            np.asarray(payload["error"], dtype=float)
            if payload["error"] is not None
            else None
        )
        finite_mask = np.isfinite(time) & np.isfinite(value)
        if np.count_nonzero(finite_mask) < 5:
            continue

        prepared = dict(payload)
        prepared["time"] = time[finite_mask]
        prepared["time_day"] = time[finite_mask] * time_factor
        prepared["value"] = value[finite_mask]
        prepared["error"] = error[finite_mask] if error is not None else None
        payloads.append(prepared)

    if not payloads:
        raise BLSPlotError("No light-curve series has at least 5 finite points")
    return schema, payloads


def _weights_from_errors(error: np.ndarray | None, sample_count: int) -> np.ndarray:
    if error is None:
        return np.ones(sample_count, dtype=np.float64)

    error = np.asarray(error, dtype=np.float64)
    valid = np.isfinite(error) & (error > 0)
    if not np.any(valid):
        return np.ones(sample_count, dtype=np.float64)

    filled_error = np.where(valid, error, float(np.nanmedian(error[valid])))
    weights = 1.0 / np.square(filled_error)
    if not np.all(np.isfinite(weights)) or np.sum(weights) <= 0:
        return np.ones(sample_count, dtype=np.float64)
    return weights


def _select_period_search_series(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    return max(payloads, key=lambda payload: payload["time_day"].size)


def _daily_alias_filter_report() -> dict[str, Any]:
    return {
        "period_windows_day": [
            {
                "period_day": float(period_day),
                "tolerance_day": float(tolerance_day),
            }
            for period_day, tolerance_day in DAILY_ALIAS_PERIOD_WINDOWS_DAY
        ],
    }


def _dip_signal(value: np.ndarray, unit: str) -> np.ndarray:
    baseline = float(np.nanmedian(value))
    if unit == "mag":
        return value - baseline
    return baseline - value


def _duration_grid(
    min_duration_day: float,
    max_duration_day: float,
    duration_grid_size: int,
) -> np.ndarray:
    if min_duration_day <= 0 or max_duration_day <= 0:
        raise BLSPlotError("BLS durations must be positive")
    if min_duration_day > max_duration_day:
        raise BLSPlotError("min_duration_day must be <= max_duration_day")
    if duration_grid_size < 1:
        raise BLSPlotError("duration_grid_size must be at least 1")
    if duration_grid_size == 1 or min_duration_day == max_duration_day:
        return np.asarray([min_duration_day], dtype=np.float64)
    return np.exp(
        np.linspace(
            np.log(min_duration_day),
            np.log(max_duration_day),
            duration_grid_size,
            dtype=np.float64,
        )
    )


def _default_period_chunk_size(
    sample_count: int,
    phase_chunk_size: int,
    device_type: str,
) -> int:
    target_elements = 24_000_000 if device_type == "cuda" else 3_000_000
    denominator = max(sample_count * phase_chunk_size, 1)
    return max(4, min(2048, target_elements // denominator))


def box_least_squares(
    time_day: np.ndarray,
    value: np.ndarray,
    error: np.ndarray | None = None,
    *,
    unit: str = "flux",
    min_period_day: float = DEFAULT_MIN_PERIOD_DAY,
    max_period_day: float = DEFAULT_MAX_PERIOD_DAY,
    grid_size: int = DEFAULT_GRID_SIZE,
    min_duration_day: float = DEFAULT_MIN_DURATION_DAY,
    max_duration_day: float = DEFAULT_MAX_DURATION_DAY,
    duration_grid_size: int = DEFAULT_DURATION_GRID_SIZE,
    phase_grid_size: int = DEFAULT_PHASE_GRID_SIZE,
    max_duration_fraction: float = DEFAULT_MAX_DURATION_FRACTION,
    min_transit_points: int = DEFAULT_MIN_TRANSIT_POINTS,
    period_chunk_size: int | None = None,
    phase_chunk_size: int = DEFAULT_PHASE_CHUNK_SIZE,
    device: str | None = None,
) -> dict[str, Any]:
    """Return a GPU-capable approximate BLS power spectrum and best box model.

    The implementation evaluates a logarithmic period grid. For each period it
    tests a grid of box durations and transit-center phases, using torch tensors
    on CUDA when available. The returned power is the best positive box-depth
    statistic at each period.
    """
    if min_period_day <= 0 or max_period_day <= 0:
        raise BLSPlotError("BLS periods must be positive")
    if min_period_day >= max_period_day:
        raise BLSPlotError("min_period_day must be smaller than max_period_day")
    if grid_size < 2:
        raise BLSPlotError("grid_size must be at least 2")
    if phase_grid_size < 4:
        raise BLSPlotError("phase_grid_size must be at least 4")
    if not (0 < max_duration_fraction < 1):
        raise BLSPlotError("max_duration_fraction must be between 0 and 1")
    if min_transit_points < 1:
        raise BLSPlotError("min_transit_points must be at least 1")

    time_day = np.asarray(time_day, dtype=np.float64)
    value = np.asarray(value, dtype=np.float64)
    finite_mask = np.isfinite(time_day) & np.isfinite(value)
    if error is not None:
        error = np.asarray(error, dtype=np.float64)
        error = error[finite_mask]
    time_day = time_day[finite_mask]
    value = value[finite_mask]
    if time_day.size < 5:
        raise BLSPlotError("BLS requires at least 5 finite light-curve points")
    if float(np.nanmax(time_day) - np.nanmin(time_day)) <= 0:
        raise BLSPlotError("BLS requires more than one unique observation time")

    signal = _dip_signal(value, unit)
    if not np.any(np.isfinite(signal)) or float(np.nanstd(signal)) <= 0:
        raise BLSPlotError("BLS requires non-constant finite photometry values")

    try:
        import torch
    except ImportError as exc:
        raise BLSPlotError("PyTorch is required for BLS period search") from exc

    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device if device is not None else default_device)
    dtype = torch.float64
    phase_chunk_size = max(1, min(phase_grid_size, phase_chunk_size))
    period_chunk_size = period_chunk_size or _default_period_chunk_size(
        time_day.size,
        phase_chunk_size,
        torch_device.type,
    )

    periods = np.exp(
        np.linspace(
            np.log(min_period_day),
            np.log(max_period_day),
            grid_size,
            dtype=np.float64,
        )
    )
    durations = _duration_grid(
        min_duration_day,
        max_duration_day,
        duration_grid_size,
    )
    phase_centers = np.linspace(
        0.0,
        1.0,
        phase_grid_size,
        endpoint=False,
        dtype=np.float64,
    )
    phase_centers += 0.5 / phase_grid_size

    powers = np.full(grid_size, np.nan, dtype=np.float64)
    best_depth_by_period = np.full(grid_size, np.nan, dtype=np.float64)
    best_duration_by_period = np.full(grid_size, np.nan, dtype=np.float64)
    best_center_by_period = np.full(grid_size, np.nan, dtype=np.float64)

    time_origin = float(np.min(time_day))
    time_centered = time_day - time_origin
    weights = _weights_from_errors(error, time_centered.size)

    t_tensor = torch.as_tensor(time_centered, dtype=dtype, device=torch_device)
    y_tensor = torch.as_tensor(signal, dtype=dtype, device=torch_device)
    w_tensor = torch.as_tensor(weights, dtype=dtype, device=torch_device)
    wy_tensor = w_tensor * y_tensor
    total_w = w_tensor.sum()
    total_wy = wy_tensor.sum()
    eps = torch.as_tensor(np.finfo(np.float64).eps, dtype=dtype, device=torch_device)
    min_points_tensor = torch.as_tensor(
        float(min_transit_points),
        dtype=dtype,
        device=torch_device,
    )

    with torch.no_grad():
        for start in range(0, grid_size, period_chunk_size):
            end = min(start + period_chunk_size, grid_size)
            period_tensor = torch.as_tensor(
                periods[start:end],
                dtype=dtype,
                device=torch_device,
            )
            period_count = end - start
            phase = torch.remainder(t_tensor[None, :], period_tensor[:, None])
            phase = phase / period_tensor[:, None]

            best_power = torch.full(
                (period_count,),
                float("-inf"),
                dtype=dtype,
                device=torch_device,
            )
            best_depth = torch.zeros((period_count,), dtype=dtype, device=torch_device)
            best_duration = torch.zeros((period_count,), dtype=dtype, device=torch_device)
            best_center = torch.zeros((period_count,), dtype=dtype, device=torch_device)

            for duration_day in durations:
                duration_tensor = torch.as_tensor(
                    float(duration_day),
                    dtype=dtype,
                    device=torch_device,
                )
                duration_fraction = duration_tensor / period_tensor
                valid_duration = (
                    (duration_fraction > 0)
                    & (duration_fraction <= max_duration_fraction)
                )
                if not bool(torch.any(valid_duration).detach().cpu()):
                    continue

                half_duration_fraction = 0.5 * duration_fraction
                for phase_start in range(0, phase_grid_size, phase_chunk_size):
                    phase_end = min(phase_start + phase_chunk_size, phase_grid_size)
                    center_tensor = torch.as_tensor(
                        phase_centers[phase_start:phase_end],
                        dtype=dtype,
                        device=torch_device,
                    )
                    distance = torch.abs(
                        torch.remainder(
                            phase[:, None, :] - center_tensor[None, :, None] + 0.5,
                            1.0,
                        )
                        - 0.5
                    )
                    in_transit_bool = (
                        distance <= half_duration_fraction[:, None, None]
                    ) & valid_duration[:, None, None]
                    in_transit = in_transit_bool.to(dtype)
                    in_count = in_transit.sum(dim=2)
                    in_w = (in_transit * w_tensor[None, None, :]).sum(dim=2)
                    out_w = total_w - in_w
                    sum_in = (in_transit * wy_tensor[None, None, :]).sum(dim=2)
                    mean_in = sum_in / torch.clamp(in_w, min=eps)
                    mean_out = (total_wy - sum_in) / torch.clamp(out_w, min=eps)
                    depth = mean_in - mean_out
                    effective_weight = in_w * out_w / torch.clamp(total_w, min=eps)
                    power = depth.square() * effective_weight
                    valid_box = (
                        (depth > 0)
                        & (in_count >= min_points_tensor)
                        & (out_w > eps)
                    )
                    power = torch.where(
                        valid_box,
                        power,
                        torch.full_like(power, float("-inf")),
                    )
                    center_best_power, center_best_index = torch.max(power, dim=1)
                    update_mask = center_best_power > best_power
                    if torch.any(update_mask):
                        selected_depth = depth[
                            torch.arange(period_count, device=torch_device),
                            center_best_index,
                        ]
                        selected_center = center_tensor[center_best_index]
                        best_power = torch.where(
                            update_mask,
                            center_best_power,
                            best_power,
                        )
                        best_depth = torch.where(
                            update_mask,
                            selected_depth,
                            best_depth,
                        )
                        best_duration = torch.where(
                            update_mask,
                            duration_tensor.expand_as(best_duration),
                            best_duration,
                        )
                        best_center = torch.where(
                            update_mask,
                            selected_center,
                            best_center,
                        )

            finite_best = torch.isfinite(best_power) & (best_power > float("-inf"))
            powers[start:end] = torch.where(
                finite_best,
                best_power,
                torch.full_like(best_power, float("nan")),
            ).detach().cpu().numpy()
            best_depth_by_period[start:end] = torch.where(
                finite_best,
                best_depth,
                torch.full_like(best_depth, float("nan")),
            ).detach().cpu().numpy()
            best_duration_by_period[start:end] = torch.where(
                finite_best,
                best_duration,
                torch.full_like(best_duration, float("nan")),
            ).detach().cpu().numpy()
            best_center_by_period[start:end] = torch.where(
                finite_best,
                best_center,
                torch.full_like(best_center, float("nan")),
            ).detach().cpu().numpy()

    if not np.any(np.isfinite(powers)):
        raise BLSPlotError("BLS periodogram did not produce finite power values")

    powers = mask_daily_alias_powers(periods, powers)
    if not np.any(np.isfinite(powers)):
        raise BLSPlotError(
            "BLS periodogram did not produce finite power values after "
            "ignoring daily aliases"
        )

    best_index = int(np.nanargmax(powers))
    best_period_day = float(periods[best_index])
    best_duration_day = float(best_duration_by_period[best_index])
    best_center_phase = float(best_center_by_period[best_index])
    best_start_phase = (best_center_phase - 0.5 * best_duration_day / best_period_day) % 1.0
    best_center_epoch_day = time_origin + best_center_phase * best_period_day
    best_start_epoch_day = time_origin + best_start_phase * best_period_day

    return {
        "periods": periods,
        "powers": np.clip(powers, 0.0, None),
        "best_period": best_period_day,
        "optimal_depth": float(best_depth_by_period[best_index]),
        "optimal_duration": best_duration_day,
        "transit_epoch": best_center_epoch_day,
        "transit_start_epoch": best_start_epoch_day,
        "best_power": float(powers[best_index]),
        "search_device": str(torch_device),
        "ignored_periods": _daily_alias_filter_report(),
    }


def _plot_periodogram(
    ax: Any,
    periods: np.ndarray,
    powers: np.ndarray,
    best_period_day: float,
    selected_label: str,
    max_plot_points: int = 100_000,
) -> None:
    finite_mask = np.isfinite(periods) & np.isfinite(powers)
    periods = periods[finite_mask]
    powers = powers[finite_mask]
    order = np.argsort(periods)
    periods = periods[order]
    powers = powers[order]

    if periods.size > max_plot_points:
        stride_indices = np.linspace(0, periods.size - 1, max_plot_points, dtype=int)
        peak_index = int(np.nanargmax(powers))
        plot_indices = np.unique(np.concatenate([stride_indices, [peak_index]]))
        plot_indices.sort()
        periods = periods[plot_indices]
        powers = powers[plot_indices]

    ax.plot(periods, powers, color="#1f77b4", linewidth=0.8)
    ax.axvline(best_period_day, color="#d62728", linewidth=1.0, alpha=0.9)
    ax.set_xscale("log")
    ax.set_xlabel("Period (day)")
    ax.set_ylabel("BLS power")
    ax.set_title(f"BLS periodogram: {selected_label}", fontsize=10)
    ax.grid(True, which="both", alpha=0.22, linewidth=0.6)


def _plot_folded_series(
    ax: Any,
    payloads: list[dict[str, Any]],
    period_day: float,
    transit_start_epoch_day: float,
    duration_day: float,
    color_cycle: list[str],
) -> None:
    shade_duration = min(duration_day, 0.35 * period_day)
    for index, payload in enumerate(payloads):
        color = color_cycle[index % len(color_cycle)]
        folded_time = np.mod(payload["time_day"] - transit_start_epoch_day, period_day)
        x_values = np.concatenate([folded_time, folded_time + period_day])
        y_values = np.concatenate([payload["value"], payload["value"]])
        ax.scatter(
            x_values,
            y_values,
            s=11,
            alpha=0.46,
            linewidths=0,
            color=color,
            label=payload["label"],
        )

    for offset in (0.0, period_day):
        ax.axvspan(
            offset,
            offset + shade_duration,
            color="#d62728",
            alpha=0.10,
            linewidth=0,
        )
    ax.set_xlim(0.0, 2.0 * period_day)
    ax.set_xlabel("Folded time (day)")
    ax.grid(True, which="major", alpha=0.22, linewidth=0.6)
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)


def plot_BLS(
    csv_path: str | Path,
    header_parse_json_results: dict[str, Any] | str | Path,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    output_path: str | Path | None = None,
    min_period_day: float = DEFAULT_MIN_PERIOD_DAY,
    max_period_day: float = DEFAULT_MAX_PERIOD_DAY,
    grid_size: int = DEFAULT_GRID_SIZE,
    min_duration_day: float = DEFAULT_MIN_DURATION_DAY,
    max_duration_day: float = DEFAULT_MAX_DURATION_DAY,
    duration_grid_size: int = DEFAULT_DURATION_GRID_SIZE,
    phase_grid_size: int = DEFAULT_PHASE_GRID_SIZE,
    max_duration_fraction: float = DEFAULT_MAX_DURATION_FRACTION,
    min_transit_points: int = DEFAULT_MIN_TRANSIT_POINTS,
    period_chunk_size: int | None = None,
    phase_chunk_size: int = DEFAULT_PHASE_CHUNK_SIZE,
    device: str | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """Plot BLS periodogram plus P, P/2, and 2P folded light curves.

    Returns a JSON-like report with the PNG path, best period in days, optimal
    transit depth, duration, epoch, BLS power, and torch search device.
    """
    csv_path = Path(csv_path).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    if not csv_path.is_file():
        raise IsADirectoryError(csv_path)

    _, payloads = _prepare_series(csv_path, header_parse_json_results)
    selected_payload = _select_period_search_series(payloads)
    bls_result = box_least_squares(
        selected_payload["time_day"],
        selected_payload["value"],
        selected_payload["error"],
        unit=selected_payload["unit"],
        min_period_day=min_period_day,
        max_period_day=max_period_day,
        grid_size=grid_size,
        min_duration_day=min_duration_day,
        max_duration_day=max_duration_day,
        duration_grid_size=duration_grid_size,
        phase_grid_size=phase_grid_size,
        max_duration_fraction=max_duration_fraction,
        min_transit_points=min_transit_points,
        period_chunk_size=period_chunk_size,
        phase_chunk_size=phase_chunk_size,
        device=device,
    )
    best_period_day = float(bls_result["best_period"])
    png_path = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else Path(output_dir).expanduser().resolve() / f"{csv_path.stem}_BLS.png"
    )
    png_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=160)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4"])

    _plot_periodogram(
        axes[0, 0],
        bls_result["periods"],
        bls_result["powers"],
        best_period_day,
        selected_payload["label"],
    )

    folded_specs = [
        (axes[0, 1], best_period_day, "Best period P"),
        (axes[1, 0], best_period_day / 2.0, "Half period P/2"),
        (axes[1, 1], best_period_day * 2.0, "Double period 2P"),
    ]
    for ax, period_day, label in folded_specs:
        _plot_folded_series(
            ax,
            payloads,
            period_day,
            float(bls_result["transit_start_epoch"]),
            float(bls_result["optimal_duration"]),
            color_cycle,
        )
        ax.set_title(f"{label} = {period_day:.8g} day", fontsize=10)

    y_limits = _limits(
        value
        for payload in payloads
        for value in np.asarray(payload["value"], dtype=float).tolist()
    )
    units = list(dict.fromkeys(payload["unit"] for payload in payloads))
    y_unit = units[0] if len(units) == 1 else "photometry"
    for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
        if y_limits is not None:
            ax.set_ylim(*y_limits)
        ax.set_ylabel(_axis_label(y_unit))
        if any(payload["unit"] == "mag" for payload in payloads):
            ax.invert_yaxis()

    if len(payloads) > 1:
        axes[0, 1].legend(loc="best", fontsize=8, frameon=False)

    fig.suptitle(
        title
        or f"{csv_path.stem} BLS best period = {best_period_day:.8g} day, "
        f"depth = {float(bls_result['optimal_depth']):.6g}"
    )
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "result_png_path": str(png_path),
        "best_period": best_period_day,
        "period_unit": "day",
        "optimal_depth": float(bls_result["optimal_depth"]),
        "optimal_duration": float(bls_result["optimal_duration"]),
        "transit_epoch": float(bls_result["transit_epoch"]),
        "transit_start_epoch": float(bls_result["transit_start_epoch"]),
        "best_power": float(bls_result["best_power"]),
        "search_device": str(bls_result["search_device"]),
        "selected_series": selected_payload["label"],
        "ignored_periods": bls_result["ignored_periods"],
    }


plot_bls = plot_BLS


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="Path to the light-curve CSV file")
    parser.add_argument(
        "header_parse_json_results",
        help="Parsed schema as JSON text or a path to a JSON file",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory used when --output-path is not supplied",
    )
    parser.add_argument("--output-path", help="Explicit output PNG path")
    parser.add_argument(
        "--min-period-day",
        type=float,
        default=DEFAULT_MIN_PERIOD_DAY,
        help="Minimum period in days; default is 20 minutes",
    )
    parser.add_argument(
        "--max-period-day",
        type=float,
        default=DEFAULT_MAX_PERIOD_DAY,
        help="Maximum period in days",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=DEFAULT_GRID_SIZE,
        help="Number of period grid points",
    )
    parser.add_argument(
        "--min-duration-day",
        type=float,
        default=DEFAULT_MIN_DURATION_DAY,
        help="Minimum box duration in days; default is 5 minutes",
    )
    parser.add_argument(
        "--max-duration-day",
        type=float,
        default=DEFAULT_MAX_DURATION_DAY,
        help="Maximum box duration in days",
    )
    parser.add_argument(
        "--duration-grid-size",
        type=int,
        default=DEFAULT_DURATION_GRID_SIZE,
        help="Number of box duration grid points",
    )
    parser.add_argument(
        "--phase-grid-size",
        type=int,
        default=DEFAULT_PHASE_GRID_SIZE,
        help="Number of transit-center phases tested per period",
    )
    parser.add_argument(
        "--max-duration-fraction",
        type=float,
        default=DEFAULT_MAX_DURATION_FRACTION,
        help="Maximum transit duration as a fraction of period",
    )
    parser.add_argument(
        "--min-transit-points",
        type=int,
        default=DEFAULT_MIN_TRANSIT_POINTS,
        help="Minimum points inside a tested transit box",
    )
    parser.add_argument(
        "--period-chunk-size",
        type=int,
        help="Period grid chunk size for the torch BLS calculation",
    )
    parser.add_argument(
        "--phase-chunk-size",
        type=int,
        default=DEFAULT_PHASE_CHUNK_SIZE,
        help="Phase grid chunk size for the torch BLS calculation",
    )
    parser.add_argument(
        "--device",
        help="Torch device, for example 'cuda', 'cuda:0', or 'cpu'",
    )
    parser.add_argument("--title", help="Optional figure title")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = plot_BLS(
        args.csv_path,
        args.header_parse_json_results,
        output_dir=args.output_dir,
        output_path=args.output_path,
        min_period_day=args.min_period_day,
        max_period_day=args.max_period_day,
        grid_size=args.grid_size,
        min_duration_day=args.min_duration_day,
        max_duration_day=args.max_duration_day,
        duration_grid_size=args.duration_grid_size,
        phase_grid_size=args.phase_grid_size,
        max_duration_fraction=args.max_duration_fraction,
        min_transit_points=args.min_transit_points,
        period_chunk_size=args.period_chunk_size,
        phase_chunk_size=args.phase_chunk_size,
        device=args.device,
        title=args.title,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
