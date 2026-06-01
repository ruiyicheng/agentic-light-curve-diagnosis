"""Generate generalized Lomb-Scargle diagnostic plots for light curves."""

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
DEFAULT_MIN_PERIOD_DAY = 20.0 / (60.0 * 24.0)
DEFAULT_MAX_PERIOD_DAY = 2000.0
DEFAULT_GRID_SIZE = 1_000_000
DEFAULT_FOURIER_ORDER = 3
DEFAULT_PHASE_COVERAGE_BINS = 20
DEFAULT_PREWHITEN_SIGNAL_COUNT = 1
DEFAULT_PREWHITEN_HARMONIC_DIVISOR_MAX = 10
DEFAULT_PREWHITEN_HARMONIC_TOLERANCE_FRACTION = 0.01
DEFAULT_MAX_PREWHITENING_REMOVALS = 50

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
    "GLSPlotError",
    "generalized_lomb_scargle",
    "plot_GLS",
    "plot_GLS_prewhitening",
    "plot_gls",
    "plot_gls_prewhitening",
]


TIME_UNIT_TO_DAY = {
    "day": 1.0,
    "hour": 1.0 / 24.0,
    "minute": 1.0 / (24.0 * 60.0),
    "second": 1.0 / (24.0 * 60.0 * 60.0),
}


class GLSPlotError(ValueError):
    """Raised when a GLS diagnostic plot cannot be generated."""


def _time_factor_to_day(time_unit: str) -> float:
    try:
        return TIME_UNIT_TO_DAY[time_unit]
    except KeyError as exc:
        raise GLSPlotError(f"Unsupported time unit for GLS: {time_unit!r}") from exc


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
        if np.count_nonzero(finite_mask) < 3:
            continue

        prepared = dict(payload)
        prepared["time"] = time[finite_mask]
        prepared["time_day"] = time[finite_mask] * time_factor
        prepared["value"] = value[finite_mask]
        prepared["error"] = error[finite_mask] if error is not None else None
        payloads.append(prepared)

    if not payloads:
        raise GLSPlotError("No light-curve series has at least 3 finite points")
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


def _default_chunk_size(sample_count: int, device_type: str) -> int:
    target_elements = 32_000_000 if device_type == "cuda" else 8_000_000
    return max(256, min(65_536, target_elements // max(sample_count, 1)))


def generalized_lomb_scargle(
    time_day: np.ndarray,
    value: np.ndarray,
    error: np.ndarray | None = None,
    *,
    min_period_day: float = DEFAULT_MIN_PERIOD_DAY,
    max_period_day: float = DEFAULT_MAX_PERIOD_DAY,
    grid_size: int = DEFAULT_GRID_SIZE,
    chunk_size: int | None = None,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return frequency grid and floating-mean GLS power.

    The search covers periods from ``min_period_day`` to ``max_period_day`` on
    a uniform period grid. Times are centered before trigonometric
    evaluation to keep GPU float64 calculations stable for Julian dates.
    """
    if min_period_day <= 0 or max_period_day <= 0:
        raise GLSPlotError("GLS periods must be positive")
    if min_period_day >= max_period_day:
        raise GLSPlotError("min_period_day must be smaller than max_period_day")
    if grid_size < 2:
        raise GLSPlotError("grid_size must be at least 2")

    time_day = np.asarray(time_day, dtype=np.float64)
    value = np.asarray(value, dtype=np.float64)
    finite_mask = np.isfinite(time_day) & np.isfinite(value)
    if error is not None:
        error = np.asarray(error, dtype=np.float64)
        error = error[finite_mask]
    time_day = time_day[finite_mask]
    value = value[finite_mask]
    if time_day.size < 3:
        raise GLSPlotError("GLS requires at least 3 finite light-curve points")
    if float(np.nanmax(time_day) - np.nanmin(time_day)) <= 0:
        raise GLSPlotError("GLS requires more than one unique observation time")

    try:
        import torch
    except ImportError as exc:
        raise GLSPlotError("PyTorch is required for GLS period search") from exc

    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device if device is not None else default_device)
    dtype = torch.float64 if torch_device.type == "cuda" else torch.float64
    chunk_size = chunk_size or _default_chunk_size(time_day.size, torch_device.type)

    periods = np.exp(np.linspace(
        np.log(min_period_day),
        np.log(max_period_day),
        grid_size,
        dtype=np.float64,
    ))
    frequencies = 1.0 / periods
    powers = np.full(grid_size, np.nan, dtype=np.float64)

    time_centered = time_day - float(np.min(time_day))
    weights = _weights_from_errors(error, time_centered.size)
    weights = weights / float(np.sum(weights))
    weighted_mean = float(np.sum(weights * value))
    centered_value = value - weighted_mean
    variance = float(np.sum(weights * np.square(centered_value)))
    if not math.isfinite(variance) or variance <= 0:
        raise GLSPlotError("GLS requires non-constant finite photometry values")

    t_tensor = torch.as_tensor(time_centered, dtype=dtype, device=torch_device)
    y_tensor = torch.as_tensor(centered_value, dtype=dtype, device=torch_device)
    w_tensor = torch.as_tensor(weights, dtype=dtype, device=torch_device)
    wy_tensor = w_tensor * y_tensor
    variance_tensor = torch.as_tensor(variance, dtype=dtype, device=torch_device)
    two_pi = torch.as_tensor(2.0 * math.pi, dtype=dtype, device=torch_device)
    eps = torch.as_tensor(np.finfo(np.float64).eps, dtype=dtype, device=torch_device)

    with torch.no_grad():
        for start in range(0, grid_size, chunk_size):
            end = min(start + chunk_size, grid_size)
            frequency_tensor = torch.as_tensor(
                frequencies[start:end],
                dtype=dtype,
                device=torch_device,
            )
            angle = two_pi * frequency_tensor[:, None] * t_tensor[None, :]
            cos_term = torch.cos(angle)
            sin_term = torch.sin(angle)

            weighted_cos = cos_term * w_tensor
            weighted_sin = sin_term * w_tensor
            cos_mean = weighted_cos.sum(dim=1)
            sin_mean = weighted_sin.sum(dim=1)
            cos2_mean = (weighted_cos * cos_term).sum(dim=1)
            sin2_mean = (weighted_sin * sin_term).sum(dim=1)
            cossin_mean = (weighted_cos * sin_term).sum(dim=1)
            ycos_mean = (cos_term * wy_tensor).sum(dim=1)
            ysin_mean = (sin_term * wy_tensor).sum(dim=1)

            cos_var = cos2_mean - cos_mean.square()
            sin_var = sin2_mean - sin_mean.square()
            cossin_cov = cossin_mean - cos_mean * sin_mean
            determinant = cos_var * sin_var - cossin_cov.square()
            numerator = (
                sin_var * ycos_mean.square()
                + cos_var * ysin_mean.square()
                - 2.0 * cossin_cov * ycos_mean * ysin_mean
            )
            denominator = variance_tensor * determinant
            power = torch.where(
                denominator > eps,
                numerator / denominator,
                torch.full_like(denominator, float("nan")),
            )
            powers[start:end] = power.detach().cpu().numpy().astype(np.float64)

            del angle, cos_term, sin_term

    return frequencies, np.clip(powers, 0.0, None)


def _select_period_search_series(
    payloads: list[dict[str, Any]],
) -> dict[str, Any]:
    return max(payloads, key=lambda payload: payload["time_day"].size)


def _mask_daily_alias_gls_powers(
    frequencies: np.ndarray,
    powers: np.ndarray,
) -> np.ndarray:
    periods = 1.0 / np.asarray(frequencies, dtype=np.float64)
    return mask_daily_alias_powers(periods, powers)


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


def _phase_fold(time_day: np.ndarray, period_day: float, epoch_day: float) -> np.ndarray:
    return np.mod(time_day - epoch_day, period_day) / period_day


def _positive_phase_delta(start_phase: float, end_phase: float) -> float:
    return float((end_phase - start_phase) % 1.0)


def _fit_fourier_model(
    time_day: np.ndarray,
    value: np.ndarray,
    error: np.ndarray | None,
    period_day: float,
    epoch_day: float,
    *,
    harmonic_order: int = DEFAULT_FOURIER_ORDER,
) -> dict[str, Any]:
    if period_day <= 0:
        raise GLSPlotError("Fourier model period must be positive")
    if harmonic_order < 1:
        raise GLSPlotError("Fourier model harmonic_order must be at least 1")

    time_day = np.asarray(time_day, dtype=np.float64)
    value = np.asarray(value, dtype=np.float64)
    mask = np.isfinite(time_day) & np.isfinite(value)
    if error is not None:
        error = np.asarray(error, dtype=np.float64)
        model_error = error[mask]
    else:
        model_error = None
    model_time = time_day[mask]
    model_value = value[mask]
    if model_time.size < max(3, 2 * harmonic_order + 1):
        raise GLSPlotError("Fourier model requires more finite points")

    phase = _phase_fold(model_time, period_day, epoch_day)
    columns = [np.ones_like(phase)]
    for harmonic in range(1, harmonic_order + 1):
        angle = 2.0 * math.pi * harmonic * phase
        columns.extend([np.cos(angle), np.sin(angle)])
    design = np.column_stack(columns)

    weights = _weights_from_errors(model_error, model_time.size)
    sqrt_weights = np.sqrt(weights)
    weighted_design = design * sqrt_weights[:, None]
    weighted_value = model_value * sqrt_weights
    coefficients, *_ = np.linalg.lstsq(weighted_design, weighted_value, rcond=None)
    fitted_value = design @ coefficients

    harmonic_amplitudes: list[float] = []
    harmonic_phases: list[float] = []
    for harmonic in range(1, harmonic_order + 1):
        cos_coeff = float(coefficients[2 * harmonic - 1])
        sin_coeff = float(coefficients[2 * harmonic])
        amplitude = math.hypot(cos_coeff, sin_coeff)
        phase_angle = math.atan2(-sin_coeff, cos_coeff)
        harmonic_amplitudes.append(amplitude)
        harmonic_phases.append(float(phase_angle % (2.0 * math.pi)))

    return {
        "coefficients": coefficients,
        "fitted_value": fitted_value,
        "mask": mask,
        "phase": phase,
        "harmonic_amplitudes": harmonic_amplitudes,
        "harmonic_phases_rad": harmonic_phases,
    }


def _evaluate_fourier_model(
    phase: np.ndarray,
    coefficients: np.ndarray,
    harmonic_order: int,
) -> np.ndarray:
    phase = np.asarray(phase, dtype=np.float64)
    y_model = np.full_like(phase, float(coefficients[0]), dtype=np.float64)
    for harmonic in range(1, harmonic_order + 1):
        angle = 2.0 * math.pi * harmonic * phase
        y_model += (
            float(coefficients[2 * harmonic - 1]) * np.cos(angle)
            + float(coefficients[2 * harmonic]) * np.sin(angle)
        )
    return y_model


def _weighted_skewness(value: np.ndarray, error: np.ndarray | None) -> float:
    value = np.asarray(value, dtype=np.float64)
    mask = np.isfinite(value)
    if error is not None:
        error = np.asarray(error, dtype=np.float64)
        model_error = error[mask]
    else:
        model_error = None
    finite_value = value[mask]
    if finite_value.size < 3:
        return float("nan")
    weights = _weights_from_errors(model_error, finite_value.size)
    weights = weights / float(np.sum(weights))
    mean = float(np.sum(weights * finite_value))
    centered = finite_value - mean
    variance = float(np.sum(weights * np.square(centered)))
    if variance <= 0 or not math.isfinite(variance):
        return float("nan")
    third_moment = float(np.sum(weights * np.power(centered, 3)))
    return third_moment / (variance ** 1.5)


def _phase_coverage(phase: np.ndarray, *, bin_count: int) -> dict[str, Any]:
    phase = np.asarray(phase, dtype=np.float64)
    finite_phase = phase[np.isfinite(phase)]
    if finite_phase.size == 0:
        return {"bin_count": bin_count, "occupied_bin_count": 0, "fraction": 0.0}
    occupied = np.unique(np.floor(finite_phase * bin_count).astype(int).clip(0, bin_count - 1))
    return {
        "bin_count": bin_count,
        "occupied_bin_count": int(occupied.size),
        "fraction": float(occupied.size / bin_count),
    }


def _model_extrema(
    coefficients: np.ndarray,
    harmonic_order: int,
    unit: str,
) -> dict[str, float]:
    model_phase = np.linspace(0.0, 1.0, 4096, endpoint=False, dtype=np.float64)
    model_value = _evaluate_fourier_model(model_phase, coefficients, harmonic_order)
    if unit == "mag":
        phase_of_maximum = float(model_phase[int(np.nanargmin(model_value))])
        phase_of_minimum = float(model_phase[int(np.nanargmax(model_value))])
    else:
        phase_of_maximum = float(model_phase[int(np.nanargmax(model_value))])
        phase_of_minimum = float(model_phase[int(np.nanargmin(model_value))])
    return {
        "phase_of_maximum": phase_of_maximum,
        "phase_of_minimum": phase_of_minimum,
        "model_max": float(np.nanmax(model_value)),
        "model_min": float(np.nanmin(model_value)),
        "peak_to_peak_amplitude": float(np.nanmax(model_value) - np.nanmin(model_value)),
    }


def _metric_definitions(y_unit: str) -> dict[str, dict[str, str]]:
    return {
        "fourier_model": {
            "meaning": "Third-order harmonic model fitted to the phase-folded selected series.",
            "formula": "y(phi)=C+sum_{k=1..3}[a_k cos(2*pi*k*phi)+b_k sin(2*pi*k*phi)]",
            "unit": y_unit,
        },
        "A_k": {
            "meaning": "Amplitude of Fourier harmonic k.",
            "formula": "A_k=sqrt(a_k^2+b_k^2)",
            "unit": y_unit,
        },
        "psi_k": {
            "meaning": "Phase angle of Fourier harmonic k in cosine convention.",
            "formula": "psi_k=atan2(-b_k,a_k)",
            "unit": "radian",
        },
        "R21": {
            "meaning": "Second-to-first harmonic amplitude ratio.",
            "formula": "R21=A_2/A_1",
            "unit": "dimensionless",
        },
        "R31": {
            "meaning": "Third-to-first harmonic amplitude ratio.",
            "formula": "R31=A_3/A_1",
            "unit": "dimensionless",
        },
        "phi21": {
            "meaning": "Fourier phase combination for harmonic shape.",
            "formula": "phi21=(psi_2-2*psi_1) mod 2*pi",
            "unit": "radian",
        },
        "phi31": {
            "meaning": "Fourier phase combination for harmonic shape.",
            "formula": "phi31=(psi_3-3*psi_1) mod 2*pi",
            "unit": "radian",
        },
        "amplitude": {
            "meaning": "Peak-to-peak range of the fitted folded model.",
            "formula": "max(y_model)-min(y_model)",
            "unit": y_unit,
        },
        "skewness": {
            "meaning": "Weighted skewness of observed selected-series photometry values.",
            "formula": "sum(w_i*(y_i-mean_y)^3)/(sum(w_i*(y_i-mean_y)^2)^(3/2))",
            "unit": "dimensionless",
        },
        "rise_time": {
            "meaning": "Phase/time from minimum brightness to the next maximum brightness.",
            "formula": "((phase_max-phase_min) mod 1)*P",
            "unit": "phase cycle and day",
        },
        "fall_time": {
            "meaning": "Phase/time from maximum brightness to the next minimum brightness.",
            "formula": "((phase_min-phase_max) mod 1)*P",
            "unit": "phase cycle and day",
        },
        "phase_of_maximum": {
            "meaning": "Phase of maximum brightness in the fitted model; for magnitude data this is minimum magnitude.",
            "formula": "argmax brightness over y_model(phi)",
            "unit": "phase cycle in [0,1)",
        },
        "phase_coverage": {
            "meaning": "Fraction of equal phase bins containing at least one observation.",
            "formula": "N_occupied_phase_bins/N_phase_bins",
            "unit": "dimensionless",
        },
        "scatter_around_folded_model": {
            "meaning": "Weighted RMS residual around the fitted folded Fourier model.",
            "formula": "sqrt(sum(w_i*(y_i-y_model_i)^2)/sum(w_i))",
            "unit": y_unit,
        },
        "typical_photometric_uncertainty": {
            "meaning": "Median finite per-point photometric uncertainty for the selected folded series, when an error column is available.",
            "formula": "median(sigma_i)",
            "unit": y_unit,
        },
        "scatter_to_uncertainty_ratio": {
            "meaning": "Folded-model scatter divided by the typical individual photometric uncertainty; values above 1 indicate folded-light-curve width larger than per-point uncertainty.",
            "formula": "scatter_around_folded_model/median(sigma_i)",
            "unit": "dimensionless",
        },
    }


def _folded_model_metrics(
    payload: dict[str, Any],
    period_day: float,
    epoch_day: float,
    *,
    harmonic_order: int = DEFAULT_FOURIER_ORDER,
    phase_coverage_bins: int = DEFAULT_PHASE_COVERAGE_BINS,
) -> dict[str, Any]:
    fit = _fit_fourier_model(
        payload["time_day"],
        payload["value"],
        payload["error"],
        period_day,
        epoch_day,
        harmonic_order=harmonic_order,
    )
    amplitudes = fit["harmonic_amplitudes"]
    phases = fit["harmonic_phases_rad"]
    a1 = amplitudes[0] if amplitudes else float("nan")
    extrema = _model_extrema(fit["coefficients"], harmonic_order, payload["unit"])
    rise_phase = _positive_phase_delta(
        extrema["phase_of_minimum"],
        extrema["phase_of_maximum"],
    )
    fall_phase = _positive_phase_delta(
        extrema["phase_of_maximum"],
        extrema["phase_of_minimum"],
    )
    residual = fit["fitted_value"] - np.asarray(payload["value"], dtype=np.float64)[fit["mask"]]
    error = payload["error"]
    model_error = np.asarray(error, dtype=np.float64)[fit["mask"]] if error is not None else None
    weights = _weights_from_errors(model_error, residual.size)
    scatter = math.sqrt(float(np.sum(weights * np.square(residual)) / np.sum(weights)))
    finite_uncertainty = (
        model_error[np.isfinite(model_error) & (model_error > 0)]
        if model_error is not None
        else np.asarray([], dtype=np.float64)
    )
    typical_uncertainty = (
        float(np.nanmedian(finite_uncertainty))
        if finite_uncertainty.size
        else None
    )
    scatter_to_uncertainty_ratio = (
        float(scatter / typical_uncertainty)
        if typical_uncertainty is not None and typical_uncertainty > 0
        else None
    )

    return {
        "period_day": float(period_day),
        "epoch_day": float(epoch_day),
        "selected_series_label": payload["label"],
        "photometry_unit": payload["unit"],
        "fourier_order": harmonic_order,
        "R21": float(amplitudes[1] / a1) if len(amplitudes) > 1 and a1 > 0 else None,
        "R31": float(amplitudes[2] / a1) if len(amplitudes) > 2 and a1 > 0 else None,
        "phi21_rad": float((phases[1] - 2.0 * phases[0]) % (2.0 * math.pi))
        if len(phases) > 1
        else None,
        "phi31_rad": float((phases[2] - 3.0 * phases[0]) % (2.0 * math.pi))
        if len(phases) > 2
        else None,
        "amplitude": extrema["peak_to_peak_amplitude"],
        "skewness": float(_weighted_skewness(payload["value"], payload["error"])),
        "rise_time_phase": rise_phase,
        "rise_time_day": rise_phase * period_day,
        "fall_time_phase": fall_phase,
        "fall_time_day": fall_phase * period_day,
        "phase_of_maximum": extrema["phase_of_maximum"],
        "phase_of_maximum_epoch_day": float(epoch_day + extrema["phase_of_maximum"] * period_day),
        "phase_coverage": _phase_coverage(fit["phase"], bin_count=phase_coverage_bins),
        "scatter_around_folded_model": scatter,
        "typical_photometric_uncertainty": typical_uncertainty,
        "scatter_to_uncertainty_ratio": scatter_to_uncertainty_ratio,
    }


def _plot_periodogram(
    ax: Any,
    frequencies: np.ndarray,
    powers: np.ndarray,
    best_period_day: float,
    selected_label: str,
    max_plot_points: int = 100_000,
) -> None:
    periods = 1.0 / frequencies
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
    ax.set_ylabel("GLS power")
    ax.set_title(f"GLS periodogram: {selected_label}", fontsize=10)
    ax.grid(True, which="both", alpha=0.22, linewidth=0.6)


def _plot_folded_series(
    ax: Any,
    payloads: list[dict[str, Any]],
    period_day: float,
    epoch_day: float,
    color_cycle: list[str],
) -> None:
    for index, payload in enumerate(payloads):
        color = color_cycle[index % len(color_cycle)]
        folded_time = np.mod(payload["time_day"] - epoch_day, period_day)
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
        error = payload["error"]
        if error is not None:
            error = np.asarray(error, dtype=float)
            error_mask = np.isfinite(error)
            if np.any(error_mask):
                folded_error_time = folded_time[error_mask]
                error_x_values = np.concatenate(
                    [folded_error_time, folded_error_time + period_day]
                )
                error_y_values = np.concatenate(
                    [payload["value"][error_mask], payload["value"][error_mask]]
                )
                error_values = np.concatenate([error[error_mask], error[error_mask]])
                ax.errorbar(
                    error_x_values,
                    error_y_values,
                    yerr=error_values,
                    fmt="none",
                    ecolor=color,
                    elinewidth=0.55,
                    alpha=0.26,
                    capsize=0,
                )

    ax.set_xlim(0.0, 2.0 * period_day)
    ax.set_xlabel("Folded time (day)")
    ax.grid(True, which="major", alpha=0.22, linewidth=0.6)
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)


def plot_GLS(
    csv_path: str | Path,
    header_parse_json_results: dict[str, Any] | str | Path,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    output_path: str | Path | None = None,
    min_period_day: float = DEFAULT_MIN_PERIOD_DAY,
    max_period_day: float = DEFAULT_MAX_PERIOD_DAY,
    grid_size: int = DEFAULT_GRID_SIZE,
    chunk_size: int | None = None,
    device: str | None = None,
    title: str | None = None,
    harmonic_order: int = DEFAULT_FOURIER_ORDER,
    phase_coverage_bins: int = DEFAULT_PHASE_COVERAGE_BINS,
) -> dict[str, Any]:
    """Plot GLS periodogram plus P, P/2, and 2P folded light curves.

    Returns:
        ``{"result_png_path": str, "best_period": float}``, where
        ``best_period`` is measured in days.
    """
    csv_path = Path(csv_path).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    if not csv_path.is_file():
        raise IsADirectoryError(csv_path)

    schema, payloads = _prepare_series(csv_path, header_parse_json_results)
    selected_payload = _select_period_search_series(payloads)
    frequencies, powers = generalized_lomb_scargle(
        selected_payload["time_day"],
        selected_payload["value"],
        selected_payload["error"],
        min_period_day=min_period_day,
        max_period_day=max_period_day,
        grid_size=grid_size,
        chunk_size=chunk_size,
        device=device,
    )
    if not np.any(np.isfinite(powers)):
        raise GLSPlotError("GLS periodogram did not produce finite power values")

    powers = _mask_daily_alias_gls_powers(frequencies, powers)
    if not np.any(np.isfinite(powers)):
        raise GLSPlotError(
            "GLS periodogram did not produce finite power values after "
            "ignoring daily aliases"
        )

    best_index = int(np.nanargmax(powers))
    best_period_day = float(1.0 / frequencies[best_index])
    png_path = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else Path(output_dir).expanduser().resolve() / f"{csv_path.stem}_GLS.png"
    )
    png_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=160)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4"])

    _plot_periodogram(
        axes[0, 0],
        frequencies,
        powers,
        best_period_day,
        selected_payload["label"],
    )

    epoch_day = float(np.min(selected_payload["time_day"]))
    folded_specs = [
        (axes[0, 1], best_period_day, "Best period P"),
        (axes[1, 0], best_period_day / 2.0, "Half period P/2"),
        (axes[1, 1], best_period_day * 2.0, "Double period 2P"),
    ]
    for ax, period_day, label in folded_specs:
        _plot_folded_series(ax, payloads, period_day, epoch_day, color_cycle)
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
        title or f"{csv_path.stem} GLS best period = {best_period_day:.8g} day"
    )
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    folded_period_metrics = {
        "P": _folded_model_metrics(
            selected_payload,
            best_period_day,
            epoch_day,
            harmonic_order=harmonic_order,
            phase_coverage_bins=phase_coverage_bins,
        ),
        "P/2": _folded_model_metrics(
            selected_payload,
            best_period_day / 2.0,
            epoch_day,
            harmonic_order=harmonic_order,
            phase_coverage_bins=phase_coverage_bins,
        ),
        "2P": _folded_model_metrics(
            selected_payload,
            best_period_day * 2.0,
            epoch_day,
            harmonic_order=harmonic_order,
            phase_coverage_bins=phase_coverage_bins,
        ),
    }
    units = list(dict.fromkeys(payload["unit"] for payload in payloads))
    metric_unit = units[0] if len(units) == 1 else selected_payload["unit"]

    return {
        "result_png_path": str(png_path),
        "best_period": best_period_day,
        "period_unit": "day",
        "analysis_type": "GLS",
        "selected_period_search_series": selected_payload["label"],
        "ignored_periods": _daily_alias_filter_report(),
        "folded_period_metrics": folded_period_metrics,
        "metric_definitions": _metric_definitions(metric_unit),
    }


def _copy_payloads_for_residuals(payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    residual_payloads: list[dict[str, Any]] = []
    for payload in payloads:
        copied = dict(payload)
        copied["time"] = np.asarray(payload["time"], dtype=np.float64).copy()
        copied["time_day"] = np.asarray(payload["time_day"], dtype=np.float64).copy()
        copied["value"] = np.asarray(payload["value"], dtype=np.float64).copy()
        copied["error"] = (
            np.asarray(payload["error"], dtype=np.float64).copy()
            if payload["error"] is not None
            else None
        )
        residual_payloads.append(copied)
    return residual_payloads


def _subtract_periodic_signal(
    payloads: list[dict[str, Any]],
    period_day: float,
    epoch_day: float,
    *,
    harmonic_order: int,
) -> None:
    for payload in payloads:
        try:
            fit = _fit_fourier_model(
                payload["time_day"],
                payload["value"],
                payload["error"],
                period_day,
                epoch_day,
                harmonic_order=harmonic_order,
            )
        except GLSPlotError:
            continue
        value = np.asarray(payload["value"], dtype=np.float64).copy()
        model_mean = float(fit["coefficients"][0])
        value[fit["mask"]] = value[fit["mask"]] - (fit["fitted_value"] - model_mean)
        payload["value"] = value


def _strongest_gls_signal(
    payload: dict[str, Any],
    *,
    min_period_day: float,
    max_period_day: float,
    grid_size: int,
    chunk_size: int | None,
    device: str | None,
    error_prefix: str,
) -> dict[str, Any]:
    frequencies, powers = generalized_lomb_scargle(
        payload["time_day"],
        payload["value"],
        payload["error"],
        min_period_day=min_period_day,
        max_period_day=max_period_day,
        grid_size=grid_size,
        chunk_size=chunk_size,
        device=device,
    )
    if not np.any(np.isfinite(powers)):
        raise GLSPlotError(f"{error_prefix} GLS periodogram did not produce finite power values")
    powers = _mask_daily_alias_gls_powers(frequencies, powers)
    if not np.any(np.isfinite(powers)):
        raise GLSPlotError(
            f"{error_prefix} GLS periodogram did not produce finite power "
            "values after ignoring daily aliases"
        )

    best_index = int(np.nanargmax(powers))
    return {
        "frequencies": frequencies,
        "powers": powers,
        "best_index": best_index,
        "period_day": float(1.0 / frequencies[best_index]),
        "power": float(powers[best_index]),
    }


def _matching_prewhitening_harmonic(
    period_day: float,
    base_period_day: float,
    *,
    max_divisor: int = DEFAULT_PREWHITEN_HARMONIC_DIVISOR_MAX,
    tolerance_fraction: float = DEFAULT_PREWHITEN_HARMONIC_TOLERANCE_FRACTION,
) -> dict[str, Any] | None:
    if period_day <= 0 or base_period_day <= 0:
        return None

    best_match: dict[str, Any] | None = None
    for divisor in range(1, max_divisor + 1):
        target_period_day = base_period_day / divisor
        tolerance_day = target_period_day * tolerance_fraction
        offset_day = abs(period_day - target_period_day)
        if offset_day > tolerance_day:
            continue
        match = {
            "divisor": divisor,
            "target_period_day": float(target_period_day),
            "tolerance_day": float(tolerance_day),
            "offset_day": float(offset_day),
        }
        if best_match is None or offset_day < best_match["offset_day"]:
            best_match = match
    return best_match


def plot_GLS_prewhitening(
    csv_path: str | Path,
    header_parse_json_results: dict[str, Any] | str | Path,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    output_path: str | Path | None = None,
    min_period_day: float = DEFAULT_MIN_PERIOD_DAY,
    max_period_day: float = DEFAULT_MAX_PERIOD_DAY,
    grid_size: int = DEFAULT_GRID_SIZE,
    chunk_size: int | None = None,
    device: str | None = None,
    title: str | None = None,
    signal_count: int = DEFAULT_PREWHITEN_SIGNAL_COUNT,
    harmonic_order: int = DEFAULT_FOURIER_ORDER,
    phase_coverage_bins: int = DEFAULT_PHASE_COVERAGE_BINS,
    harmonic_divisor_max: int = DEFAULT_PREWHITEN_HARMONIC_DIVISOR_MAX,
    harmonic_tolerance_fraction: float = DEFAULT_PREWHITEN_HARMONIC_TOLERANCE_FRACTION,
    max_prewhitening_removals: int = DEFAULT_MAX_PREWHITENING_REMOVALS,
) -> dict[str, Any]:
    """Subtract GLS signals, including residual P/n harmonics of each base signal."""
    if signal_count < 1:
        raise GLSPlotError("signal_count must be at least 1")
    if harmonic_divisor_max < 1:
        raise GLSPlotError("harmonic_divisor_max must be at least 1")
    if harmonic_tolerance_fraction <= 0:
        raise GLSPlotError("harmonic_tolerance_fraction must be positive")
    if max_prewhitening_removals < 1:
        raise GLSPlotError("max_prewhitening_removals must be at least 1")

    csv_path = Path(csv_path).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    if not csv_path.is_file():
        raise IsADirectoryError(csv_path)

    schema, payloads = _prepare_series(csv_path, header_parse_json_results)
    residual_payloads = _copy_payloads_for_residuals(payloads)
    selected_label = _select_period_search_series(payloads)["label"]
    epoch_day = float(
        np.min(
            _select_period_search_series(residual_payloads)["time_day"]
        )
    )

    removed_signals: list[dict[str, Any]] = []
    harmonic_stop_checks: list[dict[str, Any]] = []
    stop_reason = "requested_base_signal_count_reached"
    base_signal_number = 0
    while base_signal_number < signal_count:
        if len(removed_signals) >= max_prewhitening_removals:
            stop_reason = "max_prewhitening_removals_reached"
            break

        selected_payload = next(
            payload for payload in residual_payloads if payload["label"] == selected_label
        )
        strongest_signal = _strongest_gls_signal(
            selected_payload,
            min_period_day=min_period_day,
            max_period_day=max_period_day,
            grid_size=grid_size,
            chunk_size=chunk_size,
            device=device,
            error_prefix="Prewhitening",
        )
        base_signal_number += 1
        removed_period_day = strongest_signal["period_day"]
        removed_signals.append(
            {
                "signal_number": len(removed_signals) + 1,
                "base_signal_number": base_signal_number,
                "removal_type": "base",
                "period_day": removed_period_day,
                "power": strongest_signal["power"],
                "selected_series_label": selected_label,
            }
        )
        _subtract_periodic_signal(
            residual_payloads,
            removed_period_day,
            epoch_day,
            harmonic_order=harmonic_order,
        )

        base_period_day = removed_period_day
        while len(removed_signals) < max_prewhitening_removals:
            selected_payload = next(
                payload for payload in residual_payloads if payload["label"] == selected_label
            )
            strongest_signal = _strongest_gls_signal(
                selected_payload,
                min_period_day=min_period_day,
                max_period_day=max_period_day,
                grid_size=grid_size,
                chunk_size=chunk_size,
                device=device,
                error_prefix="Prewhitening residual",
            )
            harmonic_match = _matching_prewhitening_harmonic(
                strongest_signal["period_day"],
                base_period_day,
                max_divisor=harmonic_divisor_max,
                tolerance_fraction=harmonic_tolerance_fraction,
            )
            if harmonic_match is None:
                harmonic_stop_checks.append(
                    {
                        "base_signal_number": base_signal_number,
                        "base_period_day": base_period_day,
                        "current_best_period_day": strongest_signal["period_day"],
                        "current_best_power": strongest_signal["power"],
                        "reason": "current_best_signal_not_in_base_harmonic_windows",
                    }
                )
                break

            removed_signals.append(
                {
                    "signal_number": len(removed_signals) + 1,
                    "base_signal_number": base_signal_number,
                    "removal_type": "base_period_harmonic",
                    "period_day": strongest_signal["period_day"],
                    "power": strongest_signal["power"],
                    "selected_series_label": selected_label,
                    "matched_harmonic": harmonic_match,
                }
            )
            _subtract_periodic_signal(
                residual_payloads,
                strongest_signal["period_day"],
                epoch_day,
                harmonic_order=harmonic_order,
            )

        if len(removed_signals) >= max_prewhitening_removals:
            stop_reason = "max_prewhitening_removals_reached"
            break

    selected_residual_payload = next(
        payload for payload in residual_payloads if payload["label"] == selected_label
    )
    residual_signal = _strongest_gls_signal(
        selected_residual_payload,
        min_period_day=min_period_day,
        max_period_day=max_period_day,
        grid_size=grid_size,
        chunk_size=chunk_size,
        device=device,
        error_prefix="Residual",
    )
    residual_frequencies = residual_signal["frequencies"]
    residual_powers = residual_signal["powers"]
    best_period_day = residual_signal["period_day"]
    png_path = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else Path(output_dir).expanduser().resolve()
        / f"{csv_path.stem}_GLS_prewhitening_{len(removed_signals)}.png"
    )
    png_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=160)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4"])

    _plot_periodogram(
        axes[0, 0],
        residual_frequencies,
        residual_powers,
        best_period_day,
        f"{selected_label} residual",
    )

    folded_specs = [
        (axes[0, 1], best_period_day, "Residual best period P"),
        (axes[1, 0], best_period_day / 2.0, "Residual half period P/2"),
        (axes[1, 1], best_period_day * 2.0, "Residual double period 2P"),
    ]
    for ax, period_day, label in folded_specs:
        _plot_folded_series(ax, residual_payloads, period_day, epoch_day, color_cycle)
        ax.set_title(f"{label} = {period_day:.8g} day", fontsize=10)

    y_limits = _limits(
        value
        for payload in residual_payloads
        for value in np.asarray(payload["value"], dtype=float).tolist()
    )
    units = list(dict.fromkeys(payload["unit"] for payload in residual_payloads))
    y_unit = units[0] if len(units) == 1 else "photometry"
    for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
        if y_limits is not None:
            ax.set_ylim(*y_limits)
        ax.set_ylabel(f"Residual {_axis_label(y_unit)}")
        if any(payload["unit"] == "mag" for payload in residual_payloads):
            ax.invert_yaxis()

    if len(residual_payloads) > 1:
        axes[0, 1].legend(loc="best", fontsize=8, frameon=False)

    removed_period_text = ", ".join(
        f"{signal['period_day']:.8g} d" for signal in removed_signals
    )
    fig.suptitle(
        title
        or (
            f"{csv_path.stem} GLS prewhitening: removed {len(removed_signals)} signal(s) "
            f"({removed_period_text}); residual P = {best_period_day:.8g} day"
        )
    )
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    folded_period_metrics = {
        "P": _folded_model_metrics(
            selected_residual_payload,
            best_period_day,
            epoch_day,
            harmonic_order=harmonic_order,
            phase_coverage_bins=phase_coverage_bins,
        ),
        "P/2": _folded_model_metrics(
            selected_residual_payload,
            best_period_day / 2.0,
            epoch_day,
            harmonic_order=harmonic_order,
            phase_coverage_bins=phase_coverage_bins,
        ),
        "2P": _folded_model_metrics(
            selected_residual_payload,
            best_period_day * 2.0,
            epoch_day,
            harmonic_order=harmonic_order,
            phase_coverage_bins=phase_coverage_bins,
        ),
    }
    metric_unit = units[0] if len(units) == 1 else selected_residual_payload["unit"]

    return {
        "result_png_path": str(png_path),
        "best_period": best_period_day,
        "period_unit": "day",
        "analysis_type": "GLS(prewhitening)",
        "selected_period_search_series": selected_label,
        "ignored_periods": _daily_alias_filter_report(),
        "prewhitening": {
            "signal_count": len(removed_signals),
            "requested_base_signal_count": signal_count,
            "removed_signals": removed_signals,
            "harmonic_stop_checks": harmonic_stop_checks,
            "base_harmonic_rule": {
                "periods_checked": "P/n for n=1..harmonic_divisor_max",
                "harmonic_divisor_max": harmonic_divisor_max,
                "tolerance_fraction": harmonic_tolerance_fraction,
                "tolerance_description": "absolute tolerance is (P/n) * tolerance_fraction",
                "selection_rule": (
                    "After each base period is subtracted, subtract the current "
                    "most significant residual GLS signal only if it lies within "
                    "one of that base period's P/n windows."
                ),
            },
            "stop_reason": stop_reason,
            "method": (
                "Find the strongest GLS base period in the residual selected series, "
                "fit a third-order Fourier model at that period to every series, "
                "and subtract only the periodic component around the fitted mean. "
                "After a base period P is removed, repeatedly inspect the residual "
                "GLS periodogram and also subtract the current strongest signal if "
                f"it falls within P/n +/- P/n*{harmonic_tolerance_fraction:g} "
                f"for n=1..{harmonic_divisor_max}. Stop that harmonic cascade "
                "once the current strongest residual signal is outside all P/n "
                "windows."
            ),
        },
        "folded_period_metrics": folded_period_metrics,
        "metric_definitions": _metric_definitions(metric_unit),
    }


plot_gls = plot_GLS
plot_gls_prewhitening = plot_GLS_prewhitening


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
        "--chunk-size",
        type=int,
        help="Grid chunk size for the torch GLS calculation",
    )
    parser.add_argument(
        "--device",
        help="Torch device, for example 'cuda', 'cuda:0', or 'cpu'",
    )
    parser.add_argument("--title", help="Optional figure title")
    parser.add_argument(
        "--prewhiten",
        action="store_true",
        help="Run GLS(prewhitening): subtract strongest signal(s), then plot residual GLS",
    )
    parser.add_argument(
        "--prewhiten-signal-count",
        type=int,
        default=DEFAULT_PREWHITEN_SIGNAL_COUNT,
        help="Number of strongest periodic signals to subtract when --prewhiten is used",
    )
    parser.add_argument(
        "--fourier-order",
        type=int,
        default=DEFAULT_FOURIER_ORDER,
        help="Fourier harmonic order used for morphology metrics and prewhitening",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.prewhiten:
        report = plot_GLS_prewhitening(
            args.csv_path,
            args.header_parse_json_results,
            output_dir=args.output_dir,
            output_path=args.output_path,
            min_period_day=args.min_period_day,
            max_period_day=args.max_period_day,
            grid_size=args.grid_size,
            chunk_size=args.chunk_size,
            device=args.device,
            title=args.title,
            signal_count=args.prewhiten_signal_count,
            harmonic_order=args.fourier_order,
        )
    else:
        report = plot_GLS(
            args.csv_path,
            args.header_parse_json_results,
            output_dir=args.output_dir,
            output_path=args.output_path,
            min_period_day=args.min_period_day,
            max_period_day=args.max_period_day,
            grid_size=args.grid_size,
            chunk_size=args.chunk_size,
            device=args.device,
            title=args.title,
            harmonic_order=args.fourier_order,
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
