"""Generate zoom-in diagnostic plots for short-duration light-curve events."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "assets" / "light_curve"
DEFAULT_MAX_ZOOM_REGIONS = 3

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import parse_json  # noqa: E402
from tools.plot_whole_light_curve import (  # noqa: E402
    _axis_label,
    _coerce_schema,
    _collect_series,
    _column_indices,
    _limits,
    _read_csv_rows,
)


__all__ = [
    "ZoomInPlotError",
    "plot_zoom_in_light_curve",
    "plot_zoom_in",
]


class ZoomInPlotError(ValueError):
    """Raised when the zoom-in diagnostic plot cannot be generated."""


def _read_text_or_path(value: str | Path) -> str:
    if isinstance(value, Path):
        return value.expanduser().read_text(encoding="utf-8")

    text = value
    try:
        possible_path = Path(text).expanduser()
        if "\n" not in text and possible_path.exists() and possible_path.is_file():
            return possible_path.read_text(encoding="utf-8")
    except OSError:
        pass
    return text


def _coerce_full_light_curve_diagnosis(
    full_light_curve_diagnosis: Mapping[str, Any] | str | Path | None,
) -> dict[str, Any]:
    if full_light_curve_diagnosis is None:
        return {}
    if isinstance(full_light_curve_diagnosis, Mapping):
        return dict(full_light_curve_diagnosis)
    if isinstance(full_light_curve_diagnosis, (str, Path)):
        parsed = parse_json.parse_json(_read_text_or_path(full_light_curve_diagnosis))
        if not isinstance(parsed, Mapping):
            raise ZoomInPlotError("Full light-curve diagnosis must be a JSON object")
        return dict(parsed)
    raise ZoomInPlotError(
        "Full light-curve diagnosis must be JSON text, a JSON path, or a mapping"
    )


def _coerce_number(value: Any, field_name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ZoomInPlotError(
            f"`{field_name}` must contain numeric time values"
        ) from exc

    if not math.isfinite(number):
        raise ZoomInPlotError(f"`{field_name}` must contain finite time values")
    return number


def _is_numeric_pair(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return False
    try:
        float(value[0])
        float(value[1])
    except (TypeError, ValueError):
        return False
    return True


def _normalize_range_pair(
    start_value: Any,
    end_value: Any,
    field_name: str,
) -> list[float]:
    start = _coerce_number(start_value, field_name)
    end = _coerce_number(end_value, field_name)
    if start > end:
        start, end = end, start
    if start == end:
        pad = max(abs(start) * 1e-7, 1e-7)
        start -= pad
        end += pad
    return [start, end]


def _range_from_mapping(value: Mapping[str, Any], field_name: str) -> list[float]:
    start = value.get("t_start", value.get("start", value.get("time_start")))
    end = value.get("t_end", value.get("end", value.get("time_end")))
    if start is None or end is None:
        raise ZoomInPlotError(
            f"`{field_name}` mapping ranges must include t_start/start and t_end/end"
        )
    return _normalize_range_pair(start, end, field_name)


def _normalize_zoom_ranges(value: Any) -> list[list[float]]:
    field_name = "require_zoom_in"
    if value is None:
        return []

    if isinstance(value, str):
        normalized = value.strip()
        if (
            not normalized
            or normalized.lower() in {"none", "no", "n/a", "na", "[]"}
        ):
            return []
        try:
            return _normalize_zoom_ranges(parse_json.parse_json(normalized))
        except parse_json.JSONParseError as exc:
            raise ZoomInPlotError(
                f"`{field_name}` must be JSON ranges or a list of ranges"
            ) from exc

    if isinstance(value, Mapping):
        if "require_zoom_in" in value:
            return _normalize_zoom_ranges(value["require_zoom_in"])
        return [_range_from_mapping(value, field_name)]

    if not isinstance(value, (list, tuple)):
        raise ZoomInPlotError(f"`{field_name}` must be a list of time ranges")

    if not value:
        return []
    if _is_numeric_pair(value):
        return [_normalize_range_pair(value[0], value[1], field_name)]

    ranges: list[list[float]] = []
    for item in value:
        if isinstance(item, Mapping):
            ranges.append(_range_from_mapping(item, field_name))
        elif _is_numeric_pair(item):
            ranges.append(_normalize_range_pair(item[0], item[1], field_name))
        else:
            raise ZoomInPlotError(
                f"`{field_name}` must contain ranges in [t_start, t_end] format"
            )
    return ranges


def _prepare_series(
    csv_path: Path,
    header_parse_json_results: dict[str, Any] | str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    schema = _coerce_schema(header_parse_json_results)
    columns, rows = _read_csv_rows(csv_path)
    index_by_column = _column_indices(columns, schema)
    payloads = _collect_series(rows, schema, index_by_column)

    prepared_payloads: list[dict[str, Any]] = []
    for payload in payloads:
        time = np.asarray(payload["time"], dtype=float)
        value = np.asarray(payload["value"], dtype=float)
        finite_mask = np.isfinite(time) & np.isfinite(value)
        if not np.any(finite_mask):
            continue

        error = (
            np.asarray(payload["error"], dtype=float)
            if payload["error"] is not None
            else None
        )
        prepared = dict(payload)
        prepared["time"] = time[finite_mask]
        prepared["value"] = value[finite_mask]
        prepared["error"] = error[finite_mask] if error is not None else None
        prepared_payloads.append(prepared)

    if not prepared_payloads:
        raise ZoomInPlotError("No finite light-curve points found to plot")
    return schema, prepared_payloads


def _series_values(
    payloads: list[dict[str, Any]],
    time_range: tuple[float, float] | list[float] | None = None,
) -> tuple[list[float], list[float]]:
    times: list[float] = []
    values: list[float] = []
    for payload in payloads:
        time = np.asarray(payload["time"], dtype=float)
        value = np.asarray(payload["value"], dtype=float)
        mask = np.isfinite(time) & np.isfinite(value)
        if time_range is not None:
            start, end = float(time_range[0]), float(time_range[1])
            mask &= (time >= start) & (time <= end)
        if not np.any(mask):
            continue
        times.extend(time[mask].tolist())
        values.extend(value[mask].tolist())
    return times, values


def _series_key(payload: Mapping[str, Any]) -> str:
    return str(payload.get("label") or "").strip()


def _build_series_color_map(
    payloads: list[dict[str, Any]],
    color_cycle: list[str],
) -> dict[str, str]:
    if not color_cycle:
        color_cycle = ["#1f77b4"]

    color_by_series: dict[str, str] = {}
    for payload in payloads:
        key = _series_key(payload)
        if key and key not in color_by_series:
            color_by_series[key] = color_cycle[len(color_by_series) % len(color_cycle)]
    return color_by_series


def _plot_payloads(
    ax: Any,
    payloads: list[dict[str, Any]],
    color_by_series: Mapping[str, str],
    *,
    time_range: tuple[float, float] | list[float] | None = None,
    point_size: float = 14.0,
    alpha: float = 0.82,
) -> bool:
    plotted_any = False
    fallback_colors = list(color_by_series.values()) or ["#1f77b4"]
    for index, payload in enumerate(payloads):
        series_key = _series_key(payload)
        color = color_by_series.get(
            series_key,
            fallback_colors[index % len(fallback_colors)],
        )
        time = np.asarray(payload["time"], dtype=float)
        value = np.asarray(payload["value"], dtype=float)
        mask = np.isfinite(time) & np.isfinite(value)
        if time_range is not None:
            start, end = float(time_range[0]), float(time_range[1])
            mask &= (time >= start) & (time <= end)
        if not np.any(mask):
            continue

        plotted_any = True
        ax.scatter(
            time[mask],
            value[mask],
            s=point_size,
            alpha=alpha,
            linewidths=0,
            color=color,
            label=payload["label"],
        )

        error = payload["error"]
        if error is None:
            continue
        error = np.asarray(error, dtype=float)
        error_mask = mask & np.isfinite(error)
        if np.any(error_mask):
            ax.errorbar(
                time[error_mask],
                value[error_mask],
                yerr=error[error_mask],
                fmt="none",
                ecolor=color,
                elinewidth=0.55,
                alpha=0.32,
                capsize=0,
            )

    return plotted_any


def _plain_time_axis(ax: Any) -> None:
    from matplotlib.ticker import ScalarFormatter

    formatter = ScalarFormatter(useOffset=False, useMathText=False)
    formatter.set_scientific(False)
    ax.xaxis.set_major_formatter(formatter)
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)


def _format_range_value(value: float) -> str:
    return f"{value:.8g}"


def _style_axis(ax: Any, *, time_unit: str, y_label: str) -> None:
    _plain_time_axis(ax)
    ax.set_xlabel(f"Time ({time_unit})")
    ax.set_ylabel(y_label)
    ax.grid(True, which="major", alpha=0.22, linewidth=0.6)
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)


def _draw_zoom_boxes(
    ax: Any,
    zoom_ranges: list[list[float]],
    range_colors: list[str],
) -> None:
    from matplotlib.patches import Rectangle

    for index, (start, end) in enumerate(zoom_ranges):
        color = range_colors[index % len(range_colors)]
        ax.axvspan(start, end, color=color, alpha=0.08, linewidth=0)
        box = Rectangle(
            (start, 0.0),
            end - start,
            1.0,
            transform=ax.get_xaxis_transform(),
            fill=False,
            edgecolor=color,
            linewidth=1.1,
            linestyle="-",
        )
        ax.add_patch(box)
        ax.text(
            (start + end) / 2.0,
            0.98,
            f"Z{index + 1}",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            color=color,
            fontsize=8,
        )


def _resolve_zoom_ranges(
    zoom_ranges: Any,
    full_light_curve_diagnosis: Mapping[str, Any] | str | Path | None,
    max_zoom_regions: int,
) -> list[list[float]]:
    if zoom_ranges is None:
        full_report = _coerce_full_light_curve_diagnosis(full_light_curve_diagnosis)
        zoom_ranges = full_report.get("require_zoom_in")

    normalized_ranges = _normalize_zoom_ranges(zoom_ranges)
    if max_zoom_regions > 0:
        normalized_ranges = normalized_ranges[:max_zoom_regions]
    if not normalized_ranges:
        raise ZoomInPlotError("No zoom-in ranges were supplied")
    return normalized_ranges


def plot_zoom_in_light_curve(
    csv_path: str | Path,
    header_parse_json_results: dict[str, Any] | str | Path,
    full_light_curve_diagnosis: Mapping[str, Any] | str | Path | None = None,
    *,
    zoom_ranges: Any = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    output_path: str | Path | None = None,
    max_zoom_regions: int = DEFAULT_MAX_ZOOM_REGIONS,
    title: str | None = None,
) -> dict[str, Any]:
    """Plot the full light curve plus up to three requested zoom-in regions.

    Args:
        csv_path: Path to the light-curve CSV file.
        header_parse_json_results: Parsed schema dict, JSON text, or JSON file path.
        full_light_curve_diagnosis: Full-plot diagnosis JSON containing
            ``require_zoom_in``. Used when ``zoom_ranges`` is not supplied.
        zoom_ranges: Optional explicit ranges in ``[[t_start, t_end], ...]`` format.
        output_dir: Directory used when ``output_path`` is not supplied.
        output_path: Optional explicit PNG path.
        max_zoom_regions: Maximum number of zoom panels to draw.
        title: Optional figure title.

    Returns:
        A JSON-like report with ``result_png_path`` and normalized
        ``zoom_ranges`` values.
    """
    csv_path = Path(csv_path).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    if not csv_path.is_file():
        raise IsADirectoryError(csv_path)
    if max_zoom_regions < 1:
        raise ZoomInPlotError("max_zoom_regions must be at least 1")

    normalized_ranges = _resolve_zoom_ranges(
        zoom_ranges,
        full_light_curve_diagnosis,
        max_zoom_regions,
    )
    schema, payloads = _prepare_series(csv_path, header_parse_json_results)

    png_path = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else Path(output_dir).expanduser().resolve() / f"{csv_path.stem}_zoom_in.png"
    )
    png_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), dpi=160)
    axes_flat = list(axes.flat)
    full_ax = axes_flat[0]
    zoom_axes = axes_flat[1:]
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4"])
    color_by_series = _build_series_color_map(payloads, color_cycle)
    range_colors = ["#d62728", "#2ca02c", "#9467bd"]

    all_times, all_values = _series_values(payloads)
    x_limits = _limits(all_times)
    y_limits = _limits(all_values)
    units = list(dict.fromkeys(payload["unit"] for payload in payloads))
    y_unit = units[0] if len(units) == 1 else "photometry"
    y_label = _axis_label(y_unit)
    invert_y = any(payload["unit"] == "mag" for payload in payloads)

    _plot_payloads(full_ax, payloads, color_by_series, point_size=12.0, alpha=0.72)
    if x_limits is not None:
        full_ax.set_xlim(*x_limits)
    if y_limits is not None:
        full_ax.set_ylim(*y_limits)
    if invert_y:
        full_ax.invert_yaxis()
    _style_axis(full_ax, time_unit=schema["time_unit"], y_label=y_label)
    full_ax.set_title("Full light curve with zoom regions", fontsize=10)
    _draw_zoom_boxes(full_ax, normalized_ranges, range_colors)
    if len(payloads) > 1:
        full_ax.legend(loc="best", fontsize=8, frameon=False)

    for index, ax in enumerate(zoom_axes):
        if index >= len(normalized_ranges):
            ax.axis("off")
            continue

        start, end = normalized_ranges[index]
        plotted_any = _plot_payloads(
            ax,
            payloads,
            color_by_series,
            time_range=(start, end),
            point_size=18.0,
            alpha=0.88,
        )
        ax.set_xlim(start, end)
        _, zoom_values = _series_values(payloads, (start, end))
        zoom_y_limits = _limits(zoom_values) or y_limits
        if zoom_y_limits is not None:
            ax.set_ylim(*zoom_y_limits)
        if invert_y:
            ax.invert_yaxis()
        _style_axis(ax, time_unit=schema["time_unit"], y_label=y_label)
        ax.set_title(
            "Zoom "
            f"{index + 1}: "
            f"{_format_range_value(start)} to {_format_range_value(end)}",
            fontsize=10,
            color=range_colors[index % len(range_colors)],
        )
        if not plotted_any:
            ax.text(
                0.5,
                0.5,
                "No data in range",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                color="#555555",
            )
        elif len(color_by_series) > 1:
            ax.legend(loc="best", fontsize=7, frameon=False)

    fig.suptitle(title or f"{csv_path.stem} zoom-in diagnostic")
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)

    return {
        "result_png_path": str(png_path),
        "zoom_ranges": normalized_ranges,
    }


def plot_zoom_in(
    zoom_ranges: Any,
    header_parse_json_results: dict[str, Any] | str | Path,
    csv_path: str | Path,
    full_light_curve_diagnosis: Mapping[str, Any] | str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Wrapper using the requested range, schema, CSV, diagnosis argument order."""
    return plot_zoom_in_light_curve(
        csv_path,
        header_parse_json_results,
        full_light_curve_diagnosis,
        zoom_ranges=zoom_ranges,
        **kwargs,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="Path to the light-curve CSV file")
    parser.add_argument(
        "header_parse_json_results",
        help="Parsed schema as JSON text or a path to a JSON file",
    )
    parser.add_argument(
        "full_light_curve_diagnosis",
        nargs="?",
        help="Full light-curve diagnosis JSON text or path containing require_zoom_in",
    )
    parser.add_argument(
        "--zoom-ranges",
        help="Explicit zoom ranges as JSON, overriding full_light_curve_diagnosis",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory used when --output-path is not supplied",
    )
    parser.add_argument("--output-path", help="Explicit output PNG path")
    parser.add_argument(
        "--max-zoom-regions",
        type=int,
        default=DEFAULT_MAX_ZOOM_REGIONS,
        help="Maximum number of zoom ranges to plot",
    )
    parser.add_argument("--title", help="Optional figure title")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    report = plot_zoom_in_light_curve(
        args.csv_path,
        args.header_parse_json_results,
        args.full_light_curve_diagnosis,
        zoom_ranges=args.zoom_ranges,
        output_dir=args.output_dir,
        output_path=args.output_path,
        max_zoom_regions=args.max_zoom_regions,
        title=args.title,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
