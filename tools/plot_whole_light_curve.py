"""Plot whole light curves from CSV files and parsed schema JSON."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "assets" / "light_curve"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import parse_json
from tools import parse_light_curve_df


COMMENT_PREFIXES = ("#", "//", "%", "!", ";")
NULL_VALUES = {"", "nan", "na", "n/a", "none", "null", "--", "..."}

__all__ = ["plot_whole_light_curve", "plot_light_curve"]


class LightCurvePlotError(ValueError):
    """Raised when a light curve cannot be plotted from the supplied schema."""


def _is_comment_line(line: str) -> bool:
    stripped = line.lstrip()
    return bool(stripped) and stripped.startswith(COMMENT_PREFIXES)


def _read_text_lines(path: Path) -> list[str]:
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            return path.read_text(encoding=encoding).splitlines()
        except UnicodeDecodeError:
            continue

    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def _coerce_schema(header_parse_json_results: dict[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(header_parse_json_results, dict):
        return parse_light_curve_df.normalize_schema(header_parse_json_results)

    text_or_path = header_parse_json_results
    if isinstance(text_or_path, Path):
        text = text_or_path.expanduser().read_text(encoding="utf-8")
    else:
        text = text_or_path
        try:
            possible_path = Path(text).expanduser()
            if "\n" not in text and possible_path.exists():
                text = possible_path.read_text(encoding="utf-8")
        except OSError:
            pass

    parsed = parse_json.parse_json(text)
    return parse_light_curve_df.normalize_schema(parsed)


def _read_csv_rows(csv_path: Path) -> tuple[list[str], list[list[str]]]:
    preview = parse_light_curve_df.extract_csv_preview(csv_path)
    columns = list(preview["columns"])
    delimiter = preview.get("delimiter", ",")
    has_header = bool(preview["has_header"])

    data_lines = [
        line
        for line in _read_text_lines(csv_path)
        if line.strip() and not _is_comment_line(line)
    ]
    if not data_lines:
        raise LightCurvePlotError(f"No CSV data rows found in {csv_path}")

    reader = csv.reader(data_lines, delimiter=delimiter, skipinitialspace=True)
    parsed_rows = [[cell.strip() for cell in row] for row in reader if row]
    rows = parsed_rows[1:] if has_header else parsed_rows
    if not rows:
        raise LightCurvePlotError(f"No light-curve data rows found in {csv_path}")

    return columns, rows


def _column_indices(columns: list[str], schema: dict[str, Any]) -> dict[str, int]:
    index_by_column = {column: index for index, column in enumerate(columns)}
    required_columns = [schema["time_column"], *schema["photometry_column"]]
    required_columns.extend(
        column for column in schema["photometry_err_column"] if column
    )
    if schema.get("filter_column"):
        required_columns.append(schema["filter_column"])
    missing_columns = [
        column for column in required_columns if column not in index_by_column
    ]
    if missing_columns:
        missing = ", ".join(sorted(set(missing_columns)))
        available = ", ".join(columns)
        raise LightCurvePlotError(
            f"Schema columns not found in {schema.get('path', 'CSV')}: "
            f"{missing}. Available columns: {available}"
        )

    return index_by_column


def _to_float(value: Any) -> float | None:
    text = str(value).strip().replace("\u2212", "-")
    if text.lower() in NULL_VALUES:
        return None

    try:
        number = float(text)
    except ValueError:
        return None

    if not math.isfinite(number):
        return None
    return number


def _series_label(column: str, band: str, unit: str) -> str:
    if band and band != "unknown":
        return f"{band} {unit} ({column})"
    return f"{column} ({unit})"


def _band_value(value: Any, fallback: str) -> str:
    band = str(value).strip()
    if band:
        return band
    return fallback if fallback else "unknown"


def _axis_label(unit: str) -> str:
    labels = {
        "mag": "Magnitude",
        "flux": "Flux",
        "normalized_flux": "Normalized Flux",
    }
    return labels.get(unit, unit.replace("_", " ").title())


def _limits(values: Iterable[float]) -> tuple[float, float] | None:
    finite_values = np.asarray([value for value in values if math.isfinite(value)])
    if finite_values.size == 0:
        return None

    lower, upper = np.nanpercentile(finite_values, [0.5, 99.5])
    if not np.isfinite(lower) or not np.isfinite(upper) or lower == upper:
        lower = float(np.nanmin(finite_values))
        upper = float(np.nanmax(finite_values))
    if lower == upper:
        margin = max(abs(lower) * 0.08, 1.0)
    else:
        margin = (upper - lower) * 0.08
    return lower - margin, upper + margin


def _collect_series(
    rows: list[list[str]],
    schema: dict[str, Any],
    index_by_column: dict[str, int],
) -> list[dict[str, Any]]:
    time_index = index_by_column[schema["time_column"]]
    filter_column = schema.get("filter_column") or ""
    filter_index = index_by_column[filter_column] if filter_column else None
    series_groups: dict[tuple[int, str], dict[str, Any]] = {}

    for series_index, (column, err_column, unit, static_band) in enumerate(
        zip(
            schema["photometry_column"],
            schema["photometry_err_column"],
            schema["photometry_unit"],
            schema["photometry_band"],
        )
    ):
        value_index = index_by_column[column]
        err_index = index_by_column[err_column] if err_column else None
        max_required_index = max(
            index
            for index in (time_index, value_index, err_index, filter_index)
            if index is not None
        )

        for row in rows:
            if len(row) <= max_required_index:
                continue

            time_value = _to_float(row[time_index])
            photometry_value = _to_float(row[value_index])
            if time_value is None or photometry_value is None:
                continue

            error_value = None
            if err_index is not None:
                error_value = _to_float(row[err_index])
                if error_value is None or error_value < 0:
                    error_value = None

            band = (
                _band_value(row[filter_index], static_band)
                if filter_index is not None
                else static_band
            )
            group_key = (series_index, band)
            group = series_groups.setdefault(
                group_key,
                {
                    "times": [],
                    "values": [],
                    "errors": [],
                    "has_error": False,
                    "unit": unit,
                    "label": _series_label(column, band, unit),
                },
            )

            group["times"].append(time_value)
            group["values"].append(photometry_value)
            group["errors"].append(error_value if error_value is not None else np.nan)
            if error_value is not None:
                group["has_error"] = True

    series_payloads: list[dict[str, Any]] = []
    for group in series_groups.values():
        if not group["times"]:
            continue

        time_array = np.asarray(group["times"], dtype=float)
        sort_order = np.argsort(time_array)
        payload = {
            "time": time_array[sort_order],
            "value": np.asarray(group["values"], dtype=float)[sort_order],
            "error": (
                np.asarray(group["errors"], dtype=float)[sort_order]
                if group["has_error"]
                else None
            ),
            "unit": group["unit"],
            "label": group["label"],
        }
        series_payloads.append(payload)

    if not series_payloads:
        raise LightCurvePlotError("No finite light-curve points found to plot")
    return series_payloads


def plot_whole_light_curve(
    csv_path: str | Path,
    header_parse_json_results: dict[str, Any] | str | Path,
    *,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    output_path: str | Path | None = None,
    title: str | None = None,
) -> Path:
    """Plot the whole light curve and return the saved PNG path.

    Args:
        csv_path: Path to the light-curve CSV file.
        header_parse_json_results: Parsed schema dict, JSON text, or JSON file path
            with the schema produced by ``parse_light_curve_df``.
        output_dir: Directory used when ``output_path`` is not supplied.
        output_path: Optional explicit PNG path.
        title: Optional plot title.
    """
    csv_path = Path(csv_path).expanduser().resolve()
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    if not csv_path.is_file():
        raise IsADirectoryError(csv_path)

    schema = _coerce_schema(header_parse_json_results)
    columns, rows = _read_csv_rows(csv_path)
    index_by_column = _column_indices(columns, schema)
    series_payloads = _collect_series(rows, schema, index_by_column)

    png_path = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else Path(output_dir).expanduser().resolve() / f"{csv_path.stem}.png"
    )
    png_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    fig, ax = plt.subplots(figsize=(12, 6), dpi=160)
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4"])
    all_times: list[float] = []
    all_values: list[float] = []

    for index, payload in enumerate(series_payloads):
        color = color_cycle[index % len(color_cycle)]
        time = payload["time"]
        value = payload["value"]
        error = payload["error"]
        ax.scatter(
            time,
            value,
            s=14,
            alpha=0.82,
            linewidths=0,
            color=color,
            label=payload["label"],
        )
        if error is not None:
            error_mask = np.isfinite(error)
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

        all_times.extend(time.tolist())
        all_values.extend(value.tolist())

    x_limits = _limits(all_times)
    y_limits = _limits(all_values)
    if x_limits is not None:
        ax.set_xlim(*x_limits)
    if y_limits is not None:
        ax.set_ylim(*y_limits)

    x_formatter = ScalarFormatter(useOffset=False, useMathText=False)
    x_formatter.set_scientific(False)
    ax.xaxis.set_major_formatter(x_formatter)
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)

    time_unit = schema["time_unit"]
    units = list(dict.fromkeys(payload["unit"] for payload in series_payloads))
    y_unit = units[0] if len(units) == 1 else "photometry"
    ax.set_xlabel(f"Time ({time_unit})")
    ax.set_ylabel(_axis_label(y_unit))
    ax.set_title(title or csv_path.stem)
    ax.grid(True, which="major", alpha=0.22, linewidth=0.6)
    ax.minorticks_on()
    ax.tick_params(which="both", direction="in", top=True, right=True)

    if any(payload["unit"] == "mag" for payload in series_payloads):
        ax.invert_yaxis()

    if len(series_payloads) > 1:
        ax.legend(loc="best", fontsize=8, frameon=False)

    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    return png_path


plot_light_curve = plot_whole_light_curve


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
    parser.add_argument("--title", help="Optional figure title")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    png_path = plot_whole_light_curve(
        args.csv_path,
        args.header_parse_json_results,
        output_dir=args.output_dir,
        output_path=args.output_path,
        title=args.title,
    )
    print(png_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
