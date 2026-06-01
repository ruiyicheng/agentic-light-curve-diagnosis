"""Build the cached Gaia HR diagram density grid used by query_catalog.py.

This is a one-off/offline step. Runtime catalog lookup only reads the saved
``assets/gaia_hr_diagram/gaia_hr_density_grid.npz`` file and plots the target
star on top of it.
"""

from __future__ import annotations

import argparse
import csv
import json
import signal
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.query_catalog import (  # noqa: E402
    DEFAULT_GAIA_HR_GRID_PATH,
    DEFAULT_GAIA_TAP_URLS,
    calculate_gaia_absolute_g_mag,
)


def _to_float_array(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _run_tap_async_with_timeout(
    service: Any,
    query: str,
    *,
    maxrec: int,
    timeout_seconds: float,
) -> Any:
    def _handle_timeout(signum: int, frame: Any) -> None:
        raise TimeoutError(f"Gaia TAP query exceeded {timeout_seconds:g}s")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, _handle_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
    try:
        return service.run_async(query, maxrec=maxrec, timeout=timeout_seconds).to_table()
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def _download_gaia_sample(
    *,
    sample_size: int,
    tap_url: str | Iterable[str],
    random_index_start: int,
    chunk_size: int,
    random_index_window_size: int | None,
    chunk_retries: int,
    retry_delay_seconds: float,
    chunk_timeout_seconds: float,
    bp_rp_range: tuple[float, float],
    mg_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    from pyvo.dal import TAPService

    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if random_index_window_size is None:
        random_index_window_size = chunk_size * 2
    if random_index_window_size <= 0:
        raise ValueError("random_index_window_size must be positive")
    if chunk_retries < 0:
        raise ValueError("chunk_retries must be non-negative")
    if retry_delay_seconds < 0:
        raise ValueError("retry_delay_seconds must be non-negative")
    if chunk_timeout_seconds <= 0:
        raise ValueError("chunk_timeout_seconds must be positive")

    tap_urls = _parse_tap_urls(tap_url)
    bp_rp_min, bp_rp_max = bp_rp_range
    mg_min, mg_max = mg_range
    bp_rp_chunks: list[np.ndarray] = []
    g_mag_chunks: list[np.ndarray] = []
    parallax_chunks: list[np.ndarray] = []
    next_random_index = int(random_index_start)

    while sum(chunk.size for chunk in bp_rp_chunks) < sample_size:
        rows_remaining = sample_size - sum(chunk.size for chunk in bp_rp_chunks)
        current_chunk_size = min(chunk_size, rows_remaining)
        window_end = next_random_index + random_index_window_size
        query = f"""
SELECT TOP {current_chunk_size}
    random_index,
    phot_g_mean_mag,
    bp_rp,
    parallax
FROM gaiadr3.gaia_source
WHERE phot_g_mean_mag IS NOT NULL
    AND bp_rp IS NOT NULL
    AND parallax IS NOT NULL
    AND parallax > 0
    AND parallax_over_error > 10
    AND ruwe < 1.4
    AND visibility_periods_used >= 8
    AND bp_rp BETWEEN {bp_rp_min:.6f} AND {bp_rp_max:.6f}
    AND (phot_g_mean_mag + 5 * LOG10(parallax) - 10) BETWEEN {mg_min:.6f} AND {mg_max:.6f}
    AND random_index >= {next_random_index}
    AND random_index < {window_end}
"""
        table = None
        errors: list[str] = []
        for attempt in range(chunk_retries + 1):
            errors = []
            for url in tap_urls:
                try:
                    table = _run_tap_async_with_timeout(
                        TAPService(url),
                        query,
                        maxrec=current_chunk_size,
                        timeout_seconds=chunk_timeout_seconds,
                    )
                    break
                except Exception as exc:
                    errors.append(f"{url}: {exc}")
            if table is not None:
                break
            if attempt < chunk_retries:
                print(
                    "Gaia TAP chunk failed; "
                    f"retrying {attempt + 1}/{chunk_retries} after {retry_delay_seconds:g}s",
                    file=sys.stderr,
                )
                time.sleep(retry_delay_seconds)
        if table is None:
            raise RuntimeError(
                "Gaia TAP query failed for all endpoints while building the HR grid. "
                "Try a smaller --chunk-size or --sample-size. Last errors: "
                + "; ".join(errors)
            )
        if len(table) == 0:
            next_random_index = window_end
            continue

        bp_rp_chunks.append(_to_float_array(table["bp_rp"]))
        g_mag_chunks.append(_to_float_array(table["phot_g_mean_mag"]))
        parallax_chunks.append(_to_float_array(table["parallax"]))
        next_random_index = window_end
        downloaded = sum(chunk.size for chunk in bp_rp_chunks)
        print(
            f"Downloaded {downloaded}/{sample_size} Gaia rows "
            f"(next random_index >= {next_random_index})",
            file=sys.stderr,
        )

    bp_rp = np.concatenate(bp_rp_chunks)
    g_mag = np.concatenate(g_mag_chunks)
    parallax = np.concatenate(parallax_chunks)
    mg = g_mag + 5.0 * np.log10(parallax) - 10.0
    return bp_rp, mg


def _parse_tap_urls(tap_url: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(tap_url, str):
        urls = [url.strip() for url in tap_url.split(",")]
    else:
        urls = [str(url).strip() for url in tap_url]
    urls = [url for url in urls if url]
    if not urls:
        raise ValueError("At least one Gaia TAP URL is required")
    return tuple(urls)


def _load_sample_csv(
    input_csv: str | Path,
    *,
    bp_rp_column: str,
    g_mag_column: str,
    parallax_column: str,
) -> tuple[np.ndarray, np.ndarray]:
    bp_rp_values: list[float] = []
    g_mag_values: list[float] = []
    parallax_values: list[float] = []
    with Path(input_csv).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        missing = [
            column
            for column in (bp_rp_column, g_mag_column, parallax_column)
            if column not in fieldnames
        ]
        if missing:
            raise ValueError(f"Input CSV is missing columns: {missing}")
        for row in reader:
            for values, column in (
                (bp_rp_values, bp_rp_column),
                (g_mag_values, g_mag_column),
                (parallax_values, parallax_column),
            ):
                try:
                    values.append(float(row[column]))
                except (TypeError, ValueError):
                    values.append(float("nan"))

    bp_rp = _to_float_array(bp_rp_values)
    g_mag = _to_float_array(g_mag_values)
    parallax = _to_float_array(parallax_values)
    mg = np.asarray(
        [
            calculate_gaia_absolute_g_mag(g_value, parallax_value)
            for g_value, parallax_value in zip(g_mag, parallax, strict=False)
        ],
        dtype=float,
    )
    return bp_rp, mg


def _filter_hr_points(
    bp_rp: np.ndarray,
    mg: np.ndarray,
    *,
    bp_rp_range: tuple[float, float],
    mg_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    mask = (
        np.isfinite(bp_rp)
        & np.isfinite(mg)
        & (bp_rp >= bp_rp_range[0])
        & (bp_rp <= bp_rp_range[1])
        & (mg >= mg_range[0])
        & (mg <= mg_range[1])
    )
    return bp_rp[mask], mg[mask]


def _save_preview_plot(
    output_png: str | Path,
    *,
    bp_rp_edges: np.ndarray,
    mg_edges: np.ndarray,
    density: np.ndarray,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    finite_positive = density[np.isfinite(density) & (density > 0)]
    norm = LogNorm(vmin=max(float(finite_positive.min()), 1.0), vmax=float(finite_positive.max())) if finite_positive.size else None

    fig, ax = plt.subplots(figsize=(8, 8.5), dpi=160)
    mesh = ax.pcolormesh(
        bp_rp_edges,
        mg_edges,
        density.T,
        cmap="magma",
        shading="auto",
        norm=norm,
    )
    fig.colorbar(mesh, ax=ax, label="Gaia DR3 density per grid cell")
    ax.set_xlabel("Gaia BP - RP")
    ax.set_ylabel("Gaia absolute G magnitude (M_G)")
    ax.set_ylim(float(mg_edges[-1]), float(mg_edges[0]))
    ax.set_title("Cached Gaia HR density grid")
    fig.tight_layout()
    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png)
    plt.close(fig)


def build_grid(
    *,
    output_path: str | Path = DEFAULT_GAIA_HR_GRID_PATH,
    input_csv: str | Path | None = None,
    sample_size: int = 1_000_000,
    tap_url: str | Iterable[str] = DEFAULT_GAIA_TAP_URLS,
    random_index_start: int = 0,
    chunk_size: int = 100_000,
    random_index_window_size: int | None = None,
    chunk_retries: int = 3,
    retry_delay_seconds: float = 10.0,
    chunk_timeout_seconds: float = 180.0,
    bp_rp_range: tuple[float, float] = (-0.7, 5.5),
    mg_range: tuple[float, float] = (-8.0, 16.0),
    bp_rp_bins: int = 360,
    mg_bins: int = 360,
    preview_png: str | Path | None = None,
    bp_rp_column: str = "bp_rp",
    g_mag_column: str = "phot_g_mean_mag",
    parallax_column: str = "parallax",
) -> Path:
    if input_csv is None:
        bp_rp, mg = _download_gaia_sample(
            sample_size=sample_size,
            tap_url=tap_url,
            random_index_start=random_index_start,
            chunk_size=chunk_size,
            random_index_window_size=random_index_window_size,
            chunk_retries=chunk_retries,
            retry_delay_seconds=retry_delay_seconds,
            chunk_timeout_seconds=chunk_timeout_seconds,
            bp_rp_range=bp_rp_range,
            mg_range=mg_range,
        )
        source = {
            "type": "gaia_tap",
            "tap_url": tap_url,
            "sample_size_requested": sample_size,
            "random_index_start": random_index_start,
            "chunk_size": chunk_size,
            "random_index_window_size": random_index_window_size or chunk_size * 2,
            "chunk_retries": chunk_retries,
            "retry_delay_seconds": retry_delay_seconds,
            "chunk_timeout_seconds": chunk_timeout_seconds,
            "quality_cuts": [
                "parallax > 0",
                "parallax_over_error > 10",
                "ruwe < 1.4",
                "visibility_periods_used >= 8",
                "bp_rp IS NOT NULL",
                "phot_g_mean_mag IS NOT NULL",
            ],
        }
    else:
        bp_rp, mg = _load_sample_csv(
            input_csv,
            bp_rp_column=bp_rp_column,
            g_mag_column=g_mag_column,
            parallax_column=parallax_column,
        )
        source = {"type": "csv", "input_csv": str(input_csv)}

    bp_rp, mg = _filter_hr_points(bp_rp, mg, bp_rp_range=bp_rp_range, mg_range=mg_range)
    if bp_rp.size == 0:
        raise RuntimeError("No finite Gaia HR points remain after filtering")

    density, bp_rp_edges, mg_edges = np.histogram2d(
        bp_rp,
        mg,
        bins=[bp_rp_bins, mg_bins],
        range=[bp_rp_range, mg_range],
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source": source,
        "points_used": int(bp_rp.size),
        "bp_rp_range": list(bp_rp_range),
        "mg_range": list(mg_range),
        "bp_rp_bins": bp_rp_bins,
        "mg_bins": mg_bins,
        "absolute_g_mag_method": "M_G = G + 5*log10(parallax_mas) - 10; no extinction correction",
    }
    np.savez_compressed(
        output_path,
        bp_rp_edges=bp_rp_edges,
        mg_edges=mg_edges,
        density=density,
        metadata_json=json.dumps(metadata),
    )

    if preview_png is not None:
        _save_preview_plot(
            preview_png,
            bp_rp_edges=bp_rp_edges,
            mg_edges=mg_edges,
            density=density,
        )

    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-path", default=str(DEFAULT_GAIA_HR_GRID_PATH))
    parser.add_argument("--input-csv", help="Use an existing Gaia CSV instead of querying TAP")
    parser.add_argument("--sample-size", type=int, default=1_000_000)
    parser.add_argument(
        "--tap-url",
        default=",".join(DEFAULT_GAIA_TAP_URLS),
        help="Gaia TAP URL, or a comma-separated fallback list",
    )
    parser.add_argument("--random-index-start", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument(
        "--random-index-window-size",
        type=int,
        help="Width of each random_index query window; defaults to 2 * --chunk-size",
    )
    parser.add_argument("--chunk-retries", type=int, default=3)
    parser.add_argument("--retry-delay-seconds", type=float, default=10.0)
    parser.add_argument("--chunk-timeout-seconds", type=float, default=180.0)
    parser.add_argument("--bp-rp-min", type=float, default=-0.7)
    parser.add_argument("--bp-rp-max", type=float, default=5.5)
    parser.add_argument("--mg-min", type=float, default=-8.0)
    parser.add_argument("--mg-max", type=float, default=16.0)
    parser.add_argument("--bp-rp-bins", type=int, default=360)
    parser.add_argument("--mg-bins", type=int, default=360)
    parser.add_argument("--preview-png")
    parser.add_argument("--bp-rp-column", default="bp_rp")
    parser.add_argument("--g-mag-column", default="phot_g_mean_mag")
    parser.add_argument("--parallax-column", default="parallax")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output_path = build_grid(
        output_path=args.output_path,
        input_csv=args.input_csv,
        sample_size=args.sample_size,
        tap_url=args.tap_url,
        random_index_start=args.random_index_start,
        chunk_size=args.chunk_size,
        random_index_window_size=args.random_index_window_size,
        chunk_retries=args.chunk_retries,
        retry_delay_seconds=args.retry_delay_seconds,
        chunk_timeout_seconds=args.chunk_timeout_seconds,
        bp_rp_range=(args.bp_rp_min, args.bp_rp_max),
        mg_range=(args.mg_min, args.mg_max),
        bp_rp_bins=args.bp_rp_bins,
        mg_bins=args.mg_bins,
        preview_png=args.preview_png,
        bp_rp_column=args.bp_rp_column,
        g_mag_column=args.g_mag_column,
        parallax_column=args.parallax_column,
    )
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
