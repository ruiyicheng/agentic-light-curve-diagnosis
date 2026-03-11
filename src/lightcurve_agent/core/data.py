"""Light curve data loading, schema detection, and metadata extraction."""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord

from lightcurve_agent.core.models import LightCurve, LightCurveFileProfile

COMMENT_PREFIXES = ("#", ";", "%", "//")
NAME_KEYS = ("name", "object", "object_name", "source", "target", "designation")
RA_KEYS = ("ra", "raj2000", "ra_deg", "ra_j2000", "alpha")
DEC_KEYS = ("dec", "dej2000", "dec_deg", "dec_j2000", "delta")
COORD_KEYS = ("coord", "coords", "coordinate", "coordinates", "skycoord")

TIME_ALIASES = (
    "time",
    "jd",
    "mjd",
    "hjd",
    "bjd",
    "phase_time",
    "date",
)
MAG_ALIASES = (
    "mag",
    "magnitude",
    "mag_auto",
    "magaper",
    "brightness",
    "intensity",
)
FLUX_ALIASES = (
    "flux",
    "flux_density",
    "counts",
    "adu",
    "electron_flux",
)
ERR_ALIASES = (
    "mag_err",
    "magerr",
    "flux_err",
    "fluxerr",
    "error",
    "err",
    "sigma",
)
BAND_ALIASES = (
    "filter",
    "band",
    "passband",
    "fid",
    "flt",
)


def _strip_comment_prefix(line: str) -> str:
    for prefix in COMMENT_PREFIXES:
        if line.startswith(prefix):
            return line[len(prefix):].strip()
    return line.strip()


def _split_raw_lines(path: Path) -> tuple[list[str], list[str]]:
    comment_lines: list[str] = []
    data_lines: list[str] = []

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if not stripped:
                continue
            if any(stripped.startswith(prefix) for prefix in COMMENT_PREFIXES):
                comment_lines.append(_strip_comment_prefix(stripped))
            else:
                data_lines.append(raw_line)

    return comment_lines, data_lines


def _sniff_delimiter(data_lines: list[str]) -> str:
    sample = "".join(data_lines[:10]).strip()
    if not sample:
        return ","
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;| ")
        return dialect.delimiter
    except csv.Error:
        return ","


def _read_table(path: Path) -> tuple[pd.DataFrame, str, dict[str, str]]:
    comment_lines, data_lines = _split_raw_lines(path)
    if not data_lines:
        raise ValueError(f"No tabular data found in {path}")

    delimiter = _sniff_delimiter(data_lines)
    buffer = io.StringIO("".join(data_lines))
    df = pd.read_csv(buffer, sep=delimiter, engine="python")
    df = df.rename(columns=lambda col: str(col).strip())
    metadata = _parse_metadata(comment_lines)
    return df, delimiter, metadata


def _normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _parse_metadata(comment_lines: list[str]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for line in comment_lines:
        if ":" in line:
            key, value = line.split(":", 1)
        elif "=" in line:
            key, value = line.split("=", 1)
        else:
            continue
        normalized = _normalize_key(key)
        cleaned = value.strip()
        if normalized and cleaned:
            metadata[normalized] = cleaned
    return metadata


def _find_matching_column(
    df: pd.DataFrame,
    aliases: tuple[str, ...],
) -> str | None:
    normalized = {_normalize_key(column): column for column in df.columns}
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]
    return None


def infer_scale_from_column(column_name: str) -> Literal["mag", "flux"]:
    """Infer whether a data column is magnitude-like or flux-like."""
    normalized = _normalize_key(column_name)
    if normalized in {_normalize_key(alias) for alias in FLUX_ALIASES}:
        return "flux"
    return "mag"


def detect_columns(
    df: pd.DataFrame,
    *,
    scale: Literal["mag", "flux", "auto"] = "auto",
) -> tuple[str, str, str | None, str | None, Literal["mag", "flux"]]:
    """Auto-detect column names from common light-curve conventions."""
    time_col = _find_matching_column(df, TIME_ALIASES)
    if time_col is None:
        raise ValueError(f"Could not auto-detect a time column from: {list(df.columns)}")

    mag_col: str | None = None
    inferred_scale: Literal["mag", "flux"] = "mag"

    if scale in {"mag", "auto"}:
        mag_col = _find_matching_column(df, MAG_ALIASES)
        if mag_col is not None:
            inferred_scale = "mag"

    if mag_col is None and scale in {"flux", "auto"}:
        mag_col = _find_matching_column(df, FLUX_ALIASES)
        if mag_col is not None:
            inferred_scale = "flux"

    if mag_col is None:
        raise ValueError(f"Could not auto-detect a magnitude/flux column from: {list(df.columns)}")

    err_col = _find_matching_column(df, ERR_ALIASES)
    band_col = _find_matching_column(df, BAND_ALIASES)

    if scale == "flux" and infer_scale_from_column(mag_col) == "mag":
        flux_candidate = _find_matching_column(df, FLUX_ALIASES)
        if flux_candidate is not None:
            mag_col = flux_candidate
            inferred_scale = "flux"
    elif scale == "mag" and infer_scale_from_column(mag_col) == "flux":
        mag_candidate = _find_matching_column(df, MAG_ALIASES)
        if mag_candidate is not None:
            mag_col = mag_candidate
            inferred_scale = "mag"

    return time_col, mag_col, err_col, band_col, inferred_scale


def _series_single_value(series: pd.Series) -> str | None:
    cleaned = series.dropna().astype(str).str.strip()
    if cleaned.empty:
        return None
    unique = cleaned.unique()
    if len(unique) == 1:
        return str(unique[0])
    return None


def _parse_coordinates(
    *,
    coordinate_text: str | None = None,
    ra_value: str | float | None = None,
    dec_value: str | float | None = None,
) -> tuple[float | None, float | None]:
    if coordinate_text:
        try:
            coord = SkyCoord(coordinate_text, unit=(u.hourangle, u.deg), frame="icrs")
        except ValueError:
            try:
                coord = SkyCoord(coordinate_text, unit=(u.deg, u.deg), frame="icrs")
            except ValueError:
                return None, None
        return float(coord.ra.deg), float(coord.dec.deg)

    if ra_value is None or dec_value is None:
        return None, None

    try:
        return float(ra_value), float(dec_value)
    except (TypeError, ValueError):
        try:
            coord = SkyCoord(str(ra_value), str(dec_value), unit=(u.hourangle, u.deg), frame="icrs")
        except ValueError:
            try:
                coord = SkyCoord(str(ra_value), str(dec_value), unit=(u.deg, u.deg), frame="icrs")
            except ValueError:
                return None, None
        return float(coord.ra.deg), float(coord.dec.deg)


def infer_source_context(
    df: pd.DataFrame,
    metadata: dict[str, str],
) -> tuple[str | None, float | None, float | None]:
    """Infer source name and coordinates from metadata or constant-value columns."""
    source_name = next((metadata[key] for key in NAME_KEYS if key in metadata), None)
    coord_text = next((metadata[key] for key in COORD_KEYS if key in metadata), None)
    ra_value = next((metadata[key] for key in RA_KEYS if key in metadata), None)
    dec_value = next((metadata[key] for key in DEC_KEYS if key in metadata), None)

    if source_name is None:
        for key in NAME_KEYS:
            column = _find_matching_column(df, (key,))
            if column:
                source_name = _series_single_value(df[column])
                if source_name:
                    break

    if coord_text is None:
        for key in COORD_KEYS:
            column = _find_matching_column(df, (key,))
            if column:
                coord_text = _series_single_value(df[column])
                if coord_text:
                    break

    if ra_value is None:
        for key in RA_KEYS:
            column = _find_matching_column(df, (key,))
            if column:
                ra_value = _series_single_value(df[column])
                if ra_value is not None:
                    break

    if dec_value is None:
        for key in DEC_KEYS:
            column = _find_matching_column(df, (key,))
            if column:
                dec_value = _series_single_value(df[column])
                if dec_value is not None:
                    break

    ra_deg, dec_deg = _parse_coordinates(
        coordinate_text=coord_text,
        ra_value=ra_value,
        dec_value=dec_value,
    )

    return source_name, ra_deg, dec_deg


def inspect_light_curve_file(
    csv_path: str | Path,
    *,
    scale: Literal["mag", "flux", "auto"] = "auto",
) -> LightCurveFileProfile:
    """Inspect a light-curve file and infer schema, scale, and source metadata."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Light curve file not found: {path}")

    df, delimiter, metadata = _read_table(path)
    time_col, mag_col, err_col, band_col, inferred_scale = detect_columns(df, scale=scale)
    source_name, ra_deg, dec_deg = infer_source_context(df, metadata)

    notes: list[str] = []
    if band_col:
        notes.append(f"Detected multi-band light curve in '{band_col}'.")
    if source_name:
        notes.append(f"Detected source name '{source_name}'.")
    if ra_deg is not None and dec_deg is not None:
        notes.append("Detected source coordinates for external catalog lookup.")

    band_labels: list[str] = []
    if band_col:
        band_labels = sorted(df[band_col].dropna().astype(str).unique().tolist())

    return LightCurveFileProfile(
        path=path,
        columns=[str(column) for column in df.columns],
        delimiter=delimiter,
        n_rows=len(df),
        preview_rows=df.head(5).to_dict(orient="records"),
        time_col=time_col,
        mag_col=mag_col,
        err_col=err_col,
        band_col=band_col,
        scale=inferred_scale,
        source_name=source_name,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        metadata=metadata,
        band_labels=band_labels,
        notes=notes,
    )


def load_light_curve(
    csv_path: str | Path,
    *,
    time_col: str | None = None,
    mag_col: str | None = None,
    err_col: str | None = "mag_err",
    band_col: str | None = None,
    band_value: str | None = None,
    scale: Literal["mag", "flux", "auto"] = "auto",
    validate: bool = True,
) -> LightCurve:
    """Load a light curve from a CSV-like file with automatic schema detection."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Light curve file not found: {path}")

    df, _delimiter, metadata = _read_table(path)
    detected = inspect_light_curve_file(path, scale=scale)

    time_col = time_col or detected.time_col
    mag_col = mag_col or detected.mag_col
    err_col = err_col if err_col in df.columns else detected.err_col
    band_col = band_col if band_col in df.columns else detected.band_col
    resolved_scale = infer_scale_from_column(mag_col) if scale == "auto" else scale
    if scale == "auto":
        resolved_scale = detected.scale

    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found. Columns: {list(df.columns)}")
    if mag_col not in df.columns:
        raise ValueError(f"Magnitude/flux column '{mag_col}' not found. Columns: {list(df.columns)}")

    if band_col and band_col not in df.columns:
        raise ValueError(f"Band column '{band_col}' not found. Columns: {list(df.columns)}")

    if band_col and band_value is not None:
        df = df[df[band_col].astype(str) == str(band_value)].copy()
        if df.empty:
            raise ValueError(f"No rows found for band '{band_value}' in column '{band_col}'")

    error = df[err_col] if err_col and err_col in df.columns else None
    band = df[band_col].astype(str) if band_col else None

    if validate:
        mask = np.isfinite(pd.to_numeric(df[time_col], errors="coerce"))
        mask &= np.isfinite(pd.to_numeric(df[mag_col], errors="coerce"))
        if error is not None:
            error_numeric = pd.to_numeric(error, errors="coerce")
            mask &= np.isfinite(error_numeric) & (error_numeric >= 0)
            error = error_numeric

        if not mask.all():
            df = df.loc[mask].copy()
            if error is not None:
                error = error.loc[df.index]
            if band is not None:
                band = band.loc[df.index]

    source_name, ra_deg, dec_deg = infer_source_context(df, metadata)

    return LightCurve(
        time=pd.to_numeric(df[time_col], errors="coerce").reset_index(drop=True),
        magnitude=pd.to_numeric(df[mag_col], errors="coerce").reset_index(drop=True),
        error=pd.to_numeric(error, errors="coerce").reset_index(drop=True) if error is not None else None,
        band=band.reset_index(drop=True) if band is not None else None,
        time_col=time_col,
        mag_col=mag_col,
        err_col=err_col or "",
        band_col=band_col or "",
        scale=resolved_scale,
        source_name=source_name,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        metadata=metadata,
    )


def normalize_multiband_light_curve(lc: LightCurve) -> LightCurve:
    """Remove per-band offsets before joint period analysis."""
    if lc.band is None or not lc.band_labels:
        return lc

    values = pd.Series(lc.magnitude, copy=True)
    global_median = float(np.nanmedian(values))
    band_series = pd.Series(lc.band).astype(str)

    for band in lc.band_labels:
        mask = band_series == band
        band_values = values.loc[mask]
        if band_values.empty:
            continue
        values.loc[mask] = band_values - float(np.nanmedian(band_values)) + global_median

    return LightCurve(
        time=lc.time.reset_index(drop=True),
        magnitude=values.reset_index(drop=True),
        error=lc.error.reset_index(drop=True) if lc.error is not None else None,
        band=lc.band.reset_index(drop=True) if lc.band is not None else None,
        time_col=lc.time_col,
        mag_col=lc.mag_col,
        err_col=lc.err_col,
        band_col=lc.band_col,
        scale=lc.scale,
        source_name=lc.source_name,
        ra_deg=lc.ra_deg,
        dec_deg=lc.dec_deg,
        metadata=lc.metadata.copy(),
    )


def auto_load_light_curve(
    csv_path: str | Path,
    scale: Literal["mag", "flux", "auto"] = "auto",
) -> LightCurve:
    """Load a light curve with automatic column detection."""
    return load_light_curve(csv_path, scale=scale)
