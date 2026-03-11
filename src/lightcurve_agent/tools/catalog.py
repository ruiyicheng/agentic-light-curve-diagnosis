"""Schema inspection and external catalog lookup tools."""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import tool

from lightcurve_agent.core.data import inspect_light_curve_file
from lightcurve_agent.interfaces.catalogs import lookup_source_catalogs


@tool
def inspect_light_curve_schema(
    csv_path: str,
    preferred_scale: str = "auto",
) -> dict:
    """Inspect a light-curve file head, infer columns, and extract source metadata."""
    try:
        profile = inspect_light_curve_file(csv_path, scale=preferred_scale)
        return {
            "path": str(profile.path),
            "columns": profile.columns,
            "delimiter": profile.delimiter,
            "n_rows": profile.n_rows,
            "preview_rows": profile.preview_rows,
            "detected_columns": {
                "time_col": profile.time_col,
                "y_col": profile.mag_col,
                "yerr_col": profile.err_col,
                "band_col": profile.band_col,
            },
            "scale": profile.scale,
            "source_context": {
                "source_name": profile.source_name,
                "ra_deg": profile.ra_deg,
                "dec_deg": profile.dec_deg,
            },
            "band_labels": profile.band_labels,
            "metadata": profile.metadata,
            "notes": profile.notes,
        }
    except Exception as exc:
        return {"error": str(exc)}


@tool
def lookup_source_context(
    csv_path: str | None = None,
    source_name: str | None = None,
    coordinate: str | None = None,
    ra_deg: float | None = None,
    dec_deg: float | None = None,
    radius_arcsec: float = 5.0,
) -> dict:
    """Resolve source coordinates and query VSX followed by Gaia."""
    try:
        if csv_path:
            profile = inspect_light_curve_file(Path(csv_path))
            source_name = source_name or profile.source_name
            ra_deg = ra_deg if ra_deg is not None else profile.ra_deg
            dec_deg = dec_deg if dec_deg is not None else profile.dec_deg

        result = lookup_source_catalogs(
            source_name=source_name,
            coordinate=coordinate,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            radius_arcsec=radius_arcsec,
        )
        return result
    except ModuleNotFoundError as exc:
        return {
            "error": (
                f"{exc.name} is not installed. Install astroquery to enable VSX and Gaia lookup."
            )
        }
    except Exception as exc:
        return {"error": str(exc)}
