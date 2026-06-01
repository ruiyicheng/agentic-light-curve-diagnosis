"""External catalog lookup helpers for VSX, Gaia, and LAMOST."""

from __future__ import annotations

import contextlib
import csv
import importlib
import io
import json
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

DEFAULT_GAIA_TAP_URLS = (
    "https://gaia.aip.de/tap",
    "https://gaia.ari.uni-heidelberg.de/tap",
)
GAIA_MAG_ERROR_COEFFICIENT = 2.5 / np.log(10.0)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LAMOST_SPECTROGRAPH_DIR = PROJECT_ROOT / "assets" / "spectrograph"
DEFAULT_GAIA_HR_DIAGRAM_DIR = PROJECT_ROOT / "assets" / "gaia_hr_diagram"
DEFAULT_GAIA_HR_GRID_PATH = DEFAULT_GAIA_HR_DIAGRAM_DIR / "gaia_hr_density_grid.npz"
LAMOST_SPECTRAL_LINES: tuple[tuple[str, float], ...] = (
    ("[O II] 3727", 3727.09),
    ("[O II] 3729", 3729.88),

    ("Ca II K", 3934.77),
    ("Ca II H", 3969.59),
    ("H-delta", 4102.90),
    ("H-gamma", 4341.69),
    ("He I", 4472.74),
    ("H-beta", 4862.69),

    ("[O III]", 4960.30),
    ("[O III]", 5008.24),

    ("Mg I b", 5176.4),   # approximate blend marker
    ("He I", 5877.25),
    ("Na D", 5894.58),    # approximate doublet/blend marker

    ("[O I]", 6302.05),
    ("[N II]", 6549.86),
    ("H-alpha", 6564.62),
    ("[N II]", 6585.27),
    ("He I", 6680.00),
    ("[S II]", 6718.30),
    ("[S II]", 6732.68),

    ("Ca II", 8500.36),
    ("Ca II", 8544.44),
    ("Ca II", 8664.52),
)

__all__ = [
    "CatalogQuery",
    "calculate_gaia_absolute_g_mag",
    "fetch_gaia_astrometry",
    "fetch_lamost_spectrum",
    "lookup_source_catalogs",
    "plot_gaia_hr_diagram",
    "plot_lamost_spectrum",
    "resolve_source_coordinates",
    "search_vsx",
]


def _import_astroquery_module(module_name: str) -> Any:
    module = importlib.import_module(module_name)
    return module


def _serialize_row(row: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key in row.colnames:
        value = row[key]
        if np.ma.is_masked(value):
            value = None
        if hasattr(value, "item"):
            try:
                value = value.item()
            except ValueError:
                pass
        payload[str(key)] = value
    return payload


def _gaia_tap_urls(configured_urls: str | Iterable[str] | None = None) -> list[str]:
    if configured_urls is not None:
        if isinstance(configured_urls, str):
            configured_urls = configured_urls.split(",")
        urls = []
        for url in configured_urls:
            url_text = str(url).strip()
            if url_text:
                urls.append(url_text)
        if urls:
            return urls

    configured_urls = os.environ.get("GAIA_TAP_URL")
    if configured_urls:
        urls = [url.strip() for url in configured_urls.split(",") if url.strip()]
        if urls:
            return urls
    return list(DEFAULT_GAIA_TAP_URLS)


def _run_gaia_tap_query(query: str, *, tap_urls: str | Iterable[str] | None = None) -> Any:
    TAPService = importlib.import_module("pyvo.dal").TAPService
    errors: list[str] = []

    for tap_url in _gaia_tap_urls(tap_urls):
        try:
            return TAPService(tap_url).run_sync(query, maxrec=1).to_table()
        except Exception as exc:
            errors.append(f"{tap_url}: {exc}")

    raise RuntimeError("Gaia TAP query failed for all endpoints: " + "; ".join(errors))


def _to_float(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return value


def calculate_gaia_absolute_g_mag(
    phot_g_mean_mag: Any,
    parallax_mas: Any,
) -> float | None:
    """Calculate Gaia absolute G magnitude from apparent G and parallax in mas."""
    g_mag = _to_float(phot_g_mean_mag)
    parallax = _to_float(parallax_mas)
    if g_mag is None or parallax is None or parallax <= 0:
        return None
    return float(g_mag + 5.0 * np.log10(parallax) - 10.0)


def _gaia_mag_error_from_flux_over_error(value: Any) -> float | None:
    flux_over_error = _to_float(value)
    if flux_over_error is None or flux_over_error <= 0:
        return None
    return float(GAIA_MAG_ERROR_COEFFICIENT / flux_over_error)


def _quadrature(*values: float | None) -> float | None:
    known_values = [value for value in values if value is not None]
    if not known_values:
        return None
    return float(np.sqrt(sum(value * value for value in known_values)))


def calculate_gaia_absolute_g_mag_error(
    *,
    phot_g_mean_flux_over_error: Any,
    parallax_mas: Any,
    parallax_error_mas: Any,
) -> float | None:
    """Return the formal uncertainty on M_G from Gaia G flux S/N and parallax."""
    g_mag_error = _gaia_mag_error_from_flux_over_error(phot_g_mean_flux_over_error)
    parallax = _to_float(parallax_mas)
    parallax_error = _to_float(parallax_error_mas)
    parallax_mag_error = None
    if parallax is not None and parallax > 0 and parallax_error is not None and parallax_error >= 0:
        parallax_mag_error = float((5.0 / np.log(10.0)) * parallax_error / parallax)
    return _quadrature(g_mag_error, parallax_mag_error)


def _gaia_bp_rp(payload: dict[str, Any]) -> float | None:
    bp_rp = _to_float(payload.get("bp_rp"))
    if bp_rp is not None:
        return bp_rp

    bp_mag = _to_float(payload.get("phot_bp_mean_mag"))
    rp_mag = _to_float(payload.get("phot_rp_mean_mag"))
    if bp_mag is None or rp_mag is None:
        return None
    return float(bp_mag - rp_mag)


def _gaia_bp_rp_error(payload: dict[str, Any]) -> float | None:
    bp_mag_error = _gaia_mag_error_from_flux_over_error(
        payload.get("phot_bp_mean_flux_over_error")
    )
    rp_mag_error = _gaia_mag_error_from_flux_over_error(
        payload.get("phot_rp_mean_flux_over_error")
    )
    if bp_mag_error is None or rp_mag_error is None:
        return None
    return _quadrature(bp_mag_error, rp_mag_error)


def _gaia_hr_grid_path(configured_path: str | Path | None = None) -> Path:
    if configured_path is not None:
        return Path(configured_path).expanduser()
    env_path = os.environ.get("GAIA_HR_GRID_PATH")
    if env_path:
        return Path(env_path).expanduser()
    return DEFAULT_GAIA_HR_GRID_PATH


def _gaia_hr_output_path(coord: SkyCoord, output_dir: str | Path | None = None) -> Path:
    destination_dir = Path(output_dir) if output_dir is not None else DEFAULT_GAIA_HR_DIAGRAM_DIR
    return destination_dir / f"{coord.ra.deg:.8f}_{coord.dec.deg:.8f}_gaia_hr.png"


def _load_gaia_hr_density_grid(grid_path: str | Path | None = None) -> dict[str, Any]:
    path = _gaia_hr_grid_path(grid_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Gaia HR density grid not found at {path}. "
            "Build it once with tools/build_gaia_hr_grid.py."
        )

    with np.load(path, allow_pickle=True) as grid:
        payload = {key: grid[key] for key in grid.files}
    if "bp_rp_edges" not in payload or "mg_edges" not in payload or "density" not in payload:
        raise ValueError(f"Gaia HR density grid has unexpected keys: {sorted(payload)}")
    return payload


def plot_gaia_hr_diagram(
    gaia_payload: dict[str, Any],
    *,
    output_path: str | Path,
    grid_path: str | Path | None = None,
    title: str | None = None,
) -> Path:
    """Plot a Gaia HR density grid and mark one target's BP-RP and M_G."""
    bp_rp = _gaia_bp_rp(gaia_payload)
    absolute_g_mag = _to_float(gaia_payload.get("absolute_g_mag"))
    if bp_rp is None or absolute_g_mag is None:
        raise ValueError("Gaia HR plot requires bp_rp and absolute_g_mag")
    target = gaia_payload.get("hr_diagram_coordinates")
    bp_rp_error = None
    absolute_g_mag_error = None
    if isinstance(target, dict):
        bp_rp_error = _to_float(target.get("bp_rp_error"))
        absolute_g_mag_error = _to_float(target.get("absolute_g_mag_error"))
    if bp_rp_error is None:
        bp_rp_error = _gaia_bp_rp_error(gaia_payload)
    if absolute_g_mag_error is None:
        absolute_g_mag_error = calculate_gaia_absolute_g_mag_error(
            phot_g_mean_flux_over_error=gaia_payload.get("phot_g_mean_flux_over_error"),
            parallax_mas=gaia_payload.get("parallax"),
            parallax_error_mas=gaia_payload.get("parallax_error"),
        )

    grid = _load_gaia_hr_density_grid(grid_path)
    bp_rp_edges = np.asarray(grid["bp_rp_edges"], dtype=float)
    mg_edges = np.asarray(grid["mg_edges"], dtype=float)
    density = np.asarray(grid["density"], dtype=float)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    finite_positive = density[np.isfinite(density) & (density > 0)]
    norm = LogNorm(vmin=max(float(finite_positive.min()), 1.0), vmax=float(finite_positive.max())) if finite_positive.size else None

    png_path = Path(output_path)
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

    if bp_rp_error is not None or absolute_g_mag_error is not None:
        ax.errorbar(
            [bp_rp],
            [absolute_g_mag],
            xerr=[bp_rp_error] if bp_rp_error is not None else None,
            yerr=[absolute_g_mag_error] if absolute_g_mag_error is not None else None,
            fmt="none",
            ecolor="black",
            elinewidth=1.2,
            capsize=3,
            zorder=3,
            label="Target uncertainty",
        )
    ax.scatter(
        [bp_rp],
        [absolute_g_mag],
        s=95,
        marker="*",
        color="#00bcd4",
        edgecolor="black",
        linewidth=0.8,
        zorder=4,
        label="Target",
    )
    bp_rp_text = f"BP-RP = {bp_rp:.3f}"
    if bp_rp_error is not None:
        bp_rp_text += f" +/- {bp_rp_error:.3f}"
    mg_text = f"M_G = {absolute_g_mag:.3f}"
    if absolute_g_mag_error is not None:
        mg_text += f" +/- {absolute_g_mag_error:.3f}"
    annotation = f"{bp_rp_text}\n{mg_text}"
    ax.annotate(
        annotation,
        xy=(bp_rp, absolute_g_mag),
        xytext=(12, 12),
        textcoords="offset points",
        fontsize=9,
        color="black",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.86, "edgecolor": "none"},
    )

    ax.set_xlabel("Gaia BP - RP")
    ax.set_ylabel("Gaia absolute G magnitude (M_G)")
    ax.set_title(title or "Gaia HR diagram")
    bp_rp_margin = max(0.2, 1.2 * bp_rp_error if bp_rp_error is not None else 0.0)
    mg_margin = max(
        0.8,
        1.2 * absolute_g_mag_error if absolute_g_mag_error is not None else 0.0,
    )
    ax.set_xlim(
        min(float(bp_rp_edges[0]), bp_rp - bp_rp_margin),
        max(float(bp_rp_edges[-1]), bp_rp + bp_rp_margin),
    )
    ax.set_ylim(
        max(float(mg_edges[-1]), absolute_g_mag + mg_margin),
        min(float(mg_edges[0]), absolute_g_mag - mg_margin),
    )
    ax.grid(True, color="white", linewidth=0.35, alpha=0.25)
    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9, edgecolor="none")
    fig.tight_layout()

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path)
    plt.close(fig)
    return png_path


def _truthy_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() not in {"", "0", "false", "none", "nan"}


def _max_known(*values: float | None) -> float | None:
    known = [value for value in values if value is not None]
    return max(known) if known else None


def _enrich_gaia_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Add diagnosis-oriented summaries to a Gaia row payload."""
    filtered = payload.copy()
    absolute_g_mag = calculate_gaia_absolute_g_mag(
        payload.get("phot_g_mean_mag"),
        payload.get("parallax"),
    )
    absolute_g_mag_error = calculate_gaia_absolute_g_mag_error(
        phot_g_mean_flux_over_error=payload.get("phot_g_mean_flux_over_error"),
        parallax_mas=payload.get("parallax"),
        parallax_error_mas=payload.get("parallax_error"),
    )
    bp_rp = _gaia_bp_rp(payload)
    bp_rp_error = _gaia_bp_rp_error(payload)
    g_mag_error = _gaia_mag_error_from_flux_over_error(
        payload.get("phot_g_mean_flux_over_error")
    )
    bp_mag_error = _gaia_mag_error_from_flux_over_error(
        payload.get("phot_bp_mean_flux_over_error")
    )
    rp_mag_error = _gaia_mag_error_from_flux_over_error(
        payload.get("phot_rp_mean_flux_over_error")
    )
    parallax = _to_float(payload.get("parallax"))
    parallax_error = _to_float(payload.get("parallax_error"))

    if absolute_g_mag is not None:
        filtered["absolute_g_mag"] = absolute_g_mag
    if absolute_g_mag_error is not None:
        filtered["absolute_g_mag_error"] = absolute_g_mag_error
    if g_mag_error is not None:
        filtered["phot_g_mean_mag_error"] = g_mag_error
    if bp_mag_error is not None:
        filtered["phot_bp_mean_mag_error"] = bp_mag_error
    if rp_mag_error is not None:
        filtered["phot_rp_mean_mag_error"] = rp_mag_error
    if bp_rp_error is not None:
        filtered["bp_rp_error"] = bp_rp_error
    if bp_rp is not None or absolute_g_mag is not None:
        filtered["hr_diagram_coordinates"] = {
            "bp_rp": bp_rp,
            "bp_rp_error": bp_rp_error,
            "absolute_g_mag": absolute_g_mag,
            "absolute_g_mag_error": absolute_g_mag_error,
            "absolute_g_mag_method": (
                "M_G = G + 5*log10(parallax_mas) - 10; no extinction correction"
                if absolute_g_mag is not None
                else None
            ),
            "uncertainty_method": (
                "Formal 1-sigma uncertainties from Gaia flux_over_error values "
                "and parallax_error; no extinction uncertainty included"
            ),
            "parallax_over_error": (
                float(parallax / parallax_error)
                if parallax is not None and parallax_error is not None and parallax_error > 0
                else None
            ),
        }

    stellar_parameters = {
        key: payload[key]
        for key in (
            "teff_gspphot",
            "teff_gspspec",
            "logg_gspphot",
            "logg_gspspec",
            "mh_gspphot",
            "distance_gspphot",
            "azero_gspphot",
            "ag_gspphot",
            "ebpminrp_gspphot",
        )
        if key in payload
    }

    classification_probabilities = {
        "star": _to_float(payload.get("classprob_dsc_combmod_star")),
        "binary_star": _to_float(payload.get("classprob_dsc_combmod_binarystar")),
        "galaxy": _to_float(payload.get("classprob_dsc_combmod_galaxy")),
        "quasar": _to_float(payload.get("classprob_dsc_combmod_quasar")),
    }
    classification_probabilities = {
        key: value for key, value in classification_probabilities.items() if value is not None
    }

    galaxy_prob = classification_probabilities.get("galaxy")
    binary_prob = classification_probabilities.get("binary_star")
    star_prob = classification_probabilities.get("star")
    quasar_prob = classification_probabilities.get("quasar")

    nss_indicator = any(
        _truthy_flag(payload.get(key))
        for key in (
            "non_single_star",
            "has_nss",
            "in_nss",
            "nss_solution_type",
        )
        if key in payload
    )

    max_non_galaxy = _max_known(star_prob, binary_prob, quasar_prob)
    max_non_star = _max_known(galaxy_prob, quasar_prob)

    filtered["stellar_parameters"] = stellar_parameters
    filtered["classification_probabilities"] = classification_probabilities
    filtered["diagnosis_hints"] = {
        "is_probable_galaxy": bool(
            galaxy_prob is not None
            and galaxy_prob >= 0.5
            and (max_non_galaxy is None or galaxy_prob >= max_non_galaxy)
        ),
        "is_probable_binary": bool(
            nss_indicator
            or (binary_prob is not None and binary_prob >= 0.3)
        ),
        "is_probable_star": bool(
            star_prob is not None
            and star_prob >= 0.5
            and (max_non_star is None or star_prob >= max_non_star)
        ),
    }

    return filtered


def _parse_j_designation_coordinate(source_name: str) -> SkyCoord | None:
    """Parse J2000 coordinates embedded in source names like J000013.26-623836.1."""
    match = re.search(
        r"J(?P<rah>\d{2})(?P<ram>\d{2})(?P<ras>\d{2}(?:\.\d+)?)"
        r"(?P<sign>[+-])(?P<decd>\d{2})(?P<decm>\d{2})(?P<decs>\d{2}(?:\.\d+)?)",
        source_name,
    )
    if match is None:
        return None

    ra_text = f"{match.group('rah')}h{match.group('ram')}m{match.group('ras')}s"
    dec_text = (
        f"{match.group('sign')}{match.group('decd')}d"
        f"{match.group('decm')}m{match.group('decs')}s"
    )
    return SkyCoord(ra_text, dec_text, frame="icrs")


def resolve_source_coordinates(
    *,
    source_name: str | None = None,
    coordinate: str | None = None,
    ra_deg: float | None = None,
    dec_deg: float | None = None,
) -> dict[str, Any]:
    """Resolve a source into sky coordinates."""
    if ra_deg is not None and dec_deg is not None:
        coord = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
        return {
            "resolved_by": "coordinates",
            "source_name": source_name,
            "ra_deg": float(coord.ra.deg),
            "dec_deg": float(coord.dec.deg),
            "coordinate_text": coord.to_string("hmsdms"),
        }

    if coordinate:
        try:
            coord = SkyCoord(coordinate, unit=(u.hourangle, u.deg), frame="icrs")
        except ValueError:
            coord = SkyCoord(coordinate, unit=(u.deg, u.deg), frame="icrs")
        return {
            "resolved_by": "coordinate_text",
            "source_name": source_name,
            "ra_deg": float(coord.ra.deg),
            "dec_deg": float(coord.dec.deg),
            "coordinate_text": coord.to_string("hmsdms"),
        }

    if source_name:
        parsed_coord = _parse_j_designation_coordinate(source_name)
        if parsed_coord is not None:
            return {
                "resolved_by": "j_designation",
                "source_name": source_name,
                "ra_deg": float(parsed_coord.ra.deg),
                "dec_deg": float(parsed_coord.dec.deg),
                "coordinate_text": parsed_coord.to_string("hmsdms"),
            }

        Simbad = _import_astroquery_module("astroquery.simbad").Simbad
        result = Simbad.query_object(source_name)
        if result is None or len(result) == 0:
            raise ValueError(f"Could not resolve source name '{source_name}'")

        ra_key = "RA" if "RA" in result.colnames else "ra"
        dec_key = "DEC" if "DEC" in result.colnames else "dec"
        if ra_key not in result.colnames or dec_key not in result.colnames:
            raise ValueError(f"Simbad result for '{source_name}' did not contain RA/Dec columns")

        coord = SkyCoord(
            ra=result[ra_key][0],
            dec=result[dec_key][0],
            unit=(u.hourangle, u.deg) if ra_key == "RA" else (u.deg, u.deg),
            frame="icrs",
        )
        return {
            "resolved_by": "simbad_name_lookup",
            "source_name": source_name,
            "ra_deg": float(coord.ra.deg),
            "dec_deg": float(coord.dec.deg),
            "coordinate_text": coord.to_string("hmsdms"),
        }

    raise ValueError("Need source_name, coordinate, or ra/dec to resolve a source")


def search_vsx(
    *,
    coord: SkyCoord,
    radius_arcsec: float = 5.0,
    row_limit: int = 5,
) -> dict[str, Any]:
    """Search the VSX catalog around a coordinate."""
    try:
        Vizier = _import_astroquery_module("astroquery.vizier").Vizier
        vizier = Vizier(columns=["**", "+_r"], row_limit=row_limit)
        result = vizier.query_region(coord, radius=radius_arcsec * u.arcsec, catalog="B/vsx/vsx")

        if not result:
            return {"matches": [], "is_known_variable": False}

        table = result[0]
        matches = [_serialize_row(table[index]) for index in range(len(table))]
        return {
            "matches": matches,
            "is_known_variable": bool(matches),
        }
    except Exception as exc:
        return {
            "matches": [],
            "is_known_variable": False,
            "error": str(exc),
        }


def fetch_gaia_astrometry(
    *,
    coord: SkyCoord,
    radius_arcsec: float = 5.0,
    gaia_tap_urls: str | Iterable[str] | None = None,
) -> dict[str, Any] | None:
    """Fetch the nearest Gaia source and return diagnosis-relevant Gaia parameters."""
    try:
        radius_deg = radius_arcsec / 3600.0
        query = f"""
SELECT TOP 1
    DISTANCE(
        POINT('ICRS', gs.ra, gs.dec),
        POINT('ICRS', {coord.ra.deg:.12f}, {coord.dec.deg:.12f})
    ) AS dist,
    gs.source_id,
    gs.ra,
    gs.dec,
    gs.parallax,
    gs.parallax_error,
    gs.pmra,
    gs.pmra_error,
    gs.pmdec,
    gs.pmdec_error,
    gs.ruwe,
    gs.astrometric_excess_noise,
    gs.phot_g_mean_mag,
    gs.phot_g_mean_flux_over_error,
    gs.phot_bp_mean_mag,
    gs.phot_bp_mean_flux_over_error,
    gs.phot_rp_mean_mag,
    gs.phot_rp_mean_flux_over_error,
    gs.bp_rp,
    gs.radial_velocity,
    gs.radial_velocity_error,
    gs.non_single_star,
    ap.teff_gspphot,
    ap.logg_gspphot,
    ap.mh_gspphot,
    ap.distance_gspphot,
    ap.azero_gspphot,
    ap.ag_gspphot,
    ap.ebpminrp_gspphot,
    ap.classprob_dsc_combmod_star,
    ap.classprob_dsc_combmod_binarystar,
    ap.classprob_dsc_combmod_galaxy,
    ap.classprob_dsc_combmod_quasar
FROM gaiadr3.gaia_source AS gs
LEFT OUTER JOIN gaiadr3.astrophysical_parameters AS ap
    ON gs.source_id = ap.source_id
WHERE 1 = CONTAINS(
    POINT('ICRS', gs.ra, gs.dec),
    CIRCLE('ICRS', {coord.ra.deg:.12f}, {coord.dec.deg:.12f}, {radius_deg:.12f})
)
ORDER BY dist ASC
"""

        results = _run_gaia_tap_query(query, tap_urls=gaia_tap_urls)
        if results is None or len(results) == 0:
            return None

        if "dist" in results.colnames:
            row = results[int(results["dist"].argmin())]
        else:
            row = results[0]
        payload = _serialize_row(row)

        wanted_keys = (
            "source_id",
            "ra",
            "dec",
            "parallax",
            "parallax_error",
            "pmra",
            "pmra_error",
            "pmdec",
            "pmdec_error",
            "ruwe",
            "astrometric_excess_noise",
            "phot_g_mean_mag",
            "phot_g_mean_flux_over_error",
            "phot_bp_mean_mag",
            "phot_bp_mean_flux_over_error",
            "phot_rp_mean_mag",
            "phot_rp_mean_flux_over_error",
            "bp_rp",
            "radial_velocity",
            "radial_velocity_error",
            "teff_gspphot",
            "teff_gspspec",
            "logg_gspphot",
            "logg_gspspec",
            "mh_gspphot",
            "distance_gspphot",
            "azero_gspphot",
            "ag_gspphot",
            "ebpminrp_gspphot",
            "classprob_dsc_combmod_star",
            "classprob_dsc_combmod_binarystar",
            "classprob_dsc_combmod_galaxy",
            "classprob_dsc_combmod_quasar",
            "non_single_star",
            "has_nss",
            "in_nss",
            "nss_solution_type",
        )
        filtered = {key: payload[key] for key in wanted_keys if key in payload}
        return _enrich_gaia_payload(filtered or payload)
    except Exception as exc:
        return {"error": str(exc)}


def _build_lamost_client(
    *,
    token: str | None = None,
    dr_version: str | None = None,
    sub_version: str | None = None,
) -> Any:
    lamost_cls = importlib.import_module("pylamost").lamost
    if token is None:
        token = os.environ.get("LAMOST_TOKEN") or os.environ.get("PYLAMOST_TOKEN")
    if dr_version is None:
        dr_version = os.environ.get("LAMOST_DR_VERSION", "dr10")
    if sub_version is None:
        sub_version = os.environ.get("LAMOST_SUB_VERSION", "v2.0")

    # pylamost prints a token warning to stdout, which would corrupt CLI JSON output.
    with contextlib.redirect_stdout(io.StringIO()):
        return lamost_cls(token=token, dr_version=dr_version, sub_version=sub_version)


def _coordinate_csv_filename(coord: SkyCoord) -> str:
    return f"{coord.ra.deg:.8f}_{coord.dec.deg:.8f}.csv"


def _coordinate_png_filename(coord: SkyCoord) -> str:
    return f"{coord.ra.deg:.8f}_{coord.dec.deg:.8f}.png"


def _project_relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _extract_lamost_rows(response: Any) -> list[dict[str, Any]]:
    if isinstance(response, list):
        return [row for row in response if isinstance(row, dict)]
    if isinstance(response, dict):
        for key in ("rows", "data", "results"):
            rows = response.get(key)
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
    return []


def _lamost_row_coordinate(row: dict[str, Any]) -> SkyCoord | None:
    ra = None
    dec = None
    for key in ("ra", "ra_obs"):
        if key in row:
            ra = _to_float(row[key])
            if ra is not None:
                break
    for key in ("dec", "dec_obs"):
        if key in row:
            dec = _to_float(row[key])
            if dec is not None:
                break
    if ra is None or dec is None:
        return None
    return SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")


def _lamost_row_separation_arcsec(row: dict[str, Any], coord: SkyCoord) -> float | None:
    row_coord = _lamost_row_coordinate(row)
    if row_coord is None:
        return None
    return float(coord.separation(row_coord).arcsec)


def _normalize_gaia_source_id(value: Any) -> str | None:
    if value is None or np.ma.is_masked(value):
        return None
    if hasattr(value, "item"):
        try:
            value = value.item()
        except ValueError:
            pass
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return f"{value:.0f}" if value.is_integer() else str(value)

    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r"\d+\.0+", text):
        return text.split(".", 1)[0]
    return text


def _lamost_row_gaia_source_id(row: dict[str, Any]) -> str | None:
    for key in ("gaia_source_id", "gaia_dr3_source_id", "source_id"):
        if key in row:
            source_id = _normalize_gaia_source_id(row[key])
            if source_id is not None:
                return source_id
    return None


def _summarize_lamost_match(
    row: dict[str, Any],
    *,
    coord: SkyCoord,
    resolution: str,
) -> dict[str, Any]:
    keys = (
        "obsid",
        "mobsid",
        "uid",
        "designation",
        "obsdate",
        "class",
        "subclass",
        "ra",
        "dec",
        "ra_obs",
        "dec_obs",
        "snr",
        "snru",
        "snrg",
        "snrr",
        "snri",
        "snrz",
        "gaia_source_id",
        "gaia_g_mean_mag",
    )
    summary = {key: row[key] for key in keys if key in row}
    summary["resolution"] = resolution
    summary["separation_arcsec"] = _lamost_row_separation_arcsec(row, coord)
    return summary


def _select_nearest_lamost_match(
    rows: list[dict[str, Any]],
    *,
    coord: SkyCoord,
) -> dict[str, Any] | None:
    usable_rows = [row for row in rows if row.get("obsid") is not None]
    if not usable_rows:
        return None

    def sort_key(row: dict[str, Any]) -> float:
        separation = _lamost_row_separation_arcsec(row, coord)
        return float("inf") if separation is None else separation

    return min(usable_rows, key=sort_key)


def _lamost_spectrum_to_csv_text(spectrum_text: str) -> str:
    stripped = spectrum_text.lstrip()
    if not stripped:
        return ""

    if stripped.startswith("<"):
        raise ValueError("LAMOST returned a non-CSV response")

    if not stripped.startswith(("{", "[")):
        return spectrum_text

    payload = json.loads(stripped)
    if not isinstance(payload, dict) or not isinstance(payload.get("spectrums"), list):
        raise ValueError("LAMOST returned a JSON payload without spectra")

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["extname", "wavelength", "flux"])
    for spectrum in payload["spectrums"]:
        if not isinstance(spectrum, dict):
            continue
        extname = spectrum.get("extname")
        for point in spectrum.get("data") or []:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                writer.writerow([extname, point[0], point[1]])
    return output.getvalue()


def _normalize_csv_column_name(column_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", column_name.strip().lower())


def _select_csv_column(fieldnames: list[str], candidates: tuple[str, ...]) -> str | None:
    lookup = {_normalize_csv_column_name(fieldname): fieldname for fieldname in fieldnames}
    for candidate in candidates:
        fieldname = lookup.get(_normalize_csv_column_name(candidate))
        if fieldname is not None:
            return fieldname
    return None


def _read_lamost_spectrum_csv(csv_path: Path) -> dict[str, Any]:
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"LAMOST spectrum CSV has no header: {csv_path}")

        wavelength_column = _select_csv_column(
            reader.fieldnames,
            ("wavelength", "wave", "lambda", "lambda_obs"),
        )
        flux_column = _select_csv_column(
            reader.fieldnames,
            ("flux", "flux_raw", "flambda"),
        )
        smooth_column = _select_csv_column(
            reader.fieldnames,
            (
                "fluxsmooth7",
                "flux_smooth7",
                "smooth_flux",
                "fluxsmooth15",
                "flux_smooth15",
            ),
        )
        if wavelength_column is None or flux_column is None:
            raise ValueError(
                "LAMOST spectrum CSV needs wavelength and flux columns; "
                f"found {reader.fieldnames}"
            )

        wavelengths: list[float] = []
        fluxes: list[float] = []
        smooth_fluxes: list[float] = []
        for row in reader:
            wavelength = _to_float(row.get(wavelength_column))
            flux = _to_float(row.get(flux_column))
            if wavelength is None or flux is None:
                continue
            if not np.isfinite(wavelength) or not np.isfinite(flux):
                continue

            wavelengths.append(wavelength)
            fluxes.append(flux)

            if smooth_column is not None:
                smooth_flux = _to_float(row.get(smooth_column))
                if smooth_flux is not None and np.isfinite(smooth_flux):
                    smooth_fluxes.append(smooth_flux)
                else:
                    smooth_fluxes.append(float("nan"))

    if not wavelengths:
        raise ValueError(f"LAMOST spectrum CSV has no usable spectrum rows: {csv_path}")

    wavelength_array = np.asarray(wavelengths, dtype=float)
    sort_order = np.argsort(wavelength_array)
    payload: dict[str, Any] = {
        "wavelength": wavelength_array[sort_order],
        "flux": np.asarray(fluxes, dtype=float)[sort_order],
        "flux_column": flux_column,
    }
    if smooth_column is not None and smooth_fluxes:
        smooth_array = np.asarray(smooth_fluxes, dtype=float)[sort_order]
        if np.isfinite(smooth_array).any():
            payload["smooth_flux"] = smooth_array
            payload["smooth_column"] = smooth_column
    return payload


def _lamost_plot_ylim(*flux_arrays: np.ndarray) -> tuple[float, float] | None:
    finite_arrays = [flux_array[np.isfinite(flux_array)] for flux_array in flux_arrays]
    finite_values = [flux_array for flux_array in finite_arrays if flux_array.size]
    if not finite_values:
        return None

    values = np.concatenate(finite_values)
    lower, upper = np.nanpercentile(values, [0.5, 99.5])
    if not np.isfinite(lower) or not np.isfinite(upper) or lower == upper:
        lower = float(np.nanmin(values))
        upper = float(np.nanmax(values))
    if lower == upper:
        margin = max(abs(lower) * 0.08, 1.0)
    else:
        margin = (upper - lower) * 0.08
    return lower - margin, upper + margin


def plot_lamost_spectrum(
    csv_path: str | Path,
    *,
    output_path: str | Path | None = None,
    title: str | None = None,
) -> Path:
    """Plot a LAMOST spectrum CSV with common optical line markers."""
    csv_path = Path(csv_path)
    png_path = Path(output_path) if output_path is not None else csv_path.with_suffix(".png")
    spectrum = _read_lamost_spectrum_csv(csv_path)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wavelength = spectrum["wavelength"]
    flux = spectrum["flux"]
    smooth_flux = spectrum.get("smooth_flux")
    x_min = float(np.nanmin(wavelength))
    x_max = float(np.nanmax(wavelength))
    x_padding = max((x_max - x_min) * 0.01, 1.0)

    fig, ax = plt.subplots(figsize=(13, 6), dpi=160)
    if smooth_flux is None:
        ax.plot(wavelength, flux, color="#1f77b4", linewidth=0.75, label=spectrum["flux_column"])
    else:
        ax.plot(
            wavelength,
            flux,
            color="#6b7280",
            linewidth=0.45,
            alpha=0.45,
            label=spectrum["flux_column"],
        )
        ax.plot(wavelength, smooth_flux, color="#1f77b4", linewidth=0.9, label=spectrum["smooth_column"])

    y_limit = _lamost_plot_ylim(flux, smooth_flux) if smooth_flux is not None else _lamost_plot_ylim(flux)
    if y_limit is not None:
        ax.set_ylim(*y_limit)
    ax.set_xlim(x_min - x_padding, x_max + x_padding)

    visible_lines = [
        (label, line_wavelength)
        for label, line_wavelength in LAMOST_SPECTRAL_LINES
        if x_min <= line_wavelength <= x_max
    ]
    for index, (label, line_wavelength) in enumerate(visible_lines):
        ax.axvline(
            line_wavelength,
            color="#9f1239",
            linestyle="--",
            linewidth=0.8,
            alpha=0.65,
        )
        ax.text(
            line_wavelength,
            0.98 - (index % 3) * 0.055,
            label,
            color="#7f1d1d",
            fontsize=7,
            rotation=90,
            ha="center",
            va="top",
            transform=ax.get_xaxis_transform(),
            clip_on=True,
        )

    ax.set_title(title or f"LAMOST spectrum: {csv_path.stem}")
    ax.set_xlabel("Wavelength (Angstrom)")
    ax.set_ylabel("Flux")
    ax.grid(True, color="#e5e7eb", linewidth=0.7, alpha=0.8)
    ax.legend(
        loc="upper right",
        frameon=True,
        facecolor="white",
        framealpha=0.85,
        edgecolor="none",
    )
    fig.tight_layout()

    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path)
    plt.close(fig)
    return png_path


def _empty_lamost_result() -> dict[str, Any]:
    return {
        "available": False,
        "query_status": "failure",
        "spectrum_csv_path": None,
        "spectrum_csv_relative_path": None,
        "spectrum_png_path": None,
        "spectrum_png_relative_path": None,
    }


def fetch_lamost_spectrum(
    *,
    coord: SkyCoord,
    gaia_source_id: Any | None = None,
    radius_arcsec: float = 5.0,
    output_dir: str | Path | None = None,
    lamost_client: Any | None = None,
    lamost_token: str | None = None,
    lamost_dr_version: str | None = None,
    lamost_sub_version: str | None = None,
) -> dict[str, Any]:
    """Fetch a matching LAMOST spectrum and save it as a CSV file."""
    result = _empty_lamost_result()
    errors: dict[str, str] = {}
    expected_gaia_source_id = _normalize_gaia_source_id(gaia_source_id)
    if expected_gaia_source_id is not None:
        result["expected_gaia_source_id"] = expected_gaia_source_id

    try:
        client = lamost_client
        if client is None:
            client = _build_lamost_client(
                token=lamost_token,
                dr_version=lamost_dr_version,
                sub_version=lamost_sub_version,
            )
    except Exception as exc:
        result["error"] = str(exc)
        return result

    search_radius_deg = radius_arcsec / 3600.0
    selected_row: dict[str, Any] | None = None
    selected_resolution: str | None = None
    selected_ismed = False
    match_counts: dict[str, int] = {}
    gaia_source_match_counts: dict[str, int] = {}
    found_gaia_source_ids: set[str] = set()
    saw_lamost_rows_without_gaia_source_id = False

    for resolution, ismed in (("lrs", False), ("mrs", True)):
        try:
            response = client.conesearch(
                coord.ra.deg,
                coord.dec.deg,
                search_radius_deg,
                ismed=ismed,
                fmt="json",
            )
            rows = _extract_lamost_rows(response)
        except Exception as exc:
            errors[resolution] = str(exc)
            rows = []

        match_counts[resolution] = len(rows)
        candidate_rows = rows
        if expected_gaia_source_id is not None:
            candidate_rows = []
            for row in rows:
                row_gaia_source_id = _lamost_row_gaia_source_id(row)
                if row_gaia_source_id is None:
                    saw_lamost_rows_without_gaia_source_id = True
                    continue
                found_gaia_source_ids.add(row_gaia_source_id)
                if row_gaia_source_id == expected_gaia_source_id:
                    candidate_rows.append(row)
            gaia_source_match_counts[resolution] = len(candidate_rows)

        selected_row = _select_nearest_lamost_match(candidate_rows, coord=coord)
        if selected_row is not None:
            selected_resolution = resolution
            selected_ismed = ismed
            break

    result["match_counts"] = match_counts
    if gaia_source_match_counts:
        result["gaia_source_match_counts"] = gaia_source_match_counts
    if errors:
        result["errors"] = errors
    if selected_row is None or selected_resolution is None:
        if expected_gaia_source_id is not None and (found_gaia_source_ids or saw_lamost_rows_without_gaia_source_id):
            result["error"] = (
                "LAMOST Gaia source ID mismatch: "
                f"expected {expected_gaia_source_id}"
            )
            if found_gaia_source_ids:
                result["found_lamost_gaia_source_ids"] = sorted(found_gaia_source_ids)
            if saw_lamost_rows_without_gaia_source_id:
                result["found_lamost_rows_without_gaia_source_id"] = True
        return result

    result["match"] = _summarize_lamost_match(
        selected_row,
        coord=coord,
        resolution=selected_resolution,
    )

    try:
        spectrum_text = client.get_fits_csv(selected_row["obsid"], ismed=selected_ismed)
        csv_text = _lamost_spectrum_to_csv_text(spectrum_text)
        if not csv_text.strip():
            result["error"] = "LAMOST returned an empty spectrum payload"
            return result

        destination_dir = Path(output_dir) if output_dir is not None else DEFAULT_LAMOST_SPECTROGRAPH_DIR
        destination_dir.mkdir(parents=True, exist_ok=True)
        csv_path = destination_dir / _coordinate_csv_filename(coord)
        csv_path.write_text(csv_text, encoding="utf-8")
        png_path = destination_dir / _coordinate_png_filename(coord)

        result["available"] = True
        result["query_status"] = "success"
        result["spectrum_csv_path"] = str(csv_path)
        result["spectrum_csv_relative_path"] = _project_relative_path(csv_path)
        try:
            plot_title = f"LAMOST {selected_resolution.upper()} spectrum"
            designation = selected_row.get("designation") or selected_row.get("obsid")
            if designation:
                plot_title = f"{plot_title}: {designation}"
            saved_png_path = plot_lamost_spectrum(csv_path, output_path=png_path, title=plot_title)
            result["spectrum_png_path"] = str(saved_png_path)
            result["spectrum_png_relative_path"] = _project_relative_path(saved_png_path)
        except Exception as exc:
            result["plot_error"] = str(exc)
        return result
    except Exception as exc:
        result["error"] = str(exc)
        return result


class CatalogQuery:
    """Reusable client for resolving sources and querying VSX, Gaia, and LAMOST."""

    def __init__(
        self,
        *,
        radius_arcsec: float = 5.0,
        vsx_row_limit: int = 5,
        gaia_tap_urls: str | Iterable[str] | None = None,
        lamost_output_dir: str | Path | None = None,
        gaia_hr_grid_path: str | Path | None = None,
        gaia_hr_output_dir: str | Path | None = None,
        enable_gaia_hr_diagram: bool = True,
        lamost_client: Any | None = None,
        lamost_token: str | None = None,
        lamost_dr_version: str | None = None,
        lamost_sub_version: str | None = None,
        debug: bool = False
    ) -> None:
        self.radius_arcsec = radius_arcsec
        self.vsx_row_limit = vsx_row_limit
        self.gaia_tap_urls = tuple(_gaia_tap_urls(gaia_tap_urls)) if gaia_tap_urls is not None else None
        self.lamost_output_dir = Path(lamost_output_dir) if lamost_output_dir is not None else None
        self.gaia_hr_grid_path = _gaia_hr_grid_path(gaia_hr_grid_path)
        self.gaia_hr_output_dir = Path(gaia_hr_output_dir) if gaia_hr_output_dir is not None else None
        self.enable_gaia_hr_diagram = enable_gaia_hr_diagram
        self.lamost_token = lamost_token
        self.lamost_dr_version = lamost_dr_version
        self.lamost_sub_version = lamost_sub_version
        self._lamost_client = lamost_client
        self.debug = debug
    def _radius(self, radius_arcsec: float | None) -> float:
        return self.radius_arcsec if radius_arcsec is None else radius_arcsec

    def _get_lamost_client(self) -> Any:
        if self._lamost_client is None:
            self._lamost_client = _build_lamost_client(
                token=self.lamost_token,
                dr_version=self.lamost_dr_version,
                sub_version=self.lamost_sub_version,
            )
        return self._lamost_client

    @staticmethod
    def coord_from_resolved(resolved: dict[str, Any]) -> SkyCoord:
        return SkyCoord(ra=resolved["ra_deg"] * u.deg, dec=resolved["dec_deg"] * u.deg, frame="icrs")

    @staticmethod
    def plot_lamost_spectrum(
        csv_path: str | Path,
        *,
        output_path: str | Path | None = None,
        title: str | None = None,
    ) -> Path:
        return plot_lamost_spectrum(csv_path, output_path=output_path, title=title)

    def resolve_source_coordinates(
        self,
        *,
        source_name: str | None = None,
        coordinate: str | None = None,
        ra_deg: float | None = None,
        dec_deg: float | None = None,
    ) -> dict[str, Any]:
        return resolve_source_coordinates(
            source_name=source_name,
            coordinate=coordinate,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
        )

    def search_vsx(
        self,
        *,
        coord: SkyCoord,
        radius_arcsec: float | None = None,
        row_limit: int | None = None,
    ) -> dict[str, Any]:
        return search_vsx(
            coord=coord,
            radius_arcsec=self._radius(radius_arcsec),
            row_limit=self.vsx_row_limit if row_limit is None else row_limit,
        )

    def fetch_gaia_astrometry(
        self,
        *,
        coord: SkyCoord,
        radius_arcsec: float | None = None,
    ) -> dict[str, Any] | None:
        return fetch_gaia_astrometry(
            coord=coord,
            radius_arcsec=self._radius(radius_arcsec),
            gaia_tap_urls=self.gaia_tap_urls,
        )

    def plot_gaia_hr_diagram(
        self,
        *,
        coord: SkyCoord,
        gaia_payload: dict[str, Any] | None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "available": False,
            "grid_path": str(self.gaia_hr_grid_path),
            "grid_relative_path": _project_relative_path(self.gaia_hr_grid_path),
            "plot_path": None,
            "plot_relative_path": None,
        }
        if not self.enable_gaia_hr_diagram:
            result["query_status"] = "disabled"
            return result
        if not isinstance(gaia_payload, dict) or gaia_payload.get("error"):
            result["query_status"] = "missing_gaia"
            return result

        target = gaia_payload.get("hr_diagram_coordinates")
        if isinstance(target, dict):
            result["target"] = {
                "bp_rp": target.get("bp_rp"),
                "bp_rp_error": target.get("bp_rp_error"),
                "absolute_g_mag": target.get("absolute_g_mag"),
                "absolute_g_mag_error": target.get("absolute_g_mag_error"),
                "parallax_over_error": target.get("parallax_over_error"),
                "absolute_g_mag_method": target.get("absolute_g_mag_method"),
                "uncertainty_method": target.get("uncertainty_method"),
            }
        else:
            result["target"] = {
                "bp_rp": _gaia_bp_rp(gaia_payload),
                "bp_rp_error": _gaia_bp_rp_error(gaia_payload),
                "absolute_g_mag": gaia_payload.get("absolute_g_mag"),
                "absolute_g_mag_error": gaia_payload.get("absolute_g_mag_error"),
            }

        try:
            png_path = _gaia_hr_output_path(coord, self.gaia_hr_output_dir)
            source_id = gaia_payload.get("source_id")
            title = f"Gaia HR diagram: source {source_id}" if source_id else "Gaia HR diagram"
            saved_png_path = plot_gaia_hr_diagram(
                gaia_payload,
                output_path=png_path,
                grid_path=self.gaia_hr_grid_path,
                title=title,
            )
            result["available"] = True
            result["query_status"] = "success"
            result["plot_path"] = str(saved_png_path)
            result["plot_relative_path"] = _project_relative_path(saved_png_path)
        except Exception as exc:
            result["query_status"] = "failure"
            result["error"] = str(exc)
        return result

    def fetch_lamost_spectrum(
        self,
        *,
        coord: SkyCoord,
        gaia_source_id: Any | None = None,
        radius_arcsec: float | None = None,
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        try:
            lamost_client = self._get_lamost_client()
        except Exception as exc:
            result = _empty_lamost_result()
            expected_gaia_source_id = _normalize_gaia_source_id(gaia_source_id)
            if expected_gaia_source_id is not None:
                result["expected_gaia_source_id"] = expected_gaia_source_id
            result["error"] = str(exc)
            return result

        return fetch_lamost_spectrum(
            coord=coord,
            gaia_source_id=gaia_source_id,
            radius_arcsec=self._radius(radius_arcsec),
            output_dir=output_dir if output_dir is not None else self.lamost_output_dir,
            lamost_client=lamost_client,
        )

    def lookup_source_catalogs(
        self,
        *,
        source_name: str | None = None,
        coordinate: str | None = None,
        ra_deg: float | None = None,
        dec_deg: float | None = None,
        radius_arcsec: float | None = None,
    ) -> dict[str, Any]:
        """Resolve a source and query VSX, Gaia, and LAMOST."""
        radius = self._radius(radius_arcsec)
        resolved = self.resolve_source_coordinates(
            source_name=source_name,
            coordinate=coordinate,
            ra_deg=ra_deg,
            dec_deg=dec_deg,
        )
        coord = self.coord_from_resolved(resolved)

        vsx = self.search_vsx(coord=coord, radius_arcsec=radius)
        gaia = self.fetch_gaia_astrometry(coord=coord, radius_arcsec=radius)
        gaia_hr_diagram = self.plot_gaia_hr_diagram(coord=coord, gaia_payload=gaia)
        gaia_source_id = gaia.get("source_id") if isinstance(gaia, dict) else None
        lamost = self.fetch_lamost_spectrum(
            coord=coord,
            gaia_source_id=gaia_source_id,
            radius_arcsec=radius,
        )
        if not self.debug:
            return {
                "resolved_source": resolved,
                "vsx": vsx,
                "gaia": gaia,
                "gaia_hr_diagram": gaia_hr_diagram,
                "lamost": lamost,
                "gaia_hr_diagram_plot_path": gaia_hr_diagram.get("plot_path"),
                "lamost_spectrum_path": lamost["spectrum_csv_path"],
                "lamost_spectrum_plot_path": lamost.get("spectrum_png_path"),
                "skip_further_diagnosis": bool(vsx["is_known_variable"]),
            }
        else: # Not return the VSX result in debug mode, to encourage checking the Gaia/LAMOST results even for known variables.
            return {
                "resolved_source": resolved,
                "gaia": gaia,
                "gaia_hr_diagram": gaia_hr_diagram,
                "lamost": lamost,
                "gaia_hr_diagram_plot_path": gaia_hr_diagram.get("plot_path"),
                "lamost_spectrum_path": lamost["spectrum_csv_path"],
                "lamost_spectrum_plot_path": lamost.get("spectrum_png_path"),
            }


def lookup_source_catalogs(
    *,
    source_name: str | None = None,
    coordinate: str | None = None,
    ra_deg: float | None = None,
    dec_deg: float | None = None,
    radius_arcsec: float = 5.0,
) -> dict[str, Any]:
    """Resolve a source and query VSX, Gaia, and LAMOST.

    Prefer using CatalogQuery directly when calling from application code.
    """
    return CatalogQuery(radius_arcsec=radius_arcsec).lookup_source_catalogs(
        source_name=source_name,
        coordinate=coordinate,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
    )


if __name__ == "__main__":
    import json
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Lookup a source in VSX, Gaia, and LAMOST")
    parser.add_argument("--source-name", type=str, help="Source name to resolve")
    parser.add_argument("--coordinate", type=str, help="Coordinate string to resolve")
    parser.add_argument("--ra-deg", type=float, help="Right ascension in degrees")
    parser.add_argument("--dec-deg", type=float, help="Declination in degrees")
    parser.add_argument("--radius-arcsec", type=float, default=5.0, help="Search radius in arcseconds")
    parser.add_argument("--lamost-output-dir", type=str, help="Directory for saved LAMOST CSV/PNG files")
    parser.add_argument("--gaia-hr-grid-path", type=str, help="Path to cached Gaia HR density grid .npz")
    parser.add_argument("--gaia-hr-output-dir", type=str, help="Directory for saved Gaia HR diagram PNG files")
    parser.add_argument("--no-gaia-hr-diagram", action="store_true", help="Do not plot Gaia HR diagram")
    parser.add_argument(
        "--gaia-tap-url",
        action="append",
        dest="gaia_tap_urls",
        help="Gaia TAP endpoint URL. Can be passed more than once.",
    )

    args = parser.parse_args()
    query = CatalogQuery(
        radius_arcsec=args.radius_arcsec,
        gaia_tap_urls=args.gaia_tap_urls,
        lamost_output_dir=args.lamost_output_dir,
        gaia_hr_grid_path=args.gaia_hr_grid_path,
        gaia_hr_output_dir=args.gaia_hr_output_dir,
        enable_gaia_hr_diagram=not args.no_gaia_hr_diagram,
    )
    result = query.lookup_source_catalogs(
        source_name=args.source_name,
        coordinate=args.coordinate,
        ra_deg=args.ra_deg,
        dec_deg=args.dec_deg,
    )
    print(json.dumps(result, indent=2))
