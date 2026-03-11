"""External catalog lookup helpers for VSX and Gaia."""

from __future__ import annotations

import importlib
import re
from typing import Any

from astropy import units as u
from astropy.coordinates import SkyCoord


def _import_astroquery_module(module_name: str) -> Any:
    module = importlib.import_module(module_name)
    return module


def _serialize_row(row: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key in row.colnames:
        value = row[key]
        if hasattr(value, "item"):
            try:
                value = value.item()
            except ValueError:
                pass
        payload[str(key)] = value
    return payload


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
) -> dict[str, Any] | None:
    """Fetch the nearest Gaia source and return diagnosis-relevant Gaia parameters."""
    try:
        Gaia = _import_astroquery_module("astroquery.gaia").Gaia
        radius_deg = radius_arcsec / 3600.0
        query = f"""
SELECT TOP 1
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
    gs.phot_bp_mean_mag,
    gs.phot_rp_mean_mag,
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
ORDER BY DISTANCE(
    POINT('ICRS', gs.ra, gs.dec),
    POINT('ICRS', {coord.ra.deg:.12f}, {coord.dec.deg:.12f})
)
"""

        try:
            job = Gaia.launch_job_async(query=query)
            results = job.get_results()
        except Exception:
            job = Gaia.cone_search_async(coord, radius=radius_arcsec * u.arcsec)
            results = job.get_results()
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
            "phot_bp_mean_mag",
            "phot_rp_mean_mag",
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


def lookup_source_catalogs(
    *,
    source_name: str | None = None,
    coordinate: str | None = None,
    ra_deg: float | None = None,
    dec_deg: float | None = None,
    radius_arcsec: float = 5.0,
) -> dict[str, Any]:
    """Resolve a source and query VSX followed by Gaia."""
    resolved = resolve_source_coordinates(
        source_name=source_name,
        coordinate=coordinate,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
    )
    coord = SkyCoord(ra=resolved["ra_deg"] * u.deg, dec=resolved["dec_deg"] * u.deg, frame="icrs")

    vsx = search_vsx(coord=coord, radius_arcsec=radius_arcsec)
    gaia = fetch_gaia_astrometry(coord=coord, radius_arcsec=radius_arcsec)

    return {
        "resolved_source": resolved,
        "vsx": vsx,
        "gaia": gaia,
        "skip_further_diagnosis": bool(vsx["is_known_variable"]),
    }
