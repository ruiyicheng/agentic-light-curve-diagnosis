"""Catalog metadata diagnosis helpers.

This module owns the source-catalog lookup, metadata prompt assembly, optional
LAMOST spectrum image attachment, VLM call, and response normalization used by
the main diagnosis pipeline.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
import sys
from typing import Any, Protocol


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = PROJECT_ROOT / "skills" / "prompts" / "diagnose_metadata.md"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import image_to_base, parse_json, query_catalog  # noqa: E402


__all__ = [
    "MetadataDiagnosisError",
    "build_prompt",
    "diagnose_metadata",
    "diagnose_source_metadata",
    "lookup_source_metadata",
    "normalize_metadata_diagnosis",
    "parse_metadata_diagnosis_response",
]


class MetadataDiagnosisError(ValueError):
    """Raised when metadata diagnosis input cannot be prepared."""


class VLMClient(Protocol):
    """Minimal interface used by this metadata diagnostic step."""

    def use(
        self,
        prompt: str,
        image_base64_list: Any | None = None,
        temperature: float = 1,
    ) -> str:
        """Return the VLM response text for a prompt and optional images."""
        ...


def _load_prompt(prompt_path: str | Path) -> str:
    path = Path(prompt_path).expanduser().resolve()
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return "No prompt available."


def _metadata_image_inputs(catalog_result: Mapping[str, Any]) -> tuple[list[str], str]:
    image_paths: list[Any] = []
    status_parts: list[str] = []

    spectrum_png_path = _metadata_spectrum_png_path(catalog_result)
    if spectrum_png_path is None:
        status_parts.append("No LAMOST spectrum image is available for this source.")
    else:
        image_paths.append(spectrum_png_path)
        status_parts.append("LAMOST spectrum image is attached as image input.")

    gaia_hr_png_path = _metadata_gaia_hr_png_path(catalog_result)
    if gaia_hr_png_path is None:
        status_parts.append("No Gaia HR diagram image is available for this source.")
    else:
        image_paths.append(gaia_hr_png_path)
        status_parts.append("Gaia HR diagram image is attached as image input.")

    return [image_to_base.image_to_base64(path) for path in image_paths], " ".join(status_parts)


def _metadata_spectrum_png_path(catalog_result: Mapping[str, Any]) -> Any | None:
    lamost = catalog_result.get("lamost")
    if not isinstance(lamost, Mapping):
        return None

    spectrum_png_path = lamost.get("spectrum_png_path") or catalog_result.get(
        "lamost_spectrum_plot_path"
    )
    if lamost.get("available") and spectrum_png_path:
        return spectrum_png_path

    return None


def _metadata_gaia_hr_png_path(catalog_result: Mapping[str, Any]) -> Any | None:
    gaia_hr = catalog_result.get("gaia_hr_diagram")
    if not isinstance(gaia_hr, Mapping):
        return None

    plot_path = gaia_hr.get("plot_path") or catalog_result.get("gaia_hr_diagram_plot_path")
    if gaia_hr.get("available") and plot_path:
        return plot_path

    return None


def _metadata_image_status(catalog_result: Mapping[str, Any]) -> str:
    status_parts = []
    if _metadata_spectrum_png_path(catalog_result) is None:
        status_parts.append("No LAMOST spectrum image is available for this source.")
    else:
        status_parts.append("LAMOST spectrum image is attached as image input.")
    if _metadata_gaia_hr_png_path(catalog_result) is None:
        status_parts.append("No Gaia HR diagram image is available for this source.")
    else:
        status_parts.append("Gaia HR diagram image is attached as image input.")
    return " ".join(status_parts)


def normalize_metadata_diagnosis(metadata_report: Any) -> dict[str, Any]:
    """Normalize the metadata diagnosis object used by downstream steps."""
    if not isinstance(metadata_report, Mapping):
        return {"strong_excluded_variable_type": [], "reason": ""}

    raw_excluded_types = metadata_report.get("strong_excluded_variable_type", [])
    if raw_excluded_types is None:
        excluded_types = []
    elif isinstance(raw_excluded_types, str):
        excluded_types = [raw_excluded_types] if raw_excluded_types.strip() else []
    elif isinstance(raw_excluded_types, (list, tuple, set)):
        excluded_types = list(raw_excluded_types)
    else:
        excluded_types = []

    return {
        "strong_excluded_variable_type": [
            str(variable_type).strip()
            for variable_type in excluded_types
            if str(variable_type).strip()
        ],
        "reason": str(metadata_report.get("reason") or "").strip(),
    }


def build_prompt(
    catalog_result: Mapping[str, Any],
    candidate_set:set,
    prompt_path: str | Path = PROMPT_PATH,
    image_status: str | None = None,
) -> str:
    """Build the metadata diagnosis prompt from catalog lookup results."""
    if not isinstance(catalog_result, Mapping):
        raise MetadataDiagnosisError("Catalog result must be a JSON object")

    if image_status is None:
        image_status = _metadata_image_status(catalog_result)
    prompt_template = _load_prompt(prompt_path)

    str_candidate = ";".join(sorted(str(candidate) for candidate in candidate_set))
    prompt_template = prompt_template.replace("<candidate_pool>", str_candidate)
    return "\n".join(
        [
            prompt_template,
            json.dumps(catalog_result, ensure_ascii=False, indent=2, default=str),
            image_status,
        ]
    )


def parse_metadata_diagnosis_response(response_text: str) -> dict[str, Any]:
    """Parse and normalize a VLM response for the metadata prompt."""
    try:
        model_json = parse_json.parse_json(response_text)
    except parse_json.JSONParseError:
        return normalize_metadata_diagnosis({})
    return normalize_metadata_diagnosis(model_json)


def diagnose_metadata(
    catalog_result: Mapping[str, Any],
    candidate_set: set,
    vlm: VLMClient,
    *,
    prompt_path: str | Path = PROMPT_PATH,
    temperature: float = 1,
) -> dict[str, Any]:
    """Run metadata diagnosis from an existing catalog lookup result."""
    image_base64_list, image_status = _metadata_image_inputs(catalog_result)
    prompt = build_prompt(
        catalog_result,
        candidate_set,
        prompt_path=prompt_path,
        image_status=image_status,
    )
    response_text = vlm.use(prompt, image_base64_list, temperature=temperature)
    return parse_metadata_diagnosis_response(response_text)


def lookup_source_metadata(
    *,
    source_name: str | None = None,
    coordinate: str | None = None,
    ra_deg: float | None = None,
    dec_deg: float | None = None,
    radius_arcsec: float | None = None,
    catalog_query: query_catalog.CatalogQuery | None = None,
    debug: bool = True,
) -> dict[str, Any]:
    """Resolve a source and collect the catalog metadata used for diagnosis."""
    query = catalog_query or query_catalog.CatalogQuery(debug=debug)
    return query.lookup_source_catalogs(
        source_name=source_name,
        coordinate=coordinate,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        radius_arcsec=radius_arcsec,
    )


def diagnose_source_metadata(
    vlm: VLMClient,
    *,
    source_name: str | None = None,
    coordinate: str | None = None,
    ra_deg: float | None = None,
    dec_deg: float | None = None,
    radius_arcsec: float | None = None,
    catalog_query: query_catalog.CatalogQuery | None = None,
    debug: bool = True,
    prompt_path: str | Path = PROMPT_PATH,
    temperature: float = 1,
) -> dict[str, Any]:
    """Lookup source metadata, diagnose it, and return both results."""
    catalog_result = lookup_source_metadata(
        source_name=source_name,
        coordinate=coordinate,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        radius_arcsec=radius_arcsec,
        catalog_query=catalog_query,
        debug=debug,
    )
    return {
        "catalog_result": catalog_result,
        "metadata_diagnosis": diagnose_metadata(
            catalog_result,
            vlm,
            prompt_path=prompt_path,
            temperature=temperature,
        ),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-name", help="Source name to resolve")
    parser.add_argument("--coordinate", help="Coordinate string to resolve")
    parser.add_argument("--ra-deg", type=float, help="Right ascension in degrees")
    parser.add_argument("--dec-deg", type=float, help="Declination in degrees")
    parser.add_argument("--radius-arcsec", type=float, help="Catalog search radius")
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Include the normal VSX skip-further-diagnosis metadata",
    )
    parser.add_argument(
        "--prompt-path",
        default=str(PROMPT_PATH),
        help="Prompt template path",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1,
        help="VLM sampling temperature",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    from tools import use_vlm

    vlm = use_vlm.VLM()
    report = diagnose_source_metadata(
        vlm,
        source_name=args.source_name,
        coordinate=args.coordinate,
        ra_deg=args.ra_deg,
        dec_deg=args.dec_deg,
        radius_arcsec=args.radius_arcsec,
        debug=not args.no_debug,
        prompt_path=args.prompt_path,
        temperature=args.temperature,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
