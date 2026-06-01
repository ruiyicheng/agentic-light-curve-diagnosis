"""Infer the schema of a light-curve CSV file with a caller-provided VLM.

The parser builds a prompt from the file comments plus the first data rows,
sends it through the supplied VLM object, then validates and normalizes the
returned JSON.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
import re
import sys
from typing import Any, Protocol

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = PROJECT_ROOT / "skills" / "prompts" / "parse_light_curve_head.md"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import parse_json  # noqa: E402


COMMENT_PREFIXES = ("#", "//", "%", "!", ";")
VALID_TIME_UNITS = {"day", "hour", "minute", "second"}
VALID_PHOTOMETRY_UNITS = {"mag", "flux", "normalized_flux"}

TIME_UNIT_ALIASES = {
    "d": "day",
    "day": "day",
    "days": "day",
    "jd": "day",
    "mjd": "day",
    "bjd": "day",
    "hjd": "day",
    "h": "hour",
    "hr": "hour",
    "hrs": "hour",
    "hour": "hour",
    "hours": "hour",
    "m": "minute",
    "min": "minute",
    "mins": "minute",
    "minute": "minute",
    "minutes": "minute",
    "s": "second",
    "sec": "second",
    "secs": "second",
    "second": "second",
    "seconds": "second",
}

PHOTOMETRY_UNIT_ALIASES = {
    "mag": "mag",
    "magnitude": "mag",
    "magnitudes": "mag",
    "flux": "flux",
    "counts": "flux",
    "count": "flux",
    "adu": "flux",
    "electrons": "flux",
    "electron": "flux",
    "e-/s": "flux",
    "relative_flux": "normalized_flux",
    "relative flux": "normalized_flux",
    "rel_flux": "normalized_flux",
    "normalized_flux": "normalized_flux",
    "normalized flux": "normalized_flux",
    "normalised_flux": "normalized_flux",
    "normalised flux": "normalized_flux",
    "norm_flux": "normalized_flux",
}


class LightCurveSchemaError(ValueError):
    """Raised when the model response cannot be normalized to the schema."""


class VLMClient(Protocol):
    """Minimal interface used by this parser."""

    def use(
        self,
        prompt: str,
        image_base64_list: Any | None = None,
        temperature: float = 1,
    ) -> str:
        """Return the VLM response text for a prompt."""
        ...


def _is_comment_line(line: str) -> bool:
    stripped = line.lstrip()
    return bool(stripped) and stripped.startswith(COMMENT_PREFIXES)


def _looks_numeric(value: str) -> bool:
    value = value.strip()
    if not value:
        return True
    try:
        float(value)
    except ValueError:
        return False
    return True


def _looks_like_header(row: Sequence[str]) -> bool:
    return any(not _looks_numeric(cell) for cell in row)


def _sniff_dialect(lines: Sequence[str]) -> csv.Dialect:
    sample = "\n".join(lines[:20])
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t;| ")
    except csv.Error:
        class FallbackDialect(csv.excel):
            delimiter = ","
            skipinitialspace = True

        return FallbackDialect


def _csv_text(header: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue().strip()


def _clean_cell(cell: Any) -> str:
    return str(cell).strip()


def _collect_preview_lines(path: Path, row_count: int) -> tuple[list[str], list[str]]:
    for encoding in ("utf-8-sig", "utf-8", "latin-1"):
        comments: list[str] = []
        data_lines: list[str] = []
        try:
            with path.open(encoding=encoding) as handle:
                for raw_line in handle:
                    raw_line = raw_line.rstrip("\r\n")
                    if not raw_line.strip():
                        continue
                    if _is_comment_line(raw_line):
                        comments.append(raw_line.strip())
                        continue

                    data_lines.append(raw_line)
                    if len(data_lines) >= row_count + 1:
                        break
        except UnicodeDecodeError:
            continue

        return comments, data_lines

    comments = []
    data_lines = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            raw_line = raw_line.rstrip("\r\n")
            if not raw_line.strip():
                continue
            if _is_comment_line(raw_line):
                comments.append(raw_line.strip())
                continue

            data_lines.append(raw_line)
            if len(data_lines) >= row_count + 1:
                break
    return comments, data_lines


def extract_csv_preview(csv_path: str | Path, row_count: int = 5) -> dict[str, Any]:
    """Return comments, inferred columns, and the first rows of a CSV file."""
    path = Path(csv_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise IsADirectoryError(path)

    comments, data_lines = _collect_preview_lines(path, row_count=row_count)

    if not data_lines:
        raise LightCurveSchemaError(f"No CSV data rows found in {path}")

    dialect = _sniff_dialect(data_lines)
    parsed_rows = [
        [_clean_cell(cell) for cell in row]
        for row in csv.reader(data_lines, dialect=dialect)
        if row
    ]
    if not parsed_rows:
        raise LightCurveSchemaError(f"No parseable CSV rows found in {path}")

    first_row = parsed_rows[0]
    has_header = _looks_like_header(first_row)
    if has_header:
        columns = first_row
        rows = parsed_rows[1 : row_count + 1]
    else:
        columns = [f"column_{index}" for index in range(1, len(first_row) + 1)]
        rows = parsed_rows[:row_count]

    return {
        "path": str(path),
        "comments": comments,
        "columns": columns,
        "rows": rows,
        "head_csv": _csv_text(columns, rows),
        "has_header": has_header,
        "delimiter": getattr(dialect, "delimiter", ","),
    }


def build_prompt(
    csv_path: str | Path,
    prompt_path: str | Path = PROMPT_PATH,
    row_count: int = 5,
) -> tuple[str, dict[str, Any]]:
    """Build the prompt sent to the model and return it with preview metadata."""
    prompt_template_path = Path(prompt_path)
    if not prompt_template_path.exists():
        raise FileNotFoundError(prompt_template_path)

    prompt_template = prompt_template_path.read_text(encoding="utf-8").strip()
    preview = extract_csv_preview(csv_path, row_count=row_count)
    comments = "\n".join(preview["comments"]) if preview["comments"] else "No comments found."

    prompt = "\n\n".join(
        [
            prompt_template,
            "## CSV file path",
            preview["path"],
            "## Detected metadata",
            json.dumps(
                {
                    "has_header": preview["has_header"],
                    "delimiter": preview["delimiter"],
                    "columns": preview["columns"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            "## CSV comments",
            f"```text\n{comments}\n```",
            f"## First {row_count} data rows",
            f"```csv\n{preview['head_csv']}\n```",
        ]
    )
    return prompt, preview


def _normalize_scalar(value: Any, field_name: str) -> str:
    if value is None:
        raise LightCurveSchemaError(f"`{field_name}` is required")
    normalized = str(value).strip()
    if not normalized:
        raise LightCurveSchemaError(f"`{field_name}` cannot be empty")
    return normalized


def _normalize_list(
    value: Any,
    field_name: str,
    allow_empty_items: bool = False,
) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, Iterable):
        items = list(value)
    else:
        raise LightCurveSchemaError(f"`{field_name}` must be a string or list of strings")

    result: list[str] = []
    for item in items:
        if item is None:
            if allow_empty_items:
                result.append("")
            continue
        normalized = str(item).strip()
        if normalized or allow_empty_items:
            result.append(normalized)
    return result


def _normalize_enum(
    value: Any,
    aliases: dict[str, str],
    valid_values: set[str],
    field_name: str,
) -> str:
    normalized = re.sub(r"\s+", " ", _normalize_scalar(value, field_name).lower())
    normalized = aliases.get(normalized, normalized)
    if normalized not in valid_values:
        expected = ", ".join(sorted(valid_values))
        raise LightCurveSchemaError(
            f"`{field_name}` must be one of {expected}; got {value!r}"
    )
    return normalized


def _normalize_time_unit(value: Any) -> str:
    if value is None or not str(value).strip():
        return "day"

    normalized = re.sub(r"\s+", " ", str(value).strip().lower())
    if normalized in {"unknown", "unclear", "unspecified", "not specified", "n/a", "na"}:
        return "day"

    normalized = TIME_UNIT_ALIASES.get(normalized, normalized)
    if normalized not in VALID_TIME_UNITS:
        expected = ", ".join(sorted(VALID_TIME_UNITS))
        raise LightCurveSchemaError(
            f"`time_unit` must be one of {expected}; got {value!r}"
        )
    return normalized


def _normalize_optional_column(value: Any, field_name: str) -> str:
    if value is None:
        return ""

    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
        values = _normalize_list(value, field_name, allow_empty_items=True)
        value = next((item for item in values if item), "")

    normalized = str(value).strip()
    if normalized.lower() in {
        "unknown",
        "unclear",
        "unspecified",
        "none",
        "n/a",
        "na",
    }:
        return ""
    return normalized


def _first_present(mapping: dict[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _align_list(
    values: list[str],
    length: int,
    fill_value: str,
    repeat_single: bool,
) -> list[str]:
    if length <= 0:
        return []
    if not values:
        return [fill_value] * length
    if repeat_single and len(values) == 1 and length > 1:
        return values * length
    if len(values) < length:
        return values + [fill_value] * (length - len(values))
    return values[:length]


def normalize_schema(model_json: Any) -> dict[str, Any]:
    """Validate and normalize the JSON object returned by the model."""
    if not isinstance(model_json, dict):
        raise LightCurveSchemaError("Model response must be a JSON object")

    time_column = _normalize_scalar(model_json.get("time_column"), "time_column")
    time_unit = _normalize_time_unit(model_json.get("time_unit"))

    photometry_columns = _normalize_list(
        model_json.get("photometry_column"),
        "photometry_column",
    )
    if not photometry_columns:
        raise LightCurveSchemaError("`photometry_column` must contain at least one column")

    length = len(photometry_columns)
    photometry_err_columns = _align_list(
        _normalize_list(
            model_json.get("photometry_err_column"),
            "photometry_err_column",
            allow_empty_items=True,
        ),
        length,
        "",
        repeat_single=False,
    )
    photometry_units = _align_list(
        _normalize_list(model_json.get("photometry_unit"), "photometry_unit"),
        length,
        "flux",
        repeat_single=True,
    )
    photometry_units = [
        _normalize_enum(
            unit,
            PHOTOMETRY_UNIT_ALIASES,
            VALID_PHOTOMETRY_UNITS,
            "photometry_unit",
        )
        for unit in photometry_units
    ]
    photometry_bands = _align_list(
        _normalize_list(
            model_json.get("photometry_band"),
            "photometry_band",
            allow_empty_items=True,
        ),
        length,
        "unknown",
        repeat_single=True,
    )
    photometry_bands = [band if band else "unknown" for band in photometry_bands]
    filter_column = _normalize_optional_column(
        _first_present(
            model_json,
            (
                "filter_column",
                "photometry_filter_column",
                "photometry_band_column",
                "band_column",
                "passband_column",
                "filter_col",
            ),
        ),
        "filter_column",
    )

    schema = {
        "time_column": time_column,
        "time_unit": time_unit,
        "photometry_column": photometry_columns,
        "photometry_err_column": photometry_err_columns,
        "photometry_unit": photometry_units,
        "photometry_band": photometry_bands,
        "filter_column": filter_column,
    }

    if "mag" in schema["photometry_unit"]:
        keep_indices = [
            index
            for index, unit in enumerate(schema["photometry_unit"])
            if unit == "mag"
        ]
        for key in (
            "photometry_column",
            "photometry_err_column",
            "photometry_unit",
            "photometry_band",
        ):
            schema[key] = [schema[key][index] for index in keep_indices]

    return schema


def parse_light_curve_schema_response(response_text: str) -> dict[str, Any]:
    """Parse and normalize a VLM response for the light-curve schema prompt."""
    model_json = parse_json.parse_json(response_text)
    return normalize_schema(model_json)


def parse_light_curve_schema(
    csv_path: str | Path,
    *,
    vlm: VLMClient,
    prompt_path: str | Path = PROMPT_PATH,
    row_count: int = 5,
    temperature: float = 1,
) -> dict[str, Any]:
    """Infer and return the normalized schema JSON for a light-curve CSV file."""
    prompt, _ = build_prompt(csv_path, prompt_path=prompt_path, row_count=row_count)
    response_text = vlm.use(prompt, temperature=temperature)
    return parse_light_curve_schema_response(response_text)
