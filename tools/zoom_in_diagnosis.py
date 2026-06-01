"""Ask a caller-provided VLM to diagnose zoom-in light-curve plots."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable, Mapping
from numbers import Real
from pathlib import Path
import sys
from typing import Any, Protocol


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = PROJECT_ROOT / "skills" / "prompts" / "diagnose_zoom-in.md"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import image_to_base, parse_json  # noqa: E402


__all__ = [
    "ZoomInDiagnosisError",
    "build_prompt",
    "diagnose_zoom_in",
    "diagnose_zoom_in_plot",
    "normalize_zoom_in_diagnosis_report",
    "parse_zoom_in_diagnosis_response",
]


class ZoomInDiagnosisError(ValueError):
    """Raised when the zoom-in diagnosis response cannot be normalized."""


class VLMClient(Protocol):
    """Minimal interface used by this zoom-in diagnostic step."""

    def use(
        self,
        prompt: str,
        image_base64_list: Any | None = None,
        temperature: float = 1,
    ) -> str:
        """Return the VLM response text for a prompt and optional images."""
        ...


def _resolve_plot_path(plot_png_path: str | Path) -> Path:
    path = Path(plot_png_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise IsADirectoryError(path)
    return path


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


def _coerce_json_value(value: Any, field_name: str) -> Any:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, (str, Path)):
        parsed = parse_json.parse_json(_read_text_or_path(value))
        if isinstance(parsed, tuple):
            return list(parsed)
        return parsed
    raise ZoomInDiagnosisError(
        f"`{field_name}` must be JSON text, a JSON path, or JSON-like data"
    )


def _normalize_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {
            "true",
            "yes",
            "y",
            "1",
            "continuous",
            "continuously",
            "flux changing continuously",
            "flux changing continously",
            "well-sampled",
            "well sampled",
        }:
            return True
        if normalized in {
            "false",
            "no",
            "n",
            "0",
            "discontinuous",
            "not continuous",
            "not changing continuously",
            "not changing continously",
            "flux not changing continuously",
            "flux not changing continously",
            "not well-sampled",
            "not well sampled",
            "undersampled",
        }:
            return False

    if isinstance(value, Real) and not isinstance(value, bool):
        if value == 1:
            return True
        if value == 0:
            return False

    raise ZoomInDiagnosisError(f"`{field_name}` must be a boolean")


def _normalize_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _dedupe_strings(values: Iterable[Any]) -> list[str]:
    normalized_values: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = _normalize_string(value)
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized_values.append(text)
    return normalized_values


def _normalize_string_list(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        normalized = value.strip()
        if (
            not normalized
            or normalized.lower() in {"none", "no", "n/a", "na", "[]"}
        ):
            return []
        values = [normalized]
    elif isinstance(value, (list, tuple, set)):
        values = value
    else:
        raise ZoomInDiagnosisError(f"`{field_name}` must be a list of strings")

    return _dedupe_strings(values)


def _coerce_candidate_set(candidate_set: Any) -> list[str]:
    if candidate_set is None:
        return []

    parsed_candidate_set = candidate_set
    if isinstance(candidate_set, (str, Path)):
        text = _read_text_or_path(candidate_set)
        try:
            parsed_candidate_set = parse_json.parse_json(text)
        except parse_json.JSONParseError:
            parsed_candidate_set = [item.strip() for item in text.split(",")]

    if isinstance(parsed_candidate_set, Mapping):
        for key in ("candidate_set", "candidate_pool", "remaining_types"):
            if key in parsed_candidate_set:
                parsed_candidate_set = parsed_candidate_set[key]
                break
        else:
            raise ZoomInDiagnosisError(
                "`candidate_set` mapping must include candidate_set, "
                "candidate_pool, or remaining_types"
            )

    if isinstance(parsed_candidate_set, str):
        parsed_candidate_set = [
            item.strip() for item in parsed_candidate_set.split(",")
        ]

    if not isinstance(parsed_candidate_set, (list, tuple, set)):
        raise ZoomInDiagnosisError(
            "`candidate_set` must be a list, set, JSON array, or comma-separated string"
        )

    return _dedupe_strings(parsed_candidate_set)


def _canonicalize_candidate(
    candidate: str,
    canonical_by_key: Mapping[str, str],
) -> str | None:
    if not canonical_by_key:
        return candidate
    return canonical_by_key.get(candidate.casefold())


def _normalize_candidate_partition(
    model_json: Mapping[str, Any],
    candidate_set: Any,
) -> dict[str, list[str]]:
    candidate_pool = _coerce_candidate_set(candidate_set)
    canonical_by_key = {candidate.casefold(): candidate for candidate in candidate_pool}
    raw_values = {
        "strong_candidate_type": _normalize_string_list(
            model_json.get("strong_candidate_type"),
            "strong_candidate_type",
        ),
        "possible_candidate_type": _normalize_string_list(
            model_json.get("possible_candidate_type"),
            "possible_candidate_type",
        ),
        "strong_excluded_variable_type": _normalize_string_list(
            model_json.get("strong_excluded_variable_type"),
            "strong_excluded_variable_type",
        ),
    }

    partition = {
        "strong_candidate_type": [],
        "possible_candidate_type": [],
        "strong_excluded_variable_type": [],
    }
    assigned: set[str] = set()

    for field_name in (
        "strong_candidate_type",
        "strong_excluded_variable_type",
        "possible_candidate_type",
    ):
        for raw_candidate in raw_values[field_name]:
            candidate = _canonicalize_candidate(raw_candidate, canonical_by_key)
            if candidate is None:
                continue
            key = candidate.casefold()
            if key in assigned:
                continue
            assigned.add(key)
            partition[field_name].append(candidate)

    for candidate in candidate_pool:
        key = candidate.casefold()
        if key not in assigned:
            assigned.add(key)
            partition["possible_candidate_type"].append(candidate)

    return partition


def _coerce_number(value: Any, field_name: str) -> int | float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ZoomInDiagnosisError(
            f"`{field_name}` must contain numeric time values"
        ) from exc

    if not math.isfinite(number):
        raise ZoomInDiagnosisError(f"`{field_name}` must contain finite time values")

    return int(number) if number.is_integer() else number


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
) -> list[int | float]:
    start = _coerce_number(start_value, field_name)
    end = _coerce_number(end_value, field_name)
    if start <= end:
        return [start, end]
    return [end, start]


def _range_from_mapping(
    value: Mapping[str, Any],
    field_name: str,
) -> list[int | float]:
    start = value.get("t_start", value.get("start", value.get("time_start")))
    end = value.get("t_end", value.get("end", value.get("time_end")))
    if start is None or end is None:
        raise ZoomInDiagnosisError(
            f"`{field_name}` mapping ranges must include t_start/start and t_end/end"
        )
    return _normalize_range_pair(start, end, field_name)


def _normalize_zoom_ranges(value: Any, field_name: str) -> list[list[int | float]]:
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
            return _normalize_zoom_ranges(parse_json.parse_json(normalized), field_name)
        except parse_json.JSONParseError as exc:
            raise ZoomInDiagnosisError(
                f"`{field_name}` must be JSON ranges or a list of ranges"
            ) from exc

    if isinstance(value, Mapping):
        for key in (
            field_name,
            "need_further_zoom_in",
            "further_zoom_in",
            "require_zoom_in",
        ):
            if key in value:
                return _normalize_zoom_ranges(value[key], field_name)
        return [_range_from_mapping(value, field_name)]

    if not isinstance(value, (list, tuple)):
        raise ZoomInDiagnosisError(f"`{field_name}` must be a list of time ranges")

    if not value:
        return []
    if _is_numeric_pair(value):
        return [_normalize_range_pair(value[0], value[1], field_name)]

    ranges: list[list[int | float]] = []
    for item in value:
        if isinstance(item, Mapping):
            ranges.append(_range_from_mapping(item, field_name))
        elif _is_numeric_pair(item):
            ranges.append(_normalize_range_pair(item[0], item[1], field_name))
        else:
            raise ZoomInDiagnosisError(
                f"`{field_name}` must contain ranges in [t_start, t_end] format"
            )
    return ranges


def _flux_changing_continuously_value(model_json: Mapping[str, Any]) -> Any:
    for key in (
        "flux_changing_continuously",
        "flux changing continuously",
        "fluxChangingContinuously",
        "flux_changing_continously",
        "flux changing continously",
        "fluxChangingContinously",
        "well-sampled",
        "well_sampled",
        "well sampled",
        "wellSampled",
    ):
        if key in model_json:
            return model_json[key]
    return None


def _need_further_zoom_in_value(model_json: Mapping[str, Any]) -> Any:
    for key in (
        "need_further_zoom_in",
        "need further zoom-in",
        "need further zoom in",
        "further_zoom_in",
        "require_zoom_in",
    ):
        if key in model_json:
            return model_json[key]
    return None


def build_prompt(
    zoom_plot_png_path: str | Path,
    previous_json: Mapping[str, Any] | list[Any] | str | Path,
    candidate_set: Iterable[str] | str | Path,
    *,
    zoom_plot_report: Mapping[str, Any] | str | Path | None = None,
    prompt_path: str | Path = PROMPT_PATH,
) -> str:
    """Build the prompt text sent with the zoom-in plot image."""
    plot_path = _resolve_plot_path(zoom_plot_png_path)
    prompt_template_path = Path(prompt_path).expanduser().resolve()
    if not prompt_template_path.exists():
        raise FileNotFoundError(prompt_template_path)

    prompt_template = prompt_template_path.read_text(encoding="utf-8").strip()
    previous_results = _coerce_json_value(previous_json, "previous_json")
    candidate_pool = _coerce_candidate_set(candidate_set)

    prompt_sections = [
        prompt_template,
        "## Zoom-in plot image",
        f"The attached image is the zoom-in diagnostic plot at: {plot_path}.",
        "## Candidate pool from previous steps",
        "```json\n"
        + json.dumps(candidate_pool, ensure_ascii=False, indent=2, default=str)
        + "\n```",
        "## Previous JSON results",
        "```json\n"
        + json.dumps(previous_results, ensure_ascii=False, indent=2, default=str)
        + "\n```",
    ]

    if zoom_plot_report is not None:
        prompt_sections.extend(
            [
                "## Zoom-in plot numeric result",
                "```json\n"
                + json.dumps(
                    _coerce_json_value(zoom_plot_report, "zoom_plot_report"),
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
                + "\n```",
            ]
        )

    prompt_sections.append(
        "Return only valid JSON matching the schema above. "
        "Do not add markdown, comments, prose, or extra keys."
    )

    return "\n\n".join(prompt_sections)


def normalize_zoom_in_diagnosis_report(
    model_json: Any,
    candidate_set: Iterable[str] | str | Path | None = None,
) -> dict[str, Any]:
    """Validate and normalize the zoom-in JSON object returned by the VLM."""
    if not isinstance(model_json, Mapping):
        raise ZoomInDiagnosisError("Model response must be a JSON object")

    flux_changing_continuously = _normalize_bool(
        _flux_changing_continuously_value(model_json),
        "flux_changing_continuously",
    )
    if not flux_changing_continuously:
        candidate_pool = _coerce_candidate_set(candidate_set)
        possible_candidate_type = (
            candidate_pool
            if candidate_pool
            else _normalize_string_list(
                model_json.get("possible_candidate_type"),
                "possible_candidate_type",
            )
        )
        return {
            "strong_candidate_type": [],
            "possible_candidate_type": possible_candidate_type,
            "strong_excluded_variable_type": [],
            "flux_changing_continuously": False,
            "need_further_zoom_in": [],
            "reason": _normalize_string(model_json.get("reason")),
        }

    partition = _normalize_candidate_partition(model_json, candidate_set)
    return {
        "strong_candidate_type": partition["strong_candidate_type"],
        "possible_candidate_type": partition["possible_candidate_type"],
        "strong_excluded_variable_type": partition["strong_excluded_variable_type"],
        "flux_changing_continuously": True,
        "need_further_zoom_in": _normalize_zoom_ranges(
            _need_further_zoom_in_value(model_json),
            "need_further_zoom_in",
        ),
        "reason": _normalize_string(model_json.get("reason")),
    }


def parse_zoom_in_diagnosis_response(
    response_text: str,
    candidate_set: Iterable[str] | str | Path | None = None,
) -> dict[str, Any]:
    """Parse and normalize a VLM response for the zoom-in diagnosis prompt."""
    model_json = parse_json.parse_json(response_text)
    return normalize_zoom_in_diagnosis_report(model_json, candidate_set)


def diagnose_zoom_in(
    zoom_plot_png_path: str | Path,
    previous_json: Mapping[str, Any] | list[Any] | str | Path,
    candidate_set: Iterable[str] | str | Path,
    vlm: VLMClient,
    *,
    zoom_plot_report: Mapping[str, Any] | str | Path | None = None,
    prompt_path: str | Path = PROMPT_PATH,
    temperature: float = 1,
) -> dict[str, Any]:
    """Inspect a zoom-in plot and return the normalized diagnosis JSON."""
    plot_path = _resolve_plot_path(zoom_plot_png_path)
    prompt = build_prompt(
        plot_path,
        previous_json,
        candidate_set,
        zoom_plot_report=zoom_plot_report,
        prompt_path=prompt_path,
    )
    plot_image = image_to_base.image_to_base64(plot_path)
    response_text = vlm.use(prompt, [plot_image], temperature=temperature)
    return parse_zoom_in_diagnosis_response(response_text, candidate_set)


diagnose_zoom_in_plot = diagnose_zoom_in


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "zoom_plot_png_path",
        help="Path to the plotted zoom-in diagnostic PNG",
    )
    parser.add_argument(
        "previous_json",
        help="Previous diagnosis JSON as JSON text or a path to a JSON file",
    )
    parser.add_argument(
        "candidate_set",
        help="Candidate pool as JSON text, JSON path, or comma-separated text",
    )
    parser.add_argument(
        "--zoom-plot-report",
        help="Optional zoom plot report JSON text or path, usually from plot_zoom_in",
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
    report = diagnose_zoom_in(
        args.zoom_plot_png_path,
        args.previous_json,
        args.candidate_set,
        vlm,
        zoom_plot_report=args.zoom_plot_report,
        prompt_path=args.prompt_path,
        temperature=args.temperature,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
