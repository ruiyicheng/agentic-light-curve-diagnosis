"""Ask a caller-provided VLM to inspect a full light-curve plot.

The model receives the plotted whole-light-curve PNG plus the metadata
diagnosis JSON and returns the next analysis steps requested by README.md.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from numbers import Real
from pathlib import Path
import sys
from typing import Any, Protocol


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = PROJECT_ROOT / "skills" / "prompts" / "view_full_plot.md"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import image_to_base, parse_json  # noqa: E402


__all__ = [
    "FullPlotDiagnosisError",
    "build_prompt",
    "normalize_full_plot_report",
    "parse_full_plot_response",
    "view_full_plot",
]


class FullPlotDiagnosisError(ValueError):
    """Raised when the full-plot model response cannot be normalized."""


class VLMClient(Protocol):
    """Minimal interface used by this full-plot diagnostic step."""

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


def _coerce_metadata_report(
    metadata_diagnosis_json: Mapping[str, Any] | str | Path,
) -> dict[str, Any]:
    if isinstance(metadata_diagnosis_json, Mapping):
        return _normalize_metadata_report(metadata_diagnosis_json)

    if isinstance(metadata_diagnosis_json, Path):
        text = metadata_diagnosis_json.expanduser().read_text(encoding="utf-8")
    else:
        text = metadata_diagnosis_json
        try:
            possible_path = Path(text).expanduser()
            if "\n" not in text and possible_path.exists():
                text = possible_path.read_text(encoding="utf-8")
        except OSError:
            pass

    parsed = parse_json.parse_json(text)
    if not isinstance(parsed, Mapping):
        raise FullPlotDiagnosisError("Metadata diagnosis must be a JSON object")
    return _normalize_metadata_report(parsed)


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
        values = list(value)
    else:
        raise FullPlotDiagnosisError(f"`{field_name}` must be a list of strings")

    return [str(item).strip() for item in values if str(item).strip()]


def _normalize_metadata_report(metadata_report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "strong_excluded_variable_type": _normalize_string_list(
            metadata_report.get("strong_excluded_variable_type"),
            "strong_excluded_variable_type",
        ),
        "reason": _normalize_string(metadata_report.get("reason")),
    }


def _normalize_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "y", "1", "required", "require"}:
            return True
        if normalized in {"false", "no", "n", "0", "not required", "none"}:
            return False

    if isinstance(value, Real) and not isinstance(value, bool):
        if value == 1:
            return True
        if value == 0:
            return False

    raise FullPlotDiagnosisError(f"`{field_name}` must be a boolean")


def _normalize_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _coerce_number(value: Any, field_name: str) -> int | float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise FullPlotDiagnosisError(
            f"`{field_name}` must contain numeric time values"
        ) from exc

    if not math.isfinite(number):
        raise FullPlotDiagnosisError(f"`{field_name}` must contain finite time values")

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


def _range_from_mapping(
    value: Mapping[str, Any],
    field_name: str,
) -> list[int | float]:
    start = value.get("t_start", value.get("start", value.get("time_start")))
    end = value.get("t_end", value.get("end", value.get("time_end")))
    if start is None or end is None:
        raise FullPlotDiagnosisError(
            f"`{field_name}` mapping ranges must include t_start/start and t_end/end"
        )
    return _normalize_range_pair(start, end, field_name)


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


def _normalize_zoom_ranges(value: Any) -> list[list[int | float]]:
    field_name = "require_zoom_in"
    if value is None:
        return []

    if isinstance(value, str):
        if (
            not value.strip()
            or value.strip().lower() in {"none", "no", "n/a", "na", "[]"}
        ):
            return []
        raise FullPlotDiagnosisError(f"`{field_name}` must be a list of time ranges")

    if isinstance(value, Mapping):
        return [_range_from_mapping(value, field_name)]

    if not isinstance(value, (list, tuple)):
        raise FullPlotDiagnosisError(f"`{field_name}` must be a list of time ranges")

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
            raise FullPlotDiagnosisError(
                f"`{field_name}` must contain ranges in [t_start, t_end] format"
            )
    return ranges


def build_prompt(
    plot_png_path: str | Path,
    metadata_diagnosis_json: Mapping[str, Any] | str | Path,
    candidate_set: set,
    prompt_path: str | Path = PROMPT_PATH,
) -> str:
    """Build the prompt text sent with the full light-curve image."""
    plot_path = _resolve_plot_path(plot_png_path)
    prompt_template_path = Path(prompt_path).expanduser().resolve()
    if not prompt_template_path.exists():
        raise FileNotFoundError(prompt_template_path)

    prompt_template = prompt_template_path.read_text(encoding="utf-8").strip()

    # Replace the <candidate_pool> in prompt with the set given by candidate_set
    
    str_candidate = ''
    for candidate in candidate_set:
        str_candidate += candidate+';'
    
    prompt_template.replace("<candidate_pool>",str_candidate)


    metadata_report = _coerce_metadata_report(metadata_diagnosis_json)

    return "\n\n".join(
        [
            prompt_template,
            "## Full light-curve image",
            f"The attached image is the plotted full light curve at: {plot_path}.",
            "## Metadata diagnosis JSON",
            "```json\n"
            + json.dumps(metadata_report, ensure_ascii=False, indent=2, default=str)
            + "\n```",
        ]
    )


def normalize_full_plot_report(model_json: Any) -> dict[str, Any]:
    """Validate and normalize the full-plot JSON object returned by the VLM."""
    if not isinstance(model_json, Mapping):
        raise FullPlotDiagnosisError("Model response must be a JSON object")

    return {
        "require_GLS": _normalize_bool(model_json.get("require_GLS"), "require_GLS"),
        "require_GLS_reason": _normalize_string(model_json.get("require_GLS_reason")),
        "require_BLS": _normalize_bool(model_json.get("require_BLS"), "require_BLS"),
        "require_BLS_reason": _normalize_string(model_json.get("require_BLS_reason")),
        "require_zoom_in": _normalize_zoom_ranges(model_json.get("require_zoom_in")),
        "require_zoom_in_reason": _normalize_string(
            model_json.get("require_zoom_in_reason")
        ),
        "strong_excluded_variable_type": _normalize_string_list(
            model_json.get("strong_excluded_variable_type"),
            "strong_excluded_variable_type",
        ),
        "excluded_variable_type_reason": _normalize_string(
            model_json.get("excluded_variable_type_reason")
        ),
    }


def parse_full_plot_response(response_text: str) -> dict[str, Any]:
    """Parse and normalize a VLM response for the full light-curve prompt."""
    model_json = parse_json.parse_json(response_text)
    return normalize_full_plot_report(model_json)


def view_full_plot(
    plot_png_path: str | Path,
    metadata_diagnosis_json: Mapping[str, Any] | str | Path,
    candidate_set: set,
    vlm: VLMClient,
    *,
    prompt_path: str | Path = PROMPT_PATH,
    temperature: float = 1,
) -> dict[str, Any]:
    """Inspect the full plot and return the normalized README step JSON."""
    plot_path = _resolve_plot_path(plot_png_path)
    prompt = build_prompt(
        plot_path,
        metadata_diagnosis_json,
        candidate_set = candidate_set,
        prompt_path=prompt_path,
    )
    plot_image = image_to_base.image_to_base64(plot_path)
    response_text = vlm.use(prompt, [plot_image], temperature=temperature)
    return parse_full_plot_response(response_text)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "plot_png_path",
        help="Path to the plotted full light-curve PNG",
    )
    parser.add_argument(
        "metadata_diagnosis_json",
        help="Metadata diagnosis as JSON text or a path to a JSON file",
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
    report = view_full_plot(
        args.plot_png_path,
        args.metadata_diagnosis_json,
        vlm,
        prompt_path=args.prompt_path,
        temperature=args.temperature,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
