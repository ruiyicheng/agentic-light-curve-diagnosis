"""Ask a caller-provided VLM to diagnose a GLS diagnostic plot."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
import sys
from typing import Any, Protocol


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = PROJECT_ROOT / "skills" / "prompts" / "diagnosis_GLS.md"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import image_to_base, parse_json  # noqa: E402


__all__ = [
    "GLSDiagnosisError",
    "build_prompt",
    "diagnose_GLS",
    "diagnose_gls",
    "normalize_gls_diagnosis_report",
    "parse_gls_diagnosis_response",
]


class GLSDiagnosisError(ValueError):
    """Raised when the GLS diagnosis response cannot be normalized."""


class VLMClient(Protocol):
    """Minimal interface used by this GLS diagnostic step."""

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
    raise GLSDiagnosisError(
        f"`{field_name}` must be JSON text, a JSON path, or JSON-like data"
    )


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
        raise GLSDiagnosisError(f"`{field_name}` must be a list of strings")

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
            raise GLSDiagnosisError(
                "`candidate_set` mapping must include candidate_set, candidate_pool, or remaining_types"
            )

    if isinstance(parsed_candidate_set, str):
        parsed_candidate_set = [
            item.strip() for item in parsed_candidate_set.split(",")
        ]

    if not isinstance(parsed_candidate_set, (list, tuple, set)):
        raise GLSDiagnosisError(
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


def _normalize_best_period(value: Any) -> str:
    text = _normalize_string(value)
    if not text:
        raise GLSDiagnosisError("`best_period` must be P, 2P, P/2, or none")

    lower_text = text.lower()
    compact = re.sub(r"[\s*_,-]+", "", lower_text)

    if compact in {"p", "periodp", "bestp"}:
        return "P"
    if compact in {"2p", "2*p", "2xp", "twop", "doublep", "doubleperiod", "p*2"}:
        return "2P"
    if compact in {
        "p/2",
        "0.5p",
        "halfp",
        "halfperiod",
        "half-period",
        "p*0.5",
    }:
        return "P/2"
    if compact in {
        "none",
        "no",
        "noperiod",
        "nosignificantperiod",
        "notperiodic",
        "aperiodic",
        "insignificant",
    }:
        return "none"
    if "no significant" in lower_text or "not significant" in lower_text:
        return "none"

    raise GLSDiagnosisError("`best_period` must be P, 2P, P/2, or none")


def _coerce_float(value: Any, field_name: str) -> float:
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"none", "no", "n/a", "na", "unknown"}:
            return 0.0
        match = re.search(
            r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?",
            text,
        )
        if match is None:
            raise GLSDiagnosisError(f"`{field_name}` must be a finite number")
        number = float(match.group(0))
    else:
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise GLSDiagnosisError(f"`{field_name}` must be a finite number") from exc

    if not math.isfinite(number):
        raise GLSDiagnosisError(f"`{field_name}` must be a finite number")
    return number


def _coerce_bool(value: Any, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().casefold()
        if text in {"true", "yes", "y", "1", "required", "require"}:
            return True
        if text in {"false", "no", "n", "0", "none", "not required", "unneeded"}:
            return False
    raise GLSDiagnosisError(f"`{field_name}` must be a boolean")


def _coerce_positive_int(value: Any, field_name: str, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, str):
        match = re.search(r"\d+", value)
        if match is None:
            return default
        number = int(match.group(0))
    else:
        try:
            number = int(value)
        except (TypeError, ValueError):
            return default
    return max(1, number)


def build_prompt(
    gls_plot_png_path: str | Path,
    previous_json: Mapping[str, Any] | list[Any] | str | Path,
    candidate_set: Iterable[str] | str | Path,
    *,
    gls_plot_report: Mapping[str, Any] | str | Path | None = None,
    prompt_path: str | Path = PROMPT_PATH,
) -> str:
    """Build the prompt text sent with the GLS plot image."""
    plot_path = _resolve_plot_path(gls_plot_png_path)
    prompt_template_path = Path(prompt_path).expanduser().resolve()
    if not prompt_template_path.exists():
        raise FileNotFoundError(prompt_template_path)

    prompt_template = prompt_template_path.read_text(encoding="utf-8").strip()
    previous_results = _coerce_json_value(previous_json, "previous_json")
    candidate_pool = _coerce_candidate_set(candidate_set)

    prompt_sections = [
        prompt_template,
        "## GLS plot image",
        f"The attached image is the GLS diagnostic plot at: {plot_path}.",
        "## Candidate pool from previous steps",
        "```json\n"
        + json.dumps(candidate_pool, ensure_ascii=False, indent=2, default=str)
        + "\n```",
        "## Previous JSON results",
        "```json\n"
        + json.dumps(previous_results, ensure_ascii=False, indent=2, default=str)
        + "\n```",
    ]

    if gls_plot_report is not None:
        prompt_sections.extend(
            [
                "## GLS plot numeric result",
                "```json\n"
                + json.dumps(
                    _coerce_json_value(gls_plot_report, "gls_plot_report"),
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                )
                + "\n```",
            ]
        )

    return "\n\n".join(prompt_sections)


def normalize_gls_diagnosis_report(
    model_json: Any,
    candidate_set: Iterable[str] | str | Path | None = None,
) -> dict[str, Any]:
    """Validate and normalize the GLS JSON object returned by the VLM."""
    if not isinstance(model_json, Mapping):
        raise GLSDiagnosisError("Model response must be a JSON object")

    partition = _normalize_candidate_partition(model_json, candidate_set)
    amplitude_value = model_json.get("Amplitude", model_json.get("amplitude"))
    require_prewhitening_value = model_json.get("require_prewhitening", False)
    return {
        "strong_candidate_type": partition["strong_candidate_type"],
        "possible_candidate_type": partition["possible_candidate_type"],
        "strong_excluded_variable_type": partition["strong_excluded_variable_type"],
        "best_period": _normalize_best_period(model_json.get("best_period")),
        "Amplitude": _coerce_float(amplitude_value, "Amplitude"),
        "require_prewhitening": _coerce_bool(
            require_prewhitening_value,
            "require_prewhitening",
        ),
        "prewhitening_signal_count": _coerce_positive_int(
            model_json.get("prewhitening_signal_count"),
            "prewhitening_signal_count",
            1,
        ),
        "reason": _normalize_string(model_json.get("reason")),
    }


def parse_gls_diagnosis_response(
    response_text: str,
    candidate_set: Iterable[str] | str | Path | None = None,
) -> dict[str, Any]:
    """Parse and normalize a VLM response for the GLS diagnosis prompt."""
    model_json = parse_json.parse_json(response_text)
    return normalize_gls_diagnosis_report(model_json, candidate_set)


def diagnose_GLS(
    gls_plot_png_path: str | Path,
    previous_json: Mapping[str, Any] | list[Any] | str | Path,
    candidate_set: Iterable[str] | str | Path,
    vlm: VLMClient,
    *,
    gls_plot_report: Mapping[str, Any] | str | Path | None = None,
    prompt_path: str | Path = PROMPT_PATH,
    temperature: float = 1,
) -> dict[str, Any]:
    """Inspect a GLS plot and return the normalized diagnosis JSON."""
    plot_path = _resolve_plot_path(gls_plot_png_path)
    prompt = build_prompt(
        plot_path,
        previous_json,
        candidate_set,
        gls_plot_report=gls_plot_report,
        prompt_path=prompt_path,
    )
    plot_image = image_to_base.image_to_base64(plot_path)
    response_text = vlm.use(prompt, [plot_image], temperature=temperature)
    return parse_gls_diagnosis_response(response_text, candidate_set)


diagnose_gls = diagnose_GLS


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "gls_plot_png_path",
        help="Path to the plotted GLS diagnostic PNG",
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
        "--gls-plot-report",
        help="Optional GLS plot report JSON text or path, usually from plot_GLS",
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
    report = diagnose_GLS(
        args.gls_plot_png_path,
        args.previous_json,
        args.candidate_set,
        vlm,
        gls_plot_report=args.gls_plot_report,
        prompt_path=args.prompt_path,
        temperature=args.temperature,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
