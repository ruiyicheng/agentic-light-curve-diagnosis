"""Summarize all light-curve diagnosis JSON into a final classification."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
import sys
from typing import Any, Protocol


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPT_PATH = PROJECT_ROOT / "skills" / "prompts" / "summarize_results.md"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools import parse_json  # noqa: E402


__all__ = [
    "SummaryAllError",
    "build_prompt",
    "normalize_summary_report",
    "parse_summary_response",
    "summarize_all",
    "summarize_results",
]


class SummaryAllError(ValueError):
    """Raised when the final summary response cannot be normalized."""


class VLMClient(Protocol):
    """Minimal interface used by the final summarization step."""

    def use(
        self,
        prompt: str,
        image_base64_list: Any | None = None,
        temperature: float = 1,
    ) -> str:
        """Return the LLM response text for a prompt."""
        ...


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
    raise SummaryAllError(
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
        raise SummaryAllError(f"`{field_name}` must be a list of strings")

    return _dedupe_strings(values)


def _normalize_observation_requirements(value: Any) -> list[str]:
    if isinstance(value, Mapping):
        values = [
            f"{_normalize_string(key)}: {_normalize_string(item)}"
            for key, item in value.items()
            if _normalize_string(key)
        ]
        return _dedupe_strings(values)
    return _normalize_string_list(value, "requirement for future observation")


def _normalize_choice(value: Any, field_name: str, allowed_values: set[str]) -> str:
    normalized = _normalize_string(value).lower()
    if not normalized:
        raise SummaryAllError(f"`{field_name}` must be one of {sorted(allowed_values)}")
    if normalized in allowed_values:
        return normalized
    raise SummaryAllError(f"`{field_name}` must be one of {sorted(allowed_values)}")


def _normalize_urgency(value: Any) -> list[str]:
    allowed_values = {"urgent", "normal", "low"}
    if value is None:
        return []

    values = _normalize_string_list(value, "urgency_level_for_follow_up")
    normalized_values: list[str] = []
    for item in values:
        normalized = item.lower()
        if normalized not in allowed_values:
            raise SummaryAllError(
                "`urgency_level_for_follow_up` must contain only "
                f"{sorted(allowed_values)}"
            )
        normalized_values.append(normalized)
    return normalized_values


def _get_first_value(
    model_json: Mapping[str, Any],
    *field_names: str,
    default: Any = None,
) -> Any:
    for field_name in field_names:
        if field_name in model_json:
            return model_json[field_name]
    return default


def build_prompt(
    previous_json: Mapping[str, Any] | list[Any] | str | Path,
    candidate_set: Iterable[str] | str | Path,
    scientific_target: str,
    *,
    prompt_path: str | Path = PROMPT_PATH,
) -> str:
    """Build the prompt text sent to the model for final summarization."""
    prompt_template_path = Path(prompt_path).expanduser().resolve()
    if not prompt_template_path.exists():
        raise FileNotFoundError(prompt_template_path)

    prompt_template = prompt_template_path.read_text(encoding="utf-8").strip()
    previous_results = _coerce_json_value(previous_json, "previous_json")
    candidate_pool = _normalize_string_list(candidate_set, "candidate_set")

    return "\n\n".join(
        [
            prompt_template,
            "## Scientific target",
            scientific_target.strip(),
            "## Candidate pool from previous steps",
            "```json\n"
            + json.dumps(candidate_pool, ensure_ascii=False, indent=2, default=str)
            + "\n```",
            "## Previous diagnosis JSON",
            "```json\n"
            + json.dumps(previous_results, ensure_ascii=False, indent=2, default=str)
            + "\n```",
            "Return only valid JSON matching the schema above. "
            "Do not add markdown, comments, prose, or extra keys.",
        ]
    )


def normalize_summary_report(model_json: Any) -> dict[str, Any]:
    """Validate and normalize the final summary JSON returned by the model."""
    if not isinstance(model_json, Mapping):
        raise SummaryAllError("Model response must be a JSON object")

    most_likely_variable_type = _normalize_string(
        _get_first_value(
            model_json,
            "most_likely_variable_type",
            "final_candidate_type",
            "final candidate type",
            "variable_type",
            "classification",
        )
    )
    other_possible_variable_types = _normalize_string_list(
        _get_first_value(
            model_json,
            "other_possible_variable_types",
            "other possible variable types",
            "possible_candidate_type",
            "possible_candidate_types",
        ),
        "other_possible_variable_types",
    )
    if most_likely_variable_type:
        best_type_key = most_likely_variable_type.casefold()
        other_possible_variable_types = [
            variable_type
            for variable_type in other_possible_variable_types
            if variable_type.casefold() != best_type_key
        ]

    confidence_level = _get_first_value(
        model_json,
        "confidence_level",
        "confidence level",
        "confidence",
    )
    scientific_interest = _get_first_value(
        model_json,
        "Scientific interest",
        "scientific interest",
        "scientific_interest",
        "scientific importance",
        "scientific_importance",
    )
    future_observation = _get_first_value(
        model_json,
        "requirement for future observation",
        "requirement_for_future_observation",
        "future_observation",
        "follow_up_observation",
        "follow_up_observation_type",
    )
    urgency = _get_first_value(
        model_json,
        "urgency_level_for_follow_up",
        "urgency level for follow up",
        "urgency",
        "follow_up_urgency",
    )
    reasoning = _get_first_value(
        model_json,
        "reasoning",
        "reason",
    )

    return {
        "most_likely_variable_type": most_likely_variable_type,
        "confidence_level": _normalize_choice(
            confidence_level,
            "confidence_level",
            {"low", "medium", "high"},
        ),
        "other_possible_variable_types": other_possible_variable_types,
        "Scientific interest": _normalize_choice(
            scientific_interest,
            "Scientific interest",
            {"low", "medium", "high"},
        ),
        "requirement for future observation": _normalize_observation_requirements(
            future_observation
        ),
        "urgency_level_for_follow_up": _normalize_urgency(urgency),
        "reasoning": _normalize_string(reasoning),
    }


def parse_summary_response(response_text: str) -> dict[str, Any]:
    """Parse and normalize a model response for the final summary prompt."""
    model_json = parse_json.parse_json(response_text)
    return normalize_summary_report(model_json)


def summarize_results(
    previous_json: Mapping[str, Any] | list[Any] | str | Path,
    candidate_set: Iterable[str] | str | Path,
    scientific_target: str,
    vlm: VLMClient,
    *,
    prompt_path: str | Path = PROMPT_PATH,
    temperature: float = 1,
) -> dict[str, Any]:
    """Summarize all previous diagnosis JSON and return final normalized JSON."""
    prompt = build_prompt(
        previous_json,
        candidate_set,
        scientific_target,
        prompt_path=prompt_path,
    )
    response_text = vlm.use(prompt, temperature=temperature)
    return parse_summary_response(response_text)


summarize_all = summarize_results


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "previous_json",
        help="All previous diagnosis JSON as JSON text or a path to a JSON file",
    )
    parser.add_argument(
        "scientific_target",
        help="Scientific target used to judge scientific interest",
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
        help="LLM sampling temperature",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    from tools import use_vlm

    vlm = use_vlm.VLM()
    report = summarize_results(
        args.previous_json,
        [],
        args.scientific_target,
        vlm,
        prompt_path=args.prompt_path,
        temperature=args.temperature,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
