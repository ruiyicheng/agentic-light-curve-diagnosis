"""Utilities for parsing JSON returned by LLMs.

LLM responses are often either a raw JSON object or the same JSON wrapped in a
Markdown code fence labeled ``json``. This module accepts those common forms and
returns the decoded Python value.
"""

from __future__ import annotations

import json
import re
from typing import Any


_FENCED_CODE_BLOCK_RE = re.compile(
    r"```[ \t]*(?P<language>[A-Za-z0-9_-]*)[^\n\r`]*[\r\n]+"
    r"(?P<body>.*?)[\r\n]*```",
    re.DOTALL,
)


class JSONParseError(ValueError):
    """Raised when no valid JSON value can be extracted from an LLM response."""


def _looks_like_json_language(language: str) -> bool:
    return language.strip().lower() in {"", "json"}


def _parse_embedded_json(text: str) -> Any:
    """Parse the first valid JSON object/array embedded in ``text``."""
    decoder = json.JSONDecoder()

    for match in re.finditer(r"[\[{]", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue

        if isinstance(value, (dict, list)):
            return value

    raise JSONParseError("No valid JSON object or array found in LLM response.")


def parse_json(response: str | bytes | bytearray) -> Any:
    """Parse JSON from an LLM response.

    Accepted inputs:
    - Raw JSON, for example ``{"reason": "..."}``.
    - JSON wrapped in a Markdown fence labeled ``json``.
    - A response that contains one valid JSON object or array with extra text
      around it.

    Raises:
        TypeError: If ``response`` is not text or bytes.
        JSONParseError: If no valid JSON value can be parsed.
    """
    if isinstance(response, (bytes, bytearray)):
        text = response.decode("utf-8")
    elif isinstance(response, str):
        text = response
    else:
        raise TypeError("response must be str, bytes, or bytearray")

    text = text.strip()
    if not text:
        raise JSONParseError("Cannot parse JSON from an empty response.")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    for match in _FENCED_CODE_BLOCK_RE.finditer(text):
        if not _looks_like_json_language(match.group("language")):
            continue

        body = match.group("body").strip()
        if not body:
            continue

        try:
            return json.loads(body)
        except json.JSONDecodeError:
            continue

    return _parse_embedded_json(text)
