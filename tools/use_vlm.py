"""Demo: call an OpenAI-compatible LLM using settings from the project .env.

Expected environment keys:
- OPENAI_API_KEY or OPENROUTER_API_KEY
- OPENAI_BASE_URL
- OPENAI_MODEL

The script loads these values from ../.env when run from this file.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from numbers import Real
import os
from pathlib import Path
from typing import Any

try:
	from openai import OpenAI
except ImportError as exc:  # pragma: no cover - dependency check happens at runtime
	raise SystemExit(
		"The openai package is required for this demo. Install it with `pip install openai`."
	) from exc

PATH_SYSTEM_PROMPT = Path(__file__).resolve().parents[1] / "skills/prompts/system.md"
SYSTEM_PROMPT = PATH_SYSTEM_PROMPT.read_text(encoding="utf-8") if PATH_SYSTEM_PROMPT.exists() else "You are a helpful assistant."


ImageInput = str | Mapping[str, Any]
UserContent = str | list[dict[str, Any]]


class VLM:
    def __init__(self) -> None:
        self.client, self.model = self.build_client()
            
    def load_env_file(self,env_path: Path) -> None:
        if not env_path.exists():
            return

        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()

            if not key:
                continue

            if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                value = value[1:-1]

            os.environ.setdefault(key, value)


    def build_client(self) -> tuple[OpenAI, str]:
        project_root = Path(__file__).resolve().parents[1]
        self.load_env_file(project_root / ".env")

        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise SystemExit("Missing OPENAI_API_KEY or OPENROUTER_API_KEY in .env")

        base_url = os.getenv("OPENAI_BASE_URL")
        model = os.getenv("OPENAI_MODEL") or "gpt-4o-mini"

        client = OpenAI(api_key=api_key, base_url=base_url or None)
        return client, model


    @staticmethod
    def _image_url_from_base64(
        image_base64: str,
        default_mime_type: str = "image/png",
    ) -> dict[str, Any]:
        """Build an OpenAI-compatible image content item from raw base64 or a data URL."""
        image_base64 = image_base64.strip()
        if not image_base64:
            raise ValueError("Image base64 input cannot be empty")

        if image_base64.startswith(("data:", "http://", "https://")):
            image_url = image_base64
        else:
            image_url = f"data:{default_mime_type};base64,{image_base64}"

        return {"type": "image_url", "image_url": {"url": image_url}}

    @classmethod
    def _normalize_images(
        cls,
        image_base64_list: ImageInput | Iterable[ImageInput] | None,
    ) -> list[dict[str, Any]]:
        if image_base64_list is None:
            return []

        if isinstance(image_base64_list, (str, Mapping)):
            image_inputs: Iterable[ImageInput] = [image_base64_list]
        else:
            image_inputs = image_base64_list

        image_items: list[dict[str, Any]] = []
        for image_input in image_inputs:
            if isinstance(image_input, str):
                image_items.append(cls._image_url_from_base64(image_input))
            elif isinstance(image_input, Mapping):
                image_items.append(dict(image_input))
            else:
                raise TypeError(
                    "Each image input must be a base64/data-url string or an image content item"
                )

        return image_items

    @classmethod
    def _build_user_content(
        cls,
        prompt: str,
        image_base64_list: ImageInput | Iterable[ImageInput] | None,
    ) -> UserContent:
        image_items = cls._normalize_images(image_base64_list)
        if not image_items:
            return prompt

        return [
            {"type": "text", "text": prompt},
            *image_items,
        ]

    def use(
        self,
        prompt: str,
        image_base64_list: ImageInput | Iterable[ImageInput] | Real | None = None,
        temperature: float = 1,
    ) -> str:
        if isinstance(image_base64_list, Real):
            temperature = float(image_base64_list)
            image_base64_list = None

        user_content = self._build_user_content(prompt, image_base64_list)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            temperature=temperature,
        )

        message = response.choices[0].message.content or ""
        print(f"model={self.model}")
        print(message)
        return message


if __name__ == "__main__":
    vlm = VLM()
    vlm.use("The specturm of source shown strong H-alpha emission. What does it suggest about the source type?")
