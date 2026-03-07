"""Vision-Language Model (VLM) provider interface.

Provides an abstract base class for VLM providers, allowing easy swapping
between OpenAI, Claude, Gemini, or local models.
"""

from __future__ import annotations

import base64
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lightcurve_agent.config import Settings


class VLMProvider(ABC):
    """Abstract base class for Vision-Language Model providers."""

    @abstractmethod
    def analyze_image(self, image_path: str | Path, prompt: str) -> str:
        """Analyze an image with a text prompt.

        Args:
            image_path: Path to image file
            prompt: Text prompt for analysis

        Returns:
            Model's text response
        """
        pass

    @abstractmethod
    def analyze_images(self, image_paths: list[str | Path], prompt: str) -> str:
        """Analyze multiple images with a text prompt.

        Args:
            image_paths: List of image paths
            prompt: Text prompt for analysis

        Returns:
            Model's text response
        """
        pass


class OpenAIVLMProvider(VLMProvider):
    """OpenAI-compatible VLM provider (works with OpenAI, vLLM, etc.)."""

    def __init__(self, settings: Settings | None = None):
        from lightcurve_agent.config import get_settings

        self.settings = settings or get_settings()
        self._client = None

    @property
    def client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self.settings.openai_api_key,
                base_url=self.settings.openai_base_url,
            )
        return self._client

    def _encode_image(self, image_path: str | Path) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def analyze_image(self, image_path: str | Path, prompt: str) -> str:
        """Analyze a single image."""
        b64 = self._encode_image(image_path)

        resp = self.client.chat.completions.create(
            model=self.settings.openai_model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                    ],
                }
            ],
        )
        return resp.choices[0].message.content

    def analyze_images(self, image_paths: list[str | Path], prompt: str) -> str:
        """Analyze multiple images in one request."""
        content = [{"type": "text", "text": prompt}]

        for path in image_paths:
            b64 = self._encode_image(path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })

        resp = self.client.chat.completions.create(
            model=self.settings.openai_model,
            messages=[{"role": "user", "content": content}],
        )
        return resp.choices[0].message.content


class LangChainVLMProvider(VLMProvider):
    """LangChain-based VLM provider for more complex chains."""

    def __init__(self, settings: Settings | None = None):
        from lightcurve_agent.config import get_settings

        self.settings = settings or get_settings()
        self._llm = None

    @property
    def llm(self):
        """Lazy initialization of LangChain LLM."""
        if self._llm is None:
            from langchain_openai import ChatOpenAI

            self._llm = ChatOpenAI(
                model=self.settings.openai_model,
                openai_api_key=self.settings.openai_api_key,
                openai_api_base=self.settings.openai_base_url,
            )
        return self._llm

    def _encode_image(self, image_path: str | Path) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def analyze_image(self, image_path: str | Path, prompt: str) -> str:
        """Analyze a single image using LangChain."""
        from langchain_core.messages import HumanMessage

        b64 = self._encode_image(image_path)

        messages = [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ]
            )
        ]

        resp = self.llm.invoke(messages)
        return resp.content

    def analyze_images(self, image_paths: list[str | Path], prompt: str) -> str:
        """Analyze multiple images."""
        from langchain_core.messages import HumanMessage

        content = [{"type": "text", "text": prompt}]
        for path in image_paths:
            b64 = self._encode_image(path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            })

        messages = [HumanMessage(content=content)]
        resp = self.llm.invoke(messages)
        return resp.content


def get_vlm_provider(provider_type: str = "openai") -> VLMProvider:
    """Factory function to get VLM provider.

    Args:
        provider_type: "openai", "langchain", or future providers

    Returns:
        VLMProvider instance
    """
    if provider_type == "openai":
        return OpenAIVLMProvider()
    elif provider_type == "langchain":
        return LangChainVLMProvider()
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")
