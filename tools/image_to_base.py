"""Encode image files as data URLs for VLM requests."""

from __future__ import annotations

import base64
import mimetypes
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal


SUPPORTED_IMAGE_SUFFIXES = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


DataUrlEncoding = Literal["base32", "base64"]
ImagePath = str | Path


@dataclass(frozen=True)
class ImageSource:
    """A resolved image file with helpers for validation and MIME detection.

    ``path_or_dir`` may be either a supported image file or a directory that
    contains exactly one supported image. Requiring a single image avoids
    silently sending the wrong plot to the VLM.
    """

    path: Path
    supported_suffixes: ClassVar[frozenset[str]] = frozenset(SUPPORTED_IMAGE_SUFFIXES)

    @classmethod
    def from_path(cls, path_or_dir: str | Path) -> "ImageSource":
        """Resolve ``path_or_dir`` to a validated image source."""
        path = Path(path_or_dir).expanduser()

        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        if path.is_file():
            if not cls.is_supported(path):
                raise ValueError(f"Unsupported image file type: {path.suffix}")
            return cls(path)

        if not path.is_dir():
            raise ValueError(f"Expected an image file or directory, got: {path}")

        image_paths = sorted(child for child in path.iterdir() if cls.is_supported(child))

        if not image_paths:
            raise FileNotFoundError(f"No supported image files found in: {path}")

        if len(image_paths) > 1:
            names = ", ".join(child.name for child in image_paths)
            raise ValueError(
                f"Expected one image in {path}, found {len(image_paths)}: {names}"
            )

        return cls(image_paths[0])

    @classmethod
    def is_supported(cls, path: Path) -> bool:
        """Return whether ``path`` is a supported image file."""
        return path.is_file() and path.suffix.lower() in cls.supported_suffixes

    @property
    def mime_type(self) -> str:
        """Best-effort MIME type for the image."""
        mime_type, _ = mimetypes.guess_type(self.path)
        return mime_type or "application/octet-stream"

    def read_bytes(self) -> bytes:
        """Read the image bytes."""
        return self.path.read_bytes()


class ImageDataUrlEncoder:
    """Encode a resolved image source as base32 or base64 data URLs."""

    def __init__(self, path_or_dir: str | Path | ImageSource) -> None:
        self.source = (
            path_or_dir
            if isinstance(path_or_dir, ImageSource)
            else ImageSource.from_path(path_or_dir)
        )

    @property
    def image_path(self) -> Path:
        """The resolved image file path."""
        return self.source.path

    @property
    def mime_type(self) -> str:
        """Best-effort MIME type for the image."""
        return self.source.mime_type

    def to_data_url(self, encoding: DataUrlEncoding = "base64") -> str:
        """Return a ``data:<mime-type>;<encoding>,<payload>`` URL."""
        image_bytes = self.source.read_bytes()

        if encoding == "base32":
            encoded = base64.b32encode(image_bytes)
        elif encoding == "base64":
            encoded = base64.b64encode(image_bytes)
        else:
            raise ValueError(f"Unsupported data URL encoding: {encoding}")

        payload = encoded.decode("ascii")
        return f"data:{self.mime_type};{encoding},{payload}"

    def to_base32(self) -> str:
        """Return the image as a base32 data URL string."""
        return self.to_data_url("base32")

    def to_base64(self) -> str:
        """Return the image as a VLM-ready base64 data URL."""
        return self.to_data_url("base64")

    def to_openai_image_url(self) -> dict[str, object]:
        """Return an OpenAI-compatible ``image_url`` message content item."""
        return {"type": "image_url", "image_url": {"url": self.to_base64()}}


def _resolve_single_image(image_dir: str | Path) -> Path:
    """Return the single supported image file inside ``image_dir``."""
    return ImageSource.from_path(image_dir).path


def _mime_type(image_path: Path) -> str:
    return ImageSource(image_path).mime_type


def image_to_base32(image_dir: str | Path) -> str:
    """Return the only image in ``image_dir`` as a base32 data URL string.

    The returned value has the form ``data:<mime-type>;base32,<payload>``.
    """
    return ImageDataUrlEncoder(image_dir).to_base32()


def image_to_base64(image_dir: str | Path) -> str:
    """Return the only image in ``image_dir`` as a VLM-ready base64 data URL.

    OpenAI-compatible VLM APIs normally expect this value in an ``image_url.url``
    field, for example ``{"type": "image_url", "image_url": {"url": value}}``.
    """
    return ImageDataUrlEncoder(image_dir).to_base64()


def images_to_base64(image_paths: ImagePath | Iterable[ImagePath]) -> list[str]:
    """Return images as VLM-ready base64 data URLs."""
    if isinstance(image_paths, (str, Path)):
        image_paths = [image_paths]

    return [image_to_base64(image_path) for image_path in image_paths]


def image_to_openai_image_url(image_dir: str | Path) -> dict[str, object]:
    """Return an OpenAI-compatible image content item for one image."""
    return ImageDataUrlEncoder(image_dir).to_openai_image_url()
