"""Agent setup and configuration.

This module handles the creation of the DeepAgent with all tools configured.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deepagents import DeepAgent


def create_diagnosis_agent(
    model_name: str | None = None,
    temperature: float = 0.6,
    skills_dir: str | Path | None = None,
    root_dir: str | Path = ".",
) -> "DeepAgent":
    """Create a light curve diagnosis agent with all tools.

    Args:
        model_name: Model to use. Uses settings if None.
        temperature: Sampling temperature
        skills_dir: Directory containing skill definitions
        root_dir: Root directory for filesystem backend

    Returns:
        Configured DeepAgent instance
    """
    from deepagents import create_deep_agent
    from deepagents.backends.filesystem import FilesystemBackend
    from langchain.chat_models import init_chat_model

    from lightcurve_agent.config import get_settings
    from lightcurve_agent.tools import ALL_TOOLS

    settings = get_settings()

    # Initialize model
    model = model_name or settings.openai_model
    raw_model = init_chat_model(
        model,
        model_provider="openai",
        base_url=settings.openai_base_url,
        api_key=settings.openai_api_key,
        temperature=temperature,
        extra_body={"thinking": {"type": "disabled"}},
    )

    # Setup skills directory
    if skills_dir is None:
        skills_dir = settings.skills_dir

    # Create agent
    agent = create_deep_agent(
        model=raw_model,
        tools=ALL_TOOLS,
        backend=FilesystemBackend(root_dir=str(root_dir)),
        skills=[str(skills_dir)],
    )

    return agent
