"""Grid layout utilities for combining multiple plots."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PIL import Image

from lightcurve_agent.config import get_settings


def combine_plots(
    paths: Sequence[str | Path],
    out_path: str | Path | None = None,
    layout: tuple[int, int] = (2, 2),
) -> Path:
    """Combine multiple plots into a grid.

    Args:
        paths: Paths to image files
        out_path: Output path. If None, uses artifacts directory.
        layout: Grid layout as (rows, cols)

    Returns:
        Path to combined image
    """
    settings = get_settings()

    if out_path is None:
        out_path = settings.artifacts_dir / "combined.png"
    out_path = Path(out_path)

    imgs = [Image.open(p) for p in paths]
    if not imgs:
        raise ValueError("No images provided")

    w, h = imgs[0].size
    rows, cols = layout

    # Create canvas
    grid = Image.new("RGB", (w * cols, h * rows), color="white")

    # Paste images
    for i, img in enumerate(imgs):
        if i >= rows * cols:
            break
        row = i // cols
        col = i % cols
        grid.paste(img, (col * w, row * h))

    grid.save(out_path)
    return out_path


def create_grid(
    paths: dict[str, Path],
    out_path: str | Path | None = None,
) -> Path:
    """Create a 2x2 grid from phase-folded analysis results.

    Layout:
        [ Periodogram ] [ Phase P   ]
        [ Phase P/2   ] [ Phase 2P  ]

    Args:
        paths: Dict with keys 'periodogram_path', 'phase_folded_P_path',
               'phase_folded_P2_path', 'phase_folded_2P_path'
        out_path: Output path

    Returns:
        Path to combined image
    """
    required = [
        "periodogram_path",
        "phase_folded_P_path",
        "phase_folded_P2_path",
        "phase_folded_2P_path",
    ]

    ordered_paths = []
    for key in required:
        if key not in paths:
            raise KeyError(f"Missing required path: {key}")
        ordered_paths.append(paths[key])

    return combine_plots(ordered_paths, out_path, layout=(2, 2))


def create_zoom_grid(
    full_path: Path,
    zoom_paths: list[Path],
    out_path: str | Path | None = None,
) -> Path:
    """Create a 2x2 grid with full view and zooms.

    Layout:
        [ Full View ] [ Zoom 1 ]
        [ Zoom 2    ] [ Zoom 3 ]

    Args:
        full_path: Path to full light curve plot
        zoom_paths: List of 3 zoom plot paths
        out_path: Output path

    Returns:
        Path to combined image
    """
    if len(zoom_paths) != 3:
        raise ValueError("Exactly 3 zoom paths required")

    paths = [full_path] + list(zoom_paths)
    return combine_plots(paths, out_path, layout=(2, 2))
