"""Node implementations for ComfyUI-PixelPipeline.

Every node class lives in its own module; this file is the single place that
lists them so the package entry point stays a two-liner.
"""

from __future__ import annotations

from .export_scale import PixelExportScale
from .grid_restore import PixelGridRestore
from .palette import PixelPaletteApply, PixelPaletteExtract
from .video_grid_restore import PixelVideoGridRestore

ALL_NODES: list[type] = [
    PixelGridRestore,
    PixelVideoGridRestore,
    PixelPaletteExtract,
    PixelPaletteApply,
    PixelExportScale,
]

__all__ = [
    "ALL_NODES",
    "PixelGridRestore",
    "PixelVideoGridRestore",
    "PixelPaletteExtract",
    "PixelPaletteApply",
    "PixelExportScale",
]
