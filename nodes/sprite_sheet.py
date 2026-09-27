"""``PixelSpriteSheet`` -- pack a frame batch into one sprite-sheet image.

Ported from the tiling stage of ``webm2spritesheet.py``: frames are placed
row-major into a regular grid and trailing cells stay empty (black in RGB,
transparent in RGBA -- the reference's ffmpeg ``tile`` filter leaves them
zeroed too). Everything is a pure pixel copy in torch float: no resampling,
no quantising, so there is deliberately no uint8 round-trip.
"""

from __future__ import annotations

import math

import torch

from comfy_api.latest import io

from .grid_restore import _pick

__all__ = ["PixelSpriteSheet"]
NODE_ID = "PixelSpriteSheet"

LAYOUT_OPTIONS = ["Auto Square", "Fixed Columns"]
DEFAULT_COLUMNS = 8


def _unpack_layout(layout) -> tuple[str, dict]:
    """Normalise the ``layout`` DynamicCombo value handed to ``execute``.

    Mirrors ``grid_restore._unpack_grid_mode`` but is NOT that helper: that one
    hardcodes the inner dict key ``"grid_mode"`` and the fallback mode
    ``"Auto Detect"``, both of which differ here. Handles ``None`` (a prompt
    that never mentioned ``layout``), a bare unmatched string, and the empty
    dict the internals may produce for an option with zero nested inputs.
    """
    if isinstance(layout, str):
        return layout, {}
    if not layout:
        return "Auto Square", {}
    mode = layout.get("layout")
    return (mode if isinstance(mode, str) else "Auto Square"), layout


class PixelSpriteSheet(io.ComfyNode):
    """Row-major packing of same-size frames, with the grid geometry as outputs."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=NODE_ID,
            display_name="Pixel Sprite Sheet",
            category="image/pixel pipeline",
            description=(
                "Pack a batch of same-sized frames into one sprite sheet, row-major. 'Auto Square' "
                "picks columns = ceil(sqrt(frames)) for a near-square grid; 'Fixed Columns' fixes the "
                "column count. Trailing cells stay empty (black in RGB, transparent in RGBA). Pure "
                "pixel copy -- no resampling, no quantisation."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip=(
                        "Same-sized frames, RGB or RGBA. Normally the output of Pixel Background Remove, "
                        "so every cell is already the same centred square crop."
                    ),
                ),
                io.DynamicCombo.Input(
                    "layout",
                    options=[
                        io.DynamicCombo.Option("Auto Square", []),
                        io.DynamicCombo.Option("Fixed Columns", [
                            io.Int.Input(
                                "columns",
                                default=DEFAULT_COLUMNS,
                                min=1,
                                max=1024,
                                step=1,
                                tooltip=(
                                    "Fixed frames per row. rows = ceil(frames / columns). More columns "
                                    "than frames yields one row with empty trailing cells."
                                ),
                            ),
                        ]),
                    ],
                    tooltip=(
                        "How the grid is sized; only the selected layout's controls are shown. "
                        "'Auto Square' = columns ceil(sqrt(n)), the reference script's layout. "
                        "It is listed first, which makes it the default."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="sheet",
                    tooltip=(
                        "The packed sheet, [1, rows*H, columns*W, C]; channel count unchanged from "
                        "the input. Empty trailing cells are zero."
                    ),
                ),
                io.Int.Output(display_name="columns", tooltip="Actual grid columns."),
                io.Int.Output(display_name="rows", tooltip="Actual grid rows."),
                io.Int.Output(
                    display_name="cell_width",
                    tooltip="Cell width (= input frame width), for Unity Grid-By-Cell-Size slicing.",
                ),
                io.Int.Output(
                    display_name="cell_height",
                    tooltip="Cell height (= input frame height), for Unity Grid-By-Cell-Size slicing.",
                ),
            ],
        )

    @classmethod
    def execute(cls, image: torch.Tensor, layout=None) -> io.NodeOutput:
        mode, params = _unpack_layout(layout)
        if mode not in LAYOUT_OPTIONS:
            raise ValueError(f"{NODE_ID}: unknown layout {mode!r}; expected one of {LAYOUT_OPTIONS}.")
        if image.ndim != 4 or image.shape[-1] not in (3, 4) or image.shape[0] < 1:
            raise ValueError(f"{NODE_ID}: expected a non-empty IMAGE batch [B,H,W,3/4], "
                             f"got {tuple(image.shape)}.")
        n, height, width, channels = (int(v) for v in image.shape)

        if mode == "Auto Square":
            cols = math.ceil(math.sqrt(n))
        else:
            cols = int(_pick(params, "columns", DEFAULT_COLUMNS))
            if cols < 1:
                raise ValueError(f"{NODE_ID}: columns must be >= 1, got {cols}.")
        rows = math.ceil(n / cols)

        sheet = image.new_zeros((1, rows * height, cols * width, channels))
        for i in range(n):
            r, c = divmod(i, cols)
            sheet[0, r * height:(r + 1) * height, c * width:(c + 1) * width] = image[i]
        return io.NodeOutput(sheet, cols, rows, width, height)
