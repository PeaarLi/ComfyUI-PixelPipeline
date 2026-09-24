"""``PixelExportScale`` -- the last stage of the pixel pipeline.

Nearest-neighbour upscale of the (already reduced) pixel image back to a usable
resolution. Nearest is the only correct mode here: any interpolating filter
would reintroduce the soft gradients that the earlier ``PixelGridRestore`` stage
just spent effort removing.
"""

from __future__ import annotations

import torch

from comfy_api.latest import io

from ._image_utils import _nearest_scale_bhwc

__all__ = ["PixelExportScale"]

SCALE_MODE_OPTIONS = ["Nearest"]


class PixelExportScale(io.ComfyNode):
    """Nearest-neighbour integer upscale factor, with the resulting size as output."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="PixelExportScale",
            display_name="Pixel Export Scale",
            category="image/pixel pipeline",
            description=(
                "Nearest-neighbour integer upscale of a pixel image. Each source pixel becomes an exact "
                "scale x scale block, so the result stays crisp. Also reports the resulting width/height "
                "so a downstream Save/Resize node can be driven without guessing."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip=(
                        "Pixel image to upscale, e.g. the output of Pixel Grid Restore. RGB or RGBA; alpha "
                        "scales along with the colour channels, so a transparent sprite stays blocky."
                    ),
                ),
                io.Int.Input(
                    "scale",
                    default=4,
                    min=1,
                    max=16,
                    step=1,
                    tooltip="Integer upscale factor. 1 is a pass-through. Final size is input size * scale.",
                ),
                io.Combo.Input(
                    "mode",
                    options=SCALE_MODE_OPTIONS,
                    default="Nearest",
                    tooltip=(
                        "Resampling filter. Only 'Nearest' is offered on purpose: bilinear/bicubic/lanczos "
                        "would blur the pixel edges that this pipeline exists to preserve. The combo is kept "
                        "so further filters can be added without breaking saved workflows."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="image",
                    tooltip="Upscaled image, [B, H*scale, W*scale, C] with C unchanged from the input.",
                ),
                io.Int.Output(display_name="width", tooltip="Resulting width in pixels (= input W * scale)."),
                io.Int.Output(display_name="height", tooltip="Resulting height in pixels (= input H * scale)."),
            ],
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        scale: int = 4,
        mode: str = "Nearest",
    ) -> io.NodeOutput:
        if mode != "Nearest":
            raise ValueError(
                f"PixelExportScale: unsupported mode {mode!r}; expected one of {SCALE_MODE_OPTIONS}."
            )

        scale = int(scale)
        if scale < 1:
            raise ValueError(f"PixelExportScale: scale must be >= 1, got {scale}.")

        out = _nearest_scale_bhwc(image, scale)
        out = torch.clamp(out, 0.0, 1.0)

        _, height, width, _ = out.shape
        return io.NodeOutput(out, int(width), int(height))
