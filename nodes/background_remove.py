"""``PixelBackgroundRemove`` -- key a batch's background out, and align the frames.

Ported from the cleaning stage of the standalone ``webm2spritesheet.py`` and
adapted from video files to ComfyUI IMAGE batches. Three steps:

1. **Background colour.** Either the ``bg_color`` combo pins it ("Fixed Colour")
   or the most frequent colour in the first frame's 2 px border ring is taken
   ("Auto Detect"). Detection is first-frame-only, like the reference script --
   a representative first frame is the caller's job. Pinning the colour is the
   escape hatch for a first frame whose border is not representative, and it
   makes the keying reproducible across runs of different clips. The mode is a
   DynamicCombo rather than an empty-string-means-auto field so that which of
   the two is in play is visible on the node instead of inferred from the
   field's contents.
2. **One shared crop.** The content boxes of ALL frames are unioned and every
   frame is cropped to a single centred square around that union, so the
   animation keeps its registration instead of jittering inside per-frame
   crops. A frame's content is its ``alpha > 0`` pixels when that frame has
   real transparency, and otherwise the pixels whose summed per-channel RGB
   distance from the background colour exceeds ``color_tolerance``.

   The reference uses ``alpha < 255`` as the alpha-path content test, which
   marks the transparent *background* as content -- for an opaque sprite on a
   transparent canvas that is the whole frame, so the crop could never shrink.
   The direction is fixed here (``alpha > 0``); the colour path is untouched.
3. **Optional flood keying.** With ``outer_only`` (the default) only
   background-coloured regions *connected to the cropped frame's border* become
   transparent, so an interior island of the same colour survives -- that is
   the difference from a plain colour key, and the reason a character with
   background-coloured details does not get holes. Turning ``outer_only`` off
   keys every matching pixel instead, which also clears pockets the border
   cannot reach but punches holes wherever the sprite shares the background
   colour. Neither setting dominates, so both are offered.

   Keying is not the only consumer of the background colour: with
   ``remove_background`` off the colour still decides what counts as content,
   and therefore the crop window and the output size.

Two reference quirks are kept on purpose:

* The content box stores inclusive coordinates (``content_w = right - left``
  under-measures by one pixel). The 2 px minimum padding absorbs it, and
  "fixing" it would diverge every crop size from the tool this was ported from.
* ``round()`` is Python's banker's rounding, so a centred odd offset snaps
  down (``round(0.5) == 0``). Same call, same crops as the reference.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import label

from comfy_api.latest import io

from ._image_utils import (
    _join_rgb_alpha,
    _split_rgb_alpha,
    _torch_image_to_uint8_bhwc,
)
from ._palette_utils import parse_palette_hex
from .grid_restore import _pick

__all__ = ["PixelBackgroundRemove"]
NODE_ID = "PixelBackgroundRemove"

BG_COLOR_MODES = ["Auto Detect", "Fixed Colour"]

# Schema defaults live here so ``_selftest.py`` can assert the two cannot drift
# apart. The colour modes are the node's own; the rest are the reference
# script's (webm2spritesheet.py) except color_tolerance, see below.
DEFAULT_BG_COLOR = "#FFFFFF"   # the Fixed Colour default; white is a common sheet background
DEFAULT_COLOR_TOLERANCE = 60   # COLOR_TOLERANCE, raised from the reference's 12 -- see the tooltip
DEFAULT_PADDING_RATIO = 0.05   # PADDING_RATIO
DEFAULT_OUTER_ONLY = True      # border-connected keying, i.e. the reference's behaviour
PADDING_MIN_PX = 2             # PADDING_MIN_PX, deliberately not exposed
BORDER_PX = 2                  # border ring width used for background sampling


def _unpack_bg_color(bg_color) -> tuple[str, dict]:
    """Normalise the ``bg_color`` DynamicCombo value handed to ``execute``.

    Mirrors ``grid_restore._unpack_grid_mode`` but is NOT that helper: that one
    hardcodes the inner dict key ``"grid_mode"``. Handles ``None`` (a prompt
    that never mentioned ``bg_color``), a bare unmatched string, and the empty
    dict the internals may produce for the zero-nested-option case.
    """
    if isinstance(bg_color, str):
        return bg_color, {}
    if not bg_color:
        return "Auto Detect", {}
    mode = bg_color.get("bg_color")
    return (mode if isinstance(mode, str) else "Auto Detect"), bg_color


def _resolve_bg_color(mode: str, params: dict, rgb0: np.ndarray) -> np.ndarray:
    """The effective background colour as int16 ``(3,)``: pinned, or detected."""
    if mode == "Fixed Colour":
        colours = parse_palette_hex(_pick(params, "hex", DEFAULT_BG_COLOR), node_label=NODE_ID)
        if colours.shape[0] != 1:
            raise ValueError(
                f"{NODE_ID}: Fixed Colour needs exactly one colour, got {colours.shape[0]}. "
                f"Switch bg_color to Auto Detect to take it from the first frame instead."
            )
        return colours[0].astype(np.int16)
    return _detect_bg_color(rgb0)


def _detect_bg_color(rgb: np.ndarray) -> np.ndarray:
    """Most frequent colour in one frame's border ring, as int16 ``(3,)``."""
    b = BORDER_PX
    border = np.concatenate([
        rgb[:b].reshape(-1, 3),
        rgb[-b:].reshape(-1, 3),
        rgb[b:-b, :b].reshape(-1, 3),
        rgb[b:-b, -b:].reshape(-1, 3),
    ])
    colours, counts = np.unique(border, axis=0, return_counts=True)
    return colours[np.argmax(counts)].astype(np.int16)


def _union_content_bbox(rgb: np.ndarray, alpha: np.ndarray | None,
                        bg: np.ndarray, tolerance: int) -> tuple[int, int, int, int] | None:
    """Inclusive (left, top, right, bottom) union of every frame's content."""
    left, top, right, bottom = None, None, None, None
    for i in range(rgb.shape[0]):
        if alpha is not None and int(alpha[i].min()) < 255:
            ys, xs = np.where(alpha[i][..., 0] > 0)
        else:
            diff = np.abs(rgb[i].astype(np.int16) - bg).sum(axis=-1)
            ys, xs = np.where(diff > tolerance)
        if xs.size == 0:
            continue
        l, t = int(xs.min()), int(ys.min())
        r, b_ = int(xs.max()), int(ys.max())
        left = l if left is None else min(left, l)
        top = t if top is None else min(top, t)
        right = r if right is None else max(right, r)
        bottom = b_ if bottom is None else max(bottom, b_)
    if left is None:
        return None
    return left, top, right, bottom


def _crop_square(bbox: tuple[int, int, int, int], img_w: int, img_h: int,
                 padding_ratio: float) -> tuple[int, int, int, int]:
    """The reference's centred square window, edge clamping included."""
    left, top, right, bottom = bbox
    cx = (left + right) / 2
    cy = (top + bottom) / 2
    raw = max(right - left, bottom - top)
    padding = max(round(raw * padding_ratio), PADDING_MIN_PX)
    size = raw + padding * 2
    crop_left = max(0, min(round(cx - size / 2), img_w - size))
    crop_top = max(0, min(round(cy - size / 2), img_h - size))
    crop_w = min(size, img_w - crop_left)
    crop_h = min(size, img_h - crop_top)
    return crop_left, crop_top, crop_w, crop_h


def _flood_key(rgb: np.ndarray, alpha: np.ndarray, bg: np.ndarray, tolerance: int,
               outer_only: bool) -> None:
    """Set alpha to 0 on background-coloured pixels of a cropped frame.

    ``outer_only`` restricts that to regions connected to the frame border, which
    is what keeps an enclosed background-coloured island (and background-coloured
    detail inside the character) alive. Off, every matching pixel goes -- no
    labelling needed, so that path is the cheaper of the two.
    """
    mask = np.abs(rgb.astype(np.int16) - bg).sum(axis=-1) <= tolerance
    if outer_only:
        regions, _ = label(mask)
        border = (set(regions[0, :]) | set(regions[-1, :])
                  | set(regions[:, 0]) | set(regions[:, -1]))
        border.discard(0)
        if not border:
            return
        mask = np.isin(regions, list(border))
    alpha[mask, 0] = 0


class PixelBackgroundRemove(io.ComfyNode):
    """Resolve the background colour, apply one shared content crop, optionally key it out."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=NODE_ID,
            display_name="Pixel Background Remove",
            category="image/pixel pipeline",
            description=(
                "Key a batch's background out and align the frames: take the background colour from "
                "bg_color or detect it from the first frame's 2px border ring, crop every frame to ONE "
                "centred square around the union of all content boxes (so an animation keeps its "
                "registration), and flood-key border-connected background pixels to transparent. "
                "Interior same-coloured pixels survive the keying."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip=(
                        "Frames or images, RGB or RGBA, all the same size. When bg_color is left "
                        "empty the background is sampled from the FIRST frame's border ring only, so "
                        "frame 0 must have a representative background."
                    ),
                ),
                io.DynamicCombo.Input(
                    "bg_color",
                    options=[
                        io.DynamicCombo.Option("Auto Detect", []),
                        io.DynamicCombo.Option("Fixed Colour", [
                            io.String.Input(
                                "hex",
                                default=DEFAULT_BG_COLOR,
                                dynamic_prompts=False,
                                tooltip=(
                                    "The background colour to key. '#RRGGBB'; '#RGB' and a bare "
                                    "'RRGGBB' also work. Exactly one colour."
                                ),
                            ),
                        ]),
                    ],
                    tooltip=(
                        "Where the background colour comes from. 'Auto Detect' takes the most "
                        "frequent colour in the FIRST frame's 2px border ring, so frame 0 must have a "
                        "representative background. 'Fixed Colour' pins it instead -- use that when "
                        "frame 0 starts mid-fade or already has something over the edge, and to key "
                        "the same colour across several clips. The bg_color output echoes whichever "
                        "was actually used. It is listed first, which makes it the default."
                    ),
                ),
                io.Int.Input(
                    "color_tolerance",
                    default=DEFAULT_COLOR_TOLERANCE,
                    min=0,
                    max=255,
                    step=1,
                    tooltip=(
                        "A pixel counts as background when the summed per-channel RGB difference "
                        "(|dR|+|dG|+|dB|) from the background colour is <= this. Raised from the "
                        "reference script's 12 to 60 so that a gently varying AI background keys "
                        "without tuning. Flat backgrounds are unaffected at 60. For a strong "
                        "gradient, measure it: a Qwen-generated purple varied by 93-159 across the "
                        "frame and needed ~200. Raising it also widens what counts as content, so a "
                        "high value can eat sprite pixels that sit near the background colour."
                    ),
                ),
                io.Float.Input(
                    "padding_ratio",
                    default=DEFAULT_PADDING_RATIO,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=(
                        "Crop margin: side = max(content width, content height) + "
                        "2 * max(round(side * ratio), 2). The 2px minimum is fixed. One shared crop "
                        "is computed for the whole batch, so frames never drift apart."
                    ),
                ),
                io.Boolean.Input(
                    "remove_background",
                    default=True,
                    tooltip=(
                        "On: the background is keyed to transparent (alpha 0), as outer_only "
                        "describes. Off: no alpha is created and you get the shared crop only. Note "
                        "the colour is resolved either way -- it is what decides which pixels count "
                        "as content, so turning this off changes the crop but not the need for a "
                        "background colour."
                    ),
                ),
                io.Boolean.Input(
                    "outer_only",
                    default=DEFAULT_OUTER_ONLY,
                    tooltip=(
                        "On (default): only background-coloured regions CONNECTED TO the cropped "
                        "frame's border are keyed, so an enclosed pocket the border cannot reach -- "
                        "and background-coloured detail inside the character -- survives. Off: every "
                        "pixel close to the background colour is keyed, enclosed pockets included; "
                        "that is what fixes a leftover patch trapped between a shadow and a cape, "
                        "but it punches holes wherever the sprite itself uses the background colour."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="images",
                    tooltip=(
                        "Cropped (and optionally keyed) frames. RGBA when keying removed anything or "
                        "the input carried real transparency, RGB otherwise -- an all-opaque alpha "
                        "is dropped."
                    ),
                ),
                io.String.Output(
                    display_name="bg_color",
                    tooltip=(
                        "The background colour actually used, as uppercase #RRGGBB: the bg_color input "
                        "echoed back, or the detected one when it was left empty."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(cls, image: torch.Tensor, bg_color=None,
                color_tolerance: int = DEFAULT_COLOR_TOLERANCE,
                padding_ratio: float = DEFAULT_PADDING_RATIO,
                remove_background: bool = True,
                outer_only: bool = DEFAULT_OUTER_ONLY) -> io.NodeOutput:
        mode, params = _unpack_bg_color(bg_color)
        if mode not in BG_COLOR_MODES:
            raise ValueError(f"{NODE_ID}: unknown bg_color mode {mode!r}; "
                             f"expected one of {BG_COLOR_MODES}.")
        tolerance = int(color_tolerance)
        if not 0 <= tolerance <= 255:
            raise ValueError(f"{NODE_ID}: color_tolerance must be in 0..255, got {tolerance}.")
        ratio = float(padding_ratio)
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(f"{NODE_ID}: padding_ratio must be in 0..1, got {ratio}.")
        if image.ndim != 4 or image.shape[-1] not in (3, 4) or image.shape[0] < 1:
            raise ValueError(f"{NODE_ID}: expected a non-empty IMAGE batch [B,H,W,3/4], "
                             f"got {tuple(image.shape)}.")

        frames = _torch_image_to_uint8_bhwc(image)
        rgb, alpha = _split_rgb_alpha(frames)
        n, height, width = rgb.shape[:3]
        bg = _resolve_bg_color(mode, params, rgb[0])
        bbox = _union_content_bbox(rgb, alpha, bg, tolerance)
        if bbox is None:
            print(f"{NODE_ID}: no content found; keeping the full frame.")
            bbox = (0, 0, width, height)
        left, top, crop_w, crop_h = _crop_square(bbox, width, height, ratio)
        rgb = np.ascontiguousarray(rgb[:, top:top + crop_h, left:left + crop_w])
        alpha = None if alpha is None \
            else np.ascontiguousarray(alpha[:, top:top + crop_h, left:left + crop_w])

        if remove_background:
            if alpha is None:
                alpha = np.full((n, crop_h, crop_w, 1), 255, dtype=np.uint8)
            for i in range(n):
                _flood_key(rgb[i], alpha[i], bg, tolerance, bool(outer_only))

        # Pack convention: an all-opaque alpha is dropped, batch-wide.
        if alpha is not None and int(alpha.min()) == 255:
            alpha = None
        stacked = rgb if alpha is None else _join_rgb_alpha(rgb, alpha)
        out = torch.from_numpy(np.ascontiguousarray(stacked)).float().div_(255)
        bg_hex = f"#{int(bg[0]):02X}{int(bg[1]):02X}{int(bg[2]):02X}"
        return io.NodeOutput(out, bg_hex)
