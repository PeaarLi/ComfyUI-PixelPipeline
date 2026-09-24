"""The two palette nodes: ``PixelPaletteExtract`` and ``PixelPaletteQuantize``.

They are a matched pair. Extract answers "which colours is this sprite made of?"
and hands the answer out as a plain HEX string; Quantize answers "make this
image use only those colours" and accepts that same string back. Because the
palette travels as text it is visible in the UI, editable by hand, pasteable
into a commit message, and reusable across an entire sprite set -- which is the
only way to keep a set of sprites colour-consistent, since per-image k-means
cannot be (``cv2.kmeans`` seeds from a global RNG and openly warns that the same
image may come back with different labels).

Why the palette is a ``String`` with ``lazy=True``
--------------------------------------------------
``PixelPaletteQuantize.palette`` is declared ``optional=True, lazy=True`` and
paired with :meth:`PixelPaletteQuantize.check_lazy_status`: the node asks for
the palette **only** when the input is wired but not yet evaluated (ComfyUI
passes ``None`` for that) and never when it is unwired (the key is absent, so
the ``MISSING`` sentinel default applies). ``comfy_execution/graph.py:169-172``
drops lazy links from the initial execution block, so an unwired palette means
the entire ``PixelPaletteExtract`` subtree is pruned from the graph rather than
computed and thrown away. That is the whole point of the lazy declaration, and
it is why the two nodes can sit on the same canvas as an optional branch.

Where accent protection lives, and why it is on *both* nodes
------------------------------------------------------------
``accent_protect`` guards rare but vivid colours (an eye highlight, a lone
coloured accent) from being swallowed by the dominant clusters. It belongs to
the step that *chooses* a palette, and both nodes can choose one:

* ``PixelPaletteExtract`` always chooses, so its flag always applies.
* ``PixelPaletteQuantize`` chooses only when its ``palette`` input is empty.
  With a palette wired in it maps onto that palette verbatim -- adding colours
  would break the cross-image subset guarantee, which is the whole reason the
  palette is passed around as text in the first place. So its flag is inert on
  the connected path.

Consequence worth knowing: with default settings the 32 colours Extract emits
and the 32 colours Quantize derives on its own are **not the same set** -- the
latter may include protected accents. That is intended, not a bug, but it does
mean "Extract then Quantize" and "Quantize alone" are different pipelines.

``Per-Tile k-means`` is the default method of both nodes and has a tile-size
parameter, exposed as the advanced ``tile_size`` on each. The module constant
``_palette_utils.EXTRACT_TILE_SIZE`` is now only the default value.
"""

from __future__ import annotations

import numpy as np
import torch

from comfy_api.latest import io

from ._image_utils import _torch_image_to_uint8_rgb, _uint8_rgb_to_torch_image
from ._palette_utils import (
    DITHER_NONE,
    DITHER_OPTIONS,
    EXTRACT_METHODS,
    EXTRACT_TILE_SIZE,
    FIXED_PALETTE,
    MISSING,
    PER_TILE,
    QUANTIZE_METHODS,
    SORT_OPTIONS,
    SPACE_OPTIONS,
    extract_palette,
    format_palette_hex,
    map_to_palette,
    palette_swatch_image,
    parse_palette_hex,
    sort_palette,
)

__all__ = ["PixelPaletteExtract", "PixelPaletteQuantize"]

MAX_COLORS = 256
MIN_COLORS = 8
DEFAULT_COLORS = 128


def _blank_palette_text(text) -> bool:
    """True when the ``palette`` input should be read as "no palette supplied".

    ``MISSING`` is the unwired key; ``None`` is what ComfyUI passes for a wired
    but not-yet-evaluated lazy input; ``""`` is an untouched multiline widget.
    """
    if text is MISSING or text is None:
        return True
    return not str(text).strip()


class PixelPaletteExtract(io.ComfyNode):
    """Derive an indexed palette from an image and emit it as HEX text."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="PixelPaletteExtract",
            display_name="Pixel Palette Extract",
            category="image/pixel pipeline",
            description=(
                "Extract the palette an image is actually made of and emit it as newline-separated "
                "#RRGGBB text. Feed the string to Pixel Palette Quantize to force a whole sprite set onto "
                "one shared palette -- per-image clustering cannot do that, because cv2.kmeans seeds from a "
                "global RNG and returns different labels for the same image. Also emits a swatch strip so "
                "the palette can be eyeballed without decoding HEX."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip=(
                        "Image to sample. Normally the output of Pixel Grid Restore, but any image works; a "
                        "batch is pooled into one shared palette so every frame comes out consistent."
                    ),
                ),
                io.Int.Input(
                    "colors",
                    default=DEFAULT_COLORS,
                    min=MIN_COLORS,
                    max=MAX_COLORS,
                    step=1,
                    tooltip=(
                        "Palette size to aim for (8-256). 128 is the default because modern pixel art "
                        "regularly uses 64-256 colours; classic 16-32 is available by lowering this. The "
                        "reported count can be lower if the image does not contain that many distinct colours."
                    ),
                ),
                io.Combo.Input(
                    "method",
                    options=EXTRACT_METHODS,
                    default=PER_TILE,
                    tooltip=(
                        "'Per-Tile k-means' (default) clusters each local window and merges the results, so "
                        "adjacent-but-different tones survive instead of collapsing into one -- this is the "
                        "structure-preserving option, and it costs a bit of speed. 'Global k-means (Lab)' "
                        "clusters the whole image at once: fast, but neighbouring gradient steps get merged. "
                        "'Median Cut' / 'Max Coverage' / 'Octree' delegate to PIL: deterministic, no RNG, "
                        "always RGB (they have no Lab entry point, so 'space' does not apply to them)."
                    ),
                ),
                io.Combo.Input(
                    "space",
                    options=SPACE_OPTIONS,
                    default="Lab",
                    tooltip=(
                        "Colour space the k-means methods cluster in, and the metric used to count pixels per "
                        "entry. 'Lab' is perceptually far better behaved than RGB (RGB k-means notoriously "
                        "drops reds and greens at low colour counts). OpenCV's 8-bit Lab encoding is used "
                        "(L 0-255, a/b offset by 128), not the standard 0-100/+/-128 form. Ignored by the "
                        "PIL-backed methods."
                    ),
                ),
                io.Combo.Input(
                    "sort",
                    options=SORT_OPTIONS,
                    default="Luminance",
                    tooltip=(
                        "Order of the emitted HEX list, which also orders the swatch strip. 'Luminance' = dark "
                        "to light ramp (easiest to compare two palettes by eye). 'Hue' = greys first, then the "
                        "colour wheel. 'Frequency' = most-used colour first."
                    ),
                ),
                io.Boolean.Input(
                    "accent_protect",
                    default=True,
                    tooltip=(
                        "Protect rare but vivid colours (eye highlights, a single coloured accent) from being "
                        "absorbed by the dominant clusters, paying for them out of the colour budget. Belongs "
                        "here, on the node that *chooses* the palette: once a palette is handed to Pixel "
                        "Palette Quantize, that node maps onto it verbatim and must not add colours, or the "
                        "cross-image guarantee breaks. Quantize's own accent_protect therefore only applies "
                        "when you leave its 'palette' input empty."
                    ),
                ),
                io.Int.Input(
                    "tile_size",
                    default=16,
                    min=2,
                    max=512,
                    step=1,
                    tooltip=(
                        "Advanced. Tile edge in input pixels for the 'Per-Tile k-means' method, which clusters "
                        "each tile on its own so local gradients compress without flattening structure. "
                        "Ignored by the other methods. Match it to the pixel-art cell size (see Pixel Grid "
                        "Restore's pixel_size output) when you want one cluster per art pixel."
                    ),
                    advanced=True,
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="palette",
                    tooltip=(
                        "Newline-separated #RRGGBB, one colour per line. Visible, hand-editable, diffable, "
                        "and accepted verbatim by Pixel Palette Quantize."
                    ),
                ),
                io.Image.Output(
                    display_name="palette_image",
                    tooltip=(
                        "Swatch strip: one 16x16 block per colour, 16 per row, in the same order as the HEX "
                        "list. Every pixel of it is a palette colour, so it can be compared against a "
                        "quantized image directly."
                    ),
                ),
                io.Int.Output(
                    display_name="count",
                    tooltip="Number of entries actually returned. Never larger than 'colors'.",
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        colors: int = DEFAULT_COLORS,
        method: str = PER_TILE,
        space: str = "Lab",
        sort: str = "Luminance",
        accent_protect: bool = True,
        tile_size: int = EXTRACT_TILE_SIZE,
    ) -> io.NodeOutput:
        imgs = _torch_image_to_uint8_rgb(image)
        if imgs.ndim != 4 or imgs.shape[-1] != 3:
            raise ValueError(
                f"PixelPaletteExtract: expected IMAGE as [B,H,W,3], got {tuple(imgs.shape)}."
            )

        palette, counts = extract_palette(
            imgs,
            colors=int(colors),
            method=method,
            space=space,
            tile_size=int(tile_size),
            accent_protect=bool(accent_protect),
        )
        palette, counts = sort_palette(palette, counts, sort)

        swatch = palette_swatch_image(palette)
        return io.NodeOutput(
            format_palette_hex(palette),
            _uint8_rgb_to_torch_image(swatch),
            int(palette.shape[0]),
        )


class PixelPaletteQuantize(io.ComfyNode):
    """Snap every pixel onto a palette, optionally deriving or dithering it."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="PixelPaletteQuantize",
            display_name="Pixel Palette Quantize",
            category="image/pixel pipeline",
            description=(
                "Reduce an image to an indexed palette. The palette is either derived on the fly (method / "
                "colors / accent_protect) or supplied as HEX text on the optional 'palette' input, in which "
                "case it is applied verbatim -- that is what makes a set of sprites share one palette. The "
                "'palette' input is lazy: leave it unwired and the entire upstream Pixel Palette Extract "
                "subtree is pruned out of the execution graph instead of being computed and discarded."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="Image to quantize. Batch entries are processed independently but share one palette.",
                ),
                io.Combo.Input(
                    "method",
                    options=QUANTIZE_METHODS,
                    default=PER_TILE,
                    tooltip=(
                        "How the palette is derived. 'Per-Tile k-means' (default) keeps local structure; "
                        "'Global k-means (Lab)' is faster but flattens gradients; 'Median Cut' / "
                        "'Max Coverage' / 'Octree' use PIL and are deterministic; 'Fixed Palette (nearest)' "
                        "requires the 'palette' input and is an error without it. Ignored when 'palette' is "
                        "supplied -- the supplied palette wins, because a fixed palette is the whole point."
                    ),
                ),
                io.Int.Input(
                    "colors",
                    default=DEFAULT_COLORS,
                    min=MIN_COLORS,
                    max=MAX_COLORS,
                    step=1,
                    tooltip=(
                        "Palette size to derive (8-256). Ignored when 'palette' is supplied: the number of "
                        "colours then comes from that string. With accent_protect on, rare colours are paid for "
                        "out of this same budget, so the total never exceeds it."
                    ),
                ),
                io.Int.Input(
                    "tile_size",
                    default=16,
                    min=2,
                    max=512,
                    step=1,
                    tooltip=(
                        "Local window in pixels for 'Per-Tile k-means'. Smaller keeps finer local colour "
                        "variation (and is slower); larger drifts towards global clustering. Very large images "
                        "double it internally to bound the number of tiles. Ignored by the other methods."
                    ),
                ),
                io.Combo.Input(
                    "space",
                    options=SPACE_OPTIONS,
                    default="Lab",
                    tooltip=(
                        "Colour space for clustering *and* for the nearest-colour lookup that assigns every "
                        "pixel. This is the metric the output's accuracy is judged in, so it matters even when "
                        "the palette was derived by a PIL method. 'Floyd-Steinberg' is the one exception: "
                        "Pillow's error diffusion is defined on RGB, so that dither always maps in RGB."
                    ),
                ),
                io.Combo.Input(
                    "dither",
                    options=DITHER_OPTIONS,
                    default=DITHER_NONE,
                    tooltip=(
                        "Off by default on purpose: in regions narrower than ~8-10 px (faces, icons, lettering) "
                        "dithering reads as noise rather than as a blend. 'Bayer 2x2 / 4x4 / 8x8' are ordered "
                        "dithers -- deterministic, and the threshold depends only on pixel coordinates, so the "
                        "same sprite rendered at different sizes dithers identically. 'Floyd-Steinberg' is "
                        "Pillow's error diffusion; it depends on scan order and image size, so different sizes "
                        "of the same sprite will NOT match, and it tends to leave worm-like artefacts on AI "
                        "noise. All modes only ever output palette colours."
                    ),
                ),
                io.Boolean.Input(
                    "accent_protect",
                    default=True,
                    tooltip=(
                        "Reserve palette slots for rare but visually load-bearing colours, which clustering "
                        "normally eats first: saturated pixels covering under 0.1% of the image (a gem, a magic "
                        "FX), and near-black / near-white pixels (eye highlights, thin outlines). They may claim "
                        "up to 1/8 of 'colors', and only if they are still far from the derived palette. "
                        "Ignored when 'palette' is supplied, since adding colours to a fixed palette would break "
                        "cross-sprite consistency."
                    ),
                ),
                io.String.Input(
                    "palette",
                    optional=True,
                    lazy=True,
                    multiline=True,
                    dynamic_prompts=False,
                    default="",
                    tooltip=(
                        "Optional. HEX colours separated by newlines or commas ('#RRGGBB', '#RGB' and bare "
                        "'RRGGBB' all work). When filled or wired, the palette is applied verbatim by nearest "
                        "colour and 'method' / 'colors' / 'accent_protect' are ignored. Leave it empty and the "
                        "upstream Pixel Palette Extract subtree is pruned from the graph -- it is not executed "
                        "and then discarded."
                    ),
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="image",
                    tooltip="Quantized image. Every pixel is one of the palette colours, dithering included.",
                ),
                io.String.Output(
                    display_name="applied_palette",
                    tooltip=(
                        "The palette that was actually used, as HEX text: a supplied palette verbatim "
                        "(duplicates removed, order kept), or the derived palette sorted by luminance so the "
                        "same colour set always produces the same string."
                    ),
                ),
                io.Int.Output(
                    display_name="colors_used",
                    tooltip=(
                        "Number of distinct colours present in the output image. Can be lower than the palette "
                        "size -- a palette entry nothing mapped to simply does not appear."
                    ),
                ),
            ],
        )

    @classmethod
    def validate_inputs(cls, **kwargs):
        """Runs before ``check_lazy_status``, so it must tolerate ``palette=None``.

        Only cheap, structural checks live here: a wrong value from an API caller
        should fail before ComfyUI starts evaluating the lazy branch.
        """
        method = kwargs.get("method", PER_TILE)
        if method not in QUANTIZE_METHODS:
            return f"unknown method {method!r}; expected one of {QUANTIZE_METHODS}."
        space = kwargs.get("space", "Lab")
        if space not in SPACE_OPTIONS:
            return f"unknown space {space!r}; expected one of {SPACE_OPTIONS}."
        dither = kwargs.get("dither", DITHER_NONE)
        if dither not in DITHER_OPTIONS:
            return f"unknown dither {dither!r}; expected one of {DITHER_OPTIONS}."
        try:
            colors = int(kwargs.get("colors", DEFAULT_COLORS))
        except (TypeError, ValueError):
            return f"colors must be an integer, got {kwargs.get('colors')!r}."
        if not (MIN_COLORS <= colors <= MAX_COLORS):
            return f"colors must be between {MIN_COLORS} and {MAX_COLORS}, got {colors}."
        return True

    @classmethod
    def check_lazy_status(cls, palette=MISSING, **kwargs):
        """Ask for ``palette`` only when it is wired but not yet evaluated.

        ``comfy_execution/graph.py`` keeps lazy links out of the initial block, so
        an unwired palette never drags ``PixelPaletteExtract`` in. Returning
        ``["palette"]`` re-queues this node with that one input forced to a strong
        link -- which requires the input to actually *be* a link, hence the strict
        ``is None`` test: a constant (the untouched widget) must never be
        requested.
        """
        if palette is None:
            return ["palette"]
        return []

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        method: str = PER_TILE,
        colors: int = DEFAULT_COLORS,
        tile_size: int = 16,
        space: str = "Lab",
        dither: str = DITHER_NONE,
        accent_protect: bool = True,
        palette=MISSING,
    ) -> io.NodeOutput:
        imgs = _torch_image_to_uint8_rgb(image)
        if imgs.ndim != 4 or imgs.shape[-1] != 3:
            raise ValueError(
                f"PixelPaletteQuantize: expected IMAGE as [B,H,W,3], got {tuple(imgs.shape)}."
            )

        supplied = None
        if not _blank_palette_text(palette):
            supplied = parse_palette_hex(str(palette))

        if supplied is not None and supplied.shape[0] == 0:
            raise ValueError(
                "PixelPaletteQuantize: 'palette' was supplied but contains no colours."
            )
        if supplied is None and method == FIXED_PALETTE:
            raise ValueError(
                "PixelPaletteQuantize: method 'Fixed Palette (nearest)' needs a palette, but the 'palette' "
                "input is empty. Either fill it in / wire it up, or pick one of "
                f"{[m for m in QUANTIZE_METHODS if m != FIXED_PALETTE]} to derive one."
            )

        if supplied is not None:
            # A supplied palette is used exactly as given (deduplicated, order
            # preserved). Sorting it would silently reorder someone's palette, and
            # adding accents to it would break the cross-sprite guarantee.
            palette_rgb = supplied
        else:
            derived, _counts = extract_palette(
                imgs,
                colors=int(colors),
                method=method,
                space=space,
                tile_size=int(tile_size),
                accent_protect=bool(accent_protect),
            )
            palette_rgb, _counts = sort_palette(derived, None, "Luminance")

        if palette_rgb.shape[0] == 0:
            raise ValueError("PixelPaletteQuantize: the derived palette is empty.")

        labels = map_to_palette(imgs, palette_rgb, space=space, dither=dither)
        out_rgb = palette_rgb[labels]  # [B,H,W,3] uint8; dither can only pick palette rows

        out = torch.cat(
            [_uint8_rgb_to_torch_image(out_rgb[i]) for i in range(out_rgb.shape[0])], dim=0
        )
        out = torch.clamp(out, 0.0, 1.0)

        used = int(np.unique(labels).size)
        return io.NodeOutput(out, format_palette_hex(palette_rgb), used)
