"""The two palette nodes: ``PixelPaletteExtract`` and ``PixelPaletteApply``.

They are a matched pair, and the split between them is strict:

* :class:`PixelPaletteExtract` **chooses** a palette. It answers "which colours
  is this sprite made of?" and hands the answer out as a plain HEX string.
* :class:`PixelPaletteApply` **applies** a palette. It answers "make this image
  use only those colours" and accepts that same string back. It derives
  nothing: a palette is always supplied, either wired from Extract or typed
  into its own widget.

Because the palette travels as text it is visible in the UI, editable by hand,
pasteable into a commit message, and reusable across an entire sprite set --
which is the only way to keep a set of sprites colour-consistent, since
per-image k-means cannot be (``cv2.kmeans`` seeds from a global RNG and openly
warns that the same image may come back with different labels).

Why there is exactly one node that chooses a palette
----------------------------------------------------
An earlier version had Apply derive a palette of its own when none was wired
(``method`` / ``colors`` / ``tile_size`` / ``accent_protect``, plus a
"Fixed Palette (nearest)" method). That made four of its seven inputs silently
inert the moment a palette *was* wired, which is the common case -- the node
had two mutually exclusive parameter sets on one panel and no way to say which
one was live. The derivation path is gone. **Extract is now the only place a
palette is chosen**, which also removes a documented asymmetry: "Extract then
Apply" and "Apply alone" used to be different pipelines producing different
colour sets from the same image.

A consequence worth knowing: **``accent_protect`` and ``colors`` therefore live
only on Extract**, and the palette Apply receives is applied verbatim --
deduplicated, order preserved. Adding colours to a supplied palette would break
the cross-image subset guarantee, which is the whole reason the palette is
passed around as text in the first place.

``Per-Tile k-means`` is the default method of Extract and has a tile-size
parameter, exposed as the advanced ``tile_size``. The module constant
``_palette_utils.EXTRACT_TILE_SIZE`` is only the default value.
"""

from __future__ import annotations

import numpy as np
import torch

from comfy_api.latest import io

from ._image_utils import (
    _join_rgb_alpha,
    _split_rgb_alpha,
    _torch_image_to_uint8_bhwc,
    _uint8_rgb_to_torch_image,
)
from ._palette_utils import (
    DITHER_NONE,
    DITHER_OPTIONS,
    EXTRACT_METHODS,
    EXTRACT_TILE_SIZE,
    PER_TILE,
    SORT_OPTIONS,
    SPACE_OPTIONS,
    extract_palette,
    format_palette_hex,
    map_to_palette,
    palette_swatch_image,
    parse_palette_hex,
    sort_palette,
)

__all__ = ["PixelPaletteExtract", "PixelPaletteApply"]

MAX_COLORS = 256
MIN_COLORS = 8
DEFAULT_COLORS = 128


def _palette_text_is_empty(text) -> bool:
    """True when the ``palette`` input carries no colours.

    The input is a required multiline widget, so "no palette" is now only ever
    an empty (or whitespace-only) string. There is no unwired state to test for:
    the input is neither ``optional`` nor ``lazy`` any more, because Apply can
    no longer do anything useful without a palette.
    """
    if text is None:
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
                "#RRGGBB text. Feed the string to Pixel Palette Apply to force a whole sprite set onto "
                "one shared palette -- per-image clustering cannot do that, because cv2.kmeans seeds from a "
                "global RNG and returns different labels for the same image. Also emits a swatch strip so "
                "the palette can be eyeballed without decoding HEX. This is the only node that chooses a "
                "palette; Pixel Palette Apply only ever applies one."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip=(
                        "Image to sample. Normally the generator's own output (the image *before* Pixel "
                        "Grid Restore), but any image works; a batch is pooled into one shared palette so "
                        "every frame comes out consistent."
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
                        "regularly uses 64-256 colours; classic 16-32 is available by lowering this. This is "
                        "an UPPER BOUND, not a promise: the reported count is lower whenever the image does "
                        "not hold that many distinct colours, and 'Per-Tile k-means' in particular cannot "
                        "invent colours its tiles never produced. With accent_protect on, protected accents "
                        "are paid for out of this same budget."
                    ),
                ),
                io.Combo.Input(
                    "method",
                    options=EXTRACT_METHODS,
                    default=PER_TILE,
                    tooltip=(
                        "'Per-Tile k-means' (default) clusters each local window and merges the results, so "
                        "adjacent-but-different tones survive instead of collapsing into one -- this is the "
                        "structure-preserving option, and it costs a bit of speed. Note that in this mode "
                        "'colors' sets the size of the final merge, while each individual tile is capped at "
                        "roughly sqrt(tile area) clusters -- so raising 'colors' cannot recover detail finer "
                        "than the tiles saw. 'Global k-means (Lab)' clusters the whole image at once: fast, "
                        "but neighbouring gradient steps get merged. 'Median Cut' / 'Max Coverage' / 'Octree' "
                        "delegate to PIL: deterministic, no RNG, always RGB (they have no Lab entry point)."
                    ),
                ),
                io.Combo.Input(
                    "space",
                    options=SPACE_OPTIONS,
                    default="Lab",
                    tooltip=(
                        "Colour space the k-means methods cluster in, and the metric used to count pixels per "
                        "entry (which is what the 'Frequency' sort orders by). 'Lab' is perceptually far "
                        "better behaved than RGB (RGB k-means notoriously drops reds and greens at low colour "
                        "counts). OpenCV's 8-bit Lab encoding is used (L 0-255, a/b offset by 128), not the "
                        "standard 0-100/+/-128 form. "
                        "IMPORTANT, and easy to get wrong: this is NOT limited to the k-means methods. It "
                        "also sets the metric that accent_protect measures distance in, and the space a "
                        "multi-image batch is merged in (PIL palettes are converted into it first). Only the "
                        "*initial* per-image PIL extraction ignores it, because PIL has no Lab entry point. "
                        "So with a batch or with accent_protect on, 'space' changes the output even for "
                        "Median Cut."
                    ),
                ),
                io.Combo.Input(
                    "sort",
                    options=SORT_OPTIONS,
                    default="Luminance",
                    tooltip=(
                        "Order of the emitted HEX list, which also orders the swatch strip. 'Luminance' = dark "
                        "to light ramp (easiest to compare two palettes by eye). 'Hue' = greys first, then the "
                        "colour wheel. 'Frequency' = most-used colour first, counted in 'space'. Sorting "
                        "never changes which colours are in the palette, only their order."
                    ),
                ),
                io.Boolean.Input(
                    "accent_protect",
                    default=True,
                    tooltip=(
                        "Protect rare but vivid colours (eye highlights, a single coloured accent) from being "
                        "absorbed by the dominant clusters, paying for them out of the 'colors' budget -- so "
                        "it can lower the returned count. Belongs only on this node, the one that *chooses* a "
                        "palette: Pixel Palette Apply maps onto a supplied palette verbatim and must never add "
                        "colours, or the cross-image consistency guarantee breaks."
                    ),
                ),
                io.Int.Input(
                    "tile_size",
                    default=EXTRACT_TILE_SIZE,
                    min=2,
                    max=512,
                    step=1,
                    tooltip=(
                        "Advanced. Tile edge, in pixels OF THIS NODE'S INPUT IMAGE, for the 'Per-Tile k-means' "
                        "method, which clusters each tile on its own so local gradients compress without "
                        "flattening structure. Ignored by the other methods. "
                        "Which value you want depends entirely on what is wired into 'image'. Fed the "
                        "generator's raw output, set it to the art's cell size (matching Pixel Grid Restore's "
                        "'pixel_size' output) to get roughly one cluster per art pixel. Fed the output of "
                        "Pixel Grid Restore -- where one art pixel is already exactly one pixel -- that "
                        "number is far too large and will over-smooth; use a small value instead."
                    ),
                    advanced=True,
                ),
            ],
            outputs=[
                io.String.Output(
                    display_name="palette",
                    tooltip=(
                        "Newline-separated #RRGGBB, one colour per line. Visible, hand-editable, diffable, "
                        "and accepted verbatim by Pixel Palette Apply."
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
                    tooltip=(
                        "Number of entries actually returned. Never larger than 'colors', and often smaller "
                        "-- see the 'colors' tooltip for why."
                    ),
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
        imgs = _torch_image_to_uint8_bhwc(image)
        if imgs.ndim != 4 or imgs.shape[-1] not in (3, 4):
            raise ValueError(
                f"PixelPaletteExtract: expected IMAGE as [B,H,W,3] or [B,H,W,4], "
                f"got {tuple(imgs.shape)}."
            )

        # Alpha is a mask, not a colour. Handing RGBA to a k-means quantiser
        # would let it spend palette entries on "transparent", which is not
        # something anyone wants a colour swatch of. Accepted and ignored here.
        rgb, _alpha = _split_rgb_alpha(imgs)

        palette, counts = extract_palette(
            rgb,
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


class PixelPaletteApply(io.ComfyNode):
    """Snap every pixel onto a supplied palette. Derives nothing."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="PixelPaletteApply",
            display_name="Pixel Palette Apply",
            category="image/pixel pipeline",
            description=(
                "Reduce an image to an indexed palette supplied as HEX text. Wire Pixel Palette Extract's "
                "'palette' output in, or type / paste HEX directly into the widget. The palette is applied "
                "verbatim by nearest colour, so a whole set of sprites can be forced onto one shared "
                "palette -- and the output can never contain a colour outside it. This node never derives a "
                "palette of its own; that is Pixel Palette Extract's job."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip="Image to quantize. Batch entries are processed independently but share one palette.",
                ),
                io.String.Input(
                    "palette",
                    multiline=True,
                    dynamic_prompts=False,
                    default="",
                    tooltip=(
                        "REQUIRED. HEX colours separated by newlines or commas ('#RRGGBB', '#RGB' and bare "
                        "'RRGGBB' all work). Wire Pixel Palette Extract's 'palette' output here to share one "
                        "palette across a sprite set, or type / paste the colours in directly. Applied "
                        "verbatim: duplicates are removed and the order you gave is preserved, never "
                        "re-sorted. An empty value is an error -- use Pixel Palette Extract to choose a "
                        "palette first (or paste one)."
                    ),
                ),
                io.Combo.Input(
                    "space",
                    options=SPACE_OPTIONS,
                    default="Lab",
                    tooltip=(
                        "The metric every pixel's nearest palette colour is measured in. This is the metric "
                        "the output's accuracy is judged in, so it matters regardless of how the palette was "
                        "produced. 'Floyd-Steinberg' is the one exception: Pillow's error diffusion is "
                        "defined on RGB, so that dither always maps in RGB. OpenCV's 8-bit Lab encoding is "
                        "used (L 0-255, a/b offset by 128), not the standard 0-100/+/-128 form."
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
            ],
            outputs=[
                io.Image.Output(
                    display_name="image",
                    tooltip="Quantized image. Every pixel is one of the palette colours, dithering included.",
                ),
                io.String.Output(
                    display_name="applied_palette",
                    tooltip=(
                        "The palette that was actually used, as HEX text: the supplied palette verbatim, "
                        "with duplicates removed and order preserved. Echoing it back means a downstream node "
                        "or a saved prompt carries the exact colour set that produced this image."
                    ),
                ),
                io.Int.Output(
                    display_name="colors_used",
                    tooltip=(
                        "Number of distinct colours present in the output image. Usually lower than the "
                        "palette size -- a palette entry nothing mapped to simply does not appear."
                    ),
                ),
            ],
        )

    @classmethod
    def validate_inputs(cls, palette=None):
        """Reject an empty palette early. ``None`` is NOT empty -- see below.

        Two details of ComfyUI's validation phase drive this signature, and both
        were found the hard way:

        **1. A wired palette arrives here as ``None``.** Validation runs before
        any node executes, and ``execution.get_input_data()`` marks every
        *linked* input missing when it is called without an ``execution_list``
        -- which is exactly how ``execution.validate_inputs()`` calls it. So
        ``None`` means "this input is linked and upstream has not run yet", not
        "no palette". Treating it as empty makes a wired palette impossible to
        validate, i.e. it bricks the node for the one wiring this pack is built
        around. Only a genuinely blank *string* -- the untouched widget -- is an
        error here; ``execute()`` is what enforces the real value, once the
        upstream palette has actually been produced.

        **2. The parameter list is the error's address, so there is no
        ``**kwargs``.** ``execution.py`` derives ``validate_function_inputs``
        from ``argspec.args``, builds ``input_filtered`` from it, and then
        appends the returned string to its error list *once per entry of
        ``input_filtered``*. A ``**kwargs`` validator therefore reports the same
        message against every input on the node -- the user sees four identical
        "palette is required" errors, one per socket. Naming only ``palette``
        makes the message point at the socket that is actually wrong.

        A second, quieter cost of ``**kwargs``: ``validate_has_kwargs`` also
        gates ComfyUI's own min/max and combo-membership checks
        (``execution.py``), so it silently disabled them for the whole node.
        Since this validator no longer takes ``**kwargs``, ``space`` and
        ``dither`` are checked by ComfyUI itself -- better, and correctly
        attributed -- so there is no need to re-check them by hand.
        """
        if palette is None:
            # Linked, or key absent entirely. Not judgeable here.
            return True
        if _palette_text_is_empty(palette):
            return (
                "palette is required: wire Pixel Palette Extract's 'palette' output in, "
                "or paste HEX colours (#RRGGBB, one per line or comma-separated) into the widget."
            )
        return True

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        palette: str,
        space: str = "Lab",
        dither: str = DITHER_NONE,
    ) -> io.NodeOutput:
        imgs = _torch_image_to_uint8_bhwc(image)
        if imgs.ndim != 4 or imgs.shape[-1] not in (3, 4):
            raise ValueError(
                f"PixelPaletteApply: expected IMAGE as [B,H,W,3] or [B,H,W,4], "
                f"got {tuple(imgs.shape)}."
            )

        if _palette_text_is_empty(palette):
            raise ValueError(
                "PixelPaletteApply: the 'palette' input is empty, and this node only applies a palette -- "
                "it never derives one. Either wire Pixel Palette Extract's 'palette' output into it, or "
                "type / paste HEX colours (#RRGGBB, one per line or comma-separated) into the widget."
            )

        # Colour and alpha are separated for the whole node: the palette is a
        # colour set, and alpha is re-attached untouched at the end. A
        # transparent pixel has no colour to quantise, and quantising the mask
        # would turn a soft edge into a palette entry.
        rgb, alpha = _split_rgb_alpha(imgs)

        # A supplied palette is used exactly as given (deduplicated, order
        # preserved). Sorting it would silently reorder someone's palette.
        palette_rgb = parse_palette_hex(str(palette))
        if palette_rgb.shape[0] == 0:
            raise ValueError(
                "PixelPaletteApply: the 'palette' input contains no colours."
            )

        labels = map_to_palette(rgb, palette_rgb, space=space, dither=dither)
        out_rgb = palette_rgb[labels]  # [B,H,W,3] uint8; dither can only pick palette rows

        # Alpha passes through verbatim -- deliberately no "drop an all-opaque
        # alpha" rule here, unlike PixelGridRestore. This node's contract is
        # "map colours, touch nothing else", so an RGBA input yields an RGBA
        # output whose alpha is byte-identical to the input's.
        if alpha is not None:
            out_rgb = _join_rgb_alpha(out_rgb, alpha)

        # Already batched, so convert directly rather than per image through
        # ``_uint8_rgb_to_torch_image`` (which unsqueezes for a single image).
        out = torch.from_numpy(np.ascontiguousarray(out_rgb)).float() / 255.0
        out = torch.clamp(out, 0.0, 1.0)

        used = int(np.unique(labels).size)
        return io.NodeOutput(out, format_palette_hex(palette_rgb), used)
