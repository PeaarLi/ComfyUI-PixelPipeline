# `PixelSpritePrepare`

**Category:** `image/pixel pipeline`
**Output kind:** processing node (sprite pre-stage)

Cleans a batch of animation frames for sprite-sheet packing: detects the
background colour, crops every frame to **one shared centred square** around
the union of all content boxes, and optionally flood-keys the border-connected
background to transparency. Ported from the cleaning stage of the local
`webm2spritesheet.py` tool (Webm2Frames), adapted from video files to ComfyUI
`IMAGE` batches.

```
IMAGE -> images (IMAGE), bg_color (STRING)
```

## Inputs

| id | type | default | range | notes |
|---|---|---|---|---|
| `image` | IMAGE | — | — | Animation frames, RGB or RGBA, all the same size. The background colour is sampled from the **first frame's** border ring only, so frame 0 must have a representative background. |
| `color_tolerance` | INT | `12` | 0–255 | A pixel counts as background when the summed per-channel RGB difference from the detected colour is ≤ this. `0` keys exact matches only. |
| `padding_ratio` | FLOAT | `0.05` | 0.0–1.0 | Crop margin: `side = max(content_w, content_h) + 2 * max(round(side * ratio), 2)`. The 2 px minimum is fixed. |
| `remove_background` | BOOLEAN | `True` | — | Off = detection + shared crop only (the reference's "noclean" variant); no alpha is created. |

## Outputs

| display name | type | notes |
|---|---|---|
| `images` | IMAGE | Cropped (and optionally keyed) frames. RGBA when keying removed anything or the input carried real transparency, RGB otherwise — an all-opaque alpha is dropped, batch-wide. |
| `bg_color` | STRING | Detected background colour as uppercase `#RRGGBB` text, for inspection. |

## Why one crop for the whole batch

Per-frame tight crops make every cell a different size and let the character
drift inside them — sliced in an engine, the animation *swims*. Unioning all
frames' content boxes and cropping once to a centred square keeps every pixel
of the character at the same offset in every cell, which is the whole point of
the pre-stage. The trade-off is deliberate: cells are looser than per-frame
crops would give.

## Why flood fill, not a colour key

A plain colour key erases *every* pixel close to the background colour —
including background-coloured details inside the character (eyes, clothing
patches), leaving holes. This node labels connected background-coloured
regions (`scipy.ndimage.label`) and only erases the ones **connected to the
cropped frame's border**. An enclosed island of the exact background colour
keeps its alpha.

## Port notes

- **Content test with alpha.** The reference uses `alpha < 255` as the content
  selector, which marks the transparent *background* as content — for an
  opaque sprite on a transparent canvas that is the whole frame, so the crop
  could never shrink. This port uses `alpha > 0`. The colour-difference path
  (no transparency) is untouched.
- **Inclusive bounding box.** The reference stores `content_w = right - left`,
  under-measuring by one pixel. Kept deliberately: the 2 px minimum padding
  absorbs it, and changing it would diverge every crop size from the original
  tool.
- **`round()` is banker's rounding.** A centred odd offset snaps down
  (`round(0.5) == 0`) — same call as the reference, same crops.
- **No content found** (every frame is pure background): falls back to the
  full frame and prints one note, rather than raising. With keying on, such a
  batch comes out fully transparent.
- **Flat backgrounds are assumed.** The reference targets renders on a solid
  colour, and the tolerance is a per-pixel distance from *one* detected
  colour. AI-generated gradient or textured backgrounds can vary by 90+ summed
  RGB units across the frame — no tolerance keys those without also eating the
  sprite, so nothing gets cropped or keyed (measured on Qwen output). Flatten
  or replace such backgrounds upstream first.

## Example

3 frames 24×24, green `(10,200,30)` background, a 4×4 red square at a
different position per frame (union box 9×11 px):

| setting | result |
|---|---|
| defaults | `images` 3×15×15 RGBA (raw 11 + 2×2 padding), background keyed, `bg_color="#0AC81E"` |
| `remove_background=False` | `images` 3×15×15 RGB, same crop, no alpha |
| all-background batch | full 24×24 frames, fully transparent alpha, note printed |

## See also

- [`PixelSpriteSheet`](PixelSpriteSheet.md) — packs the cleaned frames into the sheet
- [`PixelPaletteApply`](PixelPaletteApply.md) — quantise the cleaned frames to a shared palette before packing
- [`PixelExportScale`](PixelExportScale.md) — crisp integer upscale of the finished sheet
