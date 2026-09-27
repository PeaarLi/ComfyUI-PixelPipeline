# `PixelBackgroundRemove`

**Category:** `image/pixel pipeline`
**Output kind:** processing node (background keying)

Keys a batch's background out and aligns the frames: takes the background
colour (given or detected), crops every frame to **one shared centred square**
around the union of all content boxes, and optionally flood-keys the
border-connected background to transparency. Ported from the cleaning stage of
the local `webm2spritesheet.py` tool (Webm2Frames), adapted from video files to
ComfyUI `IMAGE` batches.

```
IMAGE -> images (IMAGE), bg_color (STRING)
```

## Inputs

| id | type | default | range | notes |
|---|---|---|---|---|
| `image` | IMAGE | — | — | Frames or images, RGB or RGBA, all the same size. Under `Auto Detect` the background is sampled from the **first frame's** border ring only, so frame 0 must have a representative background. |
| `bg_color` | DYNAMICCOMBO | `Auto Detect` | — | Where the colour comes from. `Auto Detect` (first = default): most frequent colour in frame 0's 2 px border ring. `Fixed Colour`: you pin it — `hex`, a STRING, `#RRGGBB` (`#RGB` and a bare `RRGGBB` also work), exactly one colour. |
| `color_tolerance` | INT | `60` | 0–255 | A pixel counts as background when `\|dR\|+\|dG\|+\|dB\|` from the background colour is ≤ this. Raised from the reference's `12` so a gently varying AI background keys without tuning; flat backgrounds are unaffected. See [Choosing a tolerance](#choosing-a-tolerance). |
| `padding_ratio` | FLOAT | `0.05` | 0.0–1.0 | Crop margin: `side = max(content_w, content_h) + 2 * max(round(side * ratio), 2)`. The 2 px minimum is fixed. |
| `remove_background` | BOOLEAN | `True` | — | Off = the shared crop only, no alpha (the reference's "noclean" variant). The colour is still resolved either way — it decides what counts as content. |
| `outer_only` | BOOLEAN | `True` | — | On: only background-coloured regions **connected to** the cropped frame's border are keyed, so enclosed pockets survive. Off: every matching pixel goes, enclosed pockets included. See [Outer-only vs. everything](#outer-only-vs-everything). |

## Outputs

| display name | type | notes |
|---|---|---|
| `images` | IMAGE | Cropped (and optionally keyed) frames. RGBA when keying removed anything or the input carried real transparency, RGB otherwise — an all-opaque alpha is dropped, batch-wide. |
| `bg_color` | STRING | The colour actually used, as uppercase `#RRGGBB`: the `bg_color` input echoed back, or the detected one when it was left empty. |

## Specifying the colour vs. detecting it

Detection reads the first frame's border ring, which assumes frame 0 is
representative — usually true for a rendered clip, wrong for a first frame that
starts mid-fade or already has a finger over the edge. Pinning the colour also
makes a batch reproducible: the same `#RRGGBB` keys the same colour across
several clips regardless of what each one's first frame looks like.

That choice is a mode selector rather than one text field with "empty means
auto", so which of the two is in play is visible on the node instead of being
inferred from the field's contents — the same idiom `PixelGridRestore` and
`PixelVideoGridRestore` use for their grid modes.

The output echoes whichever colour was used, so it doubles as a record of what
detection chose when the mode was `Auto Detect`.

## Outer-only vs. everything

`outer_only` decides which matching pixels become transparent.

- **On (default).** `scipy.ndimage.label` groups the matching pixels and only
  the groups touching the cropped frame's border are keyed. An enclosed pocket
  survives, and so does background-coloured detail inside the character — a
  white highlight on a white background, a belt the same colour as the void.
  This is the reference script's behaviour.
- **Off.** Every pixel within `color_tolerance` of the background is keyed.
  No labelling is done, so this path is the cheaper one. It clears pockets the
  border cannot reach, at the cost of punching holes wherever the sprite itself
  falls inside the tolerance.

Neither dominates, so both are offered; the default is the conservative one
because a surviving pocket is visible and fixable, while an eaten detail looks
like a rendering bug. On frames with no enclosed pocket the two settings produce
byte-identical output.

### Measured: what the toggle actually costs

On a Qwen-generated sheet keyed at `Fixed Colour #9F48C9`, `color_tolerance 200`
— the tolerance that gradient needs — switching `outer_only` off removed

| frame | extra pixels keyed | of which genuinely flat background |
|---|---|---|
| 0 | 62 559 | **0** (nearest was 120 away from the colour) |
| 1 | 62 559 | **0** |
| 2 | 78 697 | 1 635 (2.1%) |

The 1 635 are the real pocket: a ~66×65 px patch trapped between that
character's shadow and cape, exactly what the toggle exists for. The other ~98%
are the sprite's own light armour plate, which a tolerance of 200 cannot tell
from the background — measured 162–200 away from the colour, i.e. inside the
tolerance but nowhere near the flat purple. Frames 0 and 1 have no pocket at
all, so there the toggle buys nothing and only costs.

**So `outer_only` off is not a free win.** When the problem is a trapped pocket
and the tolerance has had to go high to catch a gradient, prefer reining in the
tolerance (or pinning the colour so the gradient matters less) over turning
connectivity off; reach for the toggle when the pocket is the *only* thing
wrong, which is the case where the sprite's colours sit far from the background.

## Choosing a tolerance

The tolerance is a summed RGB distance, so it is three times more permissive
than a per-channel figure suggests: `60` allows an average of 20 per channel.

The reference script used `12`, which is right for flat colours but silently
does nothing on an AI-generated background. Measured on a Qwen sheet: the
"solid" purple varied by **93–159** summed RGB across the frame, so 12 matched
no pixel at all — nothing was cropped or keyed, with no error to explain why.
That is what the raised default is for.

The same measurement is the ceiling warning: clearing that gradient needed
**~200**, and a tolerance that loose also widens what counts as content, so a
sprite pixel near the background colour can be eaten. Raise it until the sprite
survives, and prefer `outer_only` off over a very high tolerance when the
problem is only a trapped pocket.

## Why one crop for the whole batch

Per-frame tight crops make every cell a different size and let the character
drift inside them — sliced in an engine, the animation *swims*. Unioning all
frames' content boxes and cropping once to a centred square keeps every pixel
of the character at the same offset in every cell, which is the whole point of
this stage. The trade-off is deliberate: cells are looser than per-frame crops
would give.

## Why the default is flood fill, not a plain colour key

A plain colour key erases *every* pixel close to the background colour —
including background-coloured details inside the character (eyes, clothing
patches), leaving holes. With `outer_only` on (the default) the node labels
connected background-coloured regions and only erases the ones **connected to
the cropped frame's border**, so an enclosed island keeps its alpha. Turning it
off is exactly the plain colour key, which is occasionally what you want; see
[Outer-only vs. everything](#outer-only-vs-everything).

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
| `Fixed Colour`, `#0AC81E` | byte-identical to the detection result — the mode feeds the same path |
| `Fixed Colour`, `#C81E0A` | nothing matches, so nothing is keyed and every pixel is content: 3×24×24 RGB |
| `remove_background=False` | `images` 3×15×15 RGB, same crop, no alpha |
| `outer_only=False` | same crop; an enclosed background-coloured island is keyed too |
| all-background batch | full 24×24 frames, fully transparent alpha, note printed |

## See also

- [`PixelSpriteSheet`](PixelSpriteSheet.md) — packs the keyed frames into the sheet
- [`PixelPaletteApply`](PixelPaletteApply.md) — quantise the cleaned frames to a shared palette before packing
- [`PixelExportScale`](PixelExportScale.md) — crisp integer upscale of the finished sheet
