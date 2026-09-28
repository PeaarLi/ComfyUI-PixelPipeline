# ComfyUI-PixelPipeline

Turn AI-generated "fake pixel art" into **real** pixel art: detect (or force) the
pixel grid, resample one colour per cell, optionally quantise to a shared
palette, then export at a crisp integer scale. A separate branch keys a batch's
background out and packs the frames into a **game-ready sprite sheet** — straight
from a generated video, if you like.

![Video to sprite sheet, end to end](docs/screenshots/workflow-sprite-sheet.png)

*One generated clip through the whole pack: grid-restore the video, pick and apply
one shared palette, remove the background, pack the frames.*

- **Seven nodes, one category** (`image/pixel pipeline`): grid restore for stills
  and video, palette extract/apply, export scale, background removal, sprite-sheet
  packing.
- **Strict on purpose.** A node that cannot do its job *loudly fails* instead of
  silently passing the input through — no giant un-pixelated images saved by
  accident.
- **Deterministic.** Every k-means call is seeded; run the same workflow twice
  and you get byte-identical output.
- **V3 node schema** (`comfy_api.latest`), RGBA-aware end to end (Qwen-Image-2.1's
  4-channel VAE included).

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/PeaarLi/ComfyUI-PixelPipeline
pip install -r ComfyUI-PixelPipeline/requirements.txt
```

ComfyUI Manager installs `requirements.txt` automatically.

## Quick start

**Stills:** `LoadImage` → `PixelGridRestore` → (`PixelPaletteExtract` →
`PixelPaletteApply`) → `PixelExportScale` → `SaveImage`.

**Sprite sheet:** `LoadImage` (or `LoadVideo` → `PixelVideoGridRestore`) →
`PixelBackgroundRemove` → `PixelSpriteSheet` → `PreviewImage`. The `cell_width` /
`cell_height` outputs are Unity's *Grid By Cell Size* values.

The nodes live under **`image/pixel pipeline`**. If the
library's type filter shows nothing, see
[the frontend-bug note](#known-frontend-bug-the-inputoutput-type-filters-hide-every-custom-node) —
typing `pixel` in the search box is the workaround.

## Status

| Stage | Node | Docs |
|---|---|---|
| 1. Grid restore | `PixelGridRestore` | [`docs/PixelGridRestore.md`](docs/PixelGridRestore.md) |
| 1v. Grid restore, **video** | `PixelVideoGridRestore` | [`docs/PixelVideoGridRestore.md`](docs/PixelVideoGridRestore.md) |
| 2. Palette extract | `PixelPaletteExtract` | [`docs/PixelPaletteExtract.md`](docs/PixelPaletteExtract.md) |
| 3. Palette apply | `PixelPaletteApply` | [`docs/PixelPaletteApply.md`](docs/PixelPaletteApply.md) |
| 4. Export scale | `PixelExportScale` | [`docs/PixelExportScale.md`](docs/PixelExportScale.md) |
| 5. Background remove | `PixelBackgroundRemove` | [`docs/PixelBackgroundRemove.md`](docs/PixelBackgroundRemove.md) |
| 6. Sprite sheet | `PixelSpriteSheet` | [`docs/PixelSpriteSheet.md`](docs/PixelSpriteSheet.md) |

## Pipeline

```
IMAGE
  │
  ├─► [PixelGridRestore] ──► IMAGE + grid_w + grid_h + pixel_size + detected + confidence
  │         │
  │         ├─► [PixelPaletteExtract] ──► palette (HEX) ─┐   (optional)
  │         │                                            │
  │         └─► [PixelPaletteApply] ◄────────────────────┘   (optional)
  │                   │
  └───────────────────┴──► [PixelExportScale] ──► IMAGE + width + height

IMAGE (frame batch)
  │
  └─► [PixelBackgroundRemove] ──► images (keyed, aligned) + bg_color   (optional)
            │
            └─► [PixelSpriteSheet] ──► sheet + columns + rows + cell_width + cell_height

VIDEO
  │
  └─► [PixelVideoGridRestore] ──► video (4:2:0 — straight to SaveVideo)
                                  audio + fps (for the rebuild below)
                                  grid_w + grid_h + pixel_size + detected + confidence
            │
            └─► images ──► [PixelPaletteExtract] ──► [PixelPaletteApply] ──► [PixelExportScale]
                                                                                   │
                             [PixelVideoGridRestore].audio + .fps ──► [CreateVideo] ◄─┘
                                                                          │
                                                                          └─► SaveVideo
```

The video branch is the stills pipeline with one stage in front and one rebuild
behind. **`PixelVideoGridRestore`'s `images` output — not its `video` output — is
the one the palette chain runs on**, because the video path is encoded 4:2:0 and
would not carry a palette's exact colours; the `audio` and `fps` outputs exist so
the quantised frames can be rebuilt into a video without re-ingesting the source.
See [Video caveats](#video-caveats).

The palette branch is **optional, and the two nodes have a strict division of
labour**: `PixelPaletteExtract` is the only node that *chooses* a palette, and
`PixelPaletteApply` only ever *applies* one. Wire Extract's `palette` output into
Apply, or paste hex straight into Apply's widget — either way, unwired means the
Extract subtree simply is not part of the graph.

That split is deliberate. An earlier version had Apply derive a palette of its
own when none was wired, which meant four of its inputs went silently inert the
moment one *was* wired — two mutually exclusive parameter sets on one panel with
no way to see which was live. It also made "Extract then Apply" and "Apply alone"
different pipelines that produced different colour sets from the same image.
Both problems are gone.

### Transparency (RGBA)

Qwen-Image-2.1's VAE carries four channels, so `VAEDecode` emits RGBA on every
run. The pipeline handles that end to end: `PixelGridRestore` resamples alpha
onto the *exact same* grid lines as the colour so a transparent sprite's mask
stays pixel-aligned, `PixelPaletteApply` passes alpha through untouched, and
`PixelExportScale` scales it along with the colour channels. An all-opaque alpha
is dropped by `PixelGridRestore`, so ordinary generations still come out as
plain RGB rather than 4 channels.

On the video branch, `PixelVideoGridRestore` reads either RGBA frames or the
separate `VideoComponents.alpha` slot. The restored mask is kept on `images`
and in the rebuilt video's alpha slot. File encoding may still discard alpha;
see [Video caveats](#video-caveats).

## Finding the nodes in the node library

The seven nodes live under **`image/pixel pipeline`**. They also
show up under the **Extensions** source group, because custom nodes are always
classified by `python_module`, never by category — using a core-looking category
does **not** fold a pack into the built-in "Comfy" group.

### Known frontend bug: the input/output type filters hide every custom node

If you open the node library's **input / output** type filter and tick
`IMAGE`, **none of these nodes will appear — and neither will any other custom
node.** This is a bug in the frontend, not in this pack, and it cannot be worked
around from inside a node package.

The cause, verified against the frontend source:

```
NodeSearchContent.vue      searchNode(query, filters, { limit: 64 })
nodeSearchService.ts:99    return results.slice(0, options.limit)
```

The truncation happens **after** filtering, and the result order is the raw
`/object_info` dictionary order — all core nodes first, all custom nodes last.
**307** nodes accept an `IMAGE` input and this pack's nodes rank **301st–306th**,
so the 64-slot cut-off is filled entirely by core nodes. The first custom node
with an `IMAGE` input is this pack's own `PixelGridRestore` at 301st, i.e. every
custom node is cut. Raising the limit or reordering the pack changes nothing.
(Numbers measured headlessly with `tools/measure_image_nodes.py`.)

**Workaround — type text as well as ticking the filter.** With a non-empty query
the results come from Fuse relevance ranking, the hit count drops far below 64,
and nothing gets truncated. Typing `pixel` with `IMAGE` ticked lists all six of
the pack's `IMAGE`-input nodes.

`PixelVideoGridRestore` takes a **`VIDEO`** input rather than an `IMAGE` one, so
it does not change the counts above (the pack has six `IMAGE`-input nodes either
way). But the same truncation applies if you tick `VIDEO` — a custom node with a
`VIDEO` input may simply not be listed — and the workaround is identical: type
text in the search box as well as ticking the filter.

The real fix belongs upstream: filtering should not truncate, or the limit should
only apply to unfiltered searches.

## Nodes

### `PixelGridRestore`

`IMAGE -> IMAGE, grid_w, grid_h, pixel_size, detected, confidence`

Three ways to determine the grid, presented as **one mode selector that shows
only the selected mode's parameters** — no dead knobs sitting on the node:

- **Auto Detect** — upstream FFT + gradient detection. Fast when it works, but
  it often fails on AI output: the FFT path only survives when the implied pixel
  edge is 4–20 px, and the gradient fallback finds zero peaks on hard-edged
  pixel art (Sobel responds equally on both columns of a 1 px step). Its
  parameters are `min_size`, `max_pixel_size` and `peak_width`.
- **By Pixel Size** — you give the target pixel edge length (e.g. `16`); the node
  converts it internally to a cell count with
  `grid_w = round(W / pixel_size)`, `grid_h = round(H / pixel_size)`.
  This is the reliable path and the reason the node exists.
- **By Grid Count** — you give the cell counts directly.

Other inputs: `sampling` (`Majority Cluster` / `Center Sample` / `Median`),
`backend` (`Auto` / `OpenCV` / `Lightweight` — see the notes below before
picking `Lightweight`), and two advanced parameters (`refine_intensity`,
`fix_square`) that apply to all three modes.

**`max_pixel_size` is why this node was rebuilt, and it defaults to upstream's
own value.** Upstream's `detect_grid_scale` hardcodes `max_pixel_size = 20.0` as
a *local variable* inside the function, and the validity gate is an OR of four
conditions that `min_size` and `max_ratio` cannot defeat. So an image whose cells
are genuinely wider than 20 px has its — probably correct — FFT result thrown
away, and control drops to the gradient fallback, which has **no validity gate at
all** and invents a pitch. Measured: a 1024 px image with coarse cells came back
as a **205-cell grid at exactly 5.00 px**, raised nothing, and got saved.

Because the ceiling is a local, exposing it meant porting the function
(`_detect_grid_scale_port`). At the default 20.0 the node calls upstream's own
code, so **behaviour is unchanged by construction rather than by test**; the port
runs only once you move the number.

The honest downside: raising the ceiling also lets aliased FFT estimates through,
turning some *loud failures* into *quiet wrong answers* — a 24 px-cell test image
is rejected at 20 and accepted as an aliased 21 cells at 64. That is exactly why
`confidence` stays an output-only diagnostic, and why the tooltip tells you to
watch it.

**`confidence` is a diagnostic, not a gate.** Upstream auto-detection can
return a grid that looks plausible and is wrong: a blurred gradient scores 64
cells at an implied 8.00 px edge while a real pixel image scores 125 cells at
8.192 px, and both sit inside upstream's own validity window, so a range check
cannot separate them. Grid *periodicity* can — measured 1.0–2.6 for gradients,
photos and text posters against 9.1–57.8 for real grids. Nothing is blocked on
the value; wire it to a condition node if you want to gate on it. The manual
modes are scored too — `detected=False` with a *high* confidence means your
forced grid agrees with the content, and a low one means the forced pitch does
not match the image's own periodicity. Either way the grid is applied as asked.

**Failures raise.** Upstream signals "I could not find a grid" by returning
`None, None, image` — the *same object*, with no exception. A node that only
keeps the image therefore emits the untouched input, and a downstream upscale
blows it up into a huge non-pixelated image with no warning. This node checks
`w is None` and raises, telling you to switch to `By Pixel Size` /
`By Grid Count` — or to raise `max_pixel_size` if the cells look wider than 20 px.

### `PixelVideoGridRestore`

`VIDEO -> video, images, audio, fps, grid_w, grid_h, pixel_size, detected, confidence`

Video v2 uses **one regular grid with exact manual dimensions**. It samples each
frame independently; no reference-frame substitution or per-line refinement.
`Robust Cell` selects a centre-weighted representative from current-cell samples.
`Center Sample`, `Majority Cluster` and `Median` remain available.

Auto Detect votes on per-frame boundary spectra (at most 32 probes, including
both ends). Weak evidence uses `fallback_pixel_size` and reports `detected=False`.
It does not average moving images to infer a grid.

`temporal_window=0` is off. Positive values select only current-cell colours in
flat, consistent regions. Motion, local edges, transparency changes and cuts
reject temporal evidence. No temporal RGB averaging, recursive feedback, or
smallest-colour tie rule. The new conservative filter can leave motion-region
flicker; start at 1 and compare against Off.

For detailed AI sprites, try `By Pixel Size`, 3–8, `Robust Cell`, no dithering.
Feed `images` into the existing palette/nearest-export chain; the direct video
output is unquantised. Exact palette membership is checked before encoding,
since 4:2:0 encoding changes colours. Audio and the original in-graph Fraction
are retained; `fps` is a float for downstream CreateVideo.

Saved frontend workflows are migrated before widget assignment: obsolete video
`refine_intensity` / `fix_square` values are removed and a fallback pitch is added.
The three grid-mode names, node ID and nine output slots are unchanged. The same
parameters on **image** nodes are unaffected. See the
[video reference](docs/PixelVideoGridRestore.md) for changed parameter meanings,
limitations and migration details.

### `PixelPaletteExtract`

`IMAGE -> palette (STRING), palette_image (IMAGE), count (INT)`

Extracts the palette an image is actually made of and emits it as plain
newline-separated `#RRGGBB` text: visible, hand-editable, diffable, and accepted
verbatim by `PixelPaletteApply`. Also emits a swatch strip so the palette can be
eyeballed without decoding hex. **This is the only node in the pack that chooses
a palette.**

Five extraction methods, defaulting to **per-tile k-means in Lab**, which keeps
local structure intact instead of flattening adjacent tones the way global
clustering does. Clustering is delegated to `cv2.kmeans` and
`PIL.Image.quantize` — nothing is re-implemented — and every k-means call is
seeded so the same input always produces byte-identical output.

`colors` is an **upper bound, not a promise**: the returned count is lower
whenever the image does not hold that many distinct colours. In the default
per-tile mode it sets the size of the final merge, while each tile is capped at
roughly `sqrt(tile area)` clusters — so raising it cannot recover detail finer
than the tiles saw.

Full reference: [`docs/PixelPaletteExtract.md`](docs/PixelPaletteExtract.md).

### `PixelPaletteApply`

`IMAGE -> image (IMAGE), applied_palette (STRING), colors_used (INT)`

Reduces an image to an indexed palette supplied as hex text — wired from
`PixelPaletteExtract`, or typed / pasted into its own widget. Sharing one palette
is what keeps a set of sprites colour-consistent: the output can only ever
contain colours from that palette.

`palette` is **required**. The node derives nothing, so an empty palette is an
error rather than a fallback, and `applied_palette` echoes the supplied palette
back verbatim — duplicates removed, **order preserved, never re-sorted**.

Dithering is off by default. `Bayer 2x2/4x4/8x8` are deterministic and depend
only on pixel coordinates, so a sprite rendered at two sizes dithers identically;
`Floyd-Steinberg` (Pillow error diffusion) is scan-order dependent and will *not*
match across sizes, and tends to leave worm-like artefacts on AI noise. Either
way the output can never contain a colour outside the palette.

Full reference: [`docs/PixelPaletteApply.md`](docs/PixelPaletteApply.md).

### `PixelExportScale`

`IMAGE -> IMAGE, width, height`

Nearest-neighbour integer upscale (1–16, default 4), plus the resulting
width/height so a downstream node can be driven without guessing. Only `Nearest`
is offered — any interpolating filter would undo the grid restore.

Full reference: [`docs/PixelExportScale.md`](docs/PixelExportScale.md).

### `PixelBackgroundRemove`

`IMAGE -> images (IMAGE), bg_color (STRING)`

Keys a batch's background out and aligns the frames, ported from the local
`webm2spritesheet.py` tool: the background colour is either pinned (`Fixed
Colour`) or detected from the first frame's 2 px border ring (`Auto Detect`),
every frame is cropped to **one shared centred square** around the union of all
content boxes (the animation keeps its registration instead of swimming inside
per-frame crops), and the background is flood-keyed to transparency.
`remove_background=False` gives the crop-only "noclean" variant.

The two colour controls are a mode selector for exactly the reason the grid
nodes have one — "which is it using?" should be visible on the node, not
inferred from an empty text field.

`outer_only` (default on) keys only background regions **connected to the frame
border**, so enclosed pockets and background-coloured detail inside the
character survive. Turning it off keys every matching pixel, which clears a
pocket trapped between a shadow and a cape — but it is not a free win: measured
on a Qwen sheet at the tolerance that gradient needed, the pixels it removed
beyond the default were **2% real pocket and 98% the sprite's own light armour
plate**, which that tolerance could not tell from the background, and two of the
three frames had no pocket to gain. Full numbers in
[the node reference](docs/PixelBackgroundRemove.md#measured-what-the-toggle-actually-costs).

`color_tolerance` defaults to **60**, raised from the reference's 12. On a
Qwen-generated sheet the "solid" purple varied by 93–159 summed RGB across the
frame, so the old default matched nothing and the node silently did nothing —
worth knowing, since the useful range on real AI output is far above what
"tolerance" usually suggests. A very high tolerance starts eating sprite pixels
near the background colour, so raise it until the sprite survives.

Full reference: [`docs/PixelBackgroundRemove.md`](docs/PixelBackgroundRemove.md).

### `PixelSpriteSheet`

`IMAGE -> sheet (IMAGE), columns, rows, cell_width, cell_height`

Packs a batch of same-sized frames into one sprite sheet, row-major, as a pure
pixel copy (no resampling, no quantisation). `Auto Square` picks
`columns = ceil(sqrt(n))`; `Fixed Columns` fixes the row width. Trailing cells
stay empty — transparent in RGBA, black in RGB. The `cell_width` / `cell_height`
outputs are Unity's *Grid By Cell Size* values.

Full reference: [`docs/PixelSpriteSheet.md`](docs/PixelSpriteSheet.md).

## Video caveats

Most of what follows comes from how ComfyUI encodes and loads a `VIDEO` rather
than from this pack; the memory and node-library bullets are this node's own.

- **The `video` output is 4:2:0.** `save_to` picks `yuv420p` (`yuv420p10le` at
  10-bit), so chroma is subsampled and a palette's exact colours cannot survive
  it. **Run the palette chain on `images`, then rebuild the video with a core
  `CreateVideo`** — feeding `PixelPaletteExtract` from `video` would quantise
  colours that have already been smeared.
- **Use `images`, and it is also ~41× faster.** Measured on a 124-frame clip with
  `Per-Tile k-means` in Lab and `accent_protect=True`: 2.605 s/frame on the raw
  768×768 frames against 0.063 s/frame on the restored 96×96 ones. The win is the
  pixel count, not short-circuited clustering — per-tile k-means is skipped only
  where a tile holds fewer distinct colours than its cluster target.
- **`audio` and `fps` are the rebuild kit.** `CreateVideo` needs both, and
  without them the final video would have to re-ingest the source for its
  soundtrack.
- **Alpha is carried in-graph from either input route.** RGBA frames and a
  separate alpha slot both work; the separate slot takes precedence. Alpha is
  spatially sampled per frame and never temporally filtered. The core encoder
  may discard it, and core CreateVideo may drop it before this node receives it.
- **The frame count is fixed, and asserted.** `save_to` truncates the waveform to
  `ceil(sample_rate / frame_rate * frame_count)` samples and never pads, so a node
  that changed the frame count would silently cut the soundtrack.
- **Watch the memory, not the result.** A 124×768×768×3 float batch is ~837 MiB
  before this node runs and cannot be shrunk by chunking; the float→uint8
  conversion is what `chunk_frames` bounds (~680 MB at 12 against ~3.5 GB
  unchunked).
- **The node library's type filter can hide the node** (see
  [above](#known-frontend-bug-the-inputoutput-type-filters-hide-every-custom-node)).
  It takes a `VIDEO` input, so ticking that filter may list nothing; type `pixel`
  in the search box as well.

## Determinism

Both grid nodes are deterministic. For the stills node — the one that already
shipped — that is a **behaviour change**, and it is not a small one:

- **OpenCV's samplers were not seeded, and now are.** `sample_majority` does its
  per-cell 2-means with an unseeded `cv2.kmeans(..., KMEANS_RANDOM_CENTERS)`, so
  `PixelGridRestore` was **not reproducible**: six consecutive calls on one image
  gave five results differing from the first (188–218 channel values), and two
  byte-identical frames disagreed in **74 cells**. `cv2.setRNGSeed(0x5EED_1234)`
  now runs before every sampler call, in the shared `_pixelate`, so the same input
  always yields the same output. On the pack's own reference image the numbers
  happen to be unchanged (125×122, `pixel_size` 8.192, `confidence`
  0.5112436612447103) — but where they changed, they changed, and there was no
  stable previous output to preserve.
- Only the OpenCV backend needs it: the no-CV2 module never imports cv2 and its
  2-means is a deterministic argmax/mean iteration.
- For the legacy video Majority Cluster sampler this removes random flicker, because a
  temporal filter can only clean up flicker already baked into the pixels — seed
  first, filter second.

The palette nodes are reproducible by construction:

- `cv2.kmeans` is seeded with a fixed constant (`KMEANS_SEED`) and uses
  `KMEANS_PP_CENTERS` before every call. Without the seed, repeated runs on
  identical input produce different labels — the selftest demonstrates this.
- `KMEANS_SEED` and the grid nodes' `SAMPLE_SEED` are the same value, so the pack
  has one "fixed seed" idea rather than two that could drift apart.
- Large images are subsampled with a fixed stride, never randomly.
- Palette de-duplication and sorting are order-stable, with the raw RGB triple as
  the final tiebreaker.

`_selftest.py` asserts byte-identical output for two consecutive runs of both
palette nodes, and for `PixelVideoGridRestore`'s whole output batch — the second
one fails without the sampler seed. The sprite nodes contain no RNG at all
(numpy/torch copies and `scipy` labelling only), so they are byte-identical by
construction; the selftest pins that too.

## Dependencies and licensing

This package **depends on** the PyPI package [`perfect-pixel`](https://pypi.org/project/perfect-pixel/)
(`perfect-pixel[opencv]`, currently 0.1.4, whose algorithms live at
<https://github.com/theamusing/perfectPixel>). It does **not** vendor or
redistribute any of that code — it is installed from PyPI as an ordinary
dependency, and all credit for the grid-detection algorithms belongs to its
author.

A note on that dependency's licence, for anyone auditing this bundle: the
upstream repository's README badge claims **MIT**, and the published package
metadata on PyPI declares `License: MIT`, but **the repository contains no
`LICENSE` file**. The licence is therefore asserted rather than demonstrated.
Depending on the package (rather than copying its source into this repository)
keeps the risk minimal: if upstream clarifies or changes its terms, the fix is a
version bump here, not a redistribution problem.

This package's own code — everything under `nodes/` and this repository's
glue — is licensed under the **GNU General Public License v3.0 or later**; see
[`LICENSE`](LICENSE). That is the same licence as ComfyUI itself, and GPL-3.0 is
compatible with the MIT terms the `perfect-pixel` dependency claims.

## Development

`_selftest.py` is a standalone verification script (no ComfyUI server needed):

```bash
python ComfyUI-PixelPipeline/_selftest.py
```

It compiles every module, loads the package through `importlib` under a legal
module name (the directory name contains a hyphen, so it is not importable as
written), calls `GET_SCHEMA()` on each node — which ComfyUI does at import time,
so a raising `define_schema` silently kills the whole pack — and then exercises
the pipeline against real images.

It also runs the video v2 regressions in `tests/test_video_pipeline.py` against
real ComfyUI components: frame identity, moving dark/light outlines, one-frame
highlights, scene cuts, current-frame colour support, static noise reduction,
chunk invariance, alpha through either route, exact frame rate and video encoding.
Frontend migration fixtures run with `node tests/test_video_migration.mjs`.
`tools/verify_video.py` generates real-video comparisons and performance reports
with input/code hashes; use the ComfyUI Python environment.
`tools/measure_image_nodes.py` re-measures the node-library counts quoted in
[the frontend-bug section](#known-frontend-bug-the-inputoutput-type-filters-hide-every-custom-node).

Beyond the grid checks, it covers the palette nodes:

- exact schema comparison (names, types, defaults, ranges) for both palette
  nodes, including that `PixelPaletteApply` carries no derive-path inputs;
- **the `grid_mode` runtime contract**, established by calling the same
  `comfy_api` functions `execution.py` calls rather than by trusting a
  hand-written dict — that only the selected mode's parameters become required,
  that a missing `grid_mode` behaves as Auto Detect, and that a bogus one is
  reported as an unknown mode;
- **the `max_pixel_size` port's equivalence**, measured against upstream
  `detect_grid_scale` across a corpus covering every branch (FFT accepted /
  gate-rejected / both detectors failed), plus the capability it buys and the
  aliasing caveat pinned as a documented behaviour;
- **zh locale keys vs the schemas** — the translation keys must equal what the
  frontend will look up, since a stale key set shows up as *blank* tooltips
  rather than as an English fallback;
- extraction of 32 colours from `fox_pixel_GRID.png`, and a check that the hex
  text round-trips to exactly 32 unique RGB triples;
- hex parser robustness (commas, semicolons, `0x`, 3-digit shorthand, lowercase,
  duplicates, and a rejection case);
- **cross-image consistency** — a palette from image A applied to image B leaves
  B using a subset of A's colours, and `applied_palette` byte-identical to A's;
- **both palette sources for Apply** — wired from Extract and pasted as a
  literal — plus that an empty palette raises and that `applied_palette` comes
  back verbatim with its order preserved;
- all five extract methods;
- every dither mode, asserting no dithered output escapes the palette, that
  Bayer thresholds depend only on `(y % n, x % n)`, and that the 2×2/4×4/8×8
  matrices match the canonical tables;
- **graph dependency**, driven through the real
  `comfy_execution.graph.TopologicalSort`: a wired palette must pull
  `PixelPaletteExtract` into the *initial* execution block;
- **determinism** — identical hex and byte-identical output tensors across two
  runs, for every method in both colour spaces.

And the sprite nodes:

- exact schema comparison for both, including the `layout` DynamicCombo:
  `Auto Square` first (option order is the only default lever), `columns` only
  under `Fixed Columns`, and schema defaults equal to the module constants;
- **the `layout` runtime contract**, driven through the same `comfy_api`
  internals `execution.py` uses — including the zero-nested-option edge, a
  missing `layout` behaving as Auto Square, and a bogus one raising;
- **behaviour against hand-computed reference numbers**: background detection
  and the shared square crop, border-only flood keying (an enclosed
  background-coloured island survives), the no-content full-frame fallback,
  RGBA composition, auto/fixed layouts (including `columns > n`), row-major
  placement, zeroed trailing cells, channel preservation, the prepare→sheet
  chain, and determinism.

It also writes comparison images to `output/pixel_verify_*.png`
(`pixel_verify_A_compare.png` is a four-panel sheet:
original | no dither | Bayer 4x4 | Floyd-Steinberg).

## Notes on upstream behaviour

`perfect-pixel` 0.1.4 has five quirks. Four are surfaced rather than patched
(patching would break the pinned PyPI dependency); the fifth is the one this
package works around by porting the function:

1. Its power spectrum is inverted (`mag = 1 - np.log1p(mag)`), so strong
   harmonics become valleys — detection is noise-sensitive.
2. The OpenCV backend computes grayscale with `COLOR_BGR2GRAY` on an **RGB**
   array, while the no-CV2 backend uses correct RGB weights. The two backends
   can disagree on the same image, which is why `backend` is an explicit input.
   The ported detector reproduces this bug deliberately, because the validity
   gate is calibrated against that grayscale — "fixing" it would change results.
3. `sample_center` clips out-of-range coordinates in the OpenCV backend but not
   in the no-CV2 one, where negative indices wrap and silently sample the wrong
   pixel.
4. **`max_pixel_size = 20.0` is a hardcoded local** inside `detect_grid_scale`,
   so cells wider than 20 px get their correct FFT result discarded. This one is
   *not* merely surfaced: `PixelGridRestore` ports the function to expose it as a
   parameter. See the node section above.
5. The **Lightweight (no-CV2) backend cannot auto-detect at all** on most AI
   output. `estimate_grid_fft` returns a bare `None` when it finds no harmonics,
   while its own caller (`perfect_pixel_noCV2.py:380`) unpacks the result as a
   2-tuple — so it raises `TypeError` instead of falling back to the gradient
   detector. `PixelGridRestore` now checks this *before* running detection and
   raises a `RuntimeError` naming the fix, so you get a message instead of a
   foreign traceback. Use `OpenCV` (`Auto` picks it when cv2 is installed), or
   pair `Lightweight` with `By Pixel Size` / `By Grid Count`, which skip
   detection entirely and are unaffected.

`debug=True` is never passed: that path imports `matplotlib` and calls
`plt.show()`, which blocks a headless server.

## Calling these nodes from the API

`grid_mode` is a `DynamicCombo`: its nested parameters are sent as
`"<input_id>.<nested_id>"` keys alongside the selected option key.

```jsonc
// Auto Detect
{"grid_mode": "Auto Detect", "grid_mode.min_size": 4.0,
 "grid_mode.max_pixel_size": 20.0, "grid_mode.peak_width": 6}

// By Pixel Size
{"grid_mode": "By Pixel Size", "grid_mode.pixel_size": 16}

// By Grid Count
{"grid_mode": "By Grid Count", "grid_mode.grid_w": 125, "grid_mode.grid_h": 122}
```

Omitting `grid_mode` entirely behaves as `Auto Detect` with the schema defaults.
Note this is **not** backwards compatible with the old flat form
(`{"grid_mode": "By Grid Count", "grid_w": 125}`), which no longer matches any
input.

`PixelSpriteSheet`'s `layout` is a `DynamicCombo` too:

```jsonc
// Auto Square
{"layout": "Auto Square"}

// Fixed Columns
{"layout": "Fixed Columns", "layout.columns": 8}
```

Omitting `layout` behaves as `Auto Square`.
