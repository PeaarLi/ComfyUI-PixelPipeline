# ComfyUI-PixelPipeline

Turn AI-generated "fake pixel art" into **real** pixel art: detect (or force) the
pixel grid, resample one colour per cell, optionally quantise to a shared
palette, then export at a crisp integer scale.

Built on the **V3 node schema** (`comfy_api.latest`), so nodes are declared with
`io.Schema` / `define_schema` rather than the legacy `INPUT_TYPES` dict.

## Status

| Stage | Node | Docs |
|---|---|---|
| 1. Grid restore | `PixelGridRestore` | [`docs/PixelGridRestore.md`](docs/PixelGridRestore.md) |
| 2. Palette extract | `PixelPaletteExtract` | [`docs/PixelPaletteExtract.md`](docs/PixelPaletteExtract.md) |
| 3. Palette quantise | `PixelPaletteQuantize` | [`docs/PixelPaletteQuantize.md`](docs/PixelPaletteQuantize.md) |
| 4. Export scale | `PixelExportScale` | [`docs/PixelExportScale.md`](docs/PixelExportScale.md) |

## Pipeline

```
IMAGE
  │
  ├─► [PixelGridRestore] ──► IMAGE + grid_w + grid_h + pixel_size + detected
  │         │
  │         ├─► [PixelPaletteExtract] ──► palette (HEX) ─┐   (optional)
  │         │                                            │
  │         └─► [PixelPaletteQuantize] ◄─────────────────┘   (optional)
  │                   │
  └───────────────────┴──► [PixelExportScale] ──► IMAGE + width + height
```

The palette pair is an **optional branch**. Leave `PixelPaletteQuantize.palette`
unwired and the entire `PixelPaletteExtract` subtree is pruned out of the
execution graph rather than computed and discarded — that is what the `lazy`
declaration on that input is for. See
[`docs/PixelPaletteQuantize.md`](docs/PixelPaletteQuantize.md#the-lazy-palette-input).

### Transparency (RGBA)

Qwen-Image-2.1's VAE carries four channels, so `VAEDecode` emits RGBA on every
run. The pipeline handles that end to end: `PixelGridRestore` resamples alpha
onto the *exact same* grid lines as the colour so a transparent sprite's mask
stays pixel-aligned, `PixelPaletteQuantize` passes alpha through untouched, and
`PixelExportScale` scales it along with the colour channels. An all-opaque alpha
is dropped by `PixelGridRestore`, so ordinary generations still come out as
plain RGB rather than 4 channels.

## Install

```bash
cd ComfyUI/custom_nodes
git clone <this repo>
pip install -r ComfyUI-PixelPipeline/requirements.txt
```

ComfyUI Manager installs `requirements.txt` automatically.

## Finding the nodes in the node library

The four nodes live under **`图像 / 像素管线`** (`image/pixel pipeline`). They also
show up under the **Extensions** source group, because custom nodes are always
classified by `python_module`, never by category — using a core-looking category
does **not** fold a pack into the built-in "Comfy" group.

### Known frontend bug: the input/output type filters hide every custom node

If you open the node library's **输入 / 输出** (input / output) filter and tick
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
**255** nodes accept an `IMAGE` input and this pack's nodes rank **251st**, so the
64-slot cut-off is filled entirely by core nodes. The first custom node with an
`IMAGE` input ranks 128th, i.e. it is also cut. Raising the limit or reordering
the pack changes nothing.

**Workaround — type text as well as ticking the filter.** With a non-empty query
the results come from Fuse relevance ranking, the hit count drops far below 64,
and nothing gets truncated. Typing `pixel` with `IMAGE` ticked lists all four
nodes.

The real fix belongs upstream: filtering should not truncate, or the limit should
only apply to unfiltered searches.

## Nodes

### `PixelGridRestore`

`IMAGE -> IMAGE, grid_w, grid_h, pixel_size, detected, confidence`

Three ways to determine the grid:

- **Auto Detect** — upstream FFT + gradient detection. Fast when it works, but
  it often fails on AI output: the FFT path only survives when the implied pixel
  edge is 4–20 px, and the gradient fallback finds zero peaks on hard-edged
  pixel art (Sobel responds equally on both columns of a 1 px step).
- **By Pixel Size** — you give the target pixel edge length (e.g. `16`); the node
  converts it internally to a cell count with
  `grid_w = round(W / pixel_size)`, `grid_h = round(H / pixel_size)`.
  This is the reliable path and the reason the node exists.
- **By Grid Count** — you give the cell counts directly.

Other inputs: `sampling` (`Majority Cluster` / `Center Sample` / `Median`),
`backend` (`Auto` / `OpenCV` / `Lightweight` — see the notes below before
picking `Lightweight`), and four advanced parameters (`refine_intensity`,
`min_size`, `peak_width`, `fix_square`) that were unreachable from the legacy
V1 wrapper.

**`confidence` is a diagnostic, not a gate.** Upstream auto-detection can
return a grid that looks plausible and is wrong: a blurred gradient scores 64
cells at an implied 8.00 px edge while a real pixel image scores 125 cells at
8.192 px, and both sit inside upstream's own validity window, so a range check
cannot separate them. Grid *periodicity* can — measured 1.0–2.6 for gradients,
photos and text posters against 9.1–57.8 for real grids. Nothing is blocked on
the value; wire it to a condition node if you want to gate on it. It reads 0 on
the manual modes by design, because a forced pitch generally does not match the
image's own periodicity.

**Failures raise.** Upstream signals "I could not find a grid" by returning
`None, None, image` — the *same object*, with no exception. A node that only
keeps the image therefore emits the untouched input, and a downstream upscale
blows it up into a huge non-pixelated image with no warning. This node checks
`w is None` and raises, telling you to switch to `By Pixel Size` /
`By Grid Count`.

### `PixelPaletteExtract`

`IMAGE -> palette (STRING), palette_image (IMAGE), count (INT)`

Extracts the palette an image is actually made of and emits it as plain
newline-separated `#RRGGBB` text: visible, hand-editable, diffable, and accepted
verbatim by `PixelPaletteQuantize`. Also emits a swatch strip so the palette can
be eyeballed without decoding hex.

Five extraction methods, defaulting to **per-tile k-means in Lab**, which keeps
local structure intact instead of flattening adjacent tones the way global
clustering does. Clustering is delegated to `cv2.kmeans` and
`PIL.Image.quantize` — nothing is re-implemented — and every k-means call is
seeded so the same input always produces byte-identical output.

Full reference: [`docs/PixelPaletteExtract.md`](docs/PixelPaletteExtract.md).

### `PixelPaletteQuantize`

`IMAGE -> image (IMAGE), applied_palette (STRING), colors_used (INT)`

Reduces an image to an indexed palette, derived on the fly or supplied as hex
text. Supplying a palette is what keeps a set of sprites colour-consistent: the
output can only ever contain colours from that palette.

`palette` is `optional=True, lazy=True`. Unwired, the upstream
`PixelPaletteExtract` subtree is **pruned from the execution graph** — not
executed and thrown away.

Dithering is off by default. `Bayer 2x2/4x4/8x8` are deterministic and depend
only on pixel coordinates, so a sprite rendered at two sizes dithers identically;
`Floyd-Steinberg` (Pillow error diffusion) is scan-order dependent and will *not*
match across sizes, and tends to leave worm-like artefacts on AI noise. Either
way the output can never contain a colour outside the palette.

Full reference: [`docs/PixelPaletteQuantize.md`](docs/PixelPaletteQuantize.md).

### `PixelExportScale`

`IMAGE -> IMAGE, width, height`

Nearest-neighbour integer upscale (1–16, default 4), plus the resulting
width/height so a downstream node can be driven without guessing. Only `Nearest`
is offered — any interpolating filter would undo the grid restore.

Full reference: [`docs/PixelExportScale.md`](docs/PixelExportScale.md).

## Determinism

The palette nodes are reproducible by construction:

- `cv2.kmeans` is seeded with a fixed constant (`KMEANS_SEED`) and uses
  `KMEANS_PP_CENTERS` before every call. Without the seed, repeated runs on
  identical input produce different labels — the selftest demonstrates this.
- Large images are subsampled with a fixed stride, never randomly.
- Palette de-duplication and sorting are order-stable, with the raw RGB triple as
  the final tiebreaker.

`_selftest.py` asserts byte-identical output for two consecutive runs of both
palette nodes.

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
glue — is MIT licensed; see [`LICENSE`](LICENSE).

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

Beyond the grid checks, it covers the palette nodes:

- exact schema comparison (names, types, defaults, ranges, and the
  `optional`/`lazy` markers on `PixelPaletteQuantize.palette`);
- extraction of 32 colours from `fox_pixel_GRID.png`, and a check that the hex
  text round-trips to exactly 32 unique RGB triples;
- hex parser robustness (commas, semicolons, `0x`, 3-digit shorthand, lowercase,
  duplicates, and a rejection case);
- **cross-image consistency** — a palette from image A applied to image B leaves
  B using a subset of A's colours, and `applied_palette` byte-identical to A's;
- all six quantize methods and all five extract methods;
- every dither mode, asserting no dithered output escapes the palette, that
  Bayer thresholds depend only on `(y % n, x % n)`, and that the 2×2/4×4/8×8
  matrices match the canonical tables;
- **lazy pruning**, driven through the real
  `comfy_execution.graph.TopologicalSort`: with `palette` unwired,
  `PixelPaletteExtract` never enters the execution block; asking for a wired
  palette promotes it to a strong link and pulls the node in; and asking for an
  unwired one raises `NodeInputError: ... but that value is a constant`;
- **determinism** — identical hex and byte-identical output tensors across two
  runs, for every method in both colour spaces.

It also writes comparison images to `output/pixel_verify_*.png`
(`pixel_verify_A_compare.png` is a four-panel sheet:
original | no dither | Bayer 4x4 | Floyd-Steinberg).

## Notes on upstream behaviour

`perfect-pixel` 0.1.4 has three quirks that this package surfaces rather than
patches (patching would break the pinned PyPI dependency):

1. Its power spectrum is inverted (`mag = 1 - np.log1p(mag)`), so strong
   harmonics become valleys — detection is noise-sensitive.
2. The OpenCV backend computes grayscale with `COLOR_BGR2GRAY` on an **RGB**
   array, while the no-CV2 backend uses correct RGB weights. The two backends
   can disagree on the same image, which is why `backend` is an explicit input.
3. `sample_center` clips out-of-range coordinates in the OpenCV backend but not
   in the no-CV2 one, where negative indices wrap and silently sample the wrong
   pixel.
4. The **Lightweight (no-CV2) backend cannot auto-detect at all** on most AI
   output. `estimate_grid_fft` returns a bare `None` when it finds no harmonics,
   while its own caller (`perfect_pixel_noCV2.py:380`) unpacks the result as a
   2-tuple — so it raises `TypeError` instead of falling back to the gradient
   detector. `PixelGridRestore` catches that and re-raises a `RuntimeError`
   naming the backend and the bug. Use `OpenCV` (`Auto` picks it when cv2 is
   installed), or pair `Lightweight` with `By Pixel Size` / `By Grid Count`,
   which skip detection entirely.

`debug=True` is never passed: that path imports `matplotlib` and calls
`plt.show()`, which blocks a headless server.
