# `PixelPaletteExtract`

**Category:** `image/pixel pipeline`
**Output kind:** data (root node in the pipeline graph)

Extracts the palette an image is actually made of and emits it as plain
`#RRGGBB` text.

```
IMAGE -> palette (STRING), palette_image (IMAGE), count (INT)
```

## Why the palette is text

A hex string is visible in the UI, editable by hand, pasteable into an issue or
a commit message, diffable, and accepted verbatim by
[`PixelPaletteApply`](PixelPaletteApply.md). It also avoids inventing a
custom socket type that no other node could consume.

The reason a *separate node* exists for this at all is cross-sprite
consistency: `cv2.kmeans` seeds its centres from a global RNG and its own
documentation admits the same image can come back with different labels. Deriving
a palette once and reusing the string across every sprite in a set is the only
way to keep the set visually coherent.

## Inputs

| id | type | default | range | notes |
|---|---|---|---|---|
| `image` | IMAGE | — | — | Normally the generator's own output — the image *before* `PixelGridRestore` — but any image works. For a video clip, feed the *restored* frames instead: see [For video](#for-video-feed-it-the-restored-frames). A batch is pooled into **one** shared palette. |
| `colors` | INT | `128` | 8–256 | Upper bound on the palette size to aim for. `count` can be lower — see [Behaviour notes](#behaviour-notes). |
| `method` | COMBO | `Per-Tile k-means` | see below | Extraction algorithm. |
| `space` | COMBO | `Lab` | `Lab`, `RGB` | Colour space for clustering, for counting, for accent separation, and for the batch merge. Only the three PIL methods' *initial* extraction ignores it — see [What `space` reaches](#what-space-reaches). |
| `sort` | COMBO | `Luminance` | `Luminance`, `Hue`, `Frequency` | Order of the emitted list, and of the swatch strip. |
| `accent_protect` | BOOLEAN | `True` | — | Keep rare but vivid colours (highlights, a lone accent) from being absorbed by the dominant clusters, paid for out of the colour budget. |
| `tile_size` | INT *(advanced)* | `16` | 2–512 | Window edge, in pixels **of this node's input image**, for `Per-Tile k-means`. Ignored by the other methods. |

## Outputs

| display name | type | notes |
|---|---|---|
| `palette` | STRING | Newline-separated `#RRGGBB`, one colour per line, no trailing newline. |
| `palette_image` | IMAGE | Swatch strip: one 16×16 block per colour, 16 per row, same order as the text. Every pixel of it is a palette colour. |
| `count` | INT | Number of entries returned. Never larger than `colors`. |

## Methods

| option | implementation | clusters in `space` | deterministic |
|---|---|---|---|
| `Per-Tile k-means` *(default)* | `cv2.kmeans` per local window, then a merge k-means over the pooled centres | yes | yes (seeded) |
| `Global k-means (Lab)` | one `cv2.kmeans` over the whole image (fixed-stride subsampled above 65 536 px) | yes | yes (seeded) |
| `Median Cut` | `PIL.Image.quantize(method=MEDIANCUT)` | no (always RGB) | yes |
| `Max Coverage` | `PIL.Image.quantize(method=MAXCOVERAGE)` | no (always RGB) | yes |
| `Octree` | `PIL.Image.quantize(method=FASTOCTREE)` | no (always RGB) | yes |

The last column is about the **clustering step only**, and it is narrower than
"`space` is ignored" — see the next section.

`Fixed Palette (nearest)` is deliberately **not** offered here. Applying a
palette you already have is not a way to extract one, and a dropdown entry that
always raises would be worse than its absence. Mapping onto a supplied palette
is [`PixelPaletteApply`](PixelPaletteApply.md)'s entire job.

### What `space` reaches

`Image.quantize` has no Lab entry point, so median cut / max coverage / octree
always *cluster* on RGB. That is the only thing they get away with ignoring
`space` for. Three later stages are computed in `space`, and they apply to the
PIL methods too:

- **the pixel count behind the `Frequency` sort** — `palette_counts()` maps every
  pixel to its nearest entry in `space`;
- **accent separation** — `_append_accents()` measures each candidate's distance
  to the palette in `space`, and the "one palette step" it has to clear is
  measured there as well;
- **the multi-image batch merge** — `_merge_blocks()` converts each image's PIL
  palette into `space` before pooling and re-clustering.

So on a **batched or accent-protected** run, `space` changes the output even for
Median Cut — only the *initial* per-image PIL extraction is space-blind, because
Pillow has no Lab entry point.

### Why `Per-Tile k-means` is the default

Global clustering merges adjacent-but-different tones, so gradients and narrow
detail collapse into flat blocks. Per-tile clustering keeps the local modes:
each tile contributes roughly the same number of candidate centres regardless of
how much area it covers, so a large flat background cannot outvote a small
detailed region. In practice the ordering of structure preservation is

```
per-tile k-means (perceptual space) > octree > global k-means > median cut
```

The cost is speed: per-tile is the slowest option by roughly 2×.

### Why `Lab` is the default space

RGB k-means is a well-known trap at low colour counts — at `colors=7` the result
often contains no red and no green at all, and red does not reliably appear
until around `colors=10`. OpenCV's *8-bit* Lab encoding is used (`L` scaled to
0–255, `a`/`b` offset by 128), **not** the standard 0–100 / ±128 form; distances
are plain Euclidean in that encoding.

## Behaviour notes

- **`colors` is an upper bound, not a promise.** `count` is never larger than
  `colors`; it is smaller whenever the image does not hold that many distinct
  colours (ask for 32 on an image with 20 and you get 20), and cluster centres
  that round to the same RGB triple are merged as well. In `Per-Tile k-means`
  there is a second ceiling: each tile is clustered to roughly
  `sqrt(tile area)` centres before the merge, so raising `colors` cannot recover
  detail finer than the tiles ever saw.
- **Sorting is fully deterministic.** Every sort key ends with the raw RGB
  triple, so colours that tie on the primary criterion still come out in a fixed
  order — the emitted string never depends on the row order of the input.
- **`Frequency` counts pixels per entry** across the whole (possibly batched)
  input.
- **Batches pool.** With `B > 1` each frame contributes its own palette and the
  group is re-clustered into a single shared palette, so every frame comes out
  consistent. `Count`/`Frequency` are measured over the whole batch.
- **`accent_protect` is on by default, and it can lower `count`.** Protected
  accents are paid for out of the colour budget, so `colors=32` with protection
  on can legitimately return 28 entries (Max Coverage does exactly that on the
  reference image). With `accent_protect=False` the count is exact for every
  method — both modes are pinned by `_selftest.py`.

  The flag belongs on the node that *chooses* a palette, and this is now the
  **only** node in the pack that has it.
  [`PixelPaletteApply`](PixelPaletteApply.md) maps onto a supplied palette
  verbatim and must never add colours, or the cross-image subset guarantee
  breaks. There is therefore no second palette-deriving path left for the flag to
  be inert on, and no way for two nodes to disagree about what a given `colors`
  value produces.
- **`tile_size` is measured in pixels of *this node's input image*.** The right
  value depends entirely on what is wired into `image`:

  | wired in | one art pixel is | `tile_size` to use |
  |---|---|---|
  | the generator's raw output | about one cell (e.g. 16 px) | the art's cell size — match `PixelGridRestore`'s `pixel_size` output for roughly one cluster per art pixel |
  | `PixelGridRestore`'s output | exactly one pixel | a **small** value — the cell size is far too large here and will over-smooth |

  It sits under the advanced disclosure with a default of 16, and is used only by
  `Per-Tile k-means`. `nodes/_palette_utils.py::EXTRACT_TILE_SIZE` is only that
  default value.

### For video: feed it the restored frames

For a clip, wire this node from [`PixelVideoGridRestore`](PixelVideoGridRestore.md)'s
**`images`** output — the already-restored frames — and **not** from the
generator's raw output. Two reasons, measured on a 124-frame clip with `Per-Tile
k-means` in Lab and `accent_protect=True`:

| fed in | frame size | per frame |
|---|---|---|
| the generator's raw output | 768×768 | 2.605 s |
| `PixelVideoGridRestore`'s `images` | 96×96 (768 px cells of 8) | 0.063 s |

That is about **41×**, and the win is simply the pixel count: the restored frames
have one pixel per art cell, so there are roughly 64× fewer of them (768² against
96² per frame). The palette is also cleaner, because the raw AI frames carry
anti-aliasing and noise *inside* each cell that clustering would happily spend
palette entries on. Both effects come from the same thing — the restored image is
what the palette was always meant to describe.

The one caveat is `tile_size`, which is measured in pixels of whatever is wired
in. With restored frames one art pixel is one pixel, so pass a **small** value:
the second row of the table above is the right one, and `PixelVideoGridRestore`'s
`images` behaves exactly like `PixelGridRestore`'s output in that respect.

There is a second, less obvious benefit for video. Because `extract_palette` pools
the whole batch into **one** palette, every frame of the clip is quantised against
the same colour set — so the palette itself cannot flicker. A per-frame palette
would let a colour appear on frame 40 that was absent on frame 39, and no amount
of careful quantisation downstream would hide that. Pooling is the whole reason
the video chain has a stable look; there is nothing to switch off.

## Example

```
PixelGridRestore (By Pixel Size, 16) ──► PixelPaletteExtract (32, Per-Tile, Lab, Luminance)
                                             │
                                             ├─ palette ──────► PixelPaletteApply.palette
                                             └─ palette_image ► PreviewImage (eyeball it)
```

`count == 32` on `fox_pixel_GRID.png`, and the returned string parses back to
exactly 32 unique RGB triples — both asserted by `_selftest.py`.

## See also

- [`PixelPaletteApply`](PixelPaletteApply.md) — the consumer of this output
- [`PixelGridRestore`](PixelGridRestore.md) — normally the producer of `image`
- [`PixelVideoGridRestore`](PixelVideoGridRestore.md) — the producer of `image` for a clip
