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
[`PixelPaletteQuantize`](PixelPaletteQuantize.md). It also avoids inventing a
custom socket type that no other node could consume.

The reason a *separate node* exists for this at all is cross-sprite
consistency: `cv2.kmeans` seeds its centres from a global RNG and its own
documentation admits the same image can come back with different labels. Deriving
a palette once and reusing the string across every sprite in a set is the only
way to keep the set visually coherent.

## Inputs

| id | type | default | range | notes |
|---|---|---|---|---|
| `image` | IMAGE | — | — | Normally the output of `PixelGridRestore`; any image works. A batch is pooled into **one** shared palette. |
| `colors` | INT | `128` | 8–256 | Target palette size. `count` can be lower if the image holds fewer distinct colours. |
| `method` | COMBO | `Per-Tile k-means` | see below | Extraction algorithm. |
| `space` | COMBO | `Lab` | `Lab`, `RGB` | Colour space for clustering and for counting. Ignored by the three PIL methods. |
| `sort` | COMBO | `Luminance` | `Luminance`, `Hue`, `Frequency` | Order of the emitted list, and of the swatch strip. |
| `accent_protect` | BOOLEAN | `True` | — | Keep rare but vivid colours (highlights, a lone accent) from being absorbed by the dominant clusters, paid for out of the colour budget. |
| `tile_size` | INT *(advanced)* | `16` | 2–512 | Window edge in input pixels for `Per-Tile k-means`. Ignored by the other methods. |

## Outputs

| display name | type | notes |
|---|---|---|
| `palette` | STRING | Newline-separated `#RRGGBB`, one colour per line, no trailing newline. |
| `palette_image` | IMAGE | Swatch strip: one 16×16 block per colour, 16 per row, same order as the text. Every pixel of it is a palette colour. |
| `count` | INT | Number of entries returned. Never larger than `colors`. |

## Methods

| option | implementation | ignores `space` | deterministic |
|---|---|---|---|
| `Per-Tile k-means` *(default)* | `cv2.kmeans` per local window, then a merge k-means over the pooled centres | no | yes (seeded) |
| `Global k-means (Lab)` | one `cv2.kmeans` over the whole image (fixed-stride subsampled above 65 536 px) | no | yes (seeded) |
| `Median Cut` | `PIL.Image.quantize(method=MEDIANCUT)` | yes (always RGB) | yes |
| `Max Coverage` | `PIL.Image.quantize(method=MAXCOVERAGE)` | yes (always RGB) | yes |
| `Octree` | `PIL.Image.quantize(method=FASTOCTREE)` | yes (always RGB) | yes |

`Fixed Palette (nearest)` is deliberately **not** offered here. Applying a
palette you already have is not a way to extract one, and a dropdown entry that
always raises would be worse than its absence. It belongs to
[`PixelPaletteQuantize`](PixelPaletteQuantize.md).

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

- **`count` is exact, not aspirational.** If an image contains 20 distinct
  colours and you ask for 32, you get 20. Cluster centres that round to the same
  RGB triple are also merged.
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

  The flag belongs on the node that *chooses* a palette, which is why it appears
  here **and** on [`PixelPaletteQuantize`](PixelPaletteQuantize.md). Quantize
  only chooses when its `palette` input is empty; with a palette wired in it maps
  onto that palette verbatim — adding colours would break the cross-image subset
  guarantee, so its flag is inert on that path. One consequence worth knowing:
  with defaults, the 32 colours Extract emits and the 32 colours Quantize derives
  on its own are **not the same set**, because the latter may include protected
  accents.
- **`tile_size` is exposed here as well.** `Per-Tile k-means` is the default
  method, so its window needs to be reachable; it sits under the advanced
  disclosure with a default of 16. `nodes/_palette_utils.py::EXTRACT_TILE_SIZE`
  is now only that default value. Matching it to a sprite's real cell size (see
  `PixelGridRestore`'s `pixel_size` output) is what makes "one cluster per art
  pixel" meaningful.

## Example

```
PixelGridRestore (By Pixel Size, 16) ──► PixelPaletteExtract (32, Per-Tile, Lab, Luminance)
                                             │
                                             ├─ palette ──────► PixelPaletteQuantize.palette
                                             └─ palette_image ► PreviewImage (eyeball it)
```

`count == 32` on `fox_pixel_GRID.png`, and the returned string parses back to
exactly 32 unique RGB triples — both asserted by `_selftest.py`.

## See also

- [`PixelPaletteQuantize`](PixelPaletteQuantize.md) — the consumer of this output
- [`PixelGridRestore`](PixelGridRestore.md) — normally the producer of `image`
