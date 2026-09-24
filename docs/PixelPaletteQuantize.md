# `PixelPaletteQuantize`

**Category:** `image/pixel pipeline`
**Output kind:** processing node

Reduces an image to an indexed palette. The palette is either derived on the fly
or supplied as hex text, in which case it is applied verbatim.

```
IMAGE -> image (IMAGE), applied_palette (STRING), colors_used (INT)
```

## Inputs

| id | type | default | range | notes |
|---|---|---|---|---|
| `image` | IMAGE | — | — | Batch entries are processed independently but share one palette. |
| `method` | COMBO | `Per-Tile k-means` | see below | How to derive the palette. Ignored when `palette` is supplied. |
| `colors` | INT | `128` | 8–256 | Palette size to derive. Ignored when `palette` is supplied. |
| `tile_size` | INT | `16` | 2–512 | Local window for `Per-Tile k-means`. Very large images double it internally to bound the tile count. |
| `space` | COMBO | `Lab` | `Lab`, `RGB` | Space for clustering **and** for the nearest-colour lookup. |
| `dither` | COMBO | `None` | `None`, `Bayer 2x2`, `Bayer 4x4`, `Bayer 8x8`, `Floyd-Steinberg` | See [Dithering](#dithering). |
| `accent_protect` | BOOLEAN | `True` | — | Reserve slots for rare, load-bearing colours. Ignored when `palette` is supplied. |
| `palette` | STRING | `""` | — | **Optional, lazy.** Hex colours (newlines, commas, semicolons or spaces). |

## Outputs

| display name | type | notes |
|---|---|---|
| `image` | IMAGE | Quantized image. Every pixel is a palette colour, dithering included. |
| `applied_palette` | STRING | The palette actually used. A supplied palette comes back verbatim (duplicates removed, order preserved); a derived one comes back sorted by luminance. |
| `colors_used` | INT | Number of distinct colours present **in the output**. Can be lower than the palette size — an entry that nothing mapped to simply does not appear. |

## Supplying a palette

When `palette` carries text (typed into the widget or wired to
[`PixelPaletteExtract`](PixelPaletteExtract.md)), the palette is applied verbatim
by nearest colour in `space`, and `method` / `colors` / `accent_protect` are
bypassed. This is the mode that guarantees a sprite set stays consistent.

Accepted syntax — all of these parse identically:

```
#1F1715
#06202C
```
```
#102030, #405060
```
```
102030 405060
```
```
#123 456          (3-digit shorthand, expanded to #112233 / #445566)
0x102030          (0x prefix tolerated)
```

Duplicates are removed (first occurrence wins). A token that is not a colour
raises a `ValueError` naming it — a silently truncated palette would be worse
than a failure.

## Methods

| option | implementation |
|---|---|
| `Per-Tile k-means` *(default)* | `cv2.kmeans` per local window, then a merge k-means over the pooled centres |
| `Global k-means (Lab)` | one `cv2.kmeans` over the whole image |
| `Median Cut` | `PIL.Image.quantize(method=MEDIANCUT)` |
| `Max Coverage` | `PIL.Image.quantize(method=MAXCOVERAGE)` |
| `Octree` | `PIL.Image.quantize(method=FASTOCTREE)` |
| `Fixed Palette (nearest)` | nearest-neighbour onto `palette`; **raises** if `palette` is empty |

`cv2.kmeans` is seeded with the fixed constant
`nodes/_palette_utils.py::KMEANS_SEED` and uses `KMEANS_PP_CENTERS` before every
call, so the same input always produces byte-identical output. Large images are
subsampled with a fixed stride, never randomly. See the
[determinism section](#determinism) below.

The three PIL methods work on RGB only — `Image.quantize` has no Lab entry
point, so `space` does not affect how the palette is derived. It *does* still
affect the nearest-colour lookup that follows.

## Dithering

Dithering only ever changes the **lookup coordinate**, never the output value:
the result is always `palette[nearest_index]`, so a dithered image can never
contain a colour that is not in the palette. (`_selftest.py` asserts this for
every mode.)

| mode | deterministic | size-independent | notes |
|---|---|---|---|
| `None` *(default)* | — | — | Off by default on purpose: in regions narrower than ~8–10 px (faces, icons, lettering) dithering reads as noise rather than as a blend. |
| `Bayer 2x2` / `4x4` / `8x8` | yes | **yes** | Ordered dithering. The threshold depends only on `(y % n, x % n)`, so the same sprite rendered at different sizes dithers identically. |
| `Floyd-Steinberg` | yes | **no** | Pillow's error diffusion. Scan-order dependent, so a sprite at two sizes will not match; it also leaves worm-like artefacts on the noise floor of AI output. Always maps in RGB, regardless of `space`. |

The Bayer amplitude is not a hand-tuned constant: it is `0.9 ×` the palette's
own mean nearest-neighbour step, measured in the selected working space. That is
the smallest nudge that can move a pixel into a neighbouring entry, and it makes
`Lab` and `RGB` dither at comparable visual strength without per-space magic
numbers.

Bayer offsets are added to all three channels of the working-space colour. In
`Lab` that shifts lightness and both chroma axes together; in `RGB` it shifts
along the grey diagonal. Both were chosen so the perturbation is symmetric and
space-agnostic.

## Accent protection

Clustering eats small, visually critical details first. With
`accent_protect=True` (the default) colours that are both **rare** (under 0.1% of
the pixels) and **distinctive** are reserved palette slots, up to `colors / 8` of
them:

- **chromatic pops** — HSV saturation ≥ 0.55 (a red gem, a magic-FX pixel);
- **tonal extremes** — value within 6% of black or white (eye highlights, thin
  outlines that survive on one limb).

A candidate is only kept if it is still at least half a palette step away from
the derived palette, so it does not waste a slot on a colour that is already
represented. Accents are paid for out of the same `colors` budget — the main
cluster count is reduced to make room — so the returned palette never exceeds
`colors`.

## The lazy `palette` input

`palette` is declared `optional=True, lazy=True`, and
`check_lazy_status()` asks for it **only** when it is wired but not yet evaluated
(ComfyUI passes `None` for that state). Three distinct states are handled:

| state | value seen | behaviour |
|---|---|---|
| not wired | key absent → `MISSING` sentinel | no request; the upstream `PixelPaletteExtract` subtree is pruned out of the execution graph |
| wired, not evaluated | `None` | return `["palette"]` → ComfyUI promotes the link and re-queues this node |
| wired, evaluated / typed in | the string | used as a fixed palette |

`comfy_execution/graph.py:169-172` holds lazy links out of the initial execution
block, so an unwired palette branch is **pruned, not computed and discarded**.
`_selftest.py` drives the real `TopologicalSort` to prove this, and also proves
the mirror-image mistake is impossible: asking for an unwired widget raises
`NodeInputError: ... but that value is a constant`, which is exactly why the test
must be `palette is None` and not a falsiness check.

Note that `validate_inputs()` runs **before** `check_lazy_status()`, so it must
tolerate `palette=None`. It only performs cheap structural checks (combo values,
`colors` range).

## Determinism

- Every `cv2.kmeans` call is preceded by `cv2.setRNGSeed(KMEANS_SEED)` and uses
  `KMEANS_PP_CENTERS`. Without the seed, repeated runs on identical input
  produce **different** labels — verified by the selftest.
- Sub-sampling of large images uses a fixed stride.
- Palette de-duplication and sorting are order-stable.
- `_selftest.py` asserts byte-identical output tensors across two runs, and
  identical hex strings for all five extraction methods in both colour spaces.

## Transparency (RGBA)

RGBA input is accepted, and the alpha channel is **passed through byte for
byte**. The contract of this node is "quantise colour, touch nothing else": the
palette is derived from the colour channels alone, because a k-means quantiser
handed RGBA would treat *transparent* as a fourth colour and spend palette
entries on it -- nobody wants a swatch of "transparent".

Unlike [`PixelGridRestore`](PixelGridRestore.md) there is deliberately **no**
"drop an all-opaque alpha" rule here, so an RGBA input always yields an RGBA
output. That keeps the node predictable when it is used on its own rather than
downstream of a grid restore.

## See also

- [`PixelPaletteExtract`](PixelPaletteExtract.md) — the usual source of `palette`
- [`PixelGridRestore`](PixelGridRestore.md) — normally the producer of `image`
