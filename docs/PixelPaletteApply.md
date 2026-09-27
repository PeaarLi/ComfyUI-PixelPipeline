# `PixelPaletteApply`

**Category:** `image/pixel pipeline`
**Output kind:** processing node

Reduces an image to an indexed palette supplied as hex text, applied verbatim.
The node derives nothing: a palette is always supplied, either wired from
[`PixelPaletteExtract`](PixelPaletteExtract.md) or typed / pasted into its own
widget.

```
IMAGE -> image (IMAGE), applied_palette (STRING), colors_used (INT)
```

## Inputs

| id | type | default | range | notes |
|---|---|---|---|---|
| `image` | IMAGE | — | — | Batch entries are processed independently but share one palette. |
| `palette` | STRING *(multiline)* | `""` | — | **Required.** Hex colours (newlines, commas, semicolons or spaces). Applied verbatim. |
| `space` | COMBO | `Lab` | `Lab`, `RGB` | Metric the nearest-colour lookup is measured in. |
| `dither` | COMBO | `None` | `None`, `Bayer 2x2`, `Bayer 4x4`, `Bayer 8x8`, `Floyd-Steinberg` | See [Dithering](#dithering). |

## Outputs

| display name | type | notes |
|---|---|---|
| `image` | IMAGE | Quantized image. Every pixel is a palette colour, dithering included. |
| `applied_palette` | STRING | The palette actually used — the supplied palette verbatim, duplicates removed, order preserved. |
| `colors_used` | INT | Number of distinct colours present **in the output**. Can be lower than the palette size — an entry that nothing mapped to simply does not appear. |

## The palette is applied verbatim

The palette travels as text: visible in the UI, editable by hand, pasteable into
a commit message, reusable across an entire sprite set. It is applied by nearest
colour in `space`, with **duplicates removed and the order you gave preserved**.
It is never re-sorted — sorting would silently reorder someone's palette, and the
echoed `applied_palette` must be the palette you supplied.

This is the mode that guarantees a sprite set stays consistent: the output can
never contain a colour outside the palette, so every frame quantized against the
same string shares one colour set. `_selftest.py` asserts the subset property
directly (frame B's output colour set ⊆ frame A's palette).

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

An **empty or whitespace-only** palette is an error, reported twice over:
`validate_inputs()` rejects it before the prompt is queued, and `execute()`
raises a `ValueError` naming the input if one gets through anyway. There is no
"derive a palette when none is supplied" path left — see below.

## Nothing is derived here

An earlier version had this node derive a palette of its own when none was wired
(`method` / `colors` / `tile_size` / `accent_protect`, plus a
`Fixed Palette (nearest)` method). That made four of its seven inputs silently
inert the moment a palette *was* wired, which is the common case — the node had
two mutually exclusive parameter sets on one panel and no way to say which one
was live. Those inputs are gone.

The consequences worth knowing:

- **`accent_protect` and `colors` live only on
  [`PixelPaletteExtract`](PixelPaletteExtract.md)** — the one node that *chooses*
  a palette. Adding colours to a supplied palette here would break the
  cross-image subset guarantee, which is the whole reason the palette is passed
  around as text in the first place.
- **A documented asymmetry is gone.** "Extract then Apply" and "Apply alone" used
  to be different pipelines producing different colour sets from the same image.
  Apply no longer has an "alone" behaviour to disagree with: the palette it
  applies is the palette it was given.
- **Extraction methods belong to Extract.** The dropdown of extractors
  (`Per-Tile k-means`, `Global k-means (Lab)`, `Median Cut`, `Max Coverage`,
  `Octree`) is on [`PixelPaletteExtract`](PixelPaletteExtract.md), as is the
  `sort` that orders the emitted list.

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

## No `lazy` palette input any more

`palette` used to be `optional=True, lazy=True`, with a `check_lazy_status()`
override whose only purpose was to prune the upstream `PixelPaletteExtract`
subtree out of the execution graph when the palette was unwired. A palette is
now required, so there is nothing to prune and both the `optional` / `lazy`
markers and the override are gone — this node does **not** override
`check_lazy_status`. (Not to be confused with `hasattr`, which is always true:
`io.ComfyNode` defines it on the base class. What matters is that this node does
not override it.)

The property to rely on now is the mirror image, and `_selftest.py` drives the
real `TopologicalSort` to prove it: a **wired** palette pulls
`PixelPaletteExtract` into the *initial* execution block, as a hard dependency,
rather than being promoted in by a lazy re-queue on a second pass. A **pasted**
palette needs no upstream at all — Extract stays out of the block for the
ordinary reason that nothing links to it, not because a lazy input pruned it.

## Validation

`validate_inputs(palette)` rejects a blank palette early, so a bad API prompt
fails before the graph runs. It is the only check standing between a prompt and
an empty palette, since there is no lazy re-queue to catch that case later;
`execute()` enforces the real value again once the upstream palette has actually
been produced.

Two properties of the signature are deliberate, and both were established by
driving the real `execution.validate_inputs()` rather than by reading the docs:

**`None` is accepted.** Validation runs *before* any node executes, and
`execution.get_input_data()` marks every **linked** input missing when it is
called without an `execution_list` — which is exactly how the validation phase
calls it. So a palette wired from `PixelPaletteExtract` arrives here as `None`,
meaning "linked, upstream has not run yet". Treating `None` as empty would make
a wired palette impossible to validate, i.e. it would brick the node for the one
wiring this pack is built around. Only a genuinely blank *string* — the untouched
widget — is an error.

**No `**kwargs`.** `execution.py` derives `validate_function_inputs` from the
validator's parameter names, builds `input_filtered` from it, and then appends
the returned string to its error list *once per entry of `input_filtered`*. A
`**kwargs` validator therefore reports the same message against **every** input
on the node — the user sees one identical error per socket. Naming only `palette`
makes the message point at the socket that is actually wrong.

A second, quieter cost of `**kwargs`: `validate_has_kwargs` also gates ComfyUI's
own min/max and combo-membership checks, so it silently disabled them for the
whole node. Because this validator no longer takes `**kwargs`, `space` and
`dither` are checked by ComfyUI itself — better, and correctly attributed — so
the node does not re-check them by hand.

`_selftest.py` covers all three cases (wired / typed / empty) through the real
`execution.validate_inputs()`, and pins that the empty case produces exactly one
error, attributed to `palette`.

## Determinism

- Every `cv2.kmeans` call in the package is preceded by
  `cv2.setRNGSeed(KMEANS_SEED)` and uses `KMEANS_PP_CENTERS`. That matters to
  this node because a palette derived by
  [`PixelPaletteExtract`](PixelPaletteExtract.md) without a seed would come back
  with different labels for the same image — verified by the selftest.
- Mapping itself is deterministic: nearest-neighbour with ties going to the
  lowest palette index, and the Bayer threshold is a pure function of
  `(y % n, x % n)`.
- `_selftest.py` asserts byte-identical output tensors, and a byte-identical
  `applied_palette`, across two runs.

## Transparency (RGBA)

RGBA input is accepted, and the alpha channel is **passed through byte for
byte**. The contract of this node is "map colours, touch nothing else": the
palette is a colour set, and alpha is not a colour. A transparent pixel has no
colour to quantise, and quantising the mask would turn a soft edge into a
palette entry.

Unlike [`PixelGridRestore`](PixelGridRestore.md) there is deliberately **no**
"drop an all-opaque alpha" rule here, so an RGBA input always yields an RGBA
output whose alpha is byte-identical to the input's. That keeps the node
predictable when it is used on its own rather than downstream of a grid restore.

## See also

- [`PixelPaletteExtract`](PixelPaletteExtract.md) — the usual source of `palette`, and the only node that chooses one
- [`PixelGridRestore`](PixelGridRestore.md) — normally the producer of `image`
- [`PixelExportScale`](PixelExportScale.md) — the last stage
