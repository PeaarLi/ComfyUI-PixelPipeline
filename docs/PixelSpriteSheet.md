# `PixelSpriteSheet`

**Category:** `image/pixel pipeline`
**Output kind:** processing node (packing stage)

Packs a batch of same-sized frames into one sprite-sheet image, row-major,
and reports the grid geometry so a game engine's slice settings can be driven
without guessing. Ported from the tiling stage of the local
`webm2spritesheet.py` tool (Webm2Frames).

```
IMAGE -> sheet (IMAGE), columns (INT), rows (INT), cell_width (INT), cell_height (INT)
```

## Inputs

| id | type | default | range | notes |
|---|---|---|---|---|
| `image` | IMAGE | — | — | Same-sized frames, RGB or RGBA. Normally the output of [`PixelBackgroundRemove`](PixelBackgroundRemove.md), so every cell is already the same centred square crop. |
| `layout` | DYNAMICCOMBO | `Auto Square` | — | `Auto Square` (first option = default): `columns = ceil(sqrt(n))`, the reference script's near-square layout. `Fixed Columns`: wrap after exactly `columns` frames per row (`columns`: INT, `8`, 1–1024). |

## Outputs

| display name | type | notes |
|---|---|---|
| `sheet` | IMAGE | `[1, rows*H, columns*W, C]`; channel count unchanged from the input. Empty trailing cells are zero — black in RGB, transparent in RGBA. |
| `columns` | INT | Actual grid columns. |
| `rows` | INT | Actual grid rows. |
| `cell_width` | INT | Cell width (= input frame width) — Unity's *Grid By Cell Size* value. |
| `cell_height` | INT | Cell height (= input frame height). |

## Why a pure torch copy

Nothing is resampled or re-coloured, so nothing goes through uint8: the sheet
is assembled by copying each frame tensor into a zero-initialised canvas
(`image.new_zeros`, same device and dtype). Whatever the upstream nodes
produced arrives in the sheet byte-for-byte.

## Trailing cells

`columns * rows - n` cells at the end of the grid are left zeroed, matching
the reference's ffmpeg `tile` behaviour. In RGBA that is a transparent hole;
in RGB it is black — if black is a meaningful sprite colour, run
[`PixelBackgroundRemove`](PixelBackgroundRemove.md) first so the sheet carries
alpha.

## `Fixed Columns` beyond the frame count

Honoured, not clamped: `columns=8` with 5 frames gives one row of 8 with three
empty cells, and `columns` reports `8`. An explicit column count is a layout
requirement (engine atlases often want a fixed row width), so the node does
not second-guess it.

## Example

5 frames of 8×8:

| layout | result |
|---|---|
| `Auto Square` | 3 columns × 2 rows, sheet 24×16, last cell empty |
| `Fixed Columns`, `columns=2` | 2 × 3, sheet 16×24 |
| `Fixed Columns`, `columns=8` | 8 × 1, sheet 64×8, cells 6–8 empty |
| 1 frame, `Auto Square` | 1 × 1 — the sheet is the frame |

## See also

- [`PixelBackgroundRemove`](PixelBackgroundRemove.md) — cleans and aligns the frames before packing
- [`PixelExportScale`](PixelExportScale.md) — crisp integer upscale of the finished sheet
