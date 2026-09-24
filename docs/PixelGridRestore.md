# `PixelGridRestore`

**Category:** `image/pixel pipeline`
**Output kind:** processing node (first stage)

Finds the underlying pixel grid of an AI-generated "fake pixel art" image and
resamples one colour per cell, producing a true low-resolution pixel image.

```
IMAGE -> image (IMAGE), grid_w (INT), grid_h (INT), pixel_size (FLOAT), detected (BOOLEAN), confidence (FLOAT)
```

## Inputs

| id | type | default | range | notes |
|---|---|---|---|---|
| `image` | IMAGE | — | — | Batch entries are processed individually; all must end up with the same grid size. |
| `sampling` | COMBO | `Majority Cluster` | `Majority Cluster`, `Center Sample`, `Median` | How one colour is picked per cell. |
| `grid_mode` | COMBO | `Auto Detect` | `Auto Detect`, `By Pixel Size`, `By Grid Count` | How the grid is determined. |
| `pixel_size` | INT | `16` | 1–512 | `By Pixel Size` only: target edge length of one cell, in input pixels. |
| `grid_w` | INT | `64` | 1–4096 | `By Grid Count` only: cells across the width. |
| `grid_h` | INT | `64` | 1–4096 | `By Grid Count` only: cells down the height. |
| `backend` | COMBO | `Auto` | `Auto`, `OpenCV`, `Lightweight` | Which upstream detector to use — read the tooltip first. |
| `refine_intensity` | FLOAT *(advanced)* | `0.25` | 0.0–0.5 | Grid-refinement search radius. |
| `min_size` | FLOAT *(advanced)* | `4.0` | 1.0–64.0 | Minimum accepted pixel edge, in pixels. `Auto Detect` only. |
| `peak_width` | INT *(advanced)* | `6` | 1–64 | Minimum peak width for FFT harmonic detection. `Auto Detect` only. |
| `fix_square` | BOOLEAN *(advanced)* | `True` | — | Nudge a one-cell-mismatched grid into an even square. |

## Outputs

| display name | type | notes |
|---|---|---|
| `image` | IMAGE | Resampled pixel image: one pixel per grid cell. |
| `grid_w` | INT | Cell count across the width, after upstream refinement. |
| `grid_h` | INT | Cell count down the height, after upstream refinement. |
| `pixel_size` | FLOAT | Effective edge length = input width / `grid_w`. Can be fractional (e.g. 8.192). |
| `detected` | BOOLEAN | `True` when the grid came from auto-detection, `False` when it was supplied. |
| `confidence` | FLOAT | 0..1 diagnostic: does the image really have a periodic grid at the detected pitch? Nothing is blocked on it. |

## `confidence`, and the false positive it exists to expose

Upstream auto-detection can return a grid that **looks entirely plausible and is
wrong**. A blurred gradient yields 64 cells at an implied edge of 8.00 px; the
Qwen pseudo-pixel reference yields 125 cells at 8.192 px. The two are
indistinguishable, and both sit inside upstream's own
`[min_size, max_pixel_size] = [4, 20]` window — so a range check cannot catch it.
Unlike the FFT path, the gradient fallback has *no* validity gate at all.

What separates them is whether the image actually *has* a periodic grid at that
pitch. `confidence` takes the gradient projections along both axes, FFTs them,
and measures how far the bin at the detected grid frequency stands above the
median of the rest of the spectrum — the weaker axis decides.

Raw prominence on the reference set, and the value after the linear map
`clamp((raw - 3.0) / 12.0, 0, 1)`:

| image | raw | `confidence` | verdict |
|---|---|---|---|
| smooth gradient | 1.04 | 0.00 | no grid |
| realistic photo | 2.07 | 0.00 | lots of edges, no grid |
| text-heavy poster | 2.61 | 0.00 | no grid |
| Qwen pseudo-pixel | 9.13 | 0.51 | weak but real grid |
| true pixel art (48 / 64 grid) | 45.16 / 57.83 | 1.00 | real grid |

Note the realistic photo: it carries *more* gradient energy than any pixel-art
sample, so an energy-based score would rank it the most trustworthy image in the
set. Periodicity ranks it correctly.

Two honest caveats:

* The calibration rests on those six images and **no sample falls between 2.61
  and 9.13**, so the middle of the range is interpolated, not measured.
* **`confidence` reads 0 on the manual modes.** `By Pixel Size` / `By Grid Count`
  override the pitch, and a forced pitch generally does not match the image's own
  periodicity — so 0 is the correct answer to "is there a grid at *this* pitch".
  The grid is still applied exactly as requested; this is a diagnostic, not a gate.

In a batch the output is the **minimum** across images: one untrustworthy grid is
reason to distrust the batch.

## The three grid modes

- **Auto Detect** — upstream FFT + gradient detection. Fast when it works, and it
  frequently does not on AI output: the FFT path only survives when the implied
  pixel edge is 4–20 px, and the gradient fallback finds **zero** peaks on
  hard-edged pixel art, because Sobel responds equally on both columns of a 1 px
  step and the peak test is a strict `>`.
- **By Pixel Size** — you give the target edge length (`16`), and the node
  converts it internally:

  ```
  grid_w = round(W / pixel_size)
  grid_h = round(H / pixel_size)
  ```

  Upstream's `grid_size` is a cell **count**, not an edge length — getting this
  backwards turns a request for 16 px blocks into a 16-cell image. This mode
  skips both detectors entirely and is the reliable path.
- **By Grid Count** — you give the cell counts directly.

`By Pixel Size` / `By Grid Count` are the escape hatch when detection fails, and
they are the reason this node wraps upstream at all: the legacy V1 wrapper
exposed no way to reach them.

## Failures raise

Upstream signals "I could not find a grid" by returning `None, None, image` —
the *same object reference*, with no exception. A node that keeps only the image
therefore emits the untouched input, and a downstream `PixelExportScale` blows a
1024 px input up to 4096 px with no warning at all. That is exactly the bug the
legacy V1 node has. This node checks `w is None` and raises a `RuntimeError`
that names the backend, the mode, and the two manual modes to try instead.

Batches whose members refine to different grid sizes raise a `ValueError`, since
there is nothing sane to stack.

## Backend notes

`Auto` resolves to `OpenCV` whenever cv2 is installed. The two backends are not
interchangeable:

1. The OpenCV backend computes grayscale with `COLOR_BGR2GRAY` on an **RGB**
   array (upstream bug), while the no-CV2 backend uses correct RGB luminance
   weights. **The same image can yield different grids.**
2. The Lightweight backend **cannot auto-detect at all** on most AI output:
   `estimate_grid_fft` returns a bare `None` when it finds no harmonics while its
   own caller (`perfect_pixel_noCV2.py:380`) unpacks it as a 2-tuple, so it dies
   with a `TypeError` instead of falling back to the gradient detector. This node
   catches that and re-raises a `RuntimeError` naming the bug.
   Pair `Lightweight` with `By Pixel Size` / `By Grid Count`, which skip detection
   entirely.

These are not patched: doing so would break the pinned `perfect-pixel` PyPI
dependency. See the README for the full upstream-bug list.

## Example

| input | settings | result |
|---|---|---|
| `fox_pixel_00001_.png` (1024×1024) | `Auto Detect` | 125×122, `pixel_size` 8.192, `detected=True` |
| same | `By Pixel Size`, `pixel_size=16` | square grid around 60–64, `detected=False` |
| same | `By Grid Count`, 64×64 | square 64×64 |
| a smooth blurred ramp | `Auto Detect` | **raises** `RuntimeError` (V1 passed it through silently) |

## See also

- [`PixelPaletteExtract`](PixelPaletteExtract.md) / [`PixelPaletteQuantize`](PixelPaletteQuantize.md)
- [`PixelExportScale`](PixelExportScale.md) — the last stage
