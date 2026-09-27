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
| `grid_mode` | COMBO (dynamic) | `Auto Detect` | `Auto Detect`, `By Pixel Size`, `By Grid Count` | How the grid is determined, **and** the parameters that mode needs — see [`grid_mode`](#grid_mode-and-its-per-mode-parameters). |
| `backend` | COMBO | `Auto` | `Auto`, `OpenCV`, `Lightweight` | Which upstream detector to use — read the tooltip first. |
| `refine_intensity` | FLOAT *(advanced)* | `0.25` | 0.0–0.5 | Grid-refinement search radius. Applies to all three modes. |
| `fix_square` | BOOLEAN *(advanced)* | `True` | — | Nudge a one-cell-mismatched grid into an even square. Applies to all three modes. |

### `grid_mode` and its per-mode parameters

`grid_mode` is a **dynamic combo**: the mode selector and that mode's own
parameters are one control, and only the selected mode's inputs are ever shown.
The nested inputs are:

| mode | id | type | default | range | notes |
|---|---|---|---|---|---|
| `Auto Detect` | `min_size` | FLOAT | `4.0` | 1.0–64.0 | Minimum accepted pixel edge, in pixels. |
| `Auto Detect` | `max_pixel_size` | FLOAT | `20.0` | 1.0–512.0 | Maximum accepted pixel edge, in pixels. See [`max_pixel_size`](#max_pixel_size-the-gate-that-throws-away-good-detections). |
| `Auto Detect` | `peak_width` | INT | `6` | 1–64 | Minimum peak width for FFT harmonic detection. |
| `By Pixel Size` | `pixel_size` | INT | `16` | 1–512 | Target edge length of one cell, in input pixels. |
| `By Grid Count` | `grid_w` | INT | `64` | 1–4096 | Cells across the width. |
| `By Grid Count` | `grid_h` | INT | `64` | 1–4096 | Cells down the height. |

Four facts about this control that are easy to get wrong, all verified against
the installed ComfyUI build:

- **Only the selected mode's parameters are added to the prompt.** An unselected
  mode's controls are not merely hidden — they are not part of the prompt at all,
  so they can never block one. Each mode remembers its own values while the
  session lasts, but **only the selected mode's values are saved into a
  workflow**.
- **The three `Auto Detect` parameters are Auto-only.** They are passed to
  nothing but the detector, so they are inert in the two manual modes and are
  therefore nested inside `Auto Detect` rather than sitting at the top level
  where they would always look live.
  `refine_intensity` and `fix_square` stayed at the top level because they apply
  to every mode: refinement and the square fix run on the grid the mode produced.
- **The dynamic combo input takes no `default` kwarg.** A freshly added node
  therefore selects whatever the frontend picks, and the pack's only lever is
  that `Auto Detect` is `options[0]`. `execute` also treats a missing or `None`
  `grid_mode` as `Auto Detect`, which is what a hand-written API prompt or a
  direct Python call will hand it.
- **It takes no `advanced` kwarg either.** Passing one raises `TypeError` at
  import time, which ComfyUI swallows into a single log warning — the whole pack
  then silently disappears from the node menu. Do not "tidy" one in.

## Outputs

| display name | type | notes |
|---|---|---|
| `image` | IMAGE | Resampled pixel image: one pixel per grid cell. RGBA when the input carried real transparency — see [Transparency](#transparency-rgba). |
| `grid_w` | INT | Cell count across the width, after upstream refinement. |
| `grid_h` | INT | Cell count down the height, after upstream refinement. |
| `pixel_size` | FLOAT | Effective edge length = input width / `grid_w`. Can be fractional (e.g. 8.192). |
| `detected` | BOOLEAN | `True` when the grid came from auto-detection, `False` when it was supplied. |
| `confidence` | FLOAT | 0..1 diagnostic: does the image really have a periodic grid at the detected pitch? Nothing is blocked on it. |

## `confidence`, and the false positive it exists to expose

Upstream auto-detection can return a grid that **looks entirely plausible and is
wrong**. A blurred gradient yields 64 cells at an implied edge of 8.00 px; the
Qwen pseudo-pixel reference yields 125 cells at 8.192 px. The two are
indistinguishable, and both sit inside the window upstream itself accepts —
`[min_size, max_pixel_size] = [4, 20]` by default — so a range check cannot catch
it. Unlike the FFT path, the gradient fallback has *no* validity gate at all.

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
* **The manual modes are scored too.** This section used to claim `confidence`
  always reads 0 under `By Pixel Size` / `By Grid Count`. That was wrong: the
  value is computed in every mode, and a forced grid that genuinely matches the
  content scores **high** — measured `1.000` when a true 96-cell grid was
  requested as `96×96`. So read `detected=False` together with a *high*
  confidence as "your forced grid agrees with the content", which is useful
  information rather than a contradiction. It reads low only when a forced pitch
  does not match the image's own periodicity, and that is not a failure either —
  the grid is applied exactly as requested. This is a diagnostic, not a gate.

In a batch the output is the **minimum** across images: one untrustworthy grid is
reason to distrust the batch.

## Transparency (RGBA)

Qwen-Image-2.1's VAE carries **four** channels so the model can generate and edit
transparent images natively, which means `VAEDecode` hands this node RGBA on
every run -- opaque or not. Both three and four channels are accepted.

Colour and alpha are separated for the whole node. The grid is detected and
refined from the **colour** channels only -- folding alpha into the grey
conversion would let a transparent region influence the detected pitch -- and
then the **same grid coordinates** are used to sample alpha. That is the
load-bearing detail. Sampling alpha onto its own grid, even with an identical
cell count, lets the mask edge drift off the colour's cell boundaries: a soft
halo appears and cells that should be fully transparent come back partly opaque.
`_selftest.py` asserts the mask edge lands exactly on a cell boundary.

Output channels follow what the input actually contains, decided **per batch**
(the stacked tensor needs a single channel count):

| input | output |
|---|---|
| RGB | RGB |
| RGBA, all-opaque | **RGB** -- an all-opaque alpha is dropped rather than turning every ordinary result into 4 channels |
| RGBA with real transparency | **RGBA**, with alpha resampled onto the same grid lines |

**Alpha keeps the source mask's softness.** A diffusion model's transparent
background normally has a soft edge, and the sampler picks one value per cell
from within it, so the result can contain partially-transparent cells (a faint
halo). The colour channels are strictly reduced to the palette, but alpha is
*not* binarised -- flattening a mask is a separate, deliberate decision and not
something a grid restore should make on your behalf. If you need a hard-edged
mask, threshold it downstream.

## The three grid modes

- **Auto Detect** — upstream FFT + gradient detection, using `min_size`,
  `max_pixel_size` and `peak_width`. Fast when it works, and it frequently does
  not on AI output: by default the FFT path only survives when the implied pixel
  edge is 4–20 px, and the gradient fallback finds **zero** peaks on hard-edged
  pixel art, because Sobel responds equally on both columns of a 1 px step and
  the peak test is a strict `>`.
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

## `max_pixel_size`: the gate that throws away good detections

Upstream's `detect_grid_scale` (`perfect_pixel.py:322-355`) rejects an FFT
result whose implied pixel edge falls outside a window. The upper end of that
window is `max_pixel_size = 20.0`, and it is a **hardcoded local variable**
(`:333`) — not a parameter, and not a module attribute that could be patched.
The gate is an OR of four conditions, so neither `min_size` nor upstream's aspect
ratio can defeat it.

What happens to an image whose cells are genuinely wider than 20 px is the
problem: its (correct) FFT result fails the gate, and control drops to
`estimate_grid_gradient`, which has **no validity gate at all**. Measured on a
run of this node: a 1024 px image with coarse cells came back as a **205-cell
grid at exactly 5.00 px**, raised nothing, and was saved. The file currently
sitting in `output\fox_pixel_00001_.png` does *not* hit this gate — its FFT fails
outright and the gradient path returns 128 cells — so the 205-cell number comes
from a different generation and is not reproducible from that file.

Exposing the ceiling therefore required **porting** the function:
`_detect_grid_scale_port` in `nodes/grid_restore.py`. `_detect_grid_scale`
dispatches between the two:

```
max_pixel_size == 20.0 (UPSTREAM_MAX_PIXEL_SIZE)  ->  upstream's own detect_grid_scale
otherwise                                        ->  the port
```

That fast path is why the default behaviour is byte-identical **by
construction** rather than by test — at the default ceiling there is no ported
behaviour in play at all. The port runs only once the user moves the ceiling.

Two details in the port look like tidy-up opportunities and are deliberately
not:

- **The two backends produce grayscale differently, and both are reproduced
  exactly.** OpenCV's `detect_grid_scale` calls `cv2.cvtColor(image,
  COLOR_BGR2GRAY)` on an **RGB** array (upstream bug #2 — R and B weights
  swapped); the Lightweight backend calls its own `rgb_to_gray` (correct Rec.601
  weights) and does not import cv2 at all. The FFT validity gate is calibrated
  against whichever one is in play, so "fixing" the OpenCV weights would change
  which detections pass the gate — and therefore change the default
  configuration's output, which is exactly what this refactor promises not to
  do.
- **The FFT result is not re-rounded.** The cv2 estimator returns ints; the
  no-CV2 one returns **unrounded floats**. An `int(round(...))` here would make
  the Lightweight backend diverge from upstream on success.

### What raising the ceiling actually buys

Measured with 512×512 synthetic grids (random palette, hard edges):

| image | upstream `detect_grid_scale` | port @20.0 | port @64.0 |
|---|---|---|---|
| 8 px cells | (64, 64) | (64, 64) | (64, 64) |
| 16 px cells | (32, 32) | (32, 32) | (32, 32) |
| 24 px cells | (None, None) | (None, None) | **(21, 21)** |
| 32 px cells | (None, None) | (None, None) | **(16, 16)** |

At the default the port and upstream agree exactly, which is the property the
dispatch exists to guarantee. Raising the ceiling to 64 resolves the 32 px case
to the correct 16×16 grid — and that is the *only* honest headline, because the
24 px row shows the other half of the story.

### The downside: loud failures become quiet wrong answers

Look at the 24 px row again. At the default ceiling it is **rejected** — a loud,
visible failure. At a raised ceiling it is accepted as an **aliased 21-cell
grid**: the detector has locked onto a harmonic of the true pitch, the node
raises nothing, and the resulting image is wrong. Raising the ceiling does not
make detection better; it moves the failure mode.

That is why `confidence` stays an **output-only diagnostic with no gating** on
it, and why the `max_pixel_size` tooltip tells you to watch it. If you raise the
ceiling, check `confidence` — a score near 0 on a detection you just "rescued" is
the node telling you it may have aliased.

Note that `_selftest.py` pins both directions: 32 px cells resolve to (16, 16) at
64.0, and 24 px cells become (21, 21) — the second check is there so the downside
cannot be quietly forgotten.

## Failures raise

Upstream signals "I could not find a grid" by returning `None, None, image` —
the *same object reference*, with no exception. A node that keeps only the image
therefore emits the untouched input, and a downstream `PixelExportScale` blows a
1024 px input up to 4096 px with no warning at all. That is exactly the bug the
legacy V1 node has. This node checks `w is None` and raises a `RuntimeError`
that names the backend, the mode, and the two manual modes to try instead.

Batches whose members refine to different grid sizes raise a `ValueError`, since
there is nothing sane to stack.

### `Lightweight` + `Auto Detect` is refused before it runs

The Lightweight (no-CV2) backend cannot auto-detect at all — its
`estimate_grid_fft` returns a bare `None` when it finds no harmonics while its
own caller unpacks the result as a 2-tuple, so it dies with a `TypeError` instead
of falling back. Rather than let that upstream exception escape, `execute` runs a
**pre-flight check** and raises a `RuntimeError` that names the fix: install
`perfect-pixel[opencv]` and leave `backend='Auto'` so OpenCV is picked, or switch
`grid_mode` to `By Pixel Size` / `By Grid Count`, both of which skip detection
entirely and work fine on this backend.

The check is **behavioural, not nominal**. `_auto_detect_usable(module)` calls the
backend's own `estimate_grid_fft` on a constant 8×8 image and asks whether it
returned a tuple — the cv2 module answers `(None, None)`, the no-CV2 module
answers a bare `None`. It costs one tiny FFT and it keys off behaviour rather
than a backend name or a version number, so if upstream ever fixes the bug the
restriction simply stops applying.

Two boundaries worth knowing:

- **`Lightweight` + `By Pixel Size` / `By Grid Count` is unaffected** — the check
  is guarded by "am I about to auto-detect", so the safe combination still runs.
- The node does **not** silently route `Lightweight` + `Auto Detect` through the
  port, even though the port could normalise the bare `None` and fall back to the
  gradient detector. `Auto Detect` means the same thing on every backend; making
  one backend take a different code path would make the mode's name a lie.

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
   **refuses the combination before it starts**, with a `RuntimeError` naming the
   bug and the fix — see
   [`Lightweight` + `Auto Detect`](#lightweight--auto-detect-is-refused-before-it-runs).
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
| 512×512 synthetic, 32 px cells | `Auto Detect`, `max_pixel_size=64` | (16, 16) — the correct grid at last |
| 512×512 synthetic, 24 px cells | `Auto Detect`, `max_pixel_size=64` | (21, 21) — **aliased**: accepted where the default ceiling rejected it |

The last two rows are the same measurement from opposite sides: raising the
ceiling recovers a real grid and admits a wrong one. Check `confidence`.

## See also

- [`PixelPaletteExtract`](PixelPaletteExtract.md) / [`PixelPaletteApply`](PixelPaletteApply.md)
- [`PixelExportScale`](PixelExportScale.md) — the last stage
