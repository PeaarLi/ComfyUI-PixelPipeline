"""``PixelGridRestore`` -- the first stage of the pixel pipeline.

Takes an AI-generated "fake pixel art" image, finds the underlying pixel grid,
samples one colour per cell and hands back a true low-resolution pixel image
plus the metadata the rest of the pipeline needs.

Why this node exists at all
---------------------------
The upstream ``perfect_pixel.get_perfect_pixel`` helper hides three things that
make it unusable directly from a node:

1. **It silently gives up.** On failure it does ``return None, None, image`` --
   the *same object reference*, no exception. A node that only keeps the image
   therefore emits the untouched (un-pixelised) input, and a downstream
   ``PixelExportScale`` blows it up by ``scale``. That is exactly the bug the
   legacy V1 node has. We check ``w is None`` and raise instead.
2. **Auto-detection is fragile on AI output.** The FFT path only runs when the
   harmonics land inside ``(center*0.25, center*1.75)`` and the implied pixel
   edge is 4-20 px; the gradient fallback finds zero peaks on hard-edged pixel
   art (Sobel gives equal response on both columns of a 1 px step, so the
   projection is a 2-wide plateau and peak detection uses a strict ``>``).
   The escape hatch is passing ``grid_size=(grid_w, grid_h)`` explicitly, which
   skips both detectors entirely and goes straight to ``refine_grids``. This
   node exposes that escape hatch as ``grid_mode``.
3. **``grid_size`` is a cell COUNT, not a pixel edge length.** Upstream calls it
   ``scale_col, scale_row`` and feeds it to ``refine_grids(image, size_x, ...)``
   where ``cell_w = W / grid_x``. So ``By Pixel Size`` has to convert:
   ``grid_w = round(W / pixel_size)``, ``grid_h = round(H / pixel_size)``.
   Getting this wrong turns a request for 16 px blocks into a 16-cell image.

Upstream bugs we deliberately do *not* patch (see ``backend`` tooltip)
----------------------------------------------------------------------
* ``perfect_pixel.py`` line 8 computes ``mag = 1 - np.log1p(mag)`` -- the power
  spectrum is inverted, so strong harmonics become *valleys*. Detection is
  therefore noise-sensitive and occasionally off-by-one.
* The OpenCV backend calls ``cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)`` on an
  **RGB** array (lines 216 and 323) while the noCV2 backend uses correct RGB
  luminance weights. The two backends therefore disagree on R/B weighting and
  can return different grids for the same image.
* ``sample_center`` clips out-of-range coordinates on the cv2 backend but not
  on the noCV2 one, where negative indices wrap around and silently sample the
  wrong pixel.
* The noCV2 backend **cannot auto-detect at all** on most AI output:
  ``estimate_grid_fft`` returns a bare ``None`` on failure while its own caller
  at ``perfect_pixel_noCV2.py:380`` unpacks it as a 2-tuple, so it raises
  ``TypeError`` instead of falling back to the gradient detector. Any such
  exception is re-raised here with the backend named and a concrete suggestion.

Patching them would break the pinned ``perfect-pixel`` PyPI dependency, so the
backend is exposed as an explicit input instead and documented.
"""

from __future__ import annotations

import importlib

import numpy as np
import torch

from comfy_api.latest import io

from ._image_utils import (
    _join_rgb_alpha,
    _split_rgb_alpha,
    _torch_image_to_uint8_bhwc,
    _uint8_rgb_to_torch_image,
)

__all__ = ["PixelGridRestore"]

SAMPLING_OPTIONS = ["Majority Cluster", "Center Sample", "Median"]
GRID_MODE_OPTIONS = ["Auto Detect", "By Pixel Size", "By Grid Count"]
BACKEND_OPTIONS = ["Auto", "OpenCV", "Lightweight"]

# Node-facing label -> upstream ``sample_method`` value.
_SAMPLE_METHOD = {
    "Majority Cluster": "majority",
    "Center Sample": "center",
    "Median": "median",
}

# Upstream module per backend. ``perfect_pixel.perfect_pixel`` needs cv2,
# ``perfect_pixel.perfect_pixel_noCV2`` is the pure-numpy fallback.
_BACKEND_MODULE = {
    "OpenCV": "perfect_pixel.perfect_pixel",
    "Lightweight": "perfect_pixel.perfect_pixel_noCV2",
}

_INSTALL_HINT = 'pip install "perfect-pixel[opencv]"'


# The upstream *steps* we drive ourselves instead of calling get_perfect_pixel().
#
# Why not get_perfect_pixel: it returns ``(w, h, image)`` and nothing else, so
# the grid line coordinates it computed internally are unrecoverable. We need
# them, because Qwen-Image-2.1's VAE emits RGBA and the alpha channel has to be
# resampled onto *exactly* the same grid as the colour channels -- resampling it
# separately (even with the same cell count) would let the two drift apart.
#
# These are upstream internals rather than a public API, so their absence is
# reported clearly instead of failing with an AttributeError deep in the call.
_REQUIRED_UPSTREAM_OPS = (
    "detect_grid_scale",
    "refine_grids",
    "sample_majority",
    "sample_median",
    "sample_center",
)


def _import_backend(backend: str):
    """Import the upstream module for ``backend`` and verify the steps we call."""
    if backend not in _BACKEND_MODULE:
        raise ValueError(
            f"PixelGridRestore: unknown backend {backend!r}; expected one of {BACKEND_OPTIONS}."
        )
    module_name = _BACKEND_MODULE[backend]
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(
            f"PixelGridRestore: 无法导入后端 {backend!r} ({module_name})。"
            f" 请先安装依赖: {_INSTALL_HINT}"
        ) from exc

    missing = [name for name in _REQUIRED_UPSTREAM_OPS if not hasattr(module, name)]
    if missing:
        raise RuntimeError(
            f"PixelGridRestore: 后端 {backend!r} ({module_name}) 缺少本节点需要的上游函数 "
            f"{missing}。本包直接驱动 perfect-pixel 的检测/精修/采样步骤（因为需要拿到网格坐标"
            f"才能让 alpha 与 RGB 严格对齐），因此依赖这些符号。多半是 perfect-pixel 升级后改名了；"
            f"请检查已安装版本，或改用另一个 backend。"
        )
    return module


def _resolve_backend(backend: str):
    """Resolve ``Auto`` to a concrete backend and return ``(label, module)``."""
    if backend == "Auto":
        try:
            import cv2  # noqa: F401
        except ImportError:
            resolved = "Lightweight"
        else:
            resolved = "OpenCV"
    else:
        resolved = backend
    return resolved, _import_backend(resolved)


def _backend_hint(backend_label: str, exc: BaseException) -> str:
    """Extra advice for the Lightweight backend's no-CV2 crash, when it fires."""
    if backend_label != "Lightweight" or not isinstance(exc, TypeError):
        return ""
    return (
        " The Lightweight (no-CV2) backend has a known upstream crash: its "
        "estimate_grid_fft() returns a bare None when it cannot find harmonics, while its own caller "
        "unpacks the result as a 2-tuple (perfect_pixel_noCV2.py:380) -- so it dies instead of falling "
        "back to the gradient detector. Switch backend to 'OpenCV', or use grid_mode='By Pixel Size' / "
        "'By Grid Count', which skip detection entirely."
    )


def _resolve_grid_size(
    grid_mode: str,
    pixel_size: int,
    grid_w: int,
    grid_h: int,
    width: int,
    height: int,
) -> tuple[int, int] | None:
    """Build the upstream ``grid_size`` argument, or ``None`` for auto-detect.

    The conversion for ``By Pixel Size`` is the load-bearing part: upstream's
    ``grid_size`` is a *cell count*, not a pixel edge length, so
    ``grid_w = round(W / pixel_size)``.
    """
    if grid_mode == "Auto Detect":
        return None

    if grid_mode == "By Pixel Size":
        if pixel_size < 1:
            raise ValueError(
                f"PixelGridRestore: pixel_size must be >= 1, got {pixel_size}. "
                "It is the target edge length of one pixel-art cell, in pixels."
            )
        # round() here, not ceil/floor: upstream refines the grid anyway and a
        # half-cell error is smaller than a full off-by-one cell.
        return (
            max(1, int(round(width / pixel_size))),
            max(1, int(round(height / pixel_size))),
        )

    if grid_mode == "By Grid Count":
        if grid_w < 1 or grid_h < 1:
            raise ValueError(
                f"PixelGridRestore: grid_w/grid_h must be >= 1, got ({grid_w}, {grid_h})."
            )
        return int(grid_w), int(grid_h)

    raise ValueError(
        f"PixelGridRestore: unknown grid_mode {grid_mode!r}; expected one of {GRID_MODE_OPTIONS}."
    )


def _grid_confidence(rgb: np.ndarray, grid_w: int, grid_h: int) -> float:
    """How believable is the detected grid, as a 0..1 score.

    Why this is needed: upstream auto-detection can return a **plausible-looking
    but spurious** grid. A blurred gradient yields 64 cells with an implied edge
    of 8.00 px, while the Qwen pseudo-pixel image yields 125 cells at 8.192 px --
    the numbers are indistinguishable and both sit inside upstream's own
    ``[min_size, max_pixel_size] = [4, 20]`` window. So a range check cannot
    catch it. The gradient fallback has *no* validity gate at all, unlike the FFT
    path.

    What does separate them is whether the image actually *has* a periodic grid
    of that pitch. We take the Sobel-style gradient projections along both axes,
    FFT them, and measure how far the bin at the detected grid frequency stands
    above the median of the rest of the spectrum. A real pixel-art grid puts a
    hard spike exactly there; a smooth ramp or a photograph has a flat spectrum.

    Measured on the reference set (raw prominence, before the mapping below):

        smooth gradient      1.04   no grid
        realistic photo      2.07   lots of edges, but no grid
        text-heavy poster    2.61   no grid
        Qwen pseudo-pixel    9.13   weak but real grid
        true pixel art      45.16 / 57.83

    Note the realistic photo: it has *more* gradient energy than any pixel-art
    sample, so an energy-based score would rate it as the most trustworthy image
    in the set. Periodicity gets it right.

    Caveat, stated plainly: the calibration rests on those six images and there
    is no sample between 2.61 and 9.13, so the midpoint of the mapping is
    interpolated, not measured. This is a diagnostic, not a guarantee -- which is
    why it is an output and not a gate.

    Implemented with numpy only, deliberately: the ``Lightweight`` backend exists
    for environments without cv2, so the score must not depend on cv2 either, and
    it must not differ between backends.
    """
    grey = (
        0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    ).astype(np.float32)

    # Central differences; the border stays zero, which only adds a constant DC
    # term that the projection's mean subtraction below removes anyway.
    gx = np.zeros_like(grey)
    gy = np.zeros_like(grey)
    gx[:, 1:-1] = (grey[:, 2:] - grey[:, :-2]) * 0.5
    gy[1:-1, :] = (grey[2:, :] - grey[:-2, :]) * 0.5

    def prominence(projection: np.ndarray, cells: int) -> float:
        centred = projection - projection.mean()
        spectrum = np.abs(np.fft.rfft(centred))
        k = int(round(cells))
        if k < 1 or k >= len(spectrum):
            return 0.0
        # +-1 bin, so a slightly clipped grid line still registers.
        band = spectrum[max(1, k - 1) : k + 2].max()
        baseline = np.median(spectrum[1:]) + 1e-9
        return float(band / baseline)

    # The weaker of the two axes decides: a grid is only real if both agree.
    raw = min(
        prominence(np.abs(gx).sum(axis=0), grid_w),
        prominence(np.abs(gy).sum(axis=1), grid_h),
    )

    # Linear map calibrated on the set above: <=3.0 reads as no grid, and the
    # first genuine grid sample (9.13) lands near the middle rather than at 1.0,
    # so a weak-but-real grid stays distinguishable from a clean one.
    return float(min(1.0, max(0.0, (raw - 3.0) / 12.0)))


def _fix_square_pair(
    rgb: np.ndarray,
    alpha: np.ndarray | None,
    refined_w: int,
    refined_h: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Upstream's ``fix_square``, applied identically to colour and to alpha.

    A verbatim port of ``perfect_pixel.py:413-428``: when the *refined* grid
    misses a square by exactly one cell, drop or duplicate one row/column. The
    parity branch is upstream's, not ours -- it chooses whichever fix leaves an
    even number of cells.

    Callers must only invoke this when ``abs(refined_w - refined_h) == 1``, which
    is upstream's own guard; the equal-size case would otherwise fall into the
    ``else`` branch and silently drop a row.

    ``alpha`` may be ``None`` (RGB input), in which case only ``rgb`` moves.
    """
    if refined_w > refined_h:
        if refined_w % 2 == 1:
            transform = lambda a: a[:, :-1]  # drop one column
        else:
            transform = lambda a: np.concatenate([a[:1, :], a], axis=0)  # dup first row
    else:
        if refined_h % 2 == 1:
            transform = lambda a: a[:-1, :]  # drop one row
        else:
            transform = lambda a: np.concatenate([a[:, :1], a], axis=1)  # dup first col

    rgb = transform(rgb)
    alpha = None if alpha is None else transform(alpha)
    return rgb, alpha


def _pixelate(
    module,
    rgb: np.ndarray,
    alpha: np.ndarray | None,
    sample_method: str,
    grid_size: tuple[int, int] | None,
    min_size: float,
    peak_width: int,
    refine_intensity: float,
    fix_square: bool,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Drive upstream's detection / refinement / sampling, keeping alpha aligned.

    This mirrors ``perfect_pixel.get_perfect_pixel`` (``perfect_pixel.py:384-429``)
    step for step -- same ``detect_grid_scale`` arguments including the hardcoded
    ``max_ratio=1.5``, same ``refine_grids``, same sampler, same ``fix_square``.
    It was checked to reproduce ``get_perfect_pixel`` byte-for-byte on a
    synthetic grid, and ``_selftest.py`` re-checks it against the real baseline.

    The one deliberate difference is that the grid coordinates are reused for
    alpha instead of being recomputed, which is the entire reason for not just
    calling ``get_perfect_pixel``: it never hands back those coordinates.

    Returns ``(rgb_out, alpha_out)``, or ``(None, None)`` on detection failure --
    upstream's own convention, preserved so the caller's error path is unchanged.
    """
    if grid_size is not None:
        scale_col, scale_row = grid_size
    else:
        scale_col, scale_row = module.detect_grid_scale(
            rgb, peak_width=peak_width, max_ratio=1.5, min_size=min_size
        )
        if scale_col is None or scale_row is None:
            return None, None

    size_x = int(round(scale_col))
    size_y = int(round(scale_row))
    x_coords, y_coords = module.refine_grids(rgb, size_x, size_y, refine_intensity)

    refined_w = len(x_coords) - 1
    refined_h = len(y_coords) - 1

    sampler = {
        "majority": module.sample_majority,
        "median": module.sample_median,
        "center": module.sample_center,
    }[sample_method]

    rgb_out = sampler(rgb, x_coords, y_coords)

    alpha_out = None
    if alpha is not None:
        # Replicated to 3 channels so the identical sampler can be reused; the
        # three copies are equal, so whichever one it picks back is alpha.
        alpha_out = sampler(np.repeat(alpha, 3, axis=2), x_coords, y_coords)[..., :1]

    if fix_square and abs(refined_w - refined_h) == 1:
        rgb_out, alpha_out = _fix_square_pair(rgb_out, alpha_out, refined_w, refined_h)

    return rgb_out, alpha_out


class PixelGridRestore(io.ComfyNode):
    """Detect / force a pixel grid and resample one colour per cell."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="PixelGridRestore",
            display_name="Pixel Grid Restore",
            category="image/pixel pipeline",
            description=(
                "Find the underlying pixel grid of an AI-generated pixel-art image and resample one "
                "colour per cell, producing a true low-resolution pixel image. Exposes the full upstream "
                "parameter set that the V1 perfectPixel node could not reach. Raises on grid-detection "
                "failure instead of silently passing the untouched input through."
            ),
            inputs=[
                io.Image.Input(
                    "image",
                    tooltip=(
                        "Input image batch, RGB or RGBA. Each entry is processed on its own; all entries "
                        "must end up with the same grid size. RGBA is the normal case with Qwen-Image-2.1, "
                        "whose VAE carries four channels so it can generate and edit transparent images."
                    ),
                ),
                io.Combo.Input(
                    "sampling",
                    options=SAMPLING_OPTIONS,
                    default="Majority Cluster",
                    tooltip=(
                        "How one colour is picked per cell. 'Majority Cluster' = most common colour in the "
                        "cell (safest for noisy AI output), 'Center Sample' = the single centre pixel, "
                        "'Median' = per-channel median (good against outliers/speckle; not reachable from "
                        "the V1 node)."
                    ),
                ),
                io.Combo.Input(
                    "grid_mode",
                    options=GRID_MODE_OPTIONS,
                    default="Auto Detect",
                    tooltip=(
                        "How the grid is determined. 'Auto Detect' runs upstream FFT + gradient detection, "
                        "which often fails on AI art (blocks wider than 20 px, soft edges) -- it then raises "
                        "here. 'By Pixel Size' and 'By Grid Count' pass an explicit grid and skip detection "
                        "entirely, which is the reliable path."
                    ),
                ),
                io.Int.Input(
                    "pixel_size",
                    default=16,
                    min=1,
                    max=512,
                    step=1,
                    tooltip=(
                        "Only used by grid_mode='By Pixel Size'. Target edge length of one pixel-art cell "
                        "in input pixels. Converted internally to a cell count with "
                        "grid_w = round(W / pixel_size), grid_h = round(H / pixel_size) -- upstream's "
                        "grid_size is a cell COUNT, not an edge length."
                    ),
                ),
                io.Int.Input(
                    "grid_w",
                    default=64,
                    min=1,
                    max=4096,
                    step=1,
                    tooltip="Only used by grid_mode='By Grid Count'. Number of cells across the image width.",
                ),
                io.Int.Input(
                    "grid_h",
                    default=64,
                    min=1,
                    max=4096,
                    step=1,
                    tooltip="Only used by grid_mode='By Grid Count'. Number of cells down the image height.",
                ),
                io.Combo.Input(
                    "backend",
                    options=BACKEND_OPTIONS,
                    default="Auto",
                    tooltip=(
                        "Which upstream detector implementation to use. IMPORTANT: the two backends disagree. "
                        "(1) The OpenCV backend converts gray with COLOR_BGR2GRAY on an RGB array (upstream "
                        "bug) while the Lightweight backend uses correct RGB luminance weights, so the same "
                        "image can yield different grids. (2) The Lightweight backend cannot auto-detect at "
                        "all on most AI output: its FFT estimator returns a bare None when it finds no "
                        "harmonics and its caller unpacks it as a tuple, so it raises TypeError instead of "
                        "falling back. Prefer OpenCV ('Auto' picks it whenever cv2 is installed). Lightweight "
                        "is only safe together with 'By Pixel Size' / 'By Grid Count'."
                    ),
                ),
                # --- advanced (collapsed by default) -------------------------------
                io.Float.Input(
                    "refine_intensity",
                    default=0.25,
                    min=0.0,
                    max=0.5,
                    step=0.01,
                    tooltip=(
                        "Advanced. Grid-refinement search radius. Each estimated grid line at x is snapped "
                        "within [x*(1-r), x*(1+r)]. Upstream recommends 0.0-0.5."
                    ),
                    advanced=True,
                ),
                io.Float.Input(
                    "min_size",
                    default=4.0,
                    min=1.0,
                    max=64.0,
                    step=0.5,
                    tooltip=(
                        "Advanced. Minimum accepted pixel edge length, in pixels. Only affects Auto Detect; "
                        "detections implying a smaller edge are rejected and fall back to the gradient method."
                    ),
                    advanced=True,
                ),
                io.Int.Input(
                    "peak_width",
                    default=6,
                    min=1,
                    max=64,
                    step=1,
                    tooltip="Advanced. Minimum peak width for FFT harmonic peak detection. Auto Detect only.",
                    advanced=True,
                ),
                io.Boolean.Input(
                    "fix_square",
                    default=True,
                    tooltip=(
                        "Advanced. When the refined grid differs by exactly one cell in one axis, nudge it "
                        "into an even square (drop or duplicate one row/column)."
                    ),
                    advanced=True,
                ),
            ],
            outputs=[
                io.Image.Output(
                    display_name="image",
                    tooltip=(
                        "Resampled pixel image: one pixel per detected/forced grid cell. RGBA when the input "
                        "carried real transparency (alpha is resampled onto the exact same grid lines as the "
                        "colour, so the mask stays pixel-aligned), RGB otherwise -- a fully opaque alpha is "
                        "dropped rather than turning every ordinary result into 4 channels."
                    ),
                ),
                io.Int.Output(
                    display_name="grid_w",
                    tooltip="Cell count across the width, after upstream refinement.",
                ),
                io.Int.Output(
                    display_name="grid_h",
                    tooltip="Cell count down the height, after upstream refinement.",
                ),
                io.Float.Output(
                    display_name="pixel_size",
                    tooltip="Effective pixel edge length = input width / grid_w. Can be non-integer (e.g. 8.192).",
                ),
                io.Boolean.Output(
                    display_name="detected",
                    tooltip=(
                        "True when the grid came from auto-detection, False when it was taken from an "
                        "explicit pixel_size / grid_w,grid_h."
                    ),
                ),
                io.Float.Output(
                    display_name="confidence",
                    tooltip=(
                        "0..1 diagnostic: does the image really have a periodic grid of the detected pitch? "
                        "Measures how far the detected grid frequency stands above the rest of the gradient "
                        "spectrum (weakest of the two axes). Upstream auto-detection can return a "
                        "plausible-looking spurious grid -- a blurred gradient scores a 64-cell grid at 8.00 px "
                        "edge, a real pixel image scores 125 cells at 8.192 px, and a range check cannot tell "
                        "them apart. Periodicity can: measured ~1.0-2.6 for gradients / photos / posters and "
                        "9.1-57.8 for real grids. Nothing is blocked on this value -- gate it yourself if you "
                        "want. Calibrated on six reference images with no sample between 2.6 and 9.1, so the "
                        "middle of the range is interpolated rather than measured. In a batch this is the "
                        "minimum across images. EXPECT 0 ON THE MANUAL MODES: with 'By Pixel Size' / "
                        "'By Grid Count' you are overriding the pitch, and a forced pitch usually does not "
                        "match the image's own periodicity -- so the score correctly answers 'no grid at this "
                        "pitch'. That is information, not a failure; the grid is still applied as you asked."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        sampling: str = "Majority Cluster",
        grid_mode: str = "Auto Detect",
        pixel_size: int = 16,
        grid_w: int = 64,
        grid_h: int = 64,
        backend: str = "Auto",
        refine_intensity: float = 0.25,
        min_size: float = 4.0,
        peak_width: int = 6,
        fix_square: bool = True,
    ) -> io.NodeOutput:
        try:
            sample_method = _SAMPLE_METHOD[sampling]
        except KeyError:
            raise ValueError(
                f"PixelGridRestore: unknown sampling {sampling!r}; expected one of {SAMPLING_OPTIONS}."
            ) from None

        backend_label, upstream = _resolve_backend(backend)
        detected = grid_mode == "Auto Detect"

        imgs = _torch_image_to_uint8_bhwc(image)  # numpy uint8 [B,H,W,C]
        if imgs.ndim != 4 or imgs.shape[-1] not in (3, 4):
            raise ValueError(
                f"PixelGridRestore: expected IMAGE as [B,H,W,3] or [B,H,W,4], "
                f"got {tuple(imgs.shape)}."
            )

        # Qwen-Image-2.1's VAE carries four channels so it can generate and edit
        # transparent images, so RGBA arrives here routinely -- not as an edge
        # case. Colour and alpha are separated because every colour algorithm
        # below must not see alpha (it would be read as a 4th colour, and the
        # grid detector's grey conversion would fold it into luminance).
        rgb_batch, alpha_batch = _split_rgb_alpha(imgs)
        input_has_alpha = alpha_batch is not None

        rgb_outs: list[np.ndarray] = []
        alpha_outs: list[np.ndarray | None] = []
        grid_sizes: list[tuple[int, int]] = []
        confidences: list[float] = []
        first_width = 0

        for i in range(rgb_batch.shape[0]):
            rgb = rgb_batch[i]  # [H,W,3] uint8
            alpha = alpha_batch[i] if input_has_alpha else None  # [H,W,1] uint8 | None
            height, width = rgb.shape[:2]
            if i == 0:
                first_width = width

            grid_size = _resolve_grid_size(
                grid_mode, int(pixel_size), int(grid_w), int(grid_h), width, height
            )

            try:
                out_rgb, out_alpha = _pixelate(
                    upstream,
                    rgb,
                    alpha,
                    sample_method=sample_method,
                    grid_size=grid_size,
                    min_size=float(min_size),
                    peak_width=int(peak_width),
                    refine_intensity=float(refine_intensity),
                    fix_square=bool(fix_square),
                )
            except Exception as exc:  # noqa: BLE001 -- upstream leaks raw exceptions
                raise RuntimeError(
                    f"PixelGridRestore: perfect-pixel raised {type(exc).__name__} on image {i} "
                    f"({width}x{height}), backend={backend_label!r}, grid_mode={grid_mode!r}: {exc}."
                    + _backend_hint(backend_label, exc)
                ) from exc

            # The whole reason this node exists: upstream signals failure by
            # returning nothing rather than raising. The V1 node dropped that
            # signal and emitted the untouched input.
            if out_rgb is None:
                raise RuntimeError(
                    f"PixelGridRestore: 网格检测失败 (grid detection failed) on image {i} "
                    f"({width}x{height}), backend={backend_label!r}, grid_mode={grid_mode!r}. "
                    "上游 perfect-pixel 的自动检测（FFT + 梯度后备）在这张图上找不到网格，"
                    "它不会抛异常、只会原样返回输入图；为避免静默输出未像素化的原图，本节点在此报错。"
                    "建议改用 grid_mode='By Pixel Size' 或 grid_mode='By Grid Count' 手动指定网格。"
                )

            rgb_outs.append(out_rgb)
            alpha_outs.append(out_alpha)
            out_h, out_w = out_rgb.shape[:2]
            grid_sizes.append((int(out_w), int(out_h)))
            # Scored on the *input* image (that is what the user's eye judges)
            # against the *refined* grid that was actually applied to it. Colour
            # only: grid periodicity has nothing to do with the alpha mask.
            confidences.append(_grid_confidence(rgb, out_w, out_h))

        # Batches must agree, otherwise there is nothing sane to stack.
        width_set = {gw for gw, _ in grid_sizes}
        height_set = {gh for _, gh in grid_sizes}
        if len(width_set) != 1 or len(height_set) != 1:
            raise ValueError(
                "PixelGridRestore: 批内网格尺寸不一致 (grid sizes differ across the batch): "
                f"{sorted(grid_sizes)}. Please process images one-by-one or ensure the same input sizing."
            )

        stacked_rgb = np.stack(rgb_outs, axis=0)  # [B,h,w,3]

        # Alpha is carried through only when the batch actually has transparency.
        # The VAE emits 4 channels even for a fully opaque image, and turning
        # every ordinary RGB result into RGBA would surprise every downstream
        # node, so an all-opaque alpha is dropped. Batch-wide, not per image,
        # because the stacked tensor needs one channel count.
        carries_alpha = input_has_alpha and any(
            a is not None and int(a.min()) < 255 for a in alpha_outs
        )
        if carries_alpha:
            opaque = np.full_like(stacked_rgb[..., :1], 255)
            stacked_alpha = np.stack(
                [opaque if a is None else a for a in alpha_outs], axis=0
            )
            stacked = _join_rgb_alpha(stacked_rgb, stacked_alpha)
        else:
            stacked = stacked_rgb

        # ``stacked`` is already batched [B,h,w,C], so this must NOT go through
        # ``_uint8_rgb_to_torch_image``: that helper unsqueezes for the
        # single-image case and would hand back a 5D tensor here.
        if stacked.dtype != np.uint8:
            stacked = np.clip(np.rint(stacked), 0, 255).astype(np.uint8)
        out = torch.from_numpy(np.ascontiguousarray(stacked)).float() / 255.0
        out = torch.clamp(out, 0.0, 1.0)

        out_grid_w, out_grid_h = grid_sizes[0]
        # Effective edge length is measured against the *input* width, which is
        # what makes 125 cells on a 1024 px image read as 8.192.
        effective_pixel_size = float(first_width) / float(out_grid_w) if out_grid_w else 0.0

        return io.NodeOutput(
            out,
            int(out_grid_w),
            int(out_grid_h),
            float(effective_pixel_size),
            bool(detected),
            # Minimum across the batch, not the mean: one untrustworthy grid in a
            # batch is a reason to distrust the batch.
            float(min(confidences)),
        )
