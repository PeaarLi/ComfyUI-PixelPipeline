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

from ._image_utils import _torch_image_to_uint8_rgb, _uint8_rgb_to_torch_image

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


def _import_backend(backend: str):
    """Return the upstream ``get_perfect_pixel`` for the requested backend."""
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
    return module.get_perfect_pixel


def _resolve_backend(backend: str):
    """Resolve ``Auto`` to a concrete backend and return ``(label, fn)``."""
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
                    tooltip="Input image batch. Each entry is processed on its own; all entries must end up with the same grid size.",
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
                    tooltip="Resampled pixel image: one pixel per detected/forced grid cell.",
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

        backend_label, get_perfect_pixel = _resolve_backend(backend)
        detected = grid_mode == "Auto Detect"

        imgs = _torch_image_to_uint8_rgb(image)  # numpy uint8 [B,H,W,C]
        if imgs.ndim != 4 or imgs.shape[-1] != 3:
            raise ValueError(
                f"PixelGridRestore: expected IMAGE as [B,H,W,3], got {tuple(imgs.shape)}."
            )

        outs: list[torch.Tensor] = []
        grid_sizes: list[tuple[int, int]] = []
        confidences: list[float] = []
        first_width = 0

        for i in range(imgs.shape[0]):
            rgb = imgs[i]  # [H,W,3] uint8
            height, width = rgb.shape[:2]
            if i == 0:
                first_width = width

            grid_size = _resolve_grid_size(
                grid_mode, int(pixel_size), int(grid_w), int(grid_h), width, height
            )

            # ``debug`` stays False, always: upstream's debug path imports
            # matplotlib and calls plt.show(), which blocks a headless server.
            try:
                w, h, out_rgb = get_perfect_pixel(
                    rgb,
                    sample_method=sample_method,
                    grid_size=grid_size,
                    min_size=float(min_size),
                    peak_width=int(peak_width),
                    refine_intensity=float(refine_intensity),
                    fix_square=bool(fix_square),
                    debug=False,
                )
            except Exception as exc:  # noqa: BLE001 -- upstream leaks raw exceptions
                raise RuntimeError(
                    f"PixelGridRestore: perfect-pixel raised {type(exc).__name__} on image {i} "
                    f"({width}x{height}), backend={backend_label!r}, grid_mode={grid_mode!r}: {exc}."
                    + _backend_hint(backend_label, exc)
                ) from exc

            # The whole reason this node exists: upstream signals failure by
            # returning ``None, None, image`` (same object, no exception). The V1
            # node dropped w/h and emitted the untouched input.
            if w is None or h is None:
                raise RuntimeError(
                    f"PixelGridRestore: 网格检测失败 (grid detection failed) on image {i} "
                    f"({width}x{height}), backend={backend_label!r}, grid_mode={grid_mode!r}. "
                    "上游 perfect-pixel 的自动检测（FFT + 梯度后备）在这张图上找不到网格，"
                    "它不会抛异常、只会原样返回输入图；为避免静默输出未像素化的原图，本节点在此报错。"
                    "建议改用 grid_mode='By Pixel Size' 或 grid_mode='By Grid Count' 手动指定网格。"
                )

            out_t = _uint8_rgb_to_torch_image(out_rgb)  # [1,h,w,3]
            outs.append(out_t)
            out_w, out_h = int(out_t.shape[2]), int(out_t.shape[1])
            grid_sizes.append((out_w, out_h))
            # Scored on the *input* image (that is what the user's eye judges)
            # against the *refined* grid that was actually applied to it.
            confidences.append(_grid_confidence(rgb, out_w, out_h))

        # Batches must agree, otherwise there is nothing sane to stack.
        width_set = {gw for gw, _ in grid_sizes}
        height_set = {gh for _, gh in grid_sizes}
        if len(width_set) != 1 or len(height_set) != 1:
            raise ValueError(
                "PixelGridRestore: 批内网格尺寸不一致 (grid sizes differ across the batch): "
                f"{sorted(grid_sizes)}. Please process images one-by-one or ensure the same input sizing."
            )

        out = torch.cat(outs, dim=0)  # [B,h,w,3]
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
