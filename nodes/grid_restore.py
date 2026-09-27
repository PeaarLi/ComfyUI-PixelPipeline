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
   node exposes that escape hatch as the ``By Pixel Size`` / ``By Grid Count``
   options of ``grid_mode``.
   The 20 px half of that window is the more insidious failure, because it is
   silent: ``max_pixel_size = 20.0`` is a **local variable inside**
   ``detect_grid_scale`` (``perfect_pixel.py:333``), so an image whose cells are
   genuinely wider than 20 px has its (correct) FFT result discarded by the gate
   and handed to the gradient fallback, which has **no validity gate at all**
   and invents a pitch. Measured: a 1024 px image with coarse cells came back as
   a 205-cell grid at exactly 5.00 px, raised nothing, and was saved. The gate is
   an OR of four conditions, so no combination of ``min_size`` / ``max_ratio``
   can defeat it -- which is why this node ports the function (see
   :func:`_detect_grid_scale_port`) to expose ``max_pixel_size``. That ceiling
   was hidden while ``min_size``, the parameter that *loosens* the gate, was
   exposed: half the control, and the wrong half.
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

# Schema defaults, kept in one place so ``define_schema`` and the execute-side
# unpacking of the DynamicCombo dict cannot drift apart. ``_selftest.py``
# asserts the schema defaults equal these.
DEFAULT_MIN_SIZE = 4.0
DEFAULT_MAX_PIXEL_SIZE = 20.0
DEFAULT_PEAK_WIDTH = 6
DEFAULT_PIXEL_SIZE = 16
DEFAULT_GRID_W = 64
DEFAULT_GRID_H = 64

#: Upstream's own hardcoded ceiling (``perfect_pixel.py:333``). Kept as a named
#: constant because it is the identity anchor for the whole ``max_pixel_size``
#: feature: when the user leaves the parameter at this value we call upstream's
#: own detector and the output is unchanged *by construction*, not by test.
UPSTREAM_MAX_PIXEL_SIZE = 20.0

#: Upstream's hardcoded aspect-ratio gate. Deliberately NOT exposed: non-square
#: pixels are rare in AI pixel art, and the point of this refactor is fewer
#: knobs, not more. Passed through explicitly so the port cannot drift from it.
UPSTREAM_MAX_RATIO = 1.5

#: Seed forced into OpenCV before every sampler call. See :func:`_seed_sampler`.
#: Deliberately the same value as ``_palette_utils.KMEANS_SEED`` so the pack has
#: one "fixed seed" idea -- but NOT imported from there, because that would drag
#: PIL into this module's dependency graph for a single integer. ``_selftest.py``
#: asserts the two stay equal.
SAMPLE_SEED = 0x5EED_1234

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
    "estimate_grid_fft",
    "estimate_grid_gradient",
    "refine_grids",
    "sample_majority",
    "sample_median",
    "sample_center",
)

#: Extra symbols only one backend needs. ``rgb_to_gray`` lives only in the no-CV2
#: module (the OpenCV one inlines ``cv2.cvtColor`` inside ``detect_grid_scale``),
#: so listing it per backend keeps the failure a named "missing op" rather than
#: an ``AttributeError`` deep inside the driver.
_BACKEND_EXTRA_OPS = {
    "Lightweight": ("rgb_to_gray",),
}

# ``detect_grid_scale`` is still required even though ``_detect_grid_scale``
# prefers the port whenever the ceiling is not at its default: the port mirrors
# that function, the fast path calls it, and ``_selftest.py`` diffs the two
# against each other. Its absence would mean the contract the port implements no
# longer exists.


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

    wanted = _REQUIRED_UPSTREAM_OPS + _BACKEND_EXTRA_OPS.get(backend, ())
    missing = [name for name in wanted if not hasattr(module, name)]
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


#: Why the Lightweight backend cannot auto-detect. Shared by the pre-flight
#: check in ``execute`` and by ``_backend_hint`` so the two can never drift into
#: telling the user two different stories about the same bug.
_LIGHTWEIGHT_WHY = (
    " The Lightweight (no-CV2) backend has a known upstream crash: its "
    "estimate_grid_fft() returns a bare None when it cannot find harmonics, while its own caller "
    "unpacks the result as a 2-tuple (perfect_pixel_noCV2.py:380) -- so it dies instead of falling "
    "back to the gradient detector."
)


def _backend_hint(backend_label: str, exc: BaseException) -> str:
    """Extra advice for the Lightweight backend's no-CV2 crash, when it fires."""
    if backend_label != "Lightweight" or not isinstance(exc, TypeError):
        return ""
    return _LIGHTWEIGHT_WHY + (
        " Switch backend to 'OpenCV', or use grid_mode='By Pixel Size' / "
        "'By Grid Count', which skip detection entirely."
    )


def _unpack_grid_mode(grid_mode) -> tuple[str, dict]:
    """Normalise what ComfyUI hands to ``execute`` for the ``grid_mode`` combo.

    Verified against this build by driving ``_io.get_finalized_class_inputs`` +
    ``_io.build_nested_inputs`` -- the same two calls ``execution.py`` makes::

        {"grid_mode": "Auto Detect", "grid_mode.min_size": 4}
            -> {"grid_mode": {"grid_mode": "Auto Detect", "min_size": 4}}

    So the selected option key arrives *inside* the input's own value, alongside
    that option's nested widget values -- the same shape core's
    ``SaveAudioAdvanced`` reads with ``format.get("format")``.

    Two shapes must not raise:

    * ``None`` -- the prompt carried no ``grid_mode`` key at all (hand-written
      API prompts, or calling ``execute`` from Python). Behaves as Auto Detect
      with the schema defaults.
    * a bare ``str`` -- an option key that matched no declared option. Nothing
      in the backend validates a combo's value, so it arrives as a plain string;
      it is reported as an unknown mode by the caller.
    """
    if isinstance(grid_mode, str):
        return grid_mode, {}
    if not grid_mode:
        return "Auto Detect", {}
    mode = grid_mode.get("grid_mode")
    return (mode if isinstance(mode, str) else "Auto Detect"), grid_mode


def _pick(params: dict, key: str, fallback):
    """A nested option's value, or the schema default when it was not sent.

    ``build_nested_inputs()`` writes ``None`` for a nested key the prompt omitted
    and does *not* apply the schema default (verified against this build), so the
    fallback has to be applied here. The test is an explicit ``is None`` rather
    than ``or``: ``or`` would silently rewrite a legitimate ``0`` or ``False``.
    """
    value = params.get(key, None)
    return fallback if value is None else value


def _backend_gray(module, backend_label: str, rgb: np.ndarray) -> np.ndarray:
    """The backend's own grayscale step, reproduced exactly.

    The two backends do NOT agree here, and the FFT validity gate is calibrated
    against whichever one is in play, so this must not be unified:

    * ``OpenCV`` -- ``perfect_pixel.py:323`` calls ``cv2.cvtColor`` on an **RGB**
      array with ``COLOR_BGR2GRAY``. That is upstream bug #2 (the R and B weights
      are swapped) and it is deliberately NOT "fixed" here: correcting it changes
      which detections pass the gate, and therefore changes the output of the
      default configuration, which is exactly what this refactor promises not to
      do. Fixing it is a separate decision, not a side effect of exposing a
      number.
    * ``Lightweight`` -- ``perfect_pixel_noCV2.py:376`` calls the module's own
      ``rgb_to_gray``, which uses correct Rec.601 weights. That module does not
      import cv2 at all.
    """
    if backend_label == "OpenCV":
        cv = getattr(module, "cv2", None)
        if cv is None:  # pragma: no cover - the module imports cv2 itself
            import cv2 as cv
        return cv.cvtColor(rgb, cv.COLOR_BGR2GRAY)
    return module.rgb_to_gray(rgb)


def _auto_detect_usable(module) -> bool:
    """Can this backend reach the gradient fallback in Auto Detect?

    The no-CV2 backend's ``estimate_grid_fft`` returns a bare ``None`` when it
    finds no harmonics, while its own caller at ``perfect_pixel_noCV2.py:380``
    unpacks the result as a 2-tuple -- so it raises ``TypeError`` instead of
    falling back, and cannot auto-detect most AI output at all.

    Probed *behaviourally* rather than by version or backend name, so that if
    upstream ever fixes the bug this restriction simply stops applying. A
    constant image is used because neither implementation can find harmonics in
    one: the cv2 module returns ``(None, None)``, the no-CV2 module returns
    ``None``. Costs one 8x8 FFT.
    """
    probe = np.zeros((8, 8), dtype=np.float32)
    try:
        return isinstance(module.estimate_grid_fft(probe), tuple)
    except Exception:  # noqa: BLE001 -- any failure means "cannot rely on it"
        return False


def _detect_grid_scale_port(
    module,
    backend_label: str,
    rgb: np.ndarray,
    *,
    peak_width: int,
    min_size: float,
    max_pixel_size: float,
    max_ratio: float = UPSTREAM_MAX_RATIO,
) -> tuple[int, int] | tuple[None, None]:
    """A line-for-line port of upstream's ``detect_grid_scale`` with the ceiling exposed.

    Ported from ``perfect_pixel.py:322-355`` because ``max_pixel_size`` is a
    *local variable* inside that function (``:333``) with no module-level
    attribute to patch and no parameter to pass. The gate is an OR of four
    conditions, so nothing short of replacing the function can raise it --
    ``min_size`` and ``max_ratio`` cannot defeat it.

    Everything else is reproduced verbatim, including the two ``print()``
    diagnostics (the user reads their log through those exact lines) and the
    final ``int(round(W / pixel_size))`` normalisation. Two details look like
    tidy-up opportunities and are not:

    * The FFT result is NOT re-rounded. The cv2 estimator returns ints but the
      no-CV2 one returns **unrounded floats** on success, so an ``int(round(x))``
      here would make the Lightweight backend diverge.
    * The gradient fallback gets no validity gate of its own. Upstream has none
      (see the module docstring); adding one would change the default behaviour
      for every image that falls through, which the fast path in
      :func:`_detect_grid_scale` exists to avoid.

    Returns ``(grid_w, grid_h)`` or ``(None, None)``, upstream's own convention.
    """
    gray = _backend_gray(module, backend_label, rgb)
    height, width = gray.shape

    result = module.estimate_grid_fft(gray, peak_width=peak_width)
    # No-CV2 signals failure with a bare None; cv2 with (None, None). Normalise
    # both rather than letting the bare None raise on unpack.
    grid_w, grid_h = (None, None) if result is None else result

    if grid_w is None or grid_h is None:
        print("FFT-based grid estimation failed, fallback to gradient-based method.")
        grid_w, grid_h = module.estimate_grid_gradient(gray)
    else:
        pixel_size_x = width / grid_w
        pixel_size_y = height / grid_h
        if (
            min(pixel_size_x, pixel_size_y) < min_size
            or max(pixel_size_x, pixel_size_y) > max_pixel_size
            or pixel_size_x / pixel_size_y > max_ratio
            or pixel_size_y / pixel_size_x > max_ratio
        ):
            print("Inconsistent grid size detected (FFT-based), fallback to gradient-based method.")
            grid_w, grid_h = module.estimate_grid_gradient(gray)

    if grid_w is None or grid_h is None:
        print("Gradient-based grid estimation failed.")
        return None, None

    pixel_size_x = width / grid_w
    pixel_size_y = height / grid_h

    if pixel_size_x / pixel_size_y > max_ratio or pixel_size_y / pixel_size_x > max_ratio:
        pixel_size = min(pixel_size_x, pixel_size_y)
    else:
        pixel_size = (pixel_size_x + pixel_size_y) / 2.0

    print(f"Detected pixel size: {pixel_size:.2f}")

    return int(round(width / pixel_size)), int(round(height / pixel_size))


def _detect_grid_scale(
    module,
    backend_label: str,
    rgb: np.ndarray,
    *,
    peak_width: int,
    min_size: float,
    max_pixel_size: float,
    max_ratio: float = UPSTREAM_MAX_RATIO,
) -> tuple[int, int] | tuple[None, None]:
    """Upstream's detector when it is trustworthy and unmodified; ours otherwise.

    The fast path is the entire reason "byte-identical at the default" is true
    *by construction* rather than by measurement: at
    ``max_pixel_size == UPSTREAM_MAX_PIXEL_SIZE`` the port has nothing to add, so
    upstream's own code runs and there is no ported behaviour to trust. The port
    is reached only when the user has moved the ceiling.

    The ``_auto_detect_usable`` clause is belt and braces rather than a live
    branch: by the time ``execute`` gets here its pre-flight has already refused
    a backend that cannot auto-detect, so this is always True in the node's own
    flow. It matters for direct callers (``_selftest.py`` drives this function
    across both backends) and it keeps the invariant local: never hand a
    detection job to a detector known to crash on it.
    """
    if max_pixel_size == UPSTREAM_MAX_PIXEL_SIZE and _auto_detect_usable(module):
        return module.detect_grid_scale(
            rgb, peak_width=peak_width, max_ratio=max_ratio, min_size=min_size
        )
    return _detect_grid_scale_port(
        module,
        backend_label,
        rgb,
        peak_width=peak_width,
        min_size=min_size,
        max_pixel_size=max_pixel_size,
        max_ratio=max_ratio,
    )


def _resolve_grid_size(
    mode: str,
    params: dict,
    width: int,
    height: int,
    node_label: str = "PixelGridRestore",
) -> tuple[int, int] | None:
    """Build the upstream ``grid_size`` argument, or ``None`` for auto-detect.

    The conversion for ``By Pixel Size`` is the load-bearing part: upstream's
    ``grid_size`` is a *cell count*, not a pixel edge length, so
    ``grid_w = round(W / pixel_size)``.

    ``node_label`` only affects the message prefix, so the video node's errors
    name the video node. The default keeps the still node's wording unchanged.
    """
    if mode == "Auto Detect":
        return None

    if mode == "By Pixel Size":
        pixel_size = _pick(params, "pixel_size", DEFAULT_PIXEL_SIZE)
        if pixel_size < 1:
            raise ValueError(
                f"{node_label}: pixel_size must be >= 1, got {pixel_size}. "
                "It is the target edge length of one pixel-art cell, in pixels."
            )
        # round() here, not ceil/floor: upstream refines the grid anyway and a
        # half-cell error is smaller than a full off-by-one cell.
        return (
            max(1, int(round(width / pixel_size))),
            max(1, int(round(height / pixel_size))),
        )

    if mode == "By Grid Count":
        grid_w = int(_pick(params, "grid_w", DEFAULT_GRID_W))
        grid_h = int(_pick(params, "grid_h", DEFAULT_GRID_H))
        if grid_w < 1 or grid_h < 1:
            raise ValueError(
                f"{node_label}: grid_w/grid_h must be >= 1, got ({grid_w}, {grid_h})."
            )
        return grid_w, grid_h

    raise ValueError(
        f"{node_label}: unknown grid_mode {mode!r}; expected one of {GRID_MODE_OPTIONS}."
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


def _seed_sampler(module) -> None:
    """Force a known OpenCV RNG state immediately before a sampler call.

    Why this exists: OpenCV's ``sample_majority`` does its per-cell 2-means with
    ``cv2.kmeans(..., cv2.KMEANS_RANDOM_CENTERS)`` and never seeds it, so **the
    same cell can come back differently on every call**. Measured on this install:

    * six consecutive ``sample_majority`` calls on ONE image -> five results
      differing from the first (188-218 channel values);
    * two frames with **byte-identical** content -> 74 cells differing.

    The second one is not a cosmetic wart. A batch -- or a video -- whose frames
    flicker because the *sampler* is nondeterministic cannot be repaired by any
    amount of temporal filtering afterwards: the flicker is already baked into
    the pixels. Seeding here is the largest single flicker reduction available,
    and it costs one line.

    Only the OpenCV backend needs it. ``perfect_pixel_noCV2`` never imports cv2 --
    its 2-means is a deterministic argmax/mean iteration -- so there is nothing to
    seed there, and cv2 may not even be installed. (The two backends also default
    their *numpy* seed differently: ``sample_majority`` takes ``seed=42`` in the
    cv2 module but ``seed=0`` in the no-CV2 one. That is upstream's, not ours.)

    Note this makes ``PixelGridRestore`` reproducible for the first time, which
    *changes its output* for a given input. That is a behaviour change to a
    shipped node and is called out in the README.
    """
    cv = getattr(module, "cv2", None)
    if cv is not None:
        cv.setRNGSeed(SAMPLE_SEED)


def _refine_once(
    module,
    rgb: np.ndarray,
    grid_w: int,
    grid_h: int,
    refine_intensity: float,
) -> tuple[list, list]:
    """``refine_grids`` on ONE reference frame, for reuse across many frames.

    Kept in this module so the coupling to upstream's operator name lives in one
    place. ``PixelVideoGridRestore`` calls it once and then hands the resulting
    coordinates to :func:`_pixelate` for every frame, which is what makes the
    whole clip share one geometry -- see that node for the trade-off.
    """
    return module.refine_grids(rgb, int(grid_w), int(grid_h), refine_intensity)


def _pixelate(
    module,
    backend_label: str,
    rgb: np.ndarray,
    alpha: np.ndarray | None,
    sample_method: str,
    grid_size: tuple[int, int] | None,
    min_size: float,
    max_pixel_size: float,
    peak_width: int,
    refine_intensity: float,
    fix_square: bool,
    coords: tuple[list, list] | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Drive upstream's detection / refinement / sampling, keeping alpha aligned.

    This mirrors ``perfect_pixel.get_perfect_pixel`` (``perfect_pixel.py:384-429``)
    step for step -- same ``max_ratio=1.5``, same ``refine_grids``, same sampler,
    same ``fix_square``. It was checked to reproduce ``get_perfect_pixel``
    byte-for-byte on a synthetic grid, and ``_selftest.py`` re-checks it against
    the real baseline.

    Detection goes through :func:`_detect_grid_scale` rather than straight to
    ``module.detect_grid_scale``, which at the default ``max_pixel_size`` is the
    same call -- see that function for why the ceiling has to be a parameter.

    The one deliberate difference is that the grid coordinates are reused for
    alpha instead of being recomputed, which is the entire reason for not just
    calling ``get_perfect_pixel``: it never hands back those coordinates.

    Returns ``(rgb_out, alpha_out)``, or ``(None, None)`` on detection failure --
    upstream's own convention, preserved so the caller's error path is unchanged.

    ``coords`` short-circuits both detection *and* refinement, resampling onto
    coordinates the caller already has. ``PixelVideoGridRestore`` uses it to put
    a whole clip on one shared geometry; passing it makes this function's
    sampling step byte-identical to the normal path, because it is the same code.
    """
    if coords is not None:
        x_coords, y_coords = coords
    else:
        if grid_size is not None:
            scale_col, scale_row = grid_size
        else:
            scale_col, scale_row = _detect_grid_scale(
                module,
                backend_label,
                rgb,
                peak_width=peak_width,
                min_size=min_size,
                max_pixel_size=max_pixel_size,
            )
            if scale_col is None or scale_row is None:
                return None, None

        size_x = int(round(scale_col))
        size_y = int(round(scale_row))
        x_coords, y_coords = _refine_once(module, rgb, size_x, size_y, refine_intensity)

    refined_w = len(x_coords) - 1
    refined_h = len(y_coords) - 1

    sampler = {
        "majority": module.sample_majority,
        "median": module.sample_median,
        "center": module.sample_center,
    }[sample_method]

    # Before EVERY sampler call -- see _seed_sampler for why this is not optional.
    _seed_sampler(module)
    rgb_out = sampler(rgb, x_coords, y_coords)

    alpha_out = None
    if alpha is not None:
        # Replicated to 3 channels so the identical sampler can be reused; the
        # three copies are equal, so whichever one it picks back is alpha.
        _seed_sampler(module)
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
                # The mode selector and that mode's parameters in ONE control.
                # Only the selected option's inputs are shown, so no dead knob is
                # ever on screen -- and each mode remembers its own values while
                # the session lasts.
                #
                # Two hard constraints, both verified against this build:
                #  * DynamicCombo.Input takes NO `advanced` kwarg. Passing one
                #    raises TypeError at import, which ComfyUI swallows into a
                #    single log warning -- the whole pack then silently vanishes.
                #  * It takes NO `default` kwarg either, so a freshly added node
                #    shows whatever the frontend picks. The only safe construction
                #    is to put the intended default first, which is why
                #    'Auto Detect' is options[0]; _selftest.py pins that order.
                io.DynamicCombo.Input(
                    "grid_mode",
                    options=[
                        io.DynamicCombo.Option(
                            "Auto Detect",
                            [
                                io.Float.Input(
                                    "min_size",
                                    default=DEFAULT_MIN_SIZE,
                                    min=1.0,
                                    max=64.0,
                                    step=0.5,
                                    tooltip=(
                                        "Minimum accepted pixel edge length, in pixels. A detection "
                                        "implying a smaller edge is rejected and falls through to the "
                                        "gradient detector, which has no validity gate of its own -- so "
                                        "raising this does not make detection stricter, it makes it "
                                        "depend on the ungated fallback instead."
                                    ),
                                ),
                                io.Float.Input(
                                    "max_pixel_size",
                                    default=DEFAULT_MAX_PIXEL_SIZE,
                                    min=1.0,
                                    max=512.0,
                                    step=0.5,
                                    tooltip=(
                                        "Maximum accepted pixel edge length, in pixels. 20.0 is upstream "
                                        "perfect-pixel's own hardcoded ceiling and is what this node uses "
                                        "when you leave it alone. Raise it when the art's cells are wider "
                                        "than 20 px: otherwise the FFT result -- which may well be the "
                                        "correct one -- is thrown away and the ungated gradient detector "
                                        "invents a pitch instead (observed on a 1024 px image: the real "
                                        "cells were coarser than 20 px and the fallback returned a 205-cell "
                                        "grid at 5.00 px, with nothing raising). CAUTION, measured: "
                                        "raising the ceiling also lets aliased FFT estimates through, so "
                                        "some loud failures become quiet wrong answers. Watch the "
                                        "'confidence' output. Lower it below min_size to skip the FFT "
                                        "detector entirely."
                                    ),
                                ),
                                io.Int.Input(
                                    "peak_width",
                                    default=DEFAULT_PEAK_WIDTH,
                                    min=1,
                                    max=64,
                                    step=1,
                                    tooltip=(
                                        "Minimum peak width for FFT harmonic peak detection. Lower finds "
                                        "narrower harmonics (noisier); higher demands broader ones."
                                    ),
                                ),
                            ],
                        ),
                        io.DynamicCombo.Option(
                            "By Pixel Size",
                            [
                                io.Int.Input(
                                    "pixel_size",
                                    default=DEFAULT_PIXEL_SIZE,
                                    min=1,
                                    max=512,
                                    step=1,
                                    tooltip=(
                                        "Target edge length of one pixel-art cell, in input pixels. "
                                        "Converted internally to a cell count with "
                                        "grid_w = round(W / pixel_size), grid_h = round(H / pixel_size) -- "
                                        "upstream's grid_size is a cell COUNT, not an edge length. This "
                                        "mode skips detection entirely and is the reliable path."
                                    ),
                                ),
                            ],
                        ),
                        io.DynamicCombo.Option(
                            "By Grid Count",
                            [
                                io.Int.Input(
                                    "grid_w",
                                    default=DEFAULT_GRID_W,
                                    min=1,
                                    max=4096,
                                    step=1,
                                    tooltip=(
                                        "Number of cells across the image width. Skips detection entirely. "
                                        "Use this when you know the art's resolution rather than its scale."
                                    ),
                                ),
                                io.Int.Input(
                                    "grid_h",
                                    default=DEFAULT_GRID_H,
                                    min=1,
                                    max=4096,
                                    step=1,
                                    tooltip="Number of cells down the image height. Matched with grid_w.",
                                ),
                            ],
                        ),
                    ],
                    tooltip=(
                        "How the grid is determined, and the parameters that mode needs -- only the "
                        "selected mode's controls are shown. 'Auto Detect' runs upstream FFT + gradient "
                        "detection. 'By Pixel Size' and 'By Grid Count' pass an explicit grid and skip "
                        "detection entirely, which is the reliable path. Each mode keeps its own values "
                        "while the session lasts, but only the selected mode's values are saved into a "
                        "workflow."
                    ),
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
                # Only these two stay at top level: they apply to every grid mode.
                # min_size / peak_width / max_pixel_size moved into the Auto Detect
                # option above, because _pixelate only passes them to the detector
                # -- they are inert in the two manual modes.
                io.Float.Input(
                    "refine_intensity",
                    default=0.25,
                    min=0.0,
                    max=0.5,
                    step=0.01,
                    tooltip=(
                        "Advanced. Grid-refinement search radius. Each estimated grid line at x is snapped "
                        "within [x*(1-r), x*(1+r)]. Upstream recommends 0.0-0.5. Applies to all three grid "
                        "modes, not just Auto Detect."
                    ),
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
                        "minimum across images. THE MANUAL MODES ARE SCORED TOO -- this used to claim they "
                        "always read 0, and that was wrong: the value is computed in every mode, so a forced "
                        "grid that genuinely matches the content scores high (measured 1.000 when a true "
                        "96-cell grid was requested as 96x96). Read 'detected=False' together with a HIGH "
                        "confidence as 'your forced grid agrees with the content' -- useful information. It "
                        "reads low only when a forced pitch does not match the image's own periodicity, and "
                        "that is not a failure either; the grid is applied as you asked."
                    ),
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        sampling: str = "Majority Cluster",
        grid_mode=None,
        backend: str = "Auto",
        refine_intensity: float = 0.25,
        fix_square: bool = True,
    ) -> io.NodeOutput:
        try:
            sample_method = _SAMPLE_METHOD[sampling]
        except KeyError:
            raise ValueError(
                f"PixelGridRestore: unknown sampling {sampling!r}; expected one of {SAMPLING_OPTIONS}."
            ) from None

        mode, params = _unpack_grid_mode(grid_mode)
        if mode not in GRID_MODE_OPTIONS:
            raise ValueError(
                f"PixelGridRestore: unknown grid_mode {mode!r}; expected one of {GRID_MODE_OPTIONS}."
            )

        # The Auto-only parameters. _pick falls back to the schema defaults when
        # the option's keys are absent, which is the normal case for the manual
        # modes -- they never send them.
        min_size = float(_pick(params, "min_size", DEFAULT_MIN_SIZE))
        max_pixel_size = float(_pick(params, "max_pixel_size", DEFAULT_MAX_PIXEL_SIZE))
        peak_width = int(_pick(params, "peak_width", DEFAULT_PEAK_WIDTH))

        backend_label, upstream = _resolve_backend(backend)
        detected = mode == "Auto Detect"

        # Pre-flight, before any work: the no-CV2 backend cannot auto-detect, and
        # letting it try means an upstream TypeError from inside
        # estimate_grid_fft's caller instead of a message naming the fix. Guarded
        # by `detected`, so Lightweight + a manual grid mode is untouched.
        if detected and not _auto_detect_usable(upstream):
            raise RuntimeError(
                f"PixelGridRestore: backend={backend_label!r} cannot auto-detect on this install, so "
                f"grid_mode='Auto Detect' would die inside upstream with a TypeError rather than falling "
                f"back." + _LIGHTWEIGHT_WHY +
                f" Either install OpenCV ({_INSTALL_HINT}) and leave backend='Auto' so it is picked "
                f"automatically, or switch grid_mode to 'By Pixel Size' / 'By Grid Count' -- both skip "
                f"detection entirely and work fine on this backend."
            )

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

            grid_size = _resolve_grid_size(mode, params, width, height)

            try:
                out_rgb, out_alpha = _pixelate(
                    upstream,
                    backend_label,
                    rgb,
                    alpha,
                    sample_method=sample_method,
                    grid_size=grid_size,
                    min_size=min_size,
                    max_pixel_size=max_pixel_size,
                    peak_width=peak_width,
                    refine_intensity=float(refine_intensity),
                    fix_square=bool(fix_square),
                )
            except Exception as exc:  # noqa: BLE001 -- upstream leaks raw exceptions
                raise RuntimeError(
                    f"PixelGridRestore: perfect-pixel raised {type(exc).__name__} on image {i} "
                    f"({width}x{height}), backend={backend_label!r}, grid_mode={mode!r}: {exc}."
                    + _backend_hint(backend_label, exc)
                ) from exc

            # The whole reason this node exists: upstream signals failure by
            # returning nothing rather than raising. The V1 node dropped that
            # signal and emitted the untouched input.
            if out_rgb is None:
                raise RuntimeError(
                    f"PixelGridRestore: 网格检测失败 (grid detection failed) on image {i} "
                    f"({width}x{height}), backend={backend_label!r}, grid_mode={mode!r}, "
                    f"min_size={min_size}, max_pixel_size={max_pixel_size}. "
                    "上游 perfect-pixel 的自动检测（FFT + 梯度后备）在这张图上找不到网格，"
                    "它不会抛异常、只会原样返回输入图；为避免静默输出未像素化的原图，本节点在此报错。"
                    "建议改用 grid_mode='By Pixel Size' 或 grid_mode='By Grid Count' 手动指定网格；"
                    "若方格边长明显大于 20px，请改 'Auto Detect' 的 max_pixel_size 后重试。"
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
