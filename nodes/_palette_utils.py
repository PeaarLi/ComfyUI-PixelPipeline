"""Shared palette machinery for the two palette nodes.

``PixelPaletteExtract`` chooses a palette and ``PixelPaletteApply`` applies one.
Only the extractor derives anything; both live here so the derivation and the
nearest-colour mapping it feeds cannot drift apart.

Design notes
------------

**Nothing is re-implemented.** Clustering is delegated to ``cv2.kmeans`` and
``PIL.Image.quantize``; this module only prepares their inputs, merges their
results and maps pixels onto the resulting palette.

**Determinism.** ``cv2.kmeans`` draws its initial centres from a global RNG --
its own tooltip admits the same image can come back with different labels. Every
call below is preceded by ``cv2.setRNGSeed(KMEANS_SEED)`` and uses
``KMEANS_PP_CENTERS``, so a given input always produces byte-identical output.
Sub-sampling of large images uses a fixed stride, never a random draw.

**Working space.** ``space="Lab"`` clusters on ``cv2.cvtColor(rgb,
COLOR_RGB2LAB)``, i.e. OpenCV's *8-bit* Lab encoding (L scaled to 0-255, a/b
offset by 128) -- **not** the standard 0-100 / +-128 Lab. All distances are plain
squared-Euclidean in whichever 0-255 encoding is selected, so the two spaces are
directly comparable in magnitude but not perceptually uniform.

**``space`` reaches further than the clustering step.** ``Image.quantize`` has no
Lab entry point, so median cut / max coverage / octree always *cluster* on RGB --
but that is the only thing they get away with ignoring ``space`` for. The
PIL-derived palettes are still merged across a batch in ``space``
(:func:`_merge_blocks` converts them into it first), accent separation is
measured in ``space`` (:func:`_append_accents`), and the per-entry pixel counts
that drive the ``Frequency`` sort are computed in ``space``
(:func:`palette_counts`). So a batched or accent-protected run produces
different output for the same PIL method when ``space`` changes.

**Batches.** All functions take ``[B, H, W, 3]``. When ``B > 1`` the per-image
palettes are pooled and re-clustered into a single shared palette, so one call
yields one palette that covers every frame -- which is what an indexed sprite
sheet needs.
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - depends on the host environment
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from PIL import Image

__all__ = [
    "PER_TILE",
    "GLOBAL_KMEANS",
    "MEDIAN_CUT",
    "MAX_COVERAGE",
    "OCTREE",
    "EXTRACT_METHODS",
    "SPACE_OPTIONS",
    "SORT_OPTIONS",
    "DITHER_OPTIONS",
    "DITHER_NONE",
    "DITHER_FS",
    "extract_palette",
    "format_palette_hex",
    "map_to_palette",
    "palette_swatch_image",
    "parse_palette_hex",
    "sort_palette",
]

# --------------------------------------------------------------------------- #
# Public vocabularies (node-facing labels)
# --------------------------------------------------------------------------- #
PER_TILE = "Per-Tile k-means"
GLOBAL_KMEANS = "Global k-means (Lab)"
MEDIAN_CUT = "Median Cut"
MAX_COVERAGE = "Max Coverage"
OCTREE = "Octree"

#: Every method here *produces* a palette, which is all ``PixelPaletteExtract``
#: ever does. There is deliberately no "apply a palette I already have" entry:
#: that is ``PixelPaletteApply``'s entire job, and offering it as a dropdown
#: value would make one of the two nodes' inputs silently inert.
EXTRACT_METHODS = [PER_TILE, GLOBAL_KMEANS, MEDIAN_CUT, MAX_COVERAGE, OCTREE]

SPACE_OPTIONS = ["Lab", "RGB"]
SORT_OPTIONS = ["Luminance", "Hue", "Frequency"]

DITHER_NONE = "None"
DITHER_FS = "Floyd-Steinberg"
DITHER_OPTIONS = [DITHER_NONE, "Bayer 2x2", "Bayer 4x4", "Bayer 8x8", DITHER_FS]

#: Combo label -> PIL quantize method.
_PIL_METHODS = {
    MEDIAN_CUT: Image.MEDIANCUT,
    MAX_COVERAGE: Image.MAXCOVERAGE,
    OCTREE: Image.FASTOCTREE,
}

# --------------------------------------------------------------------------- #
# Tunables
# --------------------------------------------------------------------------- #
#: Seed re-applied before *every* ``cv2.kmeans`` call. Any fixed value works;
#: what matters is that it is fixed.
KMEANS_SEED = 0x5EED_1234
_KMEANS_ATTEMPTS = 3
_KMEANS_MAX_ITER = 20
_KMEANS_EPS = 0.5

#: Global k-means is fed at most this many pixels (fixed-stride subsample).
MAX_KMEANS_SAMPLES = 65536
#: Hard cap on the pool fed to the per-tile / per-frame merge k-means.
MAX_MERGE_POOL = 65536
#: Bounded integer replication used to approximate a frequency-weighted k-means
#: (``cv2.kmeans`` has no sample-weights argument).
MERGE_REP_CAP = 16
#: Per-tile clustering target: one cluster per ~16 pixels, i.e.
#: ``tile_k ~= sqrt(pixels_in_tile)``.
TILE_K_MIN = 2
#: Per-tile loop is bounded: ``tile_size`` is doubled until the tile count fits.
MAX_TILES = 4096

#: Default tile size for ``PixelPaletteExtract``. Only a default: the node
#: exposes the window as its advanced ``tile_size`` input. Note it is measured in
#: pixels *of that node's input image*, so the right value depends on whether it
#: is fed the generator's raw output or an already grid-restored image.
EXTRACT_TILE_SIZE = 16

#: Accent protection.
ACCENT_MAX_FRACTION = 0.001  # rarer than 0.1% of the pixels
ACCENT_MIN_SAT = 0.55  # or: HSV saturation >= this ...
ACCENT_VALUE_MARGIN = 0.06  # ... or value within 6% of 0 or 1 (outlines / highlights)
ACCENT_BUDGET_DIVISOR = 8  # accents may claim at most colors/8 slots
ACCENT_MIN_DIST = 0.5  # must sit >= 0.5 * palette step away from the main palette

#: Ordered (Bayer) dithering amplitude, as a multiple of the palette's mean
#: nearest-neighbour spacing in the working space.
BAYER_AMPLITUDE = 0.9
_STEP_FALLBACK = 16.0
_STEP_MIN = 2.0
_STEP_MAX = 48.0

#: Exact-LUT index for the swatch strip produced by ``PixelPaletteExtract``.
SWATCH_BLOCK = 16
SWATCH_COLUMNS = 16

_INSTALL_HINT = 'pip install "perfect-pixel[opencv]"'


def _require_cv2():
    if cv2 is None:
        raise RuntimeError(
            "ComfyUI-PixelPipeline: OpenCV (cv2) is required but not installed in this environment."
            f" Install the dependency: {_INSTALL_HINT}"
        )
    return cv2


# --------------------------------------------------------------------------- #
# Colour space plumbing
# --------------------------------------------------------------------------- #
def _rgb_to_work(rgb: np.ndarray, space: str) -> np.ndarray:
    """uint8 RGB ``[..., 3]`` -> float32 working-space array of the same shape.

    ``cv2.cvtColor`` only accepts 2-D or 3-D images, so the array is flattened to
    ``[N, 1, 3]`` for the conversion and restored afterwards -- that is what lets
    a bare ``[K, 3]`` palette be converted the same way as an image.
    """
    rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    if space == "Lab":
        lab = _require_cv2().cvtColor(rgb.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB)
        return lab.reshape(rgb.shape).astype(np.float32)
    if space == "RGB":
        return rgb.astype(np.float32)
    raise ValueError(f"unknown space {space!r}; expected one of {SPACE_OPTIONS}.")


def _work_to_rgb(work: np.ndarray, space: str) -> np.ndarray:
    """float32 working-space ``[..., 3]`` -> uint8 RGB, clipped and rounded."""
    if space == "Lab":
        enc = np.ascontiguousarray(np.clip(np.rint(work), 0.0, 255.0).astype(np.uint8))
        rgb = _require_cv2().cvtColor(enc.reshape(-1, 1, 3), cv2.COLOR_LAB2RGB)
        return rgb.reshape(enc.shape)
    if space == "RGB":
        return np.clip(np.rint(work), 0.0, 255.0).astype(np.uint8)
    raise ValueError(f"unknown space {space!r}; expected one of {SPACE_OPTIONS}.")


# --------------------------------------------------------------------------- #
# HEX palette <-> array
# --------------------------------------------------------------------------- #
def format_palette_hex(palette: np.ndarray) -> str:
    """``[K, 3]`` uint8 -> ``"#RRGGBB\\n#RRGGBB\\n..."`` (uppercase, no trailing NL)."""
    pal = np.asarray(palette, dtype=np.uint8).reshape(-1, 3)
    return "\n".join(f"#{int(r):02X}{int(g):02X}{int(b):02X}" for r, g, b in pal)


_HEX_CHARS = set("0123456789abcdefABCDEF")


def _parse_hex_token(token: str, where: str,
                     node_label: str = "PixelPaletteApply") -> tuple[int, int, int]:
    body = token.strip().lstrip("#")
    if body[:2].lower() == "0x":
        body = body[2:]
    if len(body) == 3 and all(c in _HEX_CHARS for c in body):
        body = "".join(c * 2 for c in body)
    if len(body) != 6 or not all(c in _HEX_CHARS for c in body):
        raise ValueError(
            f"{node_label}: {where} cannot be parsed as a HEX colour: "
            f"{token!r}. Expected #RRGGBB (or RRGGBB / #RGB), one colour per line or comma separated."
        )
    return int(body[0:2], 16), int(body[2:4], 16), int(body[4:6], 16)


def parse_palette_hex(text: str, node_label: str = "PixelPaletteApply") -> np.ndarray:
    """Parse a HEX palette blob into ``[K, 3]`` uint8.

    Accepts newlines, commas, semicolons or plain whitespace as separators, with
    or without a leading ``#`` (and tolerates ``0x`` and the 3-digit short form).
    Duplicates are removed, keeping first-seen order, so the returned array is
    always a set of distinct colours. Raises ``ValueError`` on the first token
    that is not a colour -- a silently truncated palette would be worse.

    ``node_label`` only affects the message prefix, so ``PixelBackgroundRemove``'s
    single-colour errors name that node. The default keeps the palette node's
    wording unchanged.
    """
    if text is None:
        return np.zeros((0, 3), dtype=np.uint8)
    raw = str(text).replace(",", " ").replace(";", " ").split()
    toks = [t for t in raw if t.strip()]
    if not toks:
        return np.zeros((0, 3), dtype=np.uint8)
    seen: dict[tuple[int, int, int], None] = {}
    for i, tok in enumerate(toks):
        seen.setdefault(_parse_hex_token(tok, f"colour token {i + 1}", node_label))
    return np.array(list(seen.keys()), dtype=np.uint8)


# --------------------------------------------------------------------------- #
# Small array helpers
# --------------------------------------------------------------------------- #
def _unique_rows(rows: np.ndarray) -> np.ndarray:
    """Row-wise de-duplication, output order unspecified but stable."""
    rows = np.ascontiguousarray(rows)
    if rows.shape[0] <= 1:
        return rows
    return np.unique(rows, axis=0)


def _dedupe_keep_order(rows: np.ndarray) -> np.ndarray:
    """Row-wise de-duplication that keeps the first occurrence's position."""
    rows = np.ascontiguousarray(rows)
    if rows.shape[0] <= 1:
        return rows
    _, first = np.unique(rows, axis=0, return_index=True)
    return rows[np.sort(first)]


def _subsample_rows(rows: np.ndarray, limit: int) -> np.ndarray:
    """Fixed-stride subsample -- deterministic, unlike a random draw."""
    n = rows.shape[0]
    if n <= limit:
        return rows
    step = int(np.ceil(n / float(limit)))
    return np.ascontiguousarray(rows[::step])


# --------------------------------------------------------------------------- #
# Distance / nearest neighbour
# --------------------------------------------------------------------------- #
def nearest_labels(rows: np.ndarray, centers: np.ndarray, chunk: int = 32768) -> np.ndarray:
    """Index of the nearest centre for each row, squared-Euclidean distance.

    The per-row ``||a||^2`` term is constant, so it is dropped and only
    ``||b||^2 - 2ab`` is evaluated -- a single BLAS matmul per chunk instead of a
    ``[N, K, 3]`` broadcast that would need gigabytes on a 1 MP image. Ties go to
    the lowest index (``argmin`` semantics), which keeps the mapping stable.
    """
    rows = np.ascontiguousarray(rows, dtype=np.float32)
    centers = np.ascontiguousarray(centers, dtype=np.float32)
    k = centers.shape[0]
    if k == 1:
        return np.zeros(rows.shape[0], dtype=np.int32)
    c_sq = np.einsum("ij,ij->i", centers, centers)[None, :]
    out = np.empty(rows.shape[0], dtype=np.int32)
    for start in range(0, rows.shape[0], chunk):
        stop = min(start + chunk, rows.shape[0])
        block = rows[start:stop]
        out[start:stop] = np.argmin(c_sq - 2.0 * (block @ centers.T), axis=1)
    return out


def palette_step(work_palette: np.ndarray) -> float:
    """Mean distance from each palette entry to its nearest neighbour.

    Used to calibrate the ordered-dither amplitude: it is roughly "one palette
    step", the smallest nudge that can push a pixel into a different entry, and
    it is expressed in the working space so Lab and RGB dither at comparable
    visual strength without hand-tuned constants per space.
    """
    k = work_palette.shape[0]
    if k < 2:
        return _STEP_FALLBACK
    diff = work_palette[:, None, :] - work_palette[None, :, :]
    dist = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
    np.fill_diagonal(dist, np.inf)
    step = float(np.mean(np.min(dist, axis=1)))
    if not np.isfinite(step) or step <= 0.0:
        return _STEP_FALLBACK
    return float(np.clip(step, _STEP_MIN, _STEP_MAX))


# --------------------------------------------------------------------------- #
# k-means (always through cv2, always seeded)
# --------------------------------------------------------------------------- #
def _kmeans(rows: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Seeded ``cv2.kmeans`` over ``[N, 3]`` float32 -> ``(centers, labels)``."""
    cv = _require_cv2()
    rows = np.ascontiguousarray(rows, dtype=np.float32)
    k = int(max(1, min(int(k), rows.shape[0])))
    if k == 1:
        center = rows.mean(axis=0, keepdims=True).astype(np.float32)
        return center, np.zeros(rows.shape[0], dtype=np.int32)
    cv.setRNGSeed(int(seed))
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, _KMEANS_MAX_ITER, _KMEANS_EPS)
    _compactness, labels, centers = cv.kmeans(
        rows, k, None, criteria, _KMEANS_ATTEMPTS, cv.KMEANS_PP_CENTERS
    )
    return np.ascontiguousarray(centers, dtype=np.float32), labels.ravel().astype(np.int32)


def _merge_blocks(
    blocks: list[tuple[np.ndarray, np.ndarray]],
    k: int,
    space: str,
    seed: int,
) -> np.ndarray:
    """Pool several ``(centres_in_work_space, weights)`` blocks into one palette.

    The pool is a *weighted* k-means in spirit but ``cv2.kmeans`` takes no sample
    weights, so weights are emulated by bounded integer replication
    (``MERGE_REP_CAP`` copies at most). Each block is normalised against its own
    largest weight, so a block holding a whole tile or a whole frame contributes
    a comparable number of rows regardless of how many pixels it represents --
    that is the property that keeps local structure from being averaged away by
    whichever region happens to cover the most area.
    """
    parts = []
    for rows, weights in blocks:
        rows = np.ascontiguousarray(rows, dtype=np.float32)
        if rows.shape[0] == 0:
            continue
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        w_max = float(w.max()) if w.size else 0.0
        if w_max > 0.0:
            rep = np.clip(np.rint(w / w_max * MERGE_REP_CAP), 1, MERGE_REP_CAP).astype(np.int64)
        else:
            rep = np.ones(rows.shape[0], dtype=np.int64)
        parts.append(np.repeat(rows, rep, axis=0))
    if not parts:
        raise ValueError("PixelPalette: nothing to cluster (empty image?).")
    pool = _subsample_rows(np.concatenate(parts, axis=0), MAX_MERGE_POOL)
    centers, _ = _kmeans(pool, k, seed)
    return _work_to_rgb(centers, space)


def _tile_blocks(
    work: np.ndarray, colors: int, tile_size: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """One ``(centres, counts)`` block per tile of a single image."""
    height, width = work.shape[:2]
    tile = max(2, int(tile_size))
    # Bound the loop: double the tile until the tile count fits. Deterministic,
    # and it only ever kicks in on very large inputs.
    while ((height + tile - 1) // tile) * ((width + tile - 1) // tile) > MAX_TILES:
        tile *= 2

    blocks: list[tuple[np.ndarray, np.ndarray]] = []
    for y0 in range(0, height, tile):
        for x0 in range(0, width, tile):
            patch = work[y0 : y0 + tile, x0 : x0 + tile].reshape(-1, 3)
            uniq, _inv, counts = np.unique(
                patch, axis=0, return_inverse=True, return_counts=True
            )
            n_unique = uniq.shape[0]
            # ~one cluster per 16 pixels: sqrt() scales the local palette with
            # the tile area instead of spending the whole global budget per tile.
            target = max(TILE_K_MIN, int(round(np.sqrt(patch.shape[0]))))
            k = int(min(colors, target, n_unique))
            if k >= n_unique:
                blocks.append((uniq.astype(np.float32), counts))
            else:
                centers, labels = _kmeans(uniq.astype(np.float32), k, seed)
                blocks.append((centers, np.bincount(labels, minlength=k)))
    return blocks


# --------------------------------------------------------------------------- #
# PIL-backed extractors
# --------------------------------------------------------------------------- #
def _pil_palette(rgb: np.ndarray, k: int, pil_method: int) -> tuple[np.ndarray, np.ndarray]:
    """``Image.quantize`` -> ``(palette_u8 [K,3], counts [K])``.

    ``dither=NONE`` on purpose: dithering is a separate, explicit stage in
    ``PixelPaletteApply`` and must never be baked into extraction.
    """
    k = int(max(1, min(256, k)))
    image = Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8), "RGB")
    quantized = image.quantize(colors=k, method=pil_method, dither=Image.Dither.NONE)
    labels = np.asarray(quantized, dtype=np.int32)
    n_used = int(labels.max()) + 1 if labels.size else 0
    raw = quantized.getpalette() or []
    palette = np.array(raw[: n_used * 3], dtype=np.uint8).reshape(-1, 3)
    counts = np.bincount(labels.ravel(), minlength=palette.shape[0])[: palette.shape[0]]
    return palette, counts.astype(np.int64)


# --------------------------------------------------------------------------- #
# Accent protection
# --------------------------------------------------------------------------- #
def _accent_candidates(imgs: np.ndarray, budget: int) -> np.ndarray:
    """Rare colours that still deserve a palette slot, most salient first.

    The criterion is deliberately two-pronged, because both kinds of small detail
    are what clustering eats first:

    * **chromatic pops** -- saturated (HSV S >= ``ACCENT_MIN_SAT``) but covering
      less than ``ACCENT_MAX_FRACTION`` of the pixels (a red gem, a magic FX
      pixel);
    * **tonal extremes** -- within ``ACCENT_VALUE_MARGIN`` of black or white
      (eye highlights, black outline pixels that only survive on one limb).

    Colours are counted on a packed ``0xRRGGBB`` integer key, which makes the
    histogram a 1-D ``np.unique`` instead of a row-wise one -- an order of
    magnitude faster on megapixel input.
    """
    flat = np.ascontiguousarray(imgs, dtype=np.uint8).reshape(-1, 3)
    if flat.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    keys = (
        (flat[:, 0].astype(np.uint32) << 16)
        | (flat[:, 1].astype(np.uint32) << 8)
        | flat[:, 2].astype(np.uint32)
    )
    uniq, counts = np.unique(keys, return_counts=True)
    fraction = counts.astype(np.float64) / float(counts.sum())
    rare = fraction < ACCENT_MAX_FRACTION
    if not rare.any():
        return np.zeros((0, 3), dtype=np.uint8)

    uniq = uniq[rare]
    counts = counts[rare]
    fraction = fraction[rare]
    colors = np.stack(
        [(uniq >> 16) & 0xFF, (uniq >> 8) & 0xFF, uniq & 0xFF], axis=1
    ).astype(np.uint8)

    hsv = _require_cv2().cvtColor(colors.reshape(1, -1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    sat = hsv[:, 1].astype(np.float64) / 255.0
    val = hsv[:, 2].astype(np.float64) / 255.0
    chromatic = sat >= ACCENT_MIN_SAT
    extreme = (val >= 1.0 - ACCENT_VALUE_MARGIN) | (val <= ACCENT_VALUE_MARGIN)
    keep = chromatic | extreme
    if not keep.any():
        return np.zeros((0, 3), dtype=np.uint8)

    colors = colors[keep]
    sat = sat[keep]
    val = val[keep]
    fraction = fraction[keep]
    # Salience: saturated first, then bright, then rarer.
    order = np.lexsort((-fraction, -val, -sat))
    return np.ascontiguousarray(colors[order[: max(0, int(budget))]])


def _append_accents(
    palette: np.ndarray, accents: np.ndarray, space: str, keep_budget: int
) -> np.ndarray:
    """Append accents that are still far enough from the main palette."""
    if accents.shape[0] == 0 or keep_budget <= 0:
        return palette
    work_pal = _rgb_to_work(palette, space)
    work_acc = _rgb_to_work(accents, space)
    step = palette_step(work_pal)
    threshold = ACCENT_MIN_DIST * step
    diff = work_acc[:, None, :] - work_pal[None, :, :]
    dist = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff)).min(axis=1)
    chosen = accents[dist > threshold][:keep_budget]
    if chosen.shape[0] == 0:
        return palette
    return np.concatenate([palette, chosen], axis=0)


# --------------------------------------------------------------------------- #
# Public extraction entry point
# --------------------------------------------------------------------------- #
def extract_palette(
    imgs: np.ndarray,
    colors: int,
    method: str,
    space: str = "Lab",
    tile_size: int = 16,
    accent_protect: bool = False,
    seed: int = KMEANS_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive a palette from ``imgs`` (uint8 ``[B, H, W, 3]``).

    Returns ``(palette_u8 [K, 3], counts [K])`` where ``counts`` is the number of
    input pixels that map to each entry (used by the ``Frequency`` sort).

    ``K`` never exceeds ``colors``: accents, when enabled, are paid for out of
    the same budget by shrinking the main cluster count.
    """
    imgs = np.ascontiguousarray(imgs, dtype=np.uint8)
    # Deliberately stricter than the node inputs, which accept RGBA: this is an
    # internal invariant, not a user-facing restriction. Both callers strip alpha
    # (``_split_rgb_alpha``) before getting here, because a k-means quantiser
    # handed RGBA would treat "transparent" as a fourth colour and spend palette
    # entries on it. If this fires, a caller forgot to split.
    if imgs.ndim != 4 or imgs.shape[-1] != 3:
        raise ValueError(
            f"extract_palette expects uint8 RGB [B,H,W,3] (alpha must be split off "
            f"first with _split_rgb_alpha), got {tuple(imgs.shape)}."
        )
    if method not in EXTRACT_METHODS:
        raise ValueError(
            f"PixelPalette: unknown extraction method {method!r}; expected one of {EXTRACT_METHODS}."
        )
    colors = int(colors)
    batch = imgs.shape[0]

    accents = np.zeros((0, 3), dtype=np.uint8)
    accent_budget = 0
    main_k = colors
    if accent_protect:
        accent_budget = max(1, colors // ACCENT_BUDGET_DIVISOR)
        accents = _accent_candidates(imgs, accent_budget)
        if accents.shape[0]:
            main_k = max(2, colors - accents.shape[0])

    if method == PER_TILE:
        blocks: list[tuple[np.ndarray, np.ndarray]] = []
        for i in range(batch):
            blocks.extend(
                _tile_blocks(_rgb_to_work(imgs[i], space), main_k, int(tile_size), seed)
            )
        palette = _merge_blocks(blocks, main_k, space, seed)

    elif method == GLOBAL_KMEANS:
        blocks = []
        for i in range(batch):
            rows = _subsample_rows(
                _rgb_to_work(imgs[i], space).reshape(-1, 3), MAX_KMEANS_SAMPLES
            )
            centers, labels = _kmeans(rows, main_k, seed)
            blocks.append((centers, np.bincount(labels, minlength=centers.shape[0])))
        palette = (
            _work_to_rgb(blocks[0][0], space)
            if len(blocks) == 1
            else _merge_blocks(blocks, main_k, space, seed)
        )

    else:  # the three PIL methods -- RGB only, see the module docstring
        if batch == 1:
            # Already a uint8 RGB palette; routing it through _work_to_rgb would
            # misinterpret it as Lab.
            palette, _counts = _pil_palette(imgs[0], main_k, _PIL_METHODS[method])
        else:
            blocks = [
                (_rgb_to_work(pal_i, space), counts_i)
                for pal_i, counts_i in (
                    _pil_palette(imgs[i], main_k, _PIL_METHODS[method]) for i in range(batch)
                )
            ]
            palette = _merge_blocks(blocks, main_k, space, seed)

    palette = _unique_rows(palette)
    if accents.shape[0]:
        palette = _append_accents(palette, accents, space, accent_budget)
        palette = _unique_rows(palette)

    counts = palette_counts(imgs, palette, space)
    return palette, counts


def palette_counts(imgs: np.ndarray, palette: np.ndarray, space: str) -> np.ndarray:
    """Per-entry pixel counts over the whole batch, in palette order."""
    palette = np.asarray(palette, dtype=np.uint8).reshape(-1, 3)
    if palette.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    work_pal = _rgb_to_work(palette, space)
    total = np.zeros(palette.shape[0], dtype=np.int64)
    for i in range(imgs.shape[0]):
        rows = _rgb_to_work(imgs[i], space).reshape(-1, 3)
        labels = nearest_labels(rows, work_pal)
        total += np.bincount(labels, minlength=palette.shape[0])
    return total


# --------------------------------------------------------------------------- #
# Sorting
# --------------------------------------------------------------------------- #
_HUE_ACHROMATIC_S = 0.06


def sort_palette(
    palette: np.ndarray, counts: np.ndarray | None, sort_mode: str
) -> tuple[np.ndarray, np.ndarray | None]:
    """Order a palette for display, keeping ``counts`` aligned.

    Every key ends with the raw RGB triple, so two entries that tie on the
    primary criterion still come out in a fixed order -- the HEX string a node
    emits must not depend on the input's row order.
    """
    pal = np.ascontiguousarray(palette, dtype=np.uint8).reshape(-1, 3)
    k = pal.shape[0]
    if k <= 1:
        return pal, (None if counts is None else np.asarray(counts, dtype=np.int64))
    r = pal[:, 0].astype(np.int64)
    g = pal[:, 1].astype(np.int64)
    b = pal[:, 2].astype(np.int64)
    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b

    if sort_mode == "Frequency":
        if counts is None:
            keys = [(int(luma[i]), int(r[i]), int(g[i]), int(b[i])) for i in range(k)]
        else:
            c = np.asarray(counts, dtype=np.int64)
            keys = [
                (-int(c[i]), int(luma[i]), int(r[i]), int(g[i]), int(b[i])) for i in range(k)
            ]
    elif sort_mode == "Hue":
        hsv = _require_cv2().cvtColor(pal.reshape(1, k, 3), cv2.COLOR_RGB2HSV).reshape(k, 3)
        hue = hsv[:, 0].astype(np.int64) * 2  # OpenCV packs hue into 0..179
        sat = hsv[:, 1].astype(np.float64) / 255.0
        val = hsv[:, 2].astype(np.float64) / 255.0
        keys = [
            (
                0 if sat[i] < _HUE_ACHROMATIC_S else 1,  # greys first, then the hue wheel
                int(hue[i]),
                float(val[i]),
                int(r[i]),
                int(g[i]),
                int(b[i]),
            )
            for i in range(k)
        ]
    elif sort_mode == "Luminance":
        keys = [(float(luma[i]), int(r[i]), int(g[i]), int(b[i])) for i in range(k)]
    else:
        raise ValueError(f"unknown sort {sort_mode!r}; expected one of {SORT_OPTIONS}.")

    order = np.array(sorted(range(k), key=keys.__getitem__), dtype=np.int64)
    out_counts = None if counts is None else np.asarray(counts, dtype=np.int64)[order]
    return pal[order], out_counts


# --------------------------------------------------------------------------- #
# Dithering
# --------------------------------------------------------------------------- #
def _bayer_matrix(size: int) -> np.ndarray:
    """Standard recursive Bayer matrix, values in ``[0, 1)``.

    Built from the constant 2x2 seed rather than a literal table so 4x4 and 8x8
    cannot drift apart. The matrix tiles by pixel *coordinates*, never by image
    size: a 32x32 sprite and a 256x256 render of the same scene get the same
    threshold at the same pixel, which is what makes ordered dithering safe to
    use across a sprite set.
    """
    matrix = np.array([[0, 2], [3, 1]], dtype=np.float64)
    current = 2
    while current < size:
        matrix = np.block(
            [
                [4 * matrix, 4 * matrix + 2],
                [4 * matrix + 3, 4 * matrix + 1],
            ]
        )
        current *= 2
    if current != size:
        raise ValueError(f"unsupported Bayer size {size}; expected 2, 4 or 8.")
    return matrix / float(size * size)


_BAYER_CACHE = {2: _bayer_matrix(2), 4: _bayer_matrix(4), 8: _bayer_matrix(8)}


def bayer_offsets(height: int, width: int, size: int, amplitude: float) -> np.ndarray:
    """``[H, W, 1]`` working-space offsets, centred on zero."""
    matrix = _BAYER_CACHE[size]
    rows = matrix[np.arange(height) % size]
    tiled = rows[:, np.arange(width) % size]
    return ((tiled - 0.5) * amplitude)[:, :, None].astype(np.float32)


def _floyd_steinberg(rgb: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Error-diffusion dithering with a fixed palette, via Pillow.

    Pillow's error diffusion is defined on RGB, so this path ignores ``space``
    entirely (the palette is still whatever the extraction produced in that
    space). It is also scan-order dependent: the same sprite rendered at two
    sizes will *not* dither identically. That is stated in the node tooltip.

    The palette handed to Pillow is a **P-mode image, which always carries 256
    entries**, so a palette shorter than 256 must be padded somehow. Padding with
    black -- the obvious choice -- is a live bug: the padding entries then
    compete in the nearest-colour search and a dark pixel legitimately returns
    an index past the end of the real palette (observed as ``index 32 is out of
    bounds for axis 0 with size 32``). The padding is therefore a *periodic
    repetition* of the palette, which makes it harmless: entry ``i`` and entry
    ``i % K`` are the same colour, so the error diffused is identical either
    way, and folding the result modulo K recovers the intended index.
    """
    k = palette.shape[0]
    if k == 0:
        raise ValueError("PixelPaletteApply: empty palette.")
    palette = palette.reshape(-1, 3).astype(np.uint8)
    image = Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8), "RGB")
    holder = Image.new("P", (1, 1))
    tiled = np.tile(palette, (int(np.ceil(256 / k)), 1))[:256]
    holder.putpalette(tiled.reshape(-1).tolist())
    quantized = image.quantize(palette=holder, dither=Image.Dither.FLOYDSTEINBERG)
    labels = np.asarray(quantized, dtype=np.int32) % k
    return labels


def map_to_palette(
    imgs: np.ndarray,
    palette: np.ndarray,
    space: str = "Lab",
    dither: str = DITHER_NONE,
) -> np.ndarray:
    """Map every pixel of ``imgs`` onto ``palette``; returns ``[B, H, W]`` indices.

    Whatever the dither mode, the returned values are indices and the caller
    renders ``palette[labels]``, so the output can only ever contain palette
    colours. Ordered dithering perturbs the *lookup coordinate* -- by at most
    half the palette's own nearest-neighbour step, in the working space -- it
    never invents a colour.
    """
    imgs = np.ascontiguousarray(imgs, dtype=np.uint8)
    palette = np.ascontiguousarray(palette, dtype=np.uint8).reshape(-1, 3)
    if dither not in DITHER_OPTIONS:
        raise ValueError(f"unknown dither {dither!r}; expected one of {DITHER_OPTIONS}.")

    batch, height, width = imgs.shape[0], imgs.shape[1], imgs.shape[2]
    labels = np.empty((batch, height, width), dtype=np.int32)

    if dither == DITHER_FS:
        for i in range(batch):
            labels[i] = _floyd_steinberg(imgs[i], palette)
        return labels

    work_pal = _rgb_to_work(palette, space)
    offsets = None
    if dither != DITHER_NONE:
        # "Bayer 2x2" / "Bayer 4x4" / "Bayer 8x8" -> 2 / 4 / 8
        size = int(dither.split()[1].split("x")[0])
        amplitude = palette_step(work_pal) * BAYER_AMPLITUDE
        offsets = bayer_offsets(height, width, size, amplitude)

    for i in range(batch):
        rows = _rgb_to_work(imgs[i], space).reshape(-1, 3)
        if offsets is not None:
            rows = rows + offsets.reshape(-1, 1)
        labels[i] = nearest_labels(rows, work_pal).reshape(height, width)
    return labels


# --------------------------------------------------------------------------- #
# Swatch strip
# --------------------------------------------------------------------------- #
def palette_swatch_image(palette: np.ndarray) -> np.ndarray:
    """``[K, 3]`` uint8 -> ``[H, W, 3]`` uint8 swatch strip for visual checking.

    Plain blocks, no borders and no padding: every pixel of the result is a
    palette colour, so the strip can be compared against the quantized image
    without any doubt about what came from where.
    """
    pal = np.asarray(palette, dtype=np.uint8).reshape(-1, 3)
    if pal.shape[0] == 0:
        return np.zeros((SWATCH_BLOCK, SWATCH_BLOCK, 3), dtype=np.uint8)
    columns = min(int(pal.shape[0]), SWATCH_COLUMNS)
    rows = int(np.ceil(pal.shape[0] / columns))
    grid = np.zeros((rows * columns, 3), dtype=np.uint8)
    grid[: pal.shape[0]] = pal
    strip = grid.reshape(rows, columns, 3)
    strip = np.repeat(np.repeat(strip, SWATCH_BLOCK, axis=0), SWATCH_BLOCK, axis=1)
    return np.ascontiguousarray(strip)
