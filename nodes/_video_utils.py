"""Current-frame pixel sampling and conservative, non-recursive video stabilisation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


SAMPLING_OPTIONS = ["Robust Cell", "Majority Cluster", "Center Sample", "Median"]
MAX_PROBES = 32


def sample_indices(total: int, stride: int) -> np.ndarray:
    count = min(total, MAX_PROBES, max(2, (total + max(1, stride) - 1) // max(1, stride)))
    return np.unique(np.linspace(0, total - 1, count).round().astype(np.int64))


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """CIE Lab, D65, with L in [0,100] (not OpenCV's uint8 encoding)."""
    x = rgb.astype(np.float32) / 255.0
    x = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    xyz = x @ np.array([[0.4124564, 0.2126729, 0.0193339],
                        [0.3575761, 0.7151522, 0.1191920],
                        [0.1804375, 0.0721750, 0.9503041]], dtype=np.float32)
    xyz /= np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    f = np.where(xyz > (6 / 29) ** 3, np.cbrt(xyz), xyz * (29 / 6) ** 2 / 3 + 4 / 29)
    return np.stack((116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])), axis=-1)


def grid_evidence(rgb: np.ndarray, minimum: float, maximum: float,
                  repeats: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-axis spectral evidence of repeated colour boundaries, independent of phase."""
    values = rgb.astype(np.float32)
    scores = []
    for axis, size in ((1, rgb.shape[1]), (0, rgb.shape[0])):
        edges = np.abs(np.diff(values, axis=axis)).mean(axis=(0, 2) if axis == 1 else (1, 2))
        if not edges.size:
            scores.append(np.zeros(size // 2 + 1, dtype=np.float32))
            continue
        edges = np.maximum(edges - np.median(edges), 0)
        projection = np.zeros(size, dtype=np.float32)
        projection[1:] = edges
        energy = float(projection.sum())
        spectrum = np.abs(np.fft.rfft(projection)) / max(energy, 1e-6)
        count = np.arange(len(spectrum))
        valid = ((count >= max(2, repeats)) & (count >= size / max(maximum, minimum))
                 & (count <= size / max(1.0, minimum)))
        # A smooth ramp and effectively flat frames provide no periodic evidence.
        if energy < size * 0.5:
            valid[:] = False
        scores.append(np.where(valid, spectrum, 0).astype(np.float32))
    return scores[0], scores[1]


def choose_grid(evidence: list[tuple[np.ndarray, np.ndarray]], width: int, height: int,
                fallback: int) -> tuple[int, int, bool, float]:
    """Vote for paired frame candidates; reject weak/ambiguous periodicity."""
    candidates = []
    for sx, sy in evidence:
        pair, strengths = [], []
        for scores in (sx, sy):
            best = float(scores.max())
            # Harmonics of sharp boundaries share the same score. Choose the
            # lowest supported frequency, i.e. the coarsest fundamental grid.
            choices = np.flatnonzero((scores >= max(0.60, best * 0.90)))
            if not choices.size:
                break
            pair.append(int(choices[0]))
            strengths.append(float(scores[choices[0]]))
        if len(pair) == 2:
            candidates.append((pair[0], pair[1], min(strengths)))
    if candidates:
        clusters = []
        for gx, gy, _ in candidates:
            group = [c for c in candidates if abs(c[0] - gx) <= max(1, gx * 0.03)
                     and abs(c[1] - gy) <= max(1, gy * 0.03)]
            clusters.append((sum(c[2] for c in group), group))
        _, group = max(clusters, key=lambda c: c[0])
        # Require a majority of informative probes, not unanimity (cuts/fades).
        informative = sum(bool(sx.max() or sy.max()) for sx, sy in evidence)
        support = len(group) / max(1, informative)
        confidence = float(np.mean([c[2] for c in group]) * support)
        if support >= 0.6 and confidence >= 0.60:
            # Use a real paired candidate closest to the group, not two medians.
            gx, gy, _ = min(group, key=lambda c: sum(abs(c[0] - d[0]) + abs(c[1] - d[1]) for d in group))
            return gx, gy, True, confidence
    return max(1, round(width / fallback)), max(1, round(height / fallback)), False, 0.0


@dataclass
class CellFrame:
    rgb: np.ndarray
    alpha: np.ndarray | None
    candidates: np.ndarray
    lab: np.ndarray
    costs: np.ndarray
    best: np.ndarray


def sample_cells(rgb: np.ndarray, alpha: np.ndarray | None, grid_w: int, grid_h: int,
                 sampling: str, backend=None) -> CellFrame:
    """Nine bounded samples per uniform cell. Index zero is always the centre."""
    height, width = rgb.shape[:2]
    offsets = [(0.5, 0.5), (0.25, 0.5), (0.75, 0.5), (0.5, 0.25), (0.5, 0.75),
               (0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)]
    samples, masks, coordinates = [], [], []
    for ox, oy in offsets:
        xs = np.minimum(((np.arange(grid_w) + ox) * width / grid_w).astype(int), width - 1)
        ys = np.minimum(((np.arange(grid_h) + oy) * height / grid_h).astype(int), height - 1)
        samples.append(rgb[ys[:, None], xs])
        masks.append(np.ones((grid_h, grid_w), np.float32) if alpha is None
                     else alpha[ys[:, None], xs, 0].astype(np.float32) / 255)
        coordinates.append(ys[:, None] * width + xs)
    candidates = np.stack(samples, axis=-2)
    visible = np.stack(masks, axis=-1)
    weights = np.broadcast_to(np.array([4, 2, 2, 2, 2, 1, 1, 1, 1], np.float32), visible.shape).copy()
    # Small cells may visit a source pixel more than once: count it only once.
    for i in range(1, len(offsets)):
        duplicate = np.any(np.stack([coordinates[i] == coordinates[j] for j in range(i)]), axis=0)
        weights[..., i][duplicate] = 0
    weights *= visible
    weights /= np.maximum(weights.sum(axis=-1, keepdims=True), 1e-6)
    lab = rgb_to_lab(candidates)
    costs = np.zeros(visible.shape, dtype=np.float32)
    for i in range(len(offsets)):
        distance = np.linalg.norm(lab - lab[..., i:i + 1, :], axis=-1)
        costs[..., i] = (np.minimum(distance, 20) * weights).sum(axis=-1)
    costs[(visible == 0) & np.any(visible > 0, axis=-1, keepdims=True)] = np.inf
    best = costs.argmin(axis=-1)
    if sampling == "Center Sample":
        best = np.where(visible[..., 0] > 0, 0, best)
    elif sampling in ("Majority Cluster", "Median"):
        # Keep existing choices usable on the NEW regular grid. Upsampling and
        # alpha use the bounded visible candidates to avoid empty/hidden cells.
        if alpha is None and grid_w <= width and grid_h <= height:
            x = np.linspace(0, width, grid_w + 1).round().astype(int).tolist()
            y = np.linspace(0, height, grid_h + 1).round().astype(int).tolist()
            if sampling == "Majority Cluster":
                result = backend.sample_majority(rgb, x, y)
            else:
                result = backend.sample_median(rgb, x, y)
            # The baseline colour is a current-frame candidate, never history.
            candidates = np.concatenate((result[..., None, :], candidates), axis=-2)
            result_lab = rgb_to_lab(result)
            result_cost = (np.minimum(np.linalg.norm(lab - result_lab[..., None, :], axis=-1), 20) * weights).sum(-1)
            lab = np.concatenate((result_lab[..., None, :], lab), axis=-2)
            costs = np.concatenate((result_cost[..., None], costs), axis=-1)
            best = np.zeros((grid_h, grid_w), dtype=int)
    elif sampling != "Robust Cell":
        raise ValueError(f"Unknown video sampling: {sampling}")
    out = np.take_along_axis(candidates, best[..., None, None], axis=-2)[..., 0, :]
    # Alpha coverage belongs to this frame; it is never temporally filtered.
    out_alpha = None if alpha is None else np.rint(np.stack(masks, axis=-1).mean(-1) * 255).astype(np.uint8)[..., None]
    return CellFrame(out, out_alpha, candidates, lab, costs, best)


def dilate(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="edge")
    h, w = mask.shape
    return np.logical_or.reduce([padded[y:y + h, x:x + w] for y in range(3) for x in range(3)])


def stable_pair(left: CellFrame, right: CellFrame) -> np.ndarray:
    """Only flat, photometrically consistent areas may borrow temporal evidence."""
    a, b = rgb_to_lab(left.rgb), rgb_to_lab(right.rgb)
    delta = np.linalg.norm(a - b, axis=-1)
    patch_delta = np.linalg.norm(left.lab[..., -9:, :] - right.lab[..., -9:, :], axis=-1).max(-1)
    changed = (delta > 6.0) | (patch_delta > 6.0)
    # Scene cuts (including equal-histogram rearrangements) never share evidence.
    if changed.mean() > 0.25 or float(delta.mean()) > 8:
        return np.zeros(delta.shape, dtype=bool)
    for lab in (a, b):
        dy = np.linalg.norm(np.diff(lab, axis=0), axis=-1) > 8
        dx = np.linalg.norm(np.diff(lab, axis=1), axis=-1) > 8
        changed[1:] |= dy
        changed[:-1] |= dy
        changed[:, 1:] |= dx
        changed[:, :-1] |= dx
    for frame in (left, right):
        patch = frame.lab[..., -9:, :]
        changed |= np.linalg.norm(patch - patch[..., :1, :], axis=-1).max(-1) > 8
    if left.alpha is not None or right.alpha is not None:
        for frame in (left, right):
            if frame.alpha is not None:
                changed |= frame.alpha[..., 0] < 255
    return ~dilate(changed)


def stabilise(current: CellFrame, neighbours: list[tuple[CellFrame, np.ndarray]]) -> np.ndarray:
    """Select only a current-cell colour; no RGB mixing or recursive feedback."""
    if not neighbours:
        return current.rgb
    base = np.take_along_axis(current.costs, current.best[..., None], axis=-1)[..., 0]
    reference = rgb_to_lab(current.rgb)
    penalty = np.zeros_like(current.costs)
    valid_count = np.zeros(base.shape, dtype=np.float32)
    for frame, mask in neighbours:
        distance = np.linalg.norm(current.lab - rgb_to_lab(frame.rgb)[..., None, :], axis=-1)
        penalty += np.minimum(distance, 6) * mask[..., None]
        valid_count += mask
    score = current.costs + 0.75 * penalty / np.maximum(valid_count[..., None], 1)
    eligible = ((current.costs <= base[..., None] + 1.5)
                & (np.linalg.norm(current.lab - reference[..., None, :], axis=-1) <= 3.0))
    score = np.where(eligible, score, np.inf)
    winner = score.argmin(-1)
    winning_score = np.take_along_axis(score, winner[..., None], -1)[..., 0]
    baseline_score = np.take_along_axis(score, current.best[..., None], -1)[..., 0]
    replace = (valid_count > 0) & (winning_score < baseline_score - 1e-5)
    chosen = np.take_along_axis(current.candidates, winner[..., None, None], -2)[..., 0, :]
    return np.where(replace[..., None], chosen, current.rgb)
