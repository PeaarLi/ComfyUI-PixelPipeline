"""Restore video on a regular grid without importing colours from other times."""

from __future__ import annotations

from collections import deque
from fractions import Fraction

import numpy as np
import torch

from comfy_api.latest import Input, InputImpl, Types, io

from ._image_utils import _torch_image_to_uint8_bhwc
from ._video_utils import (
    SAMPLING_OPTIONS, choose_grid, grid_evidence, sample_cells, sample_indices,
    stabilise, stable_pair,
)
from .grid_restore import (
    BACKEND_OPTIONS, DEFAULT_GRID_H, DEFAULT_GRID_W, DEFAULT_MAX_PIXEL_SIZE,
    DEFAULT_MIN_SIZE, DEFAULT_PEAK_WIDTH, DEFAULT_PIXEL_SIZE, GRID_MODE_OPTIONS,
    _pick, _resolve_backend, _resolve_grid_size, _seed_sampler, _unpack_grid_mode,
)

__all__ = ["PixelVideoGridRestore"]
NODE_ID = "PixelVideoGridRestore"
DEFAULT_SAMPLE_STRIDE = 8
MAX_SAMPLE_STRIDE = 64
DEFAULT_TEMPORAL_WINDOW = 0
MAX_TEMPORAL_WINDOW = 8
DEFAULT_CHUNK_FRAMES = 12
DEFAULT_FALLBACK_PIXEL_SIZE = 8


class PixelVideoGridRestore(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id=NODE_ID,
            display_name="Pixel Video Grid Restore",
            category="image/pixel pipeline",
            description=(
                "Rebuild each frame on one regular pixel grid. Temporal stabilisation only selects "
                "current-frame colours in flat, consistent regions; moving edges keep their current "
                "frame. Auto detection votes on per-frame boundary spectra and falls back explicitly."
            ),
            inputs=[
                io.Video.Input("video", tooltip="SDR video. Frame order, count, audio and rational frame rate are preserved."),
                io.Combo.Input("sampling", options=SAMPLING_OPTIONS, default="Robust Cell",
                               tooltip="Robust Cell uses a centre-weighted colour medoid. Center Sample preserves tiny details. Legacy Majority/Median remain available."),
                io.DynamicCombo.Input("grid_mode", options=[
                    io.DynamicCombo.Option("Auto Detect", [
                        io.Int.Input("sample_stride", default=DEFAULT_SAMPLE_STRIDE, min=1, max=64,
                                     tooltip="Approximate probe spacing; at most 32 uniformly spaced frames, including both ends."),
                        io.Float.Input("min_size", default=DEFAULT_MIN_SIZE, min=1, max=64, step=0.5,
                                       tooltip="Minimum candidate cell pitch in source pixels."),
                        io.Float.Input("max_pixel_size", default=DEFAULT_MAX_PIXEL_SIZE, min=1, max=512, step=0.5,
                                       tooltip="Maximum candidate cell pitch in source pixels."),
                        io.Int.Input("peak_width", default=DEFAULT_PEAK_WIDTH, min=1, max=64,
                                     tooltip="Minimum repeated cells per axis needed as detection evidence (video v2)."),
                    ]),
                    io.DynamicCombo.Option("By Pixel Size", [
                        io.Int.Input("pixel_size", default=DEFAULT_PIXEL_SIZE, min=1, max=512,
                                     tooltip="Target cell pitch. Output dimensions are round(width/pitch), round(height/pitch). Try 3–8 for detailed AI sprites."),
                    ]),
                    io.DynamicCombo.Option("By Grid Count", [
                        io.Int.Input("grid_w", default=DEFAULT_GRID_W, min=1, max=4096,
                                     tooltip="Exact output width in cells; no content-dependent refinement."),
                        io.Int.Input("grid_h", default=DEFAULT_GRID_H, min=1, max=4096,
                                     tooltip="Exact output height in cells; rectangular grids stay rectangular."),
                    ]),
                ], tooltip="One regular full-frame grid, fixed for the whole clip. Manual counts are exact."),
                io.Int.Input("temporal_window", default=0, min=0, max=8,
                             tooltip="0 = off. Positive values look up to this many frames each side, only through consecutive stable regions. No RGB averaging, history feedback or cross-cut mixing. Start at 1."),
                io.Combo.Input("backend", options=BACKEND_OPTIONS, default="Auto",
                               tooltip="Backend for legacy Majority/Median sampling. Robust Cell, Center and detection use NumPy on either backend."),
                io.Int.Input("chunk_frames", default=12, min=0, max=256, advanced=True,
                             tooltip="Frames converted to uint8 per block; 0 = all. Temporal neighbours cross block boundaries. Does not change results."),
                io.Int.Input("fallback_pixel_size", default=8, min=1, max=512, optional=True, advanced=True,
                             tooltip="Auto Detect uses this pitch if periodicity is weak. detected=False then. It is a style choice, not an inferred source grid."),
            ],
            outputs=[
                io.Video.Output(display_name="video", tooltip="Unquantised video with original audio and exact in-graph frame rate. Encoding may change colours."),
                io.Image.Output(display_name="images", tooltip="Restored RGB/RGBA frames. Feed the palette and nearest export chain from here."),
                io.Audio.Output(display_name="audio", tooltip="Original audio, untouched."),
                io.Float.Output(display_name="fps", tooltip="Numeric frame rate for CreateVideo; use video for the exact original Fraction."),
                io.Int.Output(display_name="grid_w", tooltip="Exact output width in cells."),
                io.Int.Output(display_name="grid_h", tooltip="Exact output height in cells."),
                io.Float.Output(display_name="pixel_size", tooltip="Source width divided by output width; may be fractional."),
                io.Boolean.Output(display_name="detected", tooltip="True only when Auto has sufficient periodic evidence; false for manual grids and fallback."),
                io.Float.Output(display_name="confidence", tooltip="0..1 periodic evidence for the selected grid, not a visual quality score."),
            ],
        )

    @classmethod
    def execute(cls, video: Input.Video, sampling: str = "Robust Cell", grid_mode=None,
                backend: str = "Auto", temporal_window: int = 0, chunk_frames: int = 12,
                fallback_pixel_size: int = 8) -> io.NodeOutput:
        if sampling not in SAMPLING_OPTIONS:
            raise ValueError(f"{NODE_ID}: unknown sampling {sampling!r}.")
        if backend not in BACKEND_OPTIONS:
            raise ValueError(f"{NODE_ID}: unknown backend {backend!r}.")
        module = _resolve_backend(backend)[1] if sampling in ("Majority Cluster", "Median") else None
        mode, params = _unpack_grid_mode(grid_mode)
        if mode not in GRID_MODE_OPTIONS:
            raise ValueError(f"{NODE_ID}: unknown grid_mode {mode!r}.")
        components = video.get_components()
        frames = components.images
        if frames.ndim != 4 or frames.shape[-1] not in (3, 4) or min(frames.shape[:3]) < 1:
            raise ValueError(f"{NODE_ID}: expected non-empty [T,H,W,3/4], got {tuple(frames.shape)}.")
        total, height, width = frames.shape[:3]
        alpha = components.alpha
        if alpha is not None:
            if alpha.ndim == 3:
                alpha = alpha.unsqueeze(-1)
            if tuple(alpha.shape) != (total, height, width, 1):
                raise ValueError(f"{NODE_ID}: alpha must match [T,H,W] or [T,H,W,1].")
        if alpha is None and frames.shape[-1] == 4:
            alpha = frames[..., 3:4]
        minimum = float(_pick(params, "min_size", DEFAULT_MIN_SIZE))
        maximum = float(_pick(params, "max_pixel_size", DEFAULT_MAX_PIXEL_SIZE))
        repeats = int(_pick(params, "peak_width", DEFAULT_PEAK_WIDTH))
        probes = sample_indices(total, int(_pick(params, "sample_stride", DEFAULT_SAMPLE_STRIDE)))
        evidence = []
        for index in probes:
            probe = _torch_image_to_uint8_bhwc(frames[int(index):int(index) + 1, ..., :3])[0]
            if alpha is not None:
                visibility = alpha[int(index)].detach().cpu().numpy()
                probe = np.rint(probe * np.clip(visibility, 0, 1)).astype(np.uint8)
            evidence.append(grid_evidence(probe, minimum, maximum, repeats))
        if mode == "Auto Detect":
            gw, gh, detected, confidence = choose_grid(evidence, width, height, max(1, int(fallback_pixel_size)))
            if not detected:
                print(f"{NODE_ID}: weak periodicity; using fallback pitch {fallback_pixel_size} ({gw}x{gh}).")
        else:
            gw, gh = _resolve_grid_size(mode, params, width, height, node_label=NODE_ID)
            detected = False
            confidence = float(np.mean([min(float(sx[gw]) if gw < len(sx) else 0,
                                            float(sy[gh]) if gh < len(sy) else 0) for sx, sy in evidence]))
        radius = max(0, min(MAX_TEMPORAL_WINDOW, int(temporal_window)))
        chunk = total if chunk_frames <= 0 else max(1, int(chunk_frames))
        restored = np.empty((total, gh, gw, 3), dtype=np.uint8)
        restored_alpha = None if alpha is None else np.empty((total, gh, gw, 1), dtype=np.uint8)
        pending = deque()
        next_output = 0

        def emit(index):
            by_index = dict(pending)
            current = by_index[index]
            neighbours = []
            for direction in (-1, 1):
                valid = np.ones((gh, gw), dtype=bool)
                previous = current
                for distance in range(1, radius + 1):
                    neighbour = by_index.get(index + direction * distance)
                    if neighbour is None:
                        break
                    valid = valid & stable_pair(previous, neighbour)
                    neighbours.append((neighbour, valid))
                    previous = neighbour
            restored[index] = stabilise(current, neighbours)
            if restored_alpha is not None:
                restored_alpha[index] = current.alpha

        for start in range(0, total, chunk):
            stop = min(total, start + chunk)
            block = _torch_image_to_uint8_bhwc(frames[start:stop, ..., :3])
            alpha_block = None if alpha is None else _torch_image_to_uint8_bhwc(alpha[start:stop])
            for j, rgb in enumerate(block):
                if module is not None:
                    _seed_sampler(module)
                cells = sample_cells(rgb, None if alpha_block is None else alpha_block[j], gw, gh, sampling, module)
                index = start + j
                pending.append((index, cells))
                if index >= next_output + radius:
                    emit(next_output)
                    next_output += 1
                    while pending and pending[0][0] < next_output - radius:
                        pending.popleft()
        while next_output < total:
            emit(next_output)
            next_output += 1
            while pending and pending[0][0] < next_output - radius:
                pending.popleft()
        if restored_alpha is not None and np.all(restored_alpha == 255):
            restored_alpha = None
        stacked = restored if restored_alpha is None else np.concatenate((restored, restored_alpha), axis=-1)
        images = torch.from_numpy(np.ascontiguousarray(stacked)).float().div_(255)
        rate = components.frame_rate
        if not isinstance(rate, Fraction):
            rate = Fraction(rate)
        colour_space = video.get_color_space()
        if colour_space in ("HDR", "HDR PQ"):
            print(f"{NODE_ID}: processing is 8-bit; HDR precision is not preserved. Use an SDR source for pixel work.")
        if colour_space not in ("sRGB", "HDR", "HDR PQ"):
            colour_space = "sRGB"
        output = InputImpl.VideoFromComponents(
            Types.VideoComponents(images=images[..., :3], audio=components.audio, frame_rate=rate,
                                  metadata=components.metadata, alpha=None if restored_alpha is None else images[..., 3]),
            bit_depth=video.get_bit_depth() or 8, color_space=colour_space,
        )
        return io.NodeOutput(output, images, components.audio, float(rate), gw, gh,
                             float(width) / gw, detected, confidence)
