"""Video regressions against real ComfyUI VideoComponents, without a running server."""

import importlib.util
import io as buffer_io
import sys
import unittest
import warnings
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent.parent))
if "pixelpipe" not in sys.modules:
    spec = importlib.util.spec_from_file_location("pixelpipe", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    package = importlib.util.module_from_spec(spec)
    sys.modules["pixelpipe"] = package
    spec.loader.exec_module(package)

from comfy_api.latest import InputImpl, Types
from pixelpipe.nodes.video_grid_restore import PixelVideoGridRestore as Node
from pixelpipe.nodes._video_utils import sample_cells, sample_indices, stabilise, stable_pair, rgb_to_lab


def video(frames, alpha=None, rate=Fraction(24)):
    return InputImpl.VideoFromComponents(Types.VideoComponents(
        images=torch.from_numpy(np.ascontiguousarray(frames)).float() / 255,
        frame_rate=rate, alpha=None if alpha is None else torch.from_numpy(alpha).float() / 255,
        audio={"waveform": torch.zeros(1, 2, 3200), "sample_rate": 32000},
    ), color_space="sRGB")


def run(frames, grid=(8, 8), **kwargs):
    source = frames if not isinstance(frames, np.ndarray) else video(frames)
    result = Node.execute(video=source, grid_mode={"grid_mode": "By Grid Count", "grid_w": grid[0], "grid_h": grid[1]}, **kwargs).result
    return (result[1] * 255).round().to(torch.uint8).numpy(), result


def upscale(frames, scale=4):
    return frames.repeat(scale, axis=1).repeat(scale, axis=2)


class VideoPipelineTests(unittest.TestCase):
    def test_frame_identity_and_stride(self):
        frames = np.broadcast_to(np.arange(56, dtype=np.uint8)[:, None, None, None] * 4, (56, 32, 32, 3)).copy()
        for chunk in (0, 1, 3, 12):
            output, _ = run(frames, chunk_frames=chunk)
            np.testing.assert_array_equal(output[:, 0, 0], frames[:, 0, 0])

    def test_manual_rectangular_and_upsampled_grid(self):
        frames = np.full((2, 17, 29, 3), 137, np.uint8)
        for shape in ((7, 5), (31, 23), (1, 1)):
            output, result = run(frames, grid=shape)
            self.assertEqual(output.shape, (2, shape[1], shape[0], 3))
            self.assertEqual(result[4:6], shape)
            self.assertTrue(np.all(output == 137))

    def test_motion_short_events_cut_and_reversed_colours(self):
        for speed in (1, 2, 4):
            gt = np.full((12, 16, 48, 3), 230, np.uint8)
            for t in range(12):
                x = (t * speed) % 44
                gt[t, 4:12, x:x + 2] = 10
            gt[5, 1, 20] = [255, 20, 20]  # A real one-frame highlight.
            gt[8:] = gt[8:, :, ::-1]       # A cut with the same histogram.
            for invert in (False, True):
                source = 255 - gt if invert else gt
                for radius in (0, 1, 2, 8):
                    output, _ = run(upscale(source), grid=(48, 16), temporal_window=radius, chunk_frames=3)
                    np.testing.assert_array_equal(output, source)

    def test_no_history_colours_or_feedback(self):
        rng = np.random.default_rng(13)
        frames = np.clip(130 + rng.integers(-6, 7, (9, 48, 48, 3)), 0, 255).astype(np.uint8)
        output, _ = run(frames, temporal_window=2)
        for t in range(len(frames)):
            cells = sample_cells(frames[t], None, 8, 8, "Robust Cell")
            supported = np.any(np.all(cells.candidates == output[t, ..., None, :], axis=-1), axis=-1)
            self.assertTrue(supported.all())
        reverse, _ = run(frames[::-1].copy(), temporal_window=2)
        np.testing.assert_array_equal(output, reverse[::-1])

    def test_static_noise_improves_without_bias(self):
        rng = np.random.default_rng(5)
        # Spatial noise allows choosing a better supported current-frame sample.
        noise = rng.integers(-6, 7, (20, 64, 64, 1))
        frames = np.broadcast_to(128 + noise, (20, 64, 64, 3)).astype(np.uint8).copy()
        off, _ = run(frames, grid=(16, 16))
        on, _ = run(frames, grid=(16, 16), temporal_window=1)
        flicker_off = np.abs(np.diff(off.astype(float), axis=0)).mean()
        flicker_on = np.abs(np.diff(on.astype(float), axis=0)).mean()
        self.assertLess(flicker_on, flicker_off * 0.8, (flicker_off, flicker_on))
        self.assertLess(abs(float(on.mean()) - 128), 0.5)

    def test_stable_pair_rejects_edges_and_cut(self):
        a = np.full((64, 64, 3), 200, np.uint8)
        b = a.copy()
        b[:, 32:] = 20
        first = sample_cells(a, None, 16, 16, "Robust Cell")
        second = sample_cells(b, None, 16, 16, "Robust Cell")
        self.assertFalse(stable_pair(first, second).any())
        np.testing.assert_array_equal(stabilise(first, [(second, stable_pair(first, second))]), first.rgb)
        self.assertIs(stabilise(first, []), first.rgb)

    def test_chunk_repeat_and_short_clips(self):
        rng = np.random.default_rng(6)
        frames = rng.integers(0, 256, (7, 37, 43, 3), dtype=np.uint8)
        expected, _ = run(frames, temporal_window=2, chunk_frames=0)
        for chunk in (1, 3, 12):
            actual, _ = run(frames, temporal_window=2, chunk_frames=chunk)
            np.testing.assert_array_equal(actual, expected)
        for total in (1, 2):
            off, _ = run(frames[:total])
            on, _ = run(frames[:total], temporal_window=8)
            np.testing.assert_array_equal(off, on)

    def test_alpha_slot_rgba_and_mixed_opacity(self):
        rgb = np.full((4, 32, 32, 3), [220, 160, 80], np.uint8)
        alpha = np.full((4, 32, 32), 255, np.uint8)
        alpha[1, :, :16] = 0
        alpha[2] = 0
        output, result = run(video(rgb, alpha=alpha), temporal_window=2)
        rgba, _ = run(np.concatenate((rgb, alpha[..., None]), -1), temporal_window=2)
        np.testing.assert_array_equal(rgba, output)
        expected_alpha = alpha[:, ::4, ::4]
        np.testing.assert_array_equal(output[..., 3], expected_alpha)
        self.assertEqual(result[0].get_components().images.shape[-1], 3)
        np.testing.assert_array_equal(result[0].get_components().alpha.numpy() * 255, expected_alpha)

    def test_hidden_black_does_not_pollute_visible_colour(self):
        rgb = np.full((1, 16, 16, 3), [230, 90, 30], np.uint8)
        alpha = np.full((1, 16, 16), 255, np.uint8)
        alpha[:, ::2, ::2] = 0
        black = rgb.copy()
        black[alpha == 0] = 0
        for sampling in ("Robust Cell", "Center Sample", "Majority Cluster", "Median"):
            a, _ = run(video(rgb, alpha), grid=(4, 4), sampling=sampling)
            b, _ = run(video(black, alpha), grid=(4, 4), sampling=sampling)
            np.testing.assert_array_equal(a, b)

    def test_legacy_samplers_repeat_and_frame_identity(self):
        rng = np.random.default_rng(2)
        gt = rng.integers(0, 256, (4, 8, 8, 3), dtype=np.uint8)
        for sampling in ("Center Sample", "Median", "Majority Cluster"):
            for backend in ("OpenCV", "Lightweight"):
                a, _ = run(upscale(gt), sampling=sampling, backend=backend)
                b, _ = run(upscale(gt), sampling=sampling, backend=backend)
                np.testing.assert_array_equal(a, b)
                np.testing.assert_array_equal(a, gt)

    def test_auto_periodic_fallback_and_bounded_probes(self):
        rng = np.random.default_rng(1)
        gt = rng.integers(0, 256, (6, 8, 8, 3), dtype=np.uint8)
        frames = upscale(gt, 16)
        result = Node.execute(video=video(frames)).result
        self.assertEqual(result[4:6], (8, 8))
        self.assertTrue(result[7])
        blank = np.full((3, 35, 47, 3), 128, np.uint8)
        result = Node.execute(video=video(blank), fallback_pixel_size=5).result
        self.assertEqual(result[4:6], (9, 7))
        self.assertFalse(result[7])
        indices = sample_indices(100000, 1)
        self.assertLessEqual(len(indices), 32)
        self.assertEqual((indices[0], indices[-1]), (0, 99999))

    def test_rate_audio_encode_and_schema(self):
        source = video(np.full((3, 32, 32, 3), 150, np.uint8), rate=Fraction(30000, 1001))
        _, result = run(source)
        restored = result[0]
        self.assertEqual(restored.get_frame_rate(), Fraction(30000, 1001))
        self.assertIs(result[2], source.get_components().audio)
        self.assertEqual(restored.get_frame_count(), 3)
        buffer = buffer_io.BytesIO()
        restored.save_to(buffer, format=Types.VideoContainer.MP4)
        self.assertGreater(len(buffer.getvalue()), 100)
        self.assertEqual(len(Node.GET_SCHEMA().outputs), 9)

    def test_lab_units(self):
        lab = rgb_to_lab(np.array([[0, 0, 0], [255, 255, 255]], np.uint8))
        np.testing.assert_allclose(lab[:, 0], [0, 100], atol=0.001)
        np.testing.assert_allclose(lab[:, 1:], 0, atol=0.01)

    def test_single_pixel_input_has_no_empty_reduction(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            for shape in ((1, 1), (1, 9), (9, 1)):
                frames = np.full((1, *shape, 3), 123, np.uint8)
                result = Node.execute(video=video(frames), temporal_window=1).result
                self.assertTrue(torch.all(result[1] == 123 / 255))


if __name__ == "__main__":
    unittest.main()
