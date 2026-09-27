"""Reproducible local video comparison. Run with the ComfyUI Python environment."""

import argparse
import gc
import hashlib
import importlib.util
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent.parent))
from comfy_api.latest import InputImpl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--package", type=Path, default=ROOT)
    parser.add_argument("--sampling", default="Robust Cell")
    parser.add_argument("--pixel-size", type=int, default=3)
    parser.add_argument("--window", type=int, default=0)
    parser.add_argument("--legacy", action="store_true")
    parser.add_argument("--auto", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("pixelpipe", args.package / "__init__.py", submodule_search_locations=[str(args.package)])
    package = importlib.util.module_from_spec(spec)
    sys.modules["pixelpipe"] = package
    spec.loader.exec_module(package)
    from pixelpipe.nodes.video_grid_restore import PixelVideoGridRestore
    file_video = InputImpl.VideoFromFile(str(args.input))
    components = file_video.get_components()
    source = InputImpl.VideoFromComponents(components, color_space=file_video.get_color_space())
    parameters = dict(sampling=args.sampling, temporal_window=args.window, chunk_frames=12,
                      grid_mode={"grid_mode": "By Pixel Size", "pixel_size": args.pixel_size})
    if args.auto:
        parameters["grid_mode"] = {"grid_mode": "Auto Detect"}
    if args.legacy:
        parameters.update(refine_intensity=0, fix_square=False)
    gc.collect()
    process = psutil.Process()
    initial = process.memory_info().rss
    peak = [initial]
    done = threading.Event()

    def measure():
        while not done.wait(0.01):
            peak[0] = max(peak[0], process.memory_info().rss)

    worker = threading.Thread(target=measure)
    worker.start()
    started = time.perf_counter()
    try:
        result = PixelVideoGridRestore.execute(video=source, **parameters).result
    finally:
        elapsed = time.perf_counter() - started
        peak[0] = max(peak[0], process.memory_info().rss)
        done.set()
        worker.join()
    frames = (result[1][..., :3] * 255).round().to(dtype=torch.uint8).numpy()
    native = args.output / "frames.npz"
    np.savez_compressed(native, frames=frames)
    for index in sorted(set([0, min(3, len(frames) - 1), min(12, len(frames) - 1), len(frames) // 2, len(frames) - 1])):
        Image.fromarray(frames[index]).save(args.output / f"frame-{index:03}.png")
    # Integer nearest upscaling before 4:2:0; even padding never changes the grid.
    h, w = frames.shape[1:3]
    scale = max(1, round(736 / max(h, w)))
    command = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
               "-s", f"{w}x{h}", "-r", str(components.frame_rate), "-i", "pipe:0", "-i", str(args.input),
               "-map", "0:v", "-map", "1:a?", "-vf", f"scale=iw*{scale}:ih*{scale}:flags=neighbor,pad=ceil(iw/2)*2:ceil(ih/2)*2",
               "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac",
               "-movflags", "+faststart", str(args.output / "preview.mp4")]
    encoded = subprocess.Popen(command, stdin=subprocess.PIPE)
    encoded.communicate(frames.tobytes())
    if encoded.returncode:
        raise RuntimeError("ffmpeg encoding failed")
    stats = {
        "input": str(args.input.resolve()), "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "parameters": parameters, "grid": [result[4], result[5]], "detected": result[7], "confidence": result[8],
        "frame_count": len(frames), "frame_rate": str(components.frame_rate), "algorithm_seconds": elapsed,
        "initial_rss_mib": initial / 2 ** 20, "peak_rss_mib": peak[0] / 2 ** 20,
        "extra_peak_mib": (peak[0] - initial) / 2 ** 20,
        "dark_fraction_frame12": float((frames[min(12, len(frames) - 1)].max(-1) < 50).mean()),
        "code_sha256": {str(p.relative_to(args.package)): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in sorted((args.package / "nodes").glob("*.py"))},
        "python": sys.version, "numpy": np.__version__,
    }
    (args.output / "result.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in stats.items() if k != "code_sha256"}, indent=2))


if __name__ == "__main__":
    main()
