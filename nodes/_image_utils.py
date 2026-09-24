"""Shared torch <-> numpy image helpers.

Ported verbatim (behaviour-preserving) from the legacy V1 node
``custom_nodes/perfectPixel-ComfyUI/nodes_perfect_pixel.py`` so that every node
in this package shares one already-verified implementation.

Conventions
-----------
ComfyUI ``IMAGE`` (what our nodes receive / emit):
    ``torch.float32``, layout ``[B, H, W, C]``, values ``0.0 .. 1.0``, ``C == 3``.

``perfect_pixel`` upstream (what we must hand it / get back):
    ``numpy.uint8`` RGB, layout ``[H, W, 3]``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "_torch_image_to_uint8_rgb",
    "_uint8_rgb_to_torch_image",
    "_nearest_scale_bhwc",
]


def _torch_image_to_uint8_rgb(img: torch.Tensor) -> np.ndarray:
    """ComfyUI IMAGE -> numpy uint8 RGB.

    In:  torch ``[B, H, W, 3]`` float32, range 0..1
    Out: numpy ``[B, H, W, 3]`` uint8
    """
    img = img.detach().cpu()
    img = torch.clamp(img, 0.0, 1.0)
    img = (img * 255.0).round().to(torch.uint8)
    return img.numpy()


def _uint8_rgb_to_torch_image(rgb: np.ndarray) -> torch.Tensor:
    """numpy uint8 RGB ``[H, W, 3]`` -> torch float32 ``[1, H, W, 3]`` in 0..1."""
    if rgb.dtype != np.uint8:
        rgb = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    t = torch.from_numpy(rgb).float() / 255.0
    return t.unsqueeze(0)  # [1,H,W,3]


def _nearest_scale_bhwc(img_bhwc: torch.Tensor, scale: int) -> torch.Tensor:
    """``[B, H, W, C]`` -> ``[B, H*scale, W*scale, C]`` using nearest-neighbour.

    Nearest (not bilinear/area) is the whole point here: pixel art must stay
    crisp and every source pixel must become an exact ``scale x scale`` block.
    """
    if scale == 1:
        return img_bhwc
    b, h, w, c = img_bhwc.shape
    x = img_bhwc.permute(0, 3, 1, 2)  # [B,C,H,W]
    x = F.interpolate(x, scale_factor=scale, mode="nearest")
    return x.permute(0, 2, 3, 1)  # [B,H,W,C]
