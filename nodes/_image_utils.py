"""Shared torch <-> numpy image helpers.

Ported from the legacy V1 node ``custom_nodes/perfectPixel-ComfyUI/
nodes_perfect_pixel.py`` so that every node in this package shares one
already-verified implementation, then extended for alpha.

Conventions
-----------
ComfyUI ``IMAGE`` (what our nodes receive / emit):
    ``torch.float32``, layout ``[B, H, W, C]``, values ``0.0 .. 1.0``.

    ``C`` is **3 or 4**. Three is the common case, but Qwen-Image-2.1's VAE
    carries four channels so that it can generate and edit transparent images
    natively, and ``VAEDecode`` therefore hands us RGBA directly. Rejecting four
    channels -- which this package used to do -- makes the pipeline unusable on
    the very model it targets.

``perfect_pixel`` upstream (what we must hand it / get back):
    ``numpy.uint8``, layout ``[H, W, C]``. Its grid detection and sampling
    helpers take a colour image and are channel-count agnostic as far as this
    package relies on them; ``detect_grid_scale`` converts to grey itself.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "_torch_image_to_uint8_bhwc",
    "_torch_image_to_uint8_rgb",
    "_uint8_rgb_to_torch_image",
    "_nearest_scale_bhwc",
    "_split_rgb_alpha",
    "_join_rgb_alpha",
]


def _torch_image_to_uint8_bhwc(img: torch.Tensor) -> np.ndarray:
    """ComfyUI IMAGE -> numpy uint8 ``[B, H, W, C]``.

    Channel-agnostic on purpose: the name used to be ``..._rgb``, which was
    only ever true because the ported node was fed RGB. Nothing here assumes
    ``C == 3``.
    """
    img = img.detach().cpu()
    img = torch.clamp(img, 0.0, 1.0)
    img = (img * 255.0).round().to(torch.uint8)
    return img.numpy()


# Legacy alias, kept so existing call sites and any third-party reference to it
# keep working. Prefer the ``_bhwc`` name in new code.
_torch_image_to_uint8_rgb = _torch_image_to_uint8_bhwc


def _uint8_rgb_to_torch_image(rgb: np.ndarray) -> torch.Tensor:
    """numpy uint8 ``[H, W, C]`` -> torch float32 ``[1, H, W, C]`` in 0..1."""
    if rgb.dtype != np.uint8:
        rgb = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    t = torch.from_numpy(np.ascontiguousarray(rgb)).float() / 255.0
    return t.unsqueeze(0)  # [1,H,W,C]


def _nearest_scale_bhwc(img_bhwc: torch.Tensor, scale: int) -> torch.Tensor:
    """``[B, H, W, C]`` -> ``[B, H*scale, W*scale, C]`` using nearest-neighbour.

    Nearest (not bilinear/area) is the whole point here: pixel art must stay
    crisp and every source pixel must become an exact ``scale x scale`` block.
    ``C`` is whatever it was; alpha scales along with the colour channels, which
    is what keeps a transparent pixel-art sprite's mask blocky too.
    """
    if scale == 1:
        return img_bhwc
    b, h, w, c = img_bhwc.shape
    x = img_bhwc.permute(0, 3, 1, 2)  # [B,C,H,W]
    x = F.interpolate(x, scale_factor=scale, mode="nearest")
    return x.permute(0, 2, 3, 1)  # [B,H,W,C]


def _split_rgb_alpha(bhwc: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Split uint8 ``[B, H, W, C]`` into ``(rgb[B,H,W,3], alpha[B,H,W,1] | None)``.

    Alpha is kept as its own array rather than as a 4th colour channel because
    every algorithm in this package that reason about *colour* must not see it:
    a k-means quantiser handed RGBA would treat "transparent" as a colour, and
    the grid detector's grey conversion would fold it into the luminance.
    """
    if bhwc.ndim != 4:
        raise ValueError(f"expected a 4D image [B,H,W,C], got shape {tuple(bhwc.shape)}.")
    channels = bhwc.shape[-1]
    if channels == 3:
        return bhwc, None
    if channels == 4:
        return bhwc[..., :3], bhwc[..., 3:4]
    raise ValueError(
        f"expected 3 (RGB) or 4 (RGBA) channels, got {channels}. "
        "Qwen-Image-2.1's VAE emits RGBA; most other VAEs emit RGB."
    )


def _join_rgb_alpha(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Re-attach ``alpha[B,H,W,1]`` to ``rgb[B,H,W,3]`` -> ``[B,H,W,4]``.

    Both parts must already be at the same resolution; the caller is
    responsible for having resampled them onto one shared grid.
    """
    if rgb.shape[:-1] != alpha.shape[:-1]:
        raise ValueError(
            f"rgb and alpha must share H/W before joining, got {tuple(rgb.shape)} "
            f"vs {tuple(alpha.shape)}."
        )
    return np.concatenate([rgb, alpha], axis=-1)
