"""ComfyUI-PixelPipeline -- a strict "AI image -> pixel art" pipeline.

This package uses the **V3 node schema** (``comfy_api.latest``), not the legacy
V1 ``NODE_CLASS_MAPPINGS`` dict. ComfyUI picks the V3 path only when
``NODE_CLASS_MAPPINGS`` is *absent* from this module -- defining it would make
ComfyUI silently skip ``comfy_entrypoint`` and load nothing. Do not add it.

Everything importable lives in ``nodes/``; this file is deliberately just the
web-directory declaration plus the entry point.
"""

from __future__ import annotations

from typing_extensions import override

from comfy_api.latest import ComfyExtension, io

from .nodes import ALL_NODES

# Optional frontend extensions. The folder exists but is empty for now; batch 2
# may add JS here. ComfyUI ignores the declaration if the directory is missing.
WEB_DIRECTORY = "./web"

__all__ = ["WEB_DIRECTORY", "PixelPipelineExtension", "comfy_entrypoint"]


class PixelPipelineExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return list(ALL_NODES)


async def comfy_entrypoint() -> PixelPipelineExtension:
    """ComfyUI calls this to load the extension and its nodes."""
    return PixelPipelineExtension()
