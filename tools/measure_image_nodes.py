"""Re-measure the node-library numbers quoted in README's frontend-bug section.

Run with the ComfyUI Python environment, from anywhere:

    python tools/measure_image_nodes.py

The claim it backs -- "N nodes accept an IMAGE input, this pack's rank is M, so
the 64-slot search cut-off hides every custom node" -- went stale once already
when two IMAGE-input nodes were added, so the number is re-derivable rather
than only asserted in prose.
"""

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent.parent))

import nodes  # noqa: E402  (ComfyUI core; needs the path above)


def inputs_of(cls) -> list:
    try:
        spec = cls.INPUT_TYPES()
    except Exception:
        return []
    kinds = []
    for section in ("required", "optional", "hidden"):
        for entry in (spec.get(section) or {}).values():
            if isinstance(entry, (list, tuple)) and entry:
                kinds.append(entry[0])
    return kinds


def main() -> int:
    asyncio.run(nodes.init_extra_nodes())
    accepting, custom = [], []
    for name, cls in nodes.NODE_CLASS_MAPPINGS.items():
        if "IMAGE" not in inputs_of(cls):
            continue
        accepting.append(name)
        if "custom_nodes" in getattr(cls, "__module__", ""):
            custom.append(name)

    pack = {"PixelGridRestore", "PixelPaletteExtract", "PixelPaletteApply",
            "PixelExportScale", "PixelBackgroundRemove", "PixelSpriteSheet"}
    ranks = {n: accepting.index(n) + 1 for n in pack if n in accepting}
    print(f"nodes accepting an IMAGE input: {len(accepting)}")
    print(f"this pack's ranks (1-based):    {dict(sorted(ranks.items(), key=lambda kv: kv[1]))}")
    if custom:
        print(f"first custom node:              rank {accepting.index(custom[0]) + 1} ({custom[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
