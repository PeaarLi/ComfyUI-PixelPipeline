"""Standalone verification for ComfyUI-PixelPipeline. No ComfyUI server needed.

Run with the ComfyUI venv interpreter:

    E:\\ComfyUI\\.venv\\Scripts\\python.exe E:\\ComfyUI\\custom_nodes\\ComfyUI-PixelPipeline\\_selftest.py

It checks, in order:

1. **Syntax** -- ``py_compile`` every ``.py`` in the package.
2. **Import-time schema** -- ComfyUI calls ``GET_SCHEMA()`` on every node while
   importing the pack, and swallows any exception into a single log warning, so
   the whole pack would silently vanish. This script calls it explicitly and
   prints the resulting input/output names.
   The package directory name contains a hyphen and therefore cannot be imported
   by name; it is loaded through ``importlib`` under the legal alias
   ``pixelpipe``.
3. **Grid behaviour** -- runs the grid nodes against real images:
   - auto-detect on ``fox_pixel_00001_.png`` must yield 125x122,
   - ``By Pixel Size`` with ``pixel_size=16`` must yield a square grid near 64
     (the capability the legacy V1 node did not have at all),
   - a smooth, non-pixel image must **raise** instead of being passed through,
   - ``PixelExportScale`` with ``scale=4`` must turn 125x122 into 500x488.
4. **Palette schema** -- input/output names, types, defaults, ranges and the
   ``optional`` / ``lazy`` markers of the two palette nodes.
5. **Palette extraction** -- 32 colours from ``fox_pixel_GRID.png``, and the HEX
   text parses back to exactly 32 distinct RGB triples.
6. **Cross-image consistency** -- a palette taken from image A and applied to
   image B must leave B using a subset of A's colours.
7. **Every method** -- all six quantize methods and all five extract methods run
   and report what they produced.
8. **Dithering** -- ``None`` vs ``Bayer 4x4`` vs ``Floyd-Steinberg``; every
   dithered output must stay inside the palette, and Bayer thresholds must be a
   pure function of pixel coordinates (so they match across image sizes).
9. **Lazy pruning** -- the real ``comfy_execution.graph.TopologicalSort`` must
   leave ``PixelPaletteExtract`` out of the execution block when ``palette`` is
   unwired, and include it when it is wired.
10. **Determinism** -- the same call twice must produce byte-identical output.

Comparison images are written to ``E:\\ComfyUI\\output\\pixel_verify_*.png``.

Exit code is 0 only when every check passes.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMFY_ROOT = ROOT.parent.parent  # ...\ComfyUI
DEFAULT_BASELINE = COMFY_ROOT / "output" / "fox_pixel_00001_.png"

FAILURES: list[str] = []
CHECKS = 0


def _ok(label: str, detail: str = "") -> None:
    print(f"  [PASS] {label}" + (f" -- {detail}" if detail else ""))


def _fail(label: str, detail: str = "") -> None:
    FAILURES.append(label)
    print(f"  [FAIL] {label}" + (f" -- {detail}" if detail else ""))


def check(label: str, condition: bool, detail: str = "") -> bool:
    global CHECKS
    CHECKS += 1
    if condition:
        _ok(label, detail)
    else:
        _fail(label, detail)
    return bool(condition)


# --------------------------------------------------------------------------- #
# 1. syntax
# --------------------------------------------------------------------------- #
def section_syntax() -> None:
    print("\n=== 1. syntax (py_compile) ===")
    import py_compile

    files = sorted(p for p in ROOT.rglob("*.py") if "__pycache__" not in p.parts)
    for path in files:
        try:
            py_compile.compile(str(path), doraise=True, quiet=1)
        except py_compile.PyCompileError as exc:
            _fail(f"compile {path.relative_to(ROOT)}", str(exc))
        else:
            _ok(f"compile {path.relative_to(ROOT)}")
    print(f"  ({len(files)} files)")


# --------------------------------------------------------------------------- #
# 2. import + GET_SCHEMA
# --------------------------------------------------------------------------- #
def load_package():
    """Load the pack under the alias ``pixelpipe`` despite the hyphenated dir.

    The ComfyUI root must be on ``sys.path`` or ``import comfy_api.latest``
    fails -- inside the real server that is already the case, standalone it is
    not.
    """
    if str(COMFY_ROOT) not in sys.path:
        sys.path.insert(0, str(COMFY_ROOT))

    spec = importlib.util.spec_from_file_location(
        "pixelpipe",
        str(ROOT / "__init__.py"),
        # NOTE: ``submodule_search_locations`` is a KEYWORD-ONLY parameter.
        # Passing it as a third positional argument is a TypeError on 3.12.
        submodule_search_locations=[str(ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["pixelpipe"] = module
    spec.loader.exec_module(module)
    return module


def section_schema(module) -> list:
    print("\n=== 2. import-time GET_SCHEMA (V3 pitfall #8) ===")
    check("package has comfy_entrypoint", hasattr(module, "comfy_entrypoint"))
    check(
        "package does NOT define NODE_CLASS_MAPPINGS",
        not hasattr(module, "NODE_CLASS_MAPPINGS"),
        "defining it would make ComfyUI silently skip comfy_entrypoint",
    )
    check("WEB_DIRECTORY declared", getattr(module, "WEB_DIRECTORY", None) == "./web")

    extension = asyncio.run(module.comfy_entrypoint())
    node_list = asyncio.run(extension.get_node_list())
    check("get_node_list() returns 4 nodes", len(node_list) == 4,
          f"{[c.__name__ for c in node_list]}")

    for node_cls in node_list:
        schema = node_cls.GET_SCHEMA()  # must not raise
        print(f"\n  --- {schema.node_id} ({schema.display_name}) ---")
        print(f"      category  : {schema.category}")
        # NOTE: id / io_type live on the object, NOT in as_dict() -- as_dict()
        # only carries the per-type widget extras.
        for i in schema.inputs:
            d = i.as_dict()
            flags = [f for f in ("optional", "lazy", "advanced") if d.get(f)]
            flag_s = f" [{', '.join(flags)}]" if flags else ""
            print(
                f"      in  {i.id:<18} {i.io_type:<8} default={d.get('default')!r}"
                f" min={d.get('min')} max={d.get('max')}{flag_s}"
            )
        for o in schema.outputs:
            print(f"      out {o.id:<18} {o.io_type:<8} display_name={o.display_name!r}")
    return node_list


# --------------------------------------------------------------------------- #
# 3. behaviour
# --------------------------------------------------------------------------- #
def _load_image(path: Path):
    """PNG -> ComfyUI IMAGE tensor float32 [1,H,W,3] in 0..1."""
    import numpy as np
    import torch
    from PIL import Image

    # np.array (not asarray) so the buffer is writable -- torch warns otherwise.
    with Image.open(path) as im:
        arr = np.array(im.convert("RGB"), dtype=np.uint8)
    t = torch.from_numpy(arr).float() / 255.0
    return t.unsqueeze(0)


def _smooth_non_pixel_image(size: int = 512):
    """A deliberately smooth image: a linear ramp, gaussian-blurred.

    Chosen because it makes upstream's FFT *and* gradient detectors both fail
    (``return None, None, image``) -- verified at 256/512/1024 px. A plain
    un-blurred ramp does NOT fail, so the blur is essential.
    """
    import cv2
    import numpy as np
    import torch

    ramp = np.tile(np.linspace(0, 255, size, dtype=np.uint8), (size, 1))
    arr = np.stack([ramp] * 3, axis=-1)
    arr = cv2.GaussianBlur(arr, (0, 0), 12)
    t = torch.from_numpy(arr).float() / 255.0
    return t.unsqueeze(0)


def section_behaviour(nodes: list, baseline: Path) -> None:
    import torch
    import torch.nn.functional as F

    print("\n=== 3. behaviour ===")
    by_name = {c.__name__: c for c in nodes}
    grid = by_name["PixelGridRestore"]
    export = by_name["PixelExportScale"]

    # -- 3a. auto detect on the baseline ------------------------------------
    print("\n  [3a] PixelGridRestore Auto Detect on fox_pixel_00001_.png")
    if not baseline.is_file():
        _fail("baseline image exists", str(baseline))
        return
    _ok("baseline image exists", str(baseline))
    img = _load_image(baseline)
    print(f"      input   : {tuple(img.shape)}")
    _img, gw, gh, px, det, conf = grid.execute(
        image=img, sampling="Majority Cluster", grid_mode="Auto Detect"
    ).result
    print(f"      output  : image={tuple(_img.shape)} grid_w={gw} grid_h={gh} "
          f"pixel_size={px!r} detected={det!r} confidence={conf:.3f}")
    check("auto-detect grid_w == 125", gw == 125, f"got {gw}")
    check("auto-detect grid_h == 122", gh == 122, f"got {gh}")
    check("auto-detect detected is True", det is True, f"got {det!r}")
    check("auto-detect pixel_size ~= 8.192", abs(float(px) - 8.192) < 1e-6, f"got {px!r}")
    check("auto-detect output tensor is 1x122x125x3", tuple(_img.shape) == (1, 122, 125, 3),
          f"got {tuple(_img.shape)}")
    # confidence: the Qwen pseudo-pixel baseline has a real but weak grid
    # (raw prominence 9.13 on the calibration set -> ~0.51 after the map).
    check("auto-detect confidence is 0..1", 0.0 <= float(conf) <= 1.0, f"got {conf!r}")
    check("auto-detect confidence is mid-range for a weak real grid",
          0.35 <= float(conf) <= 0.70, f"got {conf!r}")

    # -- 3b. By Pixel Size: the capability V1 did not have -------------------
    print("\n  [3b] PixelGridRestore By Pixel Size (pixel_size=16) -- V1 could not do this")
    _img2, gw2, gh2, px2, det2, conf2 = grid.execute(
        image=img, sampling="Median", grid_mode="By Pixel Size", pixel_size=16
    ).result
    print(f"      requested grid_size=(64, 64) from round(1024/16)")
    print(f"      output  : image={tuple(_img2.shape)} grid_w={gw2} grid_h={gh2} "
          f"pixel_size={px2!r} detected={det2!r} confidence={conf2:.3f}")
    check("pixel_size=16 grid is square", gw2 == gh2, f"got {gw2}x{gh2}")
    # 64 is the requested cell count; upstream's refine pass then snaps to the
    # real cell boundaries, which on this image lands at 60. "A square grid
    # around 64" is the contract, not exactly 64.
    check("pixel_size=16 grid is ~64 cells", 55 <= gw2 <= 64, f"got {gw2}")
    check("pixel_size=16 detected is False", det2 is False, f"got {det2!r}")
    check("pixel_size=16 output is square", _img2.shape[1] == _img2.shape[2],
          f"got {tuple(_img2.shape)}")

    # -- 3b'. By Grid Count -------------------------------------------------
    print("\n  [3b'] PixelGridRestore By Grid Count (64x64)")
    _img3, gw3, gh3, px3, det3, conf3 = grid.execute(
        image=img, sampling="Center Sample", grid_mode="By Grid Count", grid_w=64, grid_h=64
    ).result
    print(f"      output  : image={tuple(_img3.shape)} grid_w={gw3} grid_h={gh3} "
          f"pixel_size={px3!r} detected={det3!r}")
    check("By Grid Count 64x64 is square", gw3 == gh3, f"got {gw3}x{gh3}")
    check("By Grid Count 64x64 is ~64 cells", 55 <= gw3 <= 64, f"got {gw3}")
    check("By Grid Count detected is False", det3 is False, f"got {det3!r}")

    # -- 3c. failure must raise, not pass through ---------------------------
    print("\n  [3c] smooth non-pixel image must RAISE (V1 silently passed it through)")
    smooth = _smooth_non_pixel_image(512)
    print(f"      input   : {tuple(smooth.shape)} (blurred linear ramp, no pixel grid)")
    raised = None
    try:
        result = grid.execute(image=smooth, sampling="Median", grid_mode="Auto Detect")
    except Exception as exc:  # noqa: BLE001 -- raising is exactly what we assert
        raised = exc
        print(f"      raised  : {type(exc).__name__}")
        print("      message : " + str(exc).replace("\n", " ")[:500])
    else:
        print(f"      NO RAISE; returned grid_w={result.result[1]} grid_h={result.result[2]}")
    check("smooth image raises", raised is not None,
          "" if raised else "upstream returned None,None,image and the node passed it through")
    check("exception is RuntimeError", isinstance(raised, RuntimeError),
          type(raised).__name__ if raised else "n/a")
    check(
        "exception message points at the manual escape hatch",
        raised is not None and "By Pixel Size" in str(raised) and "By Grid Count" in str(raised),
    )

    # -- 3c'. inconsistent batch must raise ValueError ----------------------
    # Two different 1024x1024 images that auto-detect to different grids. A
    # ComfyUI IMAGE batch is one tensor, so its members always share H/W; the
    # divergence has to come from detection, which is exactly the realistic
    # case (per-image refinement).
    print("\n  [3c'] inconsistent batch must raise ValueError")
    second = baseline.parent / "fox_source_00001_.png"
    if not second.is_file():
        print(f"      SKIP: {second} not present")
    else:
        img_b = _load_image(second)
        if img_b.shape != img.shape:
            print(f"      SKIP: {second.name} is {tuple(img_b.shape)}, not batchable with {tuple(img.shape)}")
        else:
            for tag, one in (("a", img), ("b", img_b)):
                _o, w1, h1, _p, _d, _c1 = grid.execute(
                    image=one, sampling="Median", grid_mode="Auto Detect"
                ).result
                print(f"      image {tag} ({second.name if tag == 'b' else baseline.name}) -> {w1}x{h1}")
            batch = torch.cat([img, img_b], dim=0)
            try:
                grid.execute(image=batch, sampling="Median", grid_mode="Auto Detect")
            except ValueError as exc:
                _ok("inconsistent batch raises ValueError", str(exc)[:200])
            except Exception as exc:  # noqa: BLE001
                _fail("inconsistent batch raises ValueError", f"got {type(exc).__name__}: {exc}")
            else:
                _fail("inconsistent batch raises ValueError", "no exception")

    # -- 3d. backends -------------------------------------------------------
    print("\n  [3d] backend selection")
    for be in ("Auto", "OpenCV"):
        _o, w4, h4, _p4, _d4, _c4 = grid.execute(
            image=img, sampling="Median", grid_mode="Auto Detect", backend=be
        ).result
        print(f"      backend={be:<12} -> {w4}x{h4}")
        check(f"backend={be} matches the 125x122 baseline", (w4, h4) == (125, 122), f"got {w4}x{h4}")

    # The no-CV2 backend is broken for auto-detection upstream (bare None return
    # value unpacked as a tuple). Assert we turn that into a clean RuntimeError
    # rather than letting a raw TypeError escape with a foreign traceback.
    lw_raised = None
    try:
        grid.execute(image=img, sampling="Median", grid_mode="Auto Detect", backend="Lightweight")
    except Exception as exc:  # noqa: BLE001
        lw_raised = exc
        print(f"      backend=Lightweight Auto Detect raised {type(exc).__name__}")
        print("      message : " + str(exc).replace("\n", " ")[:420])
    check("Lightweight auto-detect raises RuntimeError", isinstance(lw_raised, RuntimeError),
          type(lw_raised).__name__ if lw_raised else "no exception")
    check("Lightweight error names the upstream no-CV2 bug",
          lw_raised is not None and "estimate_grid_fft" in str(lw_raised))

    _o, w5, h5, _p5, _d5, _c5 = grid.execute(
        image=img, sampling="Median", grid_mode="By Grid Count", grid_w=125, grid_h=122,
        backend="Lightweight",
    ).result
    print(f"      backend=Lightweight By Grid Count(125,122) -> {w5}x{h5}")
    check("Lightweight works when a grid is supplied", (w5, h5) == (125, 122), f"got {w5}x{h5}")

    # -- 3e. export scale ---------------------------------------------------
    print("\n  [3e] PixelExportScale scale=4 on the 125x122 auto-detect result")
    scaled, sw, sh = export.execute(image=_img, scale=4).result
    print(f"      output  : image={tuple(scaled.shape)} width={sw} height={sh}")
    check("scale=4 width == 500", sw == 500, f"got {sw}")
    check("scale=4 height == 488", sh == 488, f"got {sh}")
    check("scale=4 tensor is 1x488x500x3", tuple(scaled.shape) == (1, 488, 500, 3),
          f"got {tuple(scaled.shape)}")
    block = scaled[0, :4, :4, :]
    check("nearest upscale produces exact 4x4 blocks",
          bool(torch.allclose(block, block[0, 0].expand_as(block))))

    scaled1, sw1, sh1 = export.execute(image=_img, scale=1).result
    check("scale=1 is a pass-through",
          (sw1, sh1) == (125, 122) and tuple(scaled1.shape) == tuple(_img.shape),
          f"{sw1}x{sh1}")


# --------------------------------------------------------------------------- #
# 4. palette node schemas
# --------------------------------------------------------------------------- #
def _schema_rows(schema):
    """(inputs, outputs) as plain tuples, for exact comparison against the spec."""
    ins = []
    for i in schema.inputs:
        d = i.as_dict()
        ins.append(
            {
                "id": i.id,
                "type": i.io_type,
                "default": d.get("default"),
                "min": d.get("min"),
                "max": d.get("max"),
                "options": list(d.get("options") or []),
                "optional": bool(d.get("optional")),
                "lazy": bool(d.get("lazy")),
                "advanced": bool(d.get("advanced")),
            }
        )
    outs = [(o.display_name, o.io_type) for o in schema.outputs]
    return ins, outs


def section_palette_schema(nodes: list) -> None:
    print("\n=== 4. palette node schemas ===")
    by_name = {c.__name__: c for c in nodes}

    extract = by_name["PixelPaletteExtract"].GET_SCHEMA()
    ins, outs = _schema_rows(extract)
    print(f"\n  --- {extract.node_id} ---")
    for row in ins:
        print(f"      in  {row['id']:<16} {row['type']:<7} default={row['default']!r} "
              f"min={row['min']} max={row['max']} options={row['options']}")
    for name, io_type in outs:
        print(f"      out {name:<16} {io_type}")
    check("Extract input ids are exactly the contract",
          [r["id"] for r in ins] == ["image", "colors", "method", "space", "sort",
                                     "accent_protect", "tile_size"],
          str([r["id"] for r in ins]))
    check("Extract input types",
          [r["type"] for r in ins] == ["IMAGE", "INT", "COMBO", "COMBO", "COMBO",
                                       "BOOLEAN", "INT"],
          str([r["type"] for r in ins]))
    check("Extract defaults: colors=128, space=Lab, sort=Luminance",
          (ins[1]["default"], ins[3]["default"], ins[4]["default"]) == (128, "Lab", "Luminance"),
          str((ins[1]["default"], ins[3]["default"], ins[4]["default"])))
    # accent_protect belongs on the node that *chooses* a palette; tile_size is
    # the Per-Tile k-means window and used to be an unreachable constant.
    check("Extract accent_protect defaults on", ins[5]["default"] is True,
          str(ins[5]["default"]))
    check("Extract tile_size is exposed with a sane range",
          (ins[6]["default"], ins[6]["min"], ins[6]["max"]) == (16, 2, 512),
          str((ins[6]["default"], ins[6]["min"], ins[6]["max"])))
    check("Extract tile_size is an advanced (collapsed) input", ins[6]["advanced"] is True,
          str(ins[6]["advanced"]))
    check("Extract colors range is 8..256", (ins[1]["min"], ins[1]["max"]) == (8, 256),
          f"{ins[1]['min']}..{ins[1]['max']}")
    check("Extract method default is Per-Tile k-means", ins[2]["default"] == "Per-Tile k-means",
          str(ins[2]["default"]))
    check("Extract method options are the 5 real extractors",
          ins[2]["options"] == ["Per-Tile k-means", "Global k-means (Lab)", "Median Cut",
                                "Max Coverage", "Octree"], str(ins[2]["options"]))
    check("Extract space options are Lab/RGB", ins[3]["options"] == ["Lab", "RGB"])
    check("Extract sort options", ins[4]["options"] == ["Luminance", "Hue", "Frequency"])
    check("Extract outputs are (palette STRING, palette_image IMAGE, count INT)",
          outs == [("palette", "STRING"), ("palette_image", "IMAGE"), ("count", "INT")], str(outs))
    check("nothing on Extract is optional or lazy",
          not any(r["optional"] or r["lazy"] for r in ins))

    quantize = by_name["PixelPaletteQuantize"].GET_SCHEMA()
    qins, qouts = _schema_rows(quantize)
    print(f"\n  --- {quantize.node_id} ---")
    for row in qins:
        flags = [f for f in ("optional", "lazy") if row[f]]
        print(f"      in  {row['id']:<16} {row['type']:<7} default={row['default']!r} "
              f"min={row['min']} max={row['max']} options={row['options']} {flags}")
    for name, io_type in qouts:
        print(f"      out {name:<16} {io_type}")
    check("Quantize input ids are exactly the contract",
          [r["id"] for r in qins] == ["image", "method", "colors", "tile_size", "space",
                                     "dither", "accent_protect", "palette"],
          str([r["id"] for r in qins]))
    check("Quantize input types",
          [r["type"] for r in qins] == ["IMAGE", "COMBO", "INT", "INT", "COMBO", "COMBO",
                                        "BOOLEAN", "STRING"],
          str([r["type"] for r in qins]))
    check("Quantize defaults: method=Per-Tile, colors=128, tile_size=16, space=Lab, "
          "dither=None, accent_protect=True",
          (qins[1]["default"], qins[2]["default"], qins[3]["default"], qins[4]["default"],
           qins[5]["default"], qins[6]["default"])
          == ("Per-Tile k-means", 128, 16, "Lab", "None", True),
          str([qins[i]["default"] for i in (1, 2, 3, 4, 5, 6)]))
    check("Quantize method options include Fixed Palette (nearest)",
          qins[1]["options"] == ["Per-Tile k-means", "Global k-means (Lab)", "Median Cut",
                                 "Max Coverage", "Octree", "Fixed Palette (nearest)"],
          str(qins[1]["options"]))
    check("Quantize dither options",
          qins[5]["options"] == ["None", "Bayer 2x2", "Bayer 4x4", "Bayer 8x8", "Floyd-Steinberg"],
          str(qins[5]["options"]))
    check("Quantize outputs are (image IMAGE, applied_palette STRING, colors_used INT)",
          qouts == [("image", "IMAGE"), ("applied_palette", "STRING"), ("colors_used", "INT")],
          str(qouts))

    palette_in = qins[7]
    check("palette input is optional", palette_in["optional"] is True,
          f"optional={palette_in['optional']}")
    check("palette input is lazy", palette_in["lazy"] is True, f"lazy={palette_in['lazy']}")
    check("palette input is a STRING carrying a text default",
          palette_in["type"] == "STRING" and palette_in["default"] == "",
          f"{palette_in['type']} default={palette_in['default']!r}")
    check("palette is the only optional/lazy input on Quantize",
          [r["id"] for r in qins if r["optional"] or r["lazy"]] == ["palette"],
          str([r["id"] for r in qins if r["optional"] or r["lazy"]]))


# --------------------------------------------------------------------------- #
# 5-10. palette behaviour
# --------------------------------------------------------------------------- #
def _to_u8(tensor) -> "np.ndarray":
    """ComfyUI IMAGE tensor -> uint8 ``[B, H, W, 3]``."""
    import numpy as np

    return (tensor.detach().cpu().numpy() * 255.0).round().astype(np.uint8)


def _pack(rgb) -> "np.ndarray":
    """``[..., 3]`` uint8 -> packed ``0xRRGGBB`` ints, for set arithmetic."""
    import numpy as np

    rgb = np.asarray(rgb, dtype=np.uint32).reshape(-1, 3)
    return (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]


def _save_rgb(path: Path, rgb) -> None:
    from PIL import Image

    Image.fromarray(rgb, "RGB").save(path)
    print(f"      wrote {path.name} ({rgb.shape[1]}x{rgb.shape[0]})")


def section_palette_behaviour(nodes: list, out_dir: Path) -> None:
    import numpy as np
    import torch

    from pixelpipe.nodes import _palette_utils as pu
    from pixelpipe.nodes.palette import PixelPaletteExtract, PixelPaletteQuantize

    by_name = {c.__name__: c for c in nodes}

    print("\n=== 5. palette extraction ===")
    a_path = out_dir / "fox_pixel_GRID.png"
    b_path = out_dir / "fox_source_00001_.png"
    a_img = _load_image(a_path)
    b_img = _load_image(b_path)
    print(f"  A = {a_path.name} {tuple(a_img.shape)}")
    print(f"  B = {b_path.name} {tuple(b_img.shape)}")

    hex_a, swatch_a, count_a = PixelPaletteExtract.execute(
        image=a_img, colors=32, method="Per-Tile k-means", space="Lab", sort="Luminance"
    ).result
    parsed_a = pu.parse_palette_hex(hex_a)
    print(f"  extracted {count_a} colours from A; HEX lines = {len(hex_a.splitlines())}")
    print("  first 4 lines:", hex_a.splitlines()[:4])
    check("A: count == 32", count_a == 32, f"got {count_a}")
    check("A: HEX has one line per colour", len(hex_a.splitlines()) == count_a,
          f"{len(hex_a.splitlines())} lines for {count_a} colours")
    check("A: HEX round-trips to exactly 32 RGB triples",
          parsed_a.shape == (32, 3) and parsed_a.dtype == np.uint8,
          f"shape={parsed_a.shape} dtype={parsed_a.dtype}")
    check("A: round-tripped values are unique", len(np.unique(parsed_a, axis=0)) == 32)
    check("A: HEX parses back byte-identically",
          np.array_equal(parsed_a, pu.parse_palette_hex(pu.format_palette_hex(parsed_a))))
    check("A: swatch strip is 16 blocks wide x 2 rows",
          tuple(swatch_a.shape) == (1, 32, 256, 3), f"got {tuple(swatch_a.shape)}")
    check("A: Luminance sort is non-decreasing",
          bool(np.all(np.diff(parsed_a.astype(np.int32) @ np.array([2126, 7152, 722])) >= 0)))
    _save_rgb(out_dir / "pixel_verify_A_swatch.png",
              _to_u8(swatch_a)[0])

    print("\n  parser robustness")
    want = np.array([[0x10, 0x20, 0x30], [0x40, 0x50, 0x60]], dtype=np.uint8)
    variants = {
        "newline+#": "#102030\n#405060\n",
        "commas": "#102030, #405060",
        "semicolons": "#102030;#405060",
        "no hash": "102030 405060",
        "0x prefix": "0x102030\n0x405060",
        "trailing blank lines": "\n#102030\n\n#405060\n\n",
    }
    for label, text in variants.items():
        check(f"parser accepts {label}", np.array_equal(pu.parse_palette_hex(text), want),
              repr(text))
    check("parser accepts the 3-digit short form (#123 -> #112233)",
          np.array_equal(pu.parse_palette_hex("#123"),
                         np.array([[0x11, 0x22, 0x33]], dtype=np.uint8)))
    check("parser accepts the 3-digit short form without a hash",
          np.array_equal(pu.parse_palette_hex("abc"),
                         np.array([[0xAA, 0xBB, 0xCC]], dtype=np.uint8)))
    check("parser accepts lowercase", np.array_equal(pu.parse_palette_hex("#102030".lower()), want[:1]))
    check("parser de-duplicates",
          pu.parse_palette_hex("#102030\n#102030\n102030").shape == (1, 3))
    garbage = None
    try:
        pu.parse_palette_hex("#102030\nnot-a-colour\n")
    except ValueError as exc:
        garbage = exc
    check("parser rejects garbage with a ValueError naming the token",
          garbage is not None and "not-a-colour" in str(garbage))

    print("\n=== 6. cross-image consistency (A's palette onto B) ===")
    out_b, applied_b, used_b = PixelPaletteQuantize.execute(
        image=b_img, palette=hex_a, space="Lab", dither="None"
    ).result
    b_rgb = _to_u8(out_b)[0]
    pal_set = set(_pack(parsed_a).tolist())
    b_used = set(_pack(b_rgb).tolist())
    print(f"  B quantized with A's palette: colors_used={used_b}, "
          f"distinct colours in output={len(b_used)}")
    print(f"  |output| = {len(b_used)}, |palette A| = {len(pal_set)}")
    check("B's output colour set is a subset of A's palette", b_used <= pal_set,
          f"{len(b_used - pal_set)} colour(s) outside: "
          f"{sorted(f'#{v:06X}' for v in (b_used - pal_set))[:5]}")
    check("B: colors_used matches the distinct colours in the output",
          used_b == len(b_used), f"reported {used_b}, counted {len(b_used)}")
    check("B: applied_palette is A's palette verbatim",
          pu.parse_palette_hex(applied_b).shape == parsed_a.shape
          and set(_pack(pu.parse_palette_hex(applied_b)).tolist()) == pal_set)
    check("B: every output pixel is a palette colour (no interpolation)", len(b_used) <= 32)
    _save_rgb(out_dir / "pixel_verify_B_from_A.png", b_rgb)

    # B's own palette, for contrast: a different 32 colours from the same family.
    hex_b, swatch_b, count_b = PixelPaletteExtract.execute(
        image=b_img, colors=32, method="Per-Tile k-means", space="Lab", sort="Luminance"
    ).result
    b_set = set(_pack(pu.parse_palette_hex(hex_b)).tolist())
    print(f"  B's own palette: {count_b} colours, "
          f"{len(b_set & pal_set)} shared with A, {len(b_set - pal_set)} unique to B")
    _save_rgb(out_dir / "pixel_verify_B_swatch.png", _to_u8(swatch_b)[0])

    print("\n=== 7. every method runs ===")
    quant_results = {}
    for method in pu.QUANTIZE_METHODS:
        args = dict(image=a_img, method=method, colors=32, tile_size=16, space="Lab")
        if method == pu.FIXED_PALETTE:
            # Designed to raise without a palette -- see the check right below.
            args["palette"] = hex_a
        try:
            _o, _ap, used = PixelPaletteQuantize.execute(**args).result
        except Exception as exc:  # noqa: BLE001
            check(f"Quantize method {method!r} runs", False, f"{type(exc).__name__}: {exc}")
            continue
        quant_results[method] = used
        print(f"  [quantize] {method:<26} colors_used={used}")
        check(f"Quantize method {method!r} runs", True, f"colors_used={used}")
        check(f"Quantize method {method!r} stays within colors=32", used <= 32,
              f"got {used}")

    raised = None
    try:
        PixelPaletteQuantize.execute(image=a_img, method=pu.FIXED_PALETTE, colors=32)
    except ValueError as exc:
        raised = exc
    check("Fixed Palette without a palette raises ValueError naming the input",
          raised is not None and "palette" in str(raised),
          type(raised).__name__ if raised else "no exception")

    # Both accent_protect modes are pinned, because the default changed the
    # answer: accents are paid for out of the colour budget, so switching
    # protection on can legitimately return fewer than `colors` entries (Max
    # Coverage drops to 28 of 32). Off, the count is exact for every method;
    # on, it must stay within the documented 8..colors window.
    for method in pu.EXTRACT_METHODS:
        row = []
        for protect in (False, True):
            try:
                _h, _s, count = PixelPaletteExtract.execute(
                    image=a_img, colors=32, method=method, space="Lab", sort="Luminance",
                    accent_protect=protect, tile_size=16,
                ).result
            except Exception as exc:  # noqa: BLE001
                check(f"Extract method {method!r} (accent_protect={protect}) runs", False,
                      f"{type(exc).__name__}: {exc}")
                row.append(None)
                continue
            check(f"Extract method {method!r} (accent_protect={protect}) runs", True,
                  f"count={count}")
            row.append(count)
        print(f"  [extract ] {method:<26} count off/on = {row[0]} / {row[1]}")
        if row[0] is not None:
            check(f"Extract method {method!r} with accent_protect=False returns exactly 32",
                  row[0] == 32, f"got {row[0]}")
        if row[1] is not None:
            check(f"Extract method {method!r} with accent_protect=True stays within 8..32",
                  8 <= row[1] <= 32, f"got {row[1]}")
    check("Extract does not offer Fixed Palette (it cannot extract one)",
          pu.FIXED_PALETTE not in pu.EXTRACT_METHODS)

    print("\n=== 8. dithering ===")
    # Every dither variant is compared against the palette the *quantizer* used
    # (accent_protect is on there by default, so it is not the same 32 colours as
    # the extractor's), which is the only meaningful reference.
    variants_out = {}
    dither_pal = None
    for mode in ("None", "Bayer 2x2", "Bayer 4x4", "Bayer 8x8", "Floyd-Steinberg"):
        out, applied, used = PixelPaletteQuantize.execute(
            image=a_img, method="Per-Tile k-means", colors=32, tile_size=16, space="Lab",
            dither=mode,
        ).result
        rgb = _to_u8(out)[0]
        variants_out[mode] = rgb
        run_pal = set(_pack(pu.parse_palette_hex(applied)).tolist())
        if dither_pal is None:
            dither_pal = run_pal
        used_set = set(_pack(rgb).tolist())
        print(f"  [dither] {mode:<18} colors_used={used:<4} "
              f"palette_size={len(run_pal):<4} outside_palette={len(used_set - run_pal)}")
        check(f"dither {mode!r} introduces no colour outside its palette",
              used_set <= run_pal,
              f"{len(used_set - run_pal)} new: "
              f"{sorted(f'#{v:06X}' for v in (used_set - run_pal))[:5]}")
        check(f"dither {mode!r} uses the same palette as the others (settings unchanged)",
              run_pal == dither_pal)

    diff_bayer = int(np.count_nonzero(
        variants_out["Bayer 4x4"].astype(np.int16) - variants_out["None"].astype(np.int16)))
    diff_fs = int(np.count_nonzero(
        variants_out["Floyd-Steinberg"].astype(np.int16)
        - variants_out["None"].astype(np.int16)))
    print(f"  pixels changed by Bayer 4x4 vs None          : {diff_bayer}")
    print(f"  pixels changed by Floyd-Steinberg vs None   : {diff_fs}")
    check("Bayer 4x4 actually dithers (some pixels changed)", diff_bayer > 0,
          f"{diff_bayer} channel values changed")
    check("Floyd-Steinberg actually dithers", diff_fs > 0, f"{diff_fs} channel values changed")
    check("Floyd-Steinberg output also stays inside the palette "
          "(Pillow emits palette indices, so it cannot invent colours here)",
          set(_pack(variants_out["Floyd-Steinberg"]).tolist()) <= dither_pal)

    # The claim in the tooltip: Bayer depends on pixel coordinates only.
    off_small = pu.bayer_offsets(64, 64, 4, 1.0)
    off_large = pu.bayer_offsets(300, 500, 4, 1.0)
    check("Bayer thresholds are a function of (y % n, x % n) only",
          bool(np.array_equal(off_small, off_large[:64, :64])))
    check("Bayer 2x2 matches the canonical matrix",
          bool(np.allclose(pu._BAYER_CACHE[2] * 4, [[0, 2], [3, 1]])))
    check("Bayer 4x4 matches the canonical matrix",
          bool(np.allclose(pu._BAYER_CACHE[4] * 16,
                           [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]])))
    check("Bayer 8x8 is the recursive doubling of 4x4",
          bool(np.allclose(pu._BAYER_CACHE[8] * 64,
                           np.block([[4 * pu._BAYER_CACHE[4] * 16, 4 * pu._BAYER_CACHE[4] * 16 + 2],
                                     [4 * pu._BAYER_CACHE[4] * 16 + 3,
                                      4 * pu._BAYER_CACHE[4] * 16 + 1]]))))

    _save_rgb(out_dir / "pixel_verify_A_quant_none.png", variants_out["None"])
    _save_rgb(out_dir / "pixel_verify_A_quant_bayer4.png", variants_out["Bayer 4x4"])
    _save_rgb(out_dir / "pixel_verify_A_quant_fs.png", variants_out["Floyd-Steinberg"])
    a_rgb = _to_u8(a_img)[0]
    sheet = np.concatenate(
        [a_rgb, variants_out["None"], variants_out["Bayer 4x4"], variants_out["Floyd-Steinberg"]],
        axis=1,
    )
    print("      compare sheet = [A original | no dither | Bayer 4x4 | Floyd-Steinberg]")
    _save_rgb(out_dir / "pixel_verify_A_compare.png", sheet)

    # A 2x-zoomed crop of the same four panels: the dither textures are only
    # readable at 1:1 or above, and this is the artefact a human actually checks.
    from PIL import Image as _PILImage

    y0, x0, size = 260, 380, 260
    panels = [sheet[y0 : y0 + size, x0 + i * sheet.shape[1] // 4 : x0 + i * sheet.shape[1] // 4 + size]
              for i in range(4)]
    sep = np.full((size, 4, 3), 255, dtype=np.uint8)
    strip = np.concatenate([np.concatenate([p, sep], axis=1) for p in panels], axis=1)
    zoom = np.array(
        _PILImage.fromarray(strip).resize((strip.shape[1] * 2, strip.shape[0] * 2), _PILImage.NEAREST)
    )
    _save_rgb(out_dir / "pixel_verify_A_crop.png", zoom)
    deep = int(np.count_nonzero(
        variants_out["Bayer 4x4"].astype(np.int16) - variants_out["None"].astype(np.int16)
    ))
    check("Bayer dithering changes a substantial number of pixels (visible, not incidental)",
          deep > 0.1 * variants_out["None"].size, f"{deep} of {variants_out['None'].size}")

    _save_rgb(
        out_dir / "pixel_verify_swatches.png",
        np.concatenate(
            [_to_u8(swatch_a)[0], np.full((2, 256, 3), 255, np.uint8), _to_u8(swatch_b)[0]], axis=0
        ),
    )

    print("\n=== 9. lazy pruning (the real graph pruner) ===")
    _section_lazy(by_name=by_name, hex_a=hex_a)

    print("\n=== 10. determinism ===")
    hex_a2, _sw2, count_a2 = PixelPaletteExtract.execute(
        image=a_img, colors=32, method="Per-Tile k-means", space="Lab", sort="Luminance"
    ).result
    check("Extract: identical HEX on a second run", hex_a == hex_a2,
          "differs" if hex_a != hex_a2 else "")
    check("Extract: identical count on a second run", count_a == count_a2)

    q1, ap1, u1 = PixelPaletteQuantize.execute(
        image=a_img, method="Per-Tile k-means", colors=32, tile_size=16, space="Lab",
        dither="Bayer 4x4",
    ).result
    q2, ap2, u2 = PixelPaletteQuantize.execute(
        image=a_img, method="Per-Tile k-means", colors=32, tile_size=16, space="Lab",
        dither="Bayer 4x4",
    ).result
    check("Quantize: applied_palette is byte-identical on a second run", ap1 == ap2)
    check("Quantize: output tensor is byte-identical on a second run",
          bool(torch.equal(q1, q2)), f"max|d| = {float((q1 - q2).abs().max())}")
    check("Quantize: colors_used is identical on a second run", u1 == u2)
    check("k-means seed is a hard-coded constant",
          pu.KMEANS_SEED == 0x5EED_1234, hex(pu.KMEANS_SEED))

    # k-means in both spaces, and the PIL methods, must all be reproducible too.
    for method in pu.EXTRACT_METHODS:
        for space in ("Lab", "RGB"):
            h1 = PixelPaletteExtract.execute(
                image=b_img, colors=16, method=method, space=space, sort="Hue"
            ).result[0]
            h2 = PixelPaletteExtract.execute(
                image=b_img, colors=16, method=method, space=space, sort="Hue"
            ).result[0]
            check(f"deterministic: {method} / {space} duplicate run matches", h1 == h2)

    print("\n  batch behaviour (one shared palette for both frames)")
    batch = torch.cat([a_img, b_img[:, :976, :1000, :]], dim=0)
    _bo, bap, bused = PixelPaletteQuantize.execute(
        image=batch, method="Per-Tile k-means", colors=32, tile_size=16, space="Lab"
    ).result
    bset = set(_pack(_to_u8(_bo)).tolist())
    print(f"  batch of {tuple(batch.shape)} -> {bused} colours, shared palette "
          f"{len(pu.parse_palette_hex(bap))} entries")
    check("a batch produces one shared palette for every frame",
          bset <= set(_pack(pu.parse_palette_hex(bap)).tolist()),
          f"{len(bset - set(_pack(pu.parse_palette_hex(bap)).tolist()))} outside")


def _section_lazy(by_name: dict, hex_a: str) -> None:
    """Drive the real ``TopologicalSort`` -- no ComfyUI server required.

    ``comfy_execution/graph.py:169-172`` skips the upstream node of a *lazy*
    input when it builds the initial execution block. That block *is* the
    execution plan, so if ``PixelPaletteExtract`` is absent from it, it is
    provably not executed -- which is a stronger statement than watching a
    counter stay at zero.
    """
    import comfy_execution.graph as graph_mod
    import nodes as comfy_nodes

    quantize = by_name["PixelPaletteQuantize"]
    extract = by_name["PixelPaletteExtract"]

    # The pruner looks node types up in the global registry.
    for cls in (by_name["PixelGridRestore"], extract, quantize, by_name["PixelExportScale"]):
        comfy_nodes.NODE_CLASS_MAPPINGS.setdefault(cls.GET_SCHEMA().node_id, cls)

    def build(palette_value):
        return {
            "1": {
                # A non-link placeholder: the pruner only follows links, and this
                # keeps the test independent of whatever loader nodes happen to be
                # registered. Node 1 is the root of the graph here.
                "class_type": "PixelGridRestore",
                "inputs": {"image": None, "sampling": "Median",
                           "grid_mode": "By Grid Count", "grid_w": 125, "grid_h": 122},
            },
            "2": {
                "class_type": "PixelPaletteExtract",
                "inputs": {"image": ["1", 0], "colors": 32, "method": "Per-Tile k-means",
                           "space": "Lab", "sort": "Luminance"},
            },
            "3": {
                "class_type": "PixelPaletteQuantize",
                "inputs": {"image": ["1", 0], "method": "Per-Tile k-means", "colors": 32,
                           "tile_size": 16, "space": "Lab", "dither": "None",
                           "accent_protect": True, "palette": palette_value},
            },
            "4": {
                "class_type": "PixelExportScale",
                "inputs": {"image": ["3", 0], "scale": 4, "mode": "Nearest"},
            },
        }

    def plan(palette_value):
        topo = graph_mod.TopologicalSort(graph_mod.DynamicPrompt(build(palette_value)))
        topo.add_node("4")
        return topo

    unwired = plan("")  # untouched widget -> a constant, not a link
    wired = plan(["2", 0])  # real link to the extract node

    print(f"  initial block, palette unwired : {sorted(unwired.pendingNodes)}")
    print(f"  initial block, palette wired   : {sorted(wired.pendingNodes)}")
    check("a lazy link is held out of the initial block whether or not it is wired",
          sorted(unwired.pendingNodes) == ["1", "3", "4"]
          and sorted(wired.pendingNodes) == ["1", "3", "4"],
          f"{sorted(unwired.pendingNodes)} / {sorted(wired.pendingNodes)}")
    check("palette unwired: PixelPaletteExtract (node 2) never enters the graph",
          "2" not in unwired.pendingNodes, f"block = {sorted(unwired.pendingNodes)}")

    # Stage two: this is exactly what execution.py does when check_lazy_status
    # answers ["palette"] on the re-queue (execution.py:517-520).
    wired.make_input_strong_link("3", "palette")
    print(f"  after check_lazy_status asks for 'palette' : {sorted(wired.pendingNodes)}")
    check("palette wired: the request promotes it to a strong link and pulls node 2 in",
          "2" in wired.pendingNodes, f"block = {sorted(wired.pendingNodes)}")

    # And the framework refuses the mirror-image mistake: asking for an unwired
    # widget raises, because a constant cannot be promoted to a link. This is the
    # reason check_lazy_status must test for None and not for falsiness.
    refused = None
    try:
        unwired.make_input_strong_link("3", "palette")
    except Exception as exc:  # noqa: BLE001 -- raising is the assertion
        refused = exc
    print(f"  requesting the unwired widget directly      : "
          f"{type(refused).__name__ if refused else 'NO ERROR'}: {refused}")
    check("requesting an unwired palette is refused by the framework (it is a constant)",
          refused is not None and "constant" in str(refused),
          str(refused) if refused else "no exception")

    # The check_lazy_status contract that makes the above work.
    check("check_lazy_status asks for palette only when it is wired-but-unevaluated",
          quantize.check_lazy_status(palette=None) == ["palette"],
          str(quantize.check_lazy_status(palette=None)))
    check("check_lazy_status is silent when palette is unconnected (MISSING)",
          quantize.check_lazy_status() == [],
          str(quantize.check_lazy_status()))
    check("check_lazy_status is silent when the caller omitted the key entirely",
          quantize.check_lazy_status(image=None, method="Per-Tile k-means") == [])
    check("check_lazy_status is silent for an untouched empty widget",
          quantize.check_lazy_status(palette="") == [])
    check("check_lazy_status is silent once a palette value arrived",
          quantize.check_lazy_status(palette=hex_a) == [])
    check("validate_inputs runs before check_lazy_status and tolerates palette=None",
          quantize.validate_inputs(palette=None, method="Per-Tile k-means", colors=32) is True)
    check("validate_inputs rejects an out-of-range colors",
          isinstance(quantize.validate_inputs(colors=999), str))
    check("validate_inputs rejects an unknown method",
          isinstance(quantize.validate_inputs(method="Nope"), str))


# --------------------------------------------------------------------------- #
def _rgba_test_image(size: int = 512, cell: int = 8):
    """A synthetic pixel-art image plus a real alpha mask, as ``[1,H,W,4]``.

    The colour half is a regular grid of distinct colours so grid detection has
    something genuine to find; the alpha half is opaque only in a centred square.
    A correct implementation therefore has to produce alpha 0 in the surrounding
    cells and 255 inside -- which is only possible if alpha is resampled onto the
    *same* grid lines as the colour.

    Returns ``(tensor, inset, cell)`` where ``inset`` is the pixel offset of the
    opaque square from each edge.
    """
    import numpy as np
    import torch

    rgb = np.zeros((size, size, 3), np.uint8)
    for y in range(0, size, cell):
        for x in range(0, size, cell):
            rgb[y : y + cell, x : x + cell] = (
                ((x // cell) * 37) % 256,
                ((y // cell) * 53) % 256,
                128,
            )

    alpha = np.zeros((size, size, 1), np.uint8)
    inset = size // 4
    alpha[inset : size - inset, inset : size - inset] = 255

    arr = np.concatenate([rgb, alpha], axis=-1)
    t = torch.from_numpy(arr).float().unsqueeze(0) / 255.0
    return t, inset, cell


def section_alpha(nodes: list) -> None:
    """RGBA coverage -- the gap that let a total-failure bug ship.

    Qwen-Image-2.1's VAE carries four channels so it can generate and edit
    transparent images natively, which means ``VAEDecode`` hands this pipeline
    RGBA on *every* run, opaque or not. The rest of this file only ever used
    three-channel PNGs, so nothing here caught the nodes rejecting four.
    """
    print("\n=== 11. alpha / RGBA (Qwen-Image-2.1's VAE emits 4 channels) ===")
    import numpy as np

    by_name = {n.GET_SCHEMA().node_id: n for n in nodes}
    grid = by_name["PixelGridRestore"]
    quant = by_name["PixelPaletteQuantize"]
    extract = by_name["PixelPaletteExtract"]
    scale = by_name["PixelExportScale"]

    rgba, inset, cell = _rgba_test_image()
    print(f"      input {tuple(rgba.shape)}, opaque square inset {inset}px, cell {cell}px")

    # -- 11a. RGBA must not raise, and alpha must survive to a 4-channel output
    try:
        out, gw, gh, px, det, conf = grid.execute(
            image=rgba, sampling="Majority Cluster", grid_mode="Auto Detect"
        ).result
        raised = None
    except Exception as exc:  # noqa: BLE001
        raised, out = exc, None

    check("11a RGBA input does not raise", raised is None,
          f"{type(raised).__name__}: {raised}" if raised else "")
    if out is None:
        return  # nothing further is testable

    out_u8 = _to_u8(out)
    print(f"      output {out_u8.shape}  grid {gw}x{gh}  confidence {conf:.3f}")
    check("11a RGBA output keeps 4 channels", out_u8.shape[-1] == 4,
          f"got {out_u8.shape}")
    check("11a grid is the synthetic 64x64", (gw, gh) == (64, 64), f"got {gw}x{gh}")

    if out_u8.shape[-1] == 4:
        a = out_u8[0, :, :, 3]
        k0, k1 = inset // cell, (rgba.shape[2] - inset) // cell
        # Two cells of margin: the mask edge lands on a grid line, and the
        # outermost cells are the ones a mis-aligned resample would smear.
        inner = a[k0 + 2 : k1 - 2, k0 + 2 : k1 - 2]
        above = a[: k0 - 2, :]
        left = a[:, : k0 - 2]
        check("11a alpha is opaque inside the mask", int(inner.min()) == 255,
              f"min={int(inner.min())}")
        check("11a alpha is transparent above the mask", int(above.max()) == 0,
              f"max={int(above.max())}")
        check("11a alpha is transparent left of the mask", int(left.max()) == 0,
              f"max={int(left.max())}")
        # The decisive one: if alpha had been resampled on its own grid rather
        # than the colour's, the mask edge would land off the cell boundary and
        # these cells would be partially transparent.
        check("11a mask edge lands exactly on the grid (no partial cells)",
              set(np.unique(a).tolist()) <= {0, 255},
              f"distinct alpha values: {sorted(np.unique(a).tolist())[:8]}")

    # -- 11b. an all-opaque alpha must be dropped, not propagated
    opaque = rgba.clone()
    opaque[..., 3] = 1.0
    out_b, *_rest = grid.execute(
        image=opaque, sampling="Majority Cluster", grid_mode="Auto Detect"
    ).result
    shape_b = tuple(_to_u8(out_b).shape)
    print(f"      all-opaque input -> {shape_b}")
    check("11b all-opaque alpha is dropped (RGB out)", len(shape_b) == 4 and shape_b[-1] == 3,
          f"got {shape_b}")

    # -- 11c. Quantize must pass alpha through byte-for-byte
    q_out, _q_pal, _q_used = quant.execute(
        image=out, method="Per-Tile k-means", colors=32, tile_size=cell,
        space="Lab", dither="None", accent_protect=True, palette="",
    ).result
    q_u8 = _to_u8(q_out)
    check("11c Quantize keeps 4 channels", q_u8.shape[-1] == 4, f"got {q_u8.shape}")
    if q_u8.shape[-1] == 4:
        check("11c Quantize leaves alpha byte-identical",
              np.array_equal(q_u8[..., 3], out_u8[..., 3]), "alpha changed")

    # -- 11d. Extract accepts RGBA (and must ignore alpha when choosing colours)
    _h, _sw, count = extract.execute(
        image=out, colors=32, method="Per-Tile k-means", space="Lab",
        sort="Luminance", accent_protect=True, tile_size=cell,
    ).result
    check("11d Extract accepts RGBA", count > 0, f"count={count}")

    # -- 11e. ExportScale carries alpha through untouched
    sc, sw_w, sc_h = scale.execute(image=out, scale=4, mode="Nearest").result
    sc_shape = tuple(sc.shape)
    print(f"      ExportScale -> {sc_shape} ({sw_w}x{sc_h})")
    check("11e ExportScale keeps 4 channels", sc_shape[-1] == 4, f"got {sc_shape}")
    check("11e ExportScale quadruples the grid", (sc_h, sw_w) == (gh * 4, gw * 4),
          f"got {sw_w}x{sc_h}")


# --------------------------------------------------------------------------- #
def main() -> int:
    # The failure message is bilingual; force UTF-8 so a cp936 console does not
    # mangle it into question marks.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("ComfyUI-PixelPipeline selftest")
    print(f"  package : {ROOT}")
    print(f"  python  : {sys.version.split()[0]} ({sys.executable})")

    section_syntax()
    try:
        module = load_package()
    except Exception:
        traceback.print_exc()
        print("\n!! package failed to import -- aborting")
        return 1
    nodes = section_schema(module)
    section_behaviour(nodes, DEFAULT_BASELINE)
    section_palette_schema(nodes)
    try:
        section_palette_behaviour(nodes, DEFAULT_BASELINE.parent)
    except Exception:
        traceback.print_exc()
        _fail("palette behaviour section crashed", "see traceback above")
    try:
        section_alpha(nodes)
    except Exception:
        traceback.print_exc()
        _fail("alpha/RGBA section crashed", "see traceback above")

    print("\n=== summary ===")
    print(f"  checks: {CHECKS}  failures: {len(FAILURES)}")
    for name in FAILURES:
        print(f"    - {name}")
    print("  RESULT: " + ("ALL PASS" if not FAILURES else "FAILED"))
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    raise SystemExit(main())
