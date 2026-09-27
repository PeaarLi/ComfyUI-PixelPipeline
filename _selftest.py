"""Standalone verification for ComfyUI-PixelPipeline. No ComfyUI server needed.

Run with the ComfyUI venv interpreter, from this directory:

    python _selftest.py

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
3f. **Grid schema** -- the ``grid_mode`` DynamicCombo: that the old flat inputs
   are gone, that ``Auto Detect`` is the FIRST option (``DynamicCombo.Input`` has
   no ``default`` kwarg, so option order is the only lever), that no nested input
   is optional/lazy/advanced, and that every schema default equals its module
   constant.
3g. **Grid runtime contract** -- the shape ``execute`` actually receives for a
   DynamicCombo, established by calling the same ``comfy_api`` functions
   ``execution.py`` calls rather than by trusting a hand-written dict. Also
   covers a missing ``grid_mode`` (behaves as Auto Detect) and a bogus one.
3h. **``max_pixel_size`` port** -- the ported detector at the default ceiling must
   match upstream ``detect_grid_scale`` on a corpus spanning every branch
   (FFT accepted / gate-rejected / both detectors failed), and must resolve 32 px
   cells that the default rejects. The 24 px aliasing case is pinned as a
   *documented* quiet-wrong-answer, not a bug.
4. **Palette schema** -- input/output names, types, defaults and ranges of both
   palette nodes, including that ``PixelPaletteApply`` carries no derive-path
   inputs and does not override ``check_lazy_status``.
4b. **zh locale keys** -- the Chinese translation keys must equal what the
   frontend will look up. Nested widgets are keyed ``<parent>_<nested>`` because
   ``normalizeI18nKey`` turns the ``.`` in ``"<parent>.<nested>"`` into ``_``, and
   a stale key set shows up as *blank* tooltips in the Vue renderer rather than
   as an English fallback.
5. **Palette extraction** -- 32 colours from ``fox_pixel_GRID.png``, and the HEX
   text parses back to exactly 32 distinct RGB triples.
6. **Cross-image consistency** -- a palette taken from image A and applied to
   image B must leave B using a subset of A's colours.
7. **Apply, both palette sources** -- a palette wired from Extract and a pasted
   HEX literal must both work, ``applied_palette`` must come back verbatim with
   order preserved (not re-sorted), and an empty palette must raise.
8. **Dithering** -- ``None`` vs ``Bayer 4x4`` vs ``Floyd-Steinberg``; every
   dithered output must stay inside the palette, and Bayer thresholds must be a
   pure function of pixel coordinates (so they match across image sizes).
9. **Graph dependency** -- the real ``comfy_execution.graph.TopologicalSort`` must
   put ``PixelPaletteExtract`` in the *initial* execution block when the palette
   is wired. This replaced a lazy-pruning check: the palette input is now an
   ordinary required one, so the property worth defending is the mirror image.
10. **Determinism** -- the same call twice must produce byte-identical output.
9b. **Apply's validation through the real framework** -- drives
   ``execution.validate_inputs`` for a wired / typed / empty palette. Only this
   level of fidelity catches the bug it exists for: validation runs *before*
   execution, so a linked input arrives as ``None``, and only the real call
   reproduces that. It also pins that an empty palette yields exactly ONE error,
   attributed to ``palette``.
11. **Alpha / RGBA** -- a synthetic RGBA clip through the grid, quantise and
   export stages: alpha must survive byte-identically and a hard mask must land
   on the grid lines.
12. **Video node schema** -- ``PixelVideoGridRestore``'s inputs, outputs, the
   grid-mode default lever, and video-only sampling choices.
13. **Video node runtime contract** -- its ``grid_mode`` driven through
   ``comfy_api``'s own machinery, the same way as the still node's.
14. **Video node behaviour** -- entirely from synthesised tensors; **no video
   file is needed**, because ``VideoFromComponents`` builds from materialised
   tensors and even encodes into an in-memory ``BytesIO``. Covers: one grid for
   the whole clip with no raise on frame disagreement; the frame rate surviving
   as an exact ``Fraction``; VIDEO images being RGB with alpha in its own slot;
   frame identity, moving edges without trails, current-frame colour support,
   static noise reduction, alpha from either input route and chunk invariance.
15. **Sprite node schema** -- ``PixelSpritePrepare``'s inputs/outputs and
   ``PixelSpriteSheet``'s ``layout`` DynamicCombo: ``Auto Square`` first (the
   default lever), ``columns`` only under ``Fixed Columns``, and schema defaults
   equal to the module constants.
16. **Sprite runtime contract** -- the ``layout`` combo driven through
   ``comfy_api``'s own machinery (including the zero-nested-option edge), plus
   ``_unpack_layout``'s None/bare-string shapes and the unknown-layout error.
17. **Sprite node behaviour** -- synthetic batches only: background detection
   and the shared square crop with hand-computed numbers, border-only flood
   keying (interior same-coloured pixels survive), the no-content fallback,
   RGBA composition, sheet layouts (auto / fixed / columns > n), row-major
   placement, empty trailing cells, channel preservation and determinism.

Comparison images are written to ``pixel_verify_*.png`` in ComfyUI's output
directory. No video file is written: the encode check uses an in-memory buffer.

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
    check("get_node_list() returns 7 nodes", len(node_list) == 7,
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


def _mode(mode: str, **params) -> dict:
    """The ``grid_mode`` value ComfyUI's build_nested_inputs() hands to execute.

    Shape verified against this build (see ``section_grid_contract``): the
    selected option's key arrives *inside* the input's own value, alongside that
    option's nested widget values.
    """
    return {"grid_mode": {"grid_mode": mode, **params}}


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
        image=img, sampling="Majority Cluster", **_mode("Auto Detect")
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
        image=img, sampling="Median", **_mode("By Pixel Size", pixel_size=16)
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
        image=img, sampling="Center Sample", **_mode("By Grid Count", grid_w=64, grid_h=64)
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
        result = grid.execute(image=smooth, sampling="Median", **_mode("Auto Detect"))
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
                    image=one, sampling="Median", **_mode("Auto Detect")
                ).result
                print(f"      image {tag} ({second.name if tag == 'b' else baseline.name}) -> {w1}x{h1}")
            batch = torch.cat([img, img_b], dim=0)
            try:
                grid.execute(image=batch, sampling="Median", **_mode("Auto Detect"))
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
            image=img, sampling="Median", backend=be, **_mode("Auto Detect")
        ).result
        print(f"      backend={be:<12} -> {w4}x{h4}")
        check(f"backend={be} matches the 125x122 baseline", (w4, h4) == (125, 122), f"got {w4}x{h4}")

    # The no-CV2 backend is broken for auto-detection upstream (bare None return
    # value unpacked as a tuple). Assert we turn that into a clean RuntimeError
    # rather than letting a raw TypeError escape with a foreign traceback.
    lw_raised = None
    try:
        grid.execute(image=img, sampling="Median", backend="Lightweight", **_mode("Auto Detect"))
    except Exception as exc:  # noqa: BLE001
        lw_raised = exc
        print(f"      backend=Lightweight Auto Detect raised {type(lw_raised).__name__}")
        print("      message : " + str(exc).replace("\n", " ")[:420])
    check("Lightweight auto-detect raises RuntimeError", isinstance(lw_raised, RuntimeError),
          type(lw_raised).__name__ if lw_raised else "no exception")
    # The pre-flight names the fix rather than letting upstream's bare-None/2-tuple
    # crash surface, so the message must carry the install hint.
    check("Lightweight error names the upstream no-CV2 bug",
          lw_raised is not None and "estimate_grid_fft" in str(lw_raised))
    check("Lightweight error carries the OpenCV install hint",
          lw_raised is not None and "perfect-pixel[opencv]" in str(lw_raised))

    _o, w5, h5, _p5, _d5, _c5 = grid.execute(
        image=img, sampling="Median", backend="Lightweight",
        **_mode("By Grid Count", grid_w=125, grid_h=122),
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
# 3f. the grid_mode DynamicCombo, and the max_pixel_size port
# --------------------------------------------------------------------------- #
def section_grid_schema(nodes: list) -> None:
    """The mode-aware control's contract, and the schema defaults it depends on."""
    from pixelpipe.nodes import grid_restore as gr

    print("\n=== 3f. PixelGridRestore schema (the grid_mode DynamicCombo) ===")
    by_name = {c.__name__: c for c in nodes}
    schema = by_name["PixelGridRestore"].GET_SCHEMA()
    ins, _outs = _schema_rows(schema)

    check("grid top-level input ids are exactly the contract",
          [r["id"] for r in ins] == ["image", "sampling", "grid_mode", "backend",
                                     "refine_intensity", "fix_square"],
          str([r["id"] for r in ins]))
    check("grid top-level input types",
          [r["type"] for r in ins] == ["IMAGE", "COMBO", "COMFY_DYNAMICCOMBO_V3", "COMBO",
                                       "FLOAT", "BOOLEAN"],
          str([r["type"] for r in ins]))
    check("the old flat grid inputs are gone",
          not ({"pixel_size", "grid_w", "grid_h", "min_size", "peak_width"}
               & {r["id"] for r in ins}),
          str([r["id"] for r in ins]))
    check("only refine_intensity and fix_square remain advanced",
          [r["id"] for r in ins if r["advanced"]] == ["refine_intensity", "fix_square"],
          str([r["id"] for r in ins if r["advanced"]]))

    opts = _dynamic_options(schema)
    check("there is exactly one DynamicCombo, with the three documented modes",
          list(opts) == ["grid_mode"]
          and list(opts["grid_mode"]) == ["Auto Detect", "By Pixel Size", "By Grid Count"],
          str(list(opts.get("grid_mode", {}))))
    # DynamicCombo.Input has no `default` kwarg, so a freshly added node shows
    # whatever the frontend picks; putting the intended default first is the only
    # lever we have. Pin it, or the node silently opens in a different mode.
    check("'Auto Detect' is the FIRST option (that is what makes it the default)",
          next(iter(opts["grid_mode"])) == "Auto Detect")

    auto = {r["id"]: r for r in opts["grid_mode"]["Auto Detect"]}
    check("Auto Detect carries exactly the parameters detection actually uses",
          list(auto) == ["min_size", "max_pixel_size", "peak_width"], str(list(auto)))
    check("max_pixel_size is exposed, defaults to upstream's own ceiling, and has room to rise",
          (auto["max_pixel_size"]["type"], auto["max_pixel_size"]["default"],
           auto["max_pixel_size"]["min"], auto["max_pixel_size"]["max"])
          == ("FLOAT", 20.0, 1.0, 512.0),
          str(auto["max_pixel_size"]))
    check("max_pixel_size default equals UPSTREAM_MAX_PIXEL_SIZE "
          "(the fast path's identity anchor)",
          auto["max_pixel_size"]["default"] == gr.UPSTREAM_MAX_PIXEL_SIZE,
          f"{auto['max_pixel_size']['default']} vs {gr.UPSTREAM_MAX_PIXEL_SIZE}")
    check("By Pixel Size carries only pixel_size",
          [r["id"] for r in opts["grid_mode"]["By Pixel Size"]] == ["pixel_size"],
          str([r["id"] for r in opts["grid_mode"]["By Pixel Size"]]))
    check("By Grid Count carries only grid_w and grid_h",
          [r["id"] for r in opts["grid_mode"]["By Grid Count"]] == ["grid_w", "grid_h"],
          str([r["id"] for r in opts["grid_mode"]["By Grid Count"]]))

    check("no nested input is optional, lazy, or advanced",
          not any(r["optional"] or r["lazy"] or r["advanced"]
                  for mode in opts["grid_mode"].values() for r in mode),
          str([(r["id"], r["optional"], r["lazy"], r["advanced"])
               for mode in opts["grid_mode"].values() for r in mode]))

    # define_schema and the execute-side unpacking read the same constants; if
    # they drift, a prompt that omits a key silently changes behaviour.
    defaults = {r["id"]: r["default"] for mode in opts["grid_mode"].values() for r in mode}
    for name, constant in (("min_size", gr.DEFAULT_MIN_SIZE),
                           ("max_pixel_size", gr.DEFAULT_MAX_PIXEL_SIZE),
                           ("peak_width", gr.DEFAULT_PEAK_WIDTH),
                           ("pixel_size", gr.DEFAULT_PIXEL_SIZE),
                           ("grid_w", gr.DEFAULT_GRID_W),
                           ("grid_h", gr.DEFAULT_GRID_H)):
        check(f"module default {name} matches the schema default",
              defaults[name] == constant, f"{defaults[name]} vs {constant}")


def section_grid_contract(nodes: list) -> None:
    """Drive ComfyUI's own machinery for the nested-input contract.

    The shape ``execute`` receives from a DynamicCombo is not documented; it was
    established by reading ``comfy_execution/execution.py`` and confirmed here by
    calling the two functions that file calls. Asserting against our own
    ``_mode()`` helper would be circular, so this goes through the real thing.
    """
    from comfy_api.latest import _io
    from pixelpipe.nodes import grid_restore as gr

    print("\n=== 3g. grid_mode runtime contract (via comfy_api internals) ===")
    by_name = {c.__name__: c for c in nodes}
    schema = by_name["PixelGridRestore"].GET_SCHEMA()
    raw = _io.create_input_dict_v1(schema.inputs)

    check("grid_mode serialises as COMFY_DYNAMICCOMBO_V3",
          raw["required"]["grid_mode"][0] == "COMFY_DYNAMICCOMBO_V3",
          str(raw["required"]["grid_mode"][0]))

    cases = (
        ("Auto Detect", {"min_size": 4.0, "max_pixel_size": 20.0, "peak_width": 6}),
        ("By Pixel Size", {"pixel_size": 16}),
        ("By Grid Count", {"grid_w": 64, "grid_h": 64}),
    )
    all_nested = {"min_size", "max_pixel_size", "peak_width", "pixel_size", "grid_w", "grid_h"}
    for mode, nested in cases:
        flat = {"grid_mode": mode, **{f"grid_mode.{k}": v for k, v in nested.items()}}
        out, _missing, v3 = _io.get_finalized_class_inputs(raw, dict(flat))
        built = _io.build_nested_inputs(dict(flat), v3)
        got = built["grid_mode"]
        prefixed = {k.split(".", 1)[1] for k in out["required"] if k.startswith("grid_mode.")}
        print(f"      {mode:<15} -> {got}")
        check(f"{mode}: execute receives the selected key inside the input's own dict",
              got.get("grid_mode") == mode, str(got))
        check(f"{mode}: nested values arrive alongside it",
              {k: v for k, v in got.items() if k != "grid_mode"} == nested, str(got))
        check(f"{mode}: exactly that mode's parameters become required",
              prefixed == set(nested), str(sorted(prefixed)))
        check(f"{mode}: no other mode's parameters leak in",
              not (all_nested - set(nested)) & prefixed, str(sorted(prefixed)))
        check(f"{mode}: _unpack_grid_mode round-trips",
              gr._unpack_grid_mode(got) == (mode, got), str(gr._unpack_grid_mode(got)))

    # Hand-written API prompts: a missing key must not raise, and a bogus mode
    # must be reported as an unknown mode rather than blowing up on a dict lookup.
    check("a prompt with no grid_mode at all behaves as Auto Detect",
          gr._unpack_grid_mode(None)[0] == "Auto Detect", str(gr._unpack_grid_mode(None)))
    check("a bare (unmatched) mode string is passed through for the caller to reject",
          gr._unpack_grid_mode("Nonsense")[0] == "Nonsense")

    raised = None
    try:
        by_name["PixelGridRestore"].execute(image=None, grid_mode="Nonsense")
    except ValueError as exc:
        raised = exc
    except Exception as exc:  # noqa: BLE001
        raised = exc
    check("an unknown grid_mode raises ValueError naming the mode",
          isinstance(raised, ValueError) and "Nonsense" in str(raised),
          f"{type(raised).__name__}: {raised}")

    # _pick must not rewrite a legitimate falsy value into the fallback.
    check("_pick returns a real 0 rather than the fallback", gr._pick({"v": 0}, "v", 99) == 0)
    check("_pick returns a real False rather than the fallback",
          gr._pick({"v": False}, "v", True) is False)
    check("_pick falls back only for a genuinely missing key",
          gr._pick({}, "v", 99) == 99 and gr._pick({"v": None}, "v", 99) == 99)


def section_grid_port(nodes: list, baseline: Path) -> None:
    """The max_pixel_size port: equivalent at the default, and it fixes the failure."""
    import numpy as np
    from pixelpipe.nodes import grid_restore as gr

    print("\n=== 3h. max_pixel_size: port equivalence and the capability it buys ===")

    def synth(cells: int, size: int = 512, seed: int = 0):
        rng = np.random.default_rng(seed)
        n = size // cells
        pal = rng.integers(0, 256, (n * n, 3), dtype=np.uint8)
        return np.ascontiguousarray(pal.reshape(n, n, 3).repeat(cells, 0).repeat(cells, 1))

    from PIL import Image

    corpus = [("synth 8px cells", synth(8)), ("synth 16px cells", synth(16)),
              ("synth 24px cells", synth(24)), ("synth 32px cells", synth(32))]
    if baseline.is_file():
        with Image.open(baseline) as im:
            corpus.append(("fox_pixel_00001_", np.array(im.convert("RGB"), dtype=np.uint8)))

    for label, img in corpus:
        for backend in ("OpenCV", "Lightweight"):
            module_name = gr._BACKEND_MODULE[backend]
            module = __import__(module_name, fromlist=["x"])
            try:
                upstream = str(module.detect_grid_scale(
                    img, peak_width=gr.DEFAULT_PEAK_WIDTH,
                    max_ratio=gr.UPSTREAM_MAX_RATIO, min_size=gr.DEFAULT_MIN_SIZE))
            except Exception as exc:  # noqa: BLE001 -- the no-CV2 backend's known crash
                upstream = f"{type(exc).__name__}"
            port = str(gr._detect_grid_scale_port(
                module, backend, img, peak_width=gr.DEFAULT_PEAK_WIDTH,
                min_size=gr.DEFAULT_MIN_SIZE,
                max_pixel_size=gr.UPSTREAM_MAX_PIXEL_SIZE))
            raised_ceiling = gr._detect_grid_scale_port(
                module, backend, img, peak_width=gr.DEFAULT_PEAK_WIDTH,
                min_size=gr.DEFAULT_MIN_SIZE, max_pixel_size=64.0)
            print(f"      {label:<17} {backend:<12} upstream={upstream:<12} "
                  f"port@20={port:<12} port@64={raised_ceiling}")
            if upstream == f"{type(TypeError()).__name__}":
                # No valid baseline: upstream crashes where the port recovers.
                check(f"port is strictly better than upstream on {label} / {backend}",
                      port != upstream, "port reproduced the crash")
            else:
                check(f"port @20.0 == upstream on {label} / {backend}",
                      port == upstream, f"{port} vs {upstream}")

    # The capability: 32px cells are rejected at the default ceiling and resolved
    # once it is raised. This is the user's own logged failure, reproduced.
    from pixelpipe.nodes.grid_restore import UPSTREAM_MAX_PIXEL_SIZE
    module = __import__(gr._BACKEND_MODULE["OpenCV"], fromlist=["x"])
    img32 = synth(32)
    at_default = gr._detect_grid_scale_port(
        module, "OpenCV", img32, peak_width=gr.DEFAULT_PEAK_WIDTH,
        min_size=gr.DEFAULT_MIN_SIZE, max_pixel_size=UPSTREAM_MAX_PIXEL_SIZE)
    at_raised = gr._detect_grid_scale_port(
        module, "OpenCV", img32, peak_width=gr.DEFAULT_PEAK_WIDTH,
        min_size=gr.DEFAULT_MIN_SIZE, max_pixel_size=64.0)
    check("32px cells fail at the default ceiling (the user's logged failure)",
          at_default == (None, None), str(at_default))
    check("32px cells resolve to a 16x16 grid once the ceiling is raised",
          at_raised == (16, 16), str(at_raised))

    # The honest downside, pinned so it cannot be quietly forgotten: raising the
    # ceiling turns some loud failures into quiet wrong answers.
    img24 = synth(24)
    check("24px cells are rejected at the default ceiling",
          gr._detect_grid_scale_port(
              module, "OpenCV", img24, peak_width=gr.DEFAULT_PEAK_WIDTH,
              min_size=gr.DEFAULT_MIN_SIZE, max_pixel_size=UPSTREAM_MAX_PIXEL_SIZE)
          == (None, None))
    aliased = gr._detect_grid_scale_port(
        module, "OpenCV", img24, peak_width=gr.DEFAULT_PEAK_WIDTH,
        min_size=gr.DEFAULT_MIN_SIZE, max_pixel_size=64.0)
    check("24px cells become a QUIET WRONG ANSWER (21 cells) when the ceiling is raised "
          "-- documented, not a bug",
          aliased == (21, 21), str(aliased))


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


def section_locale(nodes: list) -> None:
    """The zh translation keys must match what the frontend will actually look up.

    The frontend resolves a slot's text at
    ``nodeDefs.<node>.inputs.<normalizeI18nKey(slot)>.<name|tooltip>``, and
    ``normalizeI18nKey`` replaces ``.`` with ``_``. Nested DynamicCombo widgets are
    named ``"<parent>.<nested>"``, so their lookup key is
    ``"<parent>_<nested>"`` -- NOT the bare nested id.

    That makes this section load-bearing rather than cosmetic: ``/object_info``
    does not export nested inputs, so the Vue renderer falls back to an EMPTY
    string for a nested tooltip it cannot resolve. A stale key set here shows up
    as blank tooltips in the UI, which is easy to miss and easy to blame on the
    frontend.

    Asserting the key set is *exact* (not a superset) also keeps dead keys from
    accumulating after a parameter is moved or removed.
    """
    import json

    print("\n=== 4b. zh locale keys vs the schemas ===")
    path = ROOT / "locales" / "zh" / "nodeDefs.json"
    if not path.is_file():
        _fail("zh nodeDefs.json exists", str(path))
        return
    with open(path, encoding="utf-8") as fh:
        defs = json.load(fh)
    check("locale file has an entry for every registered node",
          set(defs) == {c.GET_SCHEMA().node_id for c in nodes},
          f"locale={sorted(defs)} vs nodes={sorted(c.GET_SCHEMA().node_id for c in nodes)}")

    for cls in nodes:
        schema = cls.GET_SCHEMA()
        spec = defs.get(schema.node_id)
        if spec is None:
            continue

        expected = {i.id for i in schema.inputs}
        for combo_id, options in _dynamic_options(schema).items():
            expected.discard(combo_id)
            expected.add(combo_id)  # the combo itself keeps its own plain key
            for rows in options.values():
                for row in rows:
                    expected.add(f"{combo_id}_{row['id']}")
        got = set(spec.get("inputs", {}))
        print(f"  {schema.node_id:<22} inputs: {len(got)} keys")
        check(f"{schema.node_id}: locale input keys match the schema exactly",
              got == expected,
              f"missing={sorted(expected - got)} dead={sorted(got - expected)}")
        for key, entry in spec.get("inputs", {}).items():
            check(f"{schema.node_id}.{key}: has a non-empty name and tooltip",
                  bool(str(entry.get("name", "")).strip())
                  and bool(str(entry.get("tooltip", "")).strip()),
                  str(entry)[:120])

        expected_out = {str(n) for n in range(len(schema.outputs))}
        check(f"{schema.node_id}: locale output keys are 0..{len(schema.outputs) - 1}",
              set(spec.get("outputs", {})) == expected_out,
              f"{sorted(spec.get('outputs', {}))} vs {sorted(expected_out)}")
        check(f"{schema.node_id}: has a non-empty display_name",
              bool(str(spec.get("display_name", "")).strip()))
        check(f"{schema.node_id}: has a non-empty description",
              bool(str(spec.get("description", "")).strip()))


def _dynamic_options(schema) -> "dict[str, list[dict]]":
    """``{option_key: [nested rows]}`` for every DynamicCombo input in ``schema``.

    ``_schema_rows()`` only walks top-level inputs, and a DynamicCombo serialises
    as a single row, so without this the nested parameters are invisible to both
    the printed report and the assertions.
    """
    from comfy_api.latest import io

    out: dict[str, list[dict]] = {}
    for i in schema.inputs:
        if not isinstance(i, io.DynamicCombo.Input):
            continue
        opts = {}
        for option in i.options:
            rows = []
            for nested in option.inputs:
                d = nested.as_dict()
                rows.append(
                    {
                        "id": nested.id,
                        "type": nested.io_type,
                        "default": d.get("default"),
                        "min": d.get("min"),
                        "max": d.get("max"),
                        "optional": bool(d.get("optional")),
                        "lazy": bool(d.get("lazy")),
                        "advanced": bool(d.get("advanced")),
                    }
                )
            opts[option.key] = rows
        out[i.id] = opts
    return out


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

    apply_node = by_name["PixelPaletteApply"].GET_SCHEMA()
    ains, aouts = _schema_rows(apply_node)
    print(f"\n  --- {apply_node.node_id} ---")
    for row in ains:
        flags = [f for f in ("optional", "lazy") if row[f]]
        print(f"      in  {row['id']:<16} {row['type']:<7} default={row['default']!r} "
              f"min={row['min']} max={row['max']} options={row['options']} {flags}")
    for name, io_type in aouts:
        print(f"      out {name:<16} {io_type}")

    # The whole point of the rewrite: a node that only applies a palette has no
    # derive parameters left to go silently inert.
    check("Apply input ids are exactly the contract",
          [r["id"] for r in ains] == ["image", "palette", "space", "dither"],
          str([r["id"] for r in ains]))
    check("Apply input types",
          [r["type"] for r in ains] == ["IMAGE", "STRING", "COMBO", "COMBO"],
          str([r["type"] for r in ains]))
    check("Apply has no derive-path inputs left",
          not ({"method", "colors", "tile_size", "accent_protect"}
               & {r["id"] for r in ains}),
          str([r["id"] for r in ains]))
    check("Apply defaults: palette='', space=Lab, dither=None",
          (ains[1]["default"], ains[2]["default"], ains[3]["default"]) == ("", "Lab", "None"),
          str([ains[i]["default"] for i in (1, 2, 3)]))
    check("Apply dither options",
          ains[3]["options"] == ["None", "Bayer 2x2", "Bayer 4x4", "Bayer 8x8", "Floyd-Steinberg"],
          str(ains[3]["options"]))
    check("Apply outputs are (image IMAGE, applied_palette STRING, colors_used INT)",
          aouts == [("image", "IMAGE"), ("applied_palette", "STRING"), ("colors_used", "INT")],
          str(aouts))

    palette_in = ains[1]
    check("palette is required, not optional", palette_in["optional"] is False,
          f"optional={palette_in['optional']}")
    check("palette is NOT lazy (the pruning it existed for is gone)",
          palette_in["lazy"] is False, f"lazy={palette_in['lazy']}")
    check("palette is a STRING carrying a text default",
          palette_in["type"] == "STRING" and palette_in["default"] == "",
          f"{palette_in['type']} default={palette_in['default']!r}")
    check("nothing on Apply is optional or lazy",
          not any(r["optional"] or r["lazy"] for r in ains),
          str([r["id"] for r in ains if r["optional"] or r["lazy"]]))
    # hasattr() is useless here: io.ComfyNode defines check_lazy_status on the
    # base class, so it is always present. What matters is that Apply does not
    # OVERRIDE it -- an override is the only thing that could re-introduce lazy
    # re-queueing.
    check("Apply does not override check_lazy_status",
          "check_lazy_status" not in vars(by_name["PixelPaletteApply"]),
          str([k for k in vars(by_name["PixelPaletteApply"]) if "lazy" in k]))
    check("the old PixelPaletteQuantize node class is gone",
          "PixelPaletteQuantize" not in by_name, str(sorted(by_name)))
    check("the old PixelPaletteQuantize node_id is gone",
          all(c.GET_SCHEMA().node_id != "PixelPaletteQuantize" for c in nodes))


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
    from pixelpipe.nodes.palette import PixelPaletteApply, PixelPaletteExtract

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
    out_b, applied_b, used_b = PixelPaletteApply.execute(
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

    print("\n=== 7. Apply: a wired palette and a pasted one ===")
    # Both palette sources now go through one code path -- there is no derive
    # path left for them to disagree with.
    out_w, applied_w, used_w = PixelPaletteApply.execute(
        image=a_img, palette=hex_a, space="Lab", dither="None"
    ).result
    w_set = set(_pack(_to_u8(out_w)[0]).tolist())
    print(f"  [apply] wired from Extract    colors_used={used_w} of {len(pal_set)} palette entries")
    check("Apply accepts a palette wired from Extract", used_w > 0, f"used={used_w}")
    check("Apply output is a subset of the supplied palette", w_set <= pal_set,
          f"{len(w_set - pal_set)} outside")

    pasted = "#405060,#102030"
    out_p, applied_p, used_p = PixelPaletteApply.execute(
        image=a_img, palette=pasted, space="Lab", dither="None"
    ).result
    print(f"  [apply] pasted HEX            colors_used={used_p} applied={applied_p!r}")
    check("Apply accepts a pasted HEX palette", used_p > 0, f"used={used_p}")
    check("pasted palette: output uses only those two colours",
          set(_pack(_to_u8(out_p)[0]).tolist()) <= {0x405060, 0x102030},
          str(sorted(set(_pack(_to_u8(out_p)[0]).tolist()))))
    # Verbatim means order is preserved, not re-sorted by luminance: #405060 is
    # lighter than #102030, so a luminance sort would swap them.
    check("applied_palette is the supplied palette verbatim (order preserved, not re-sorted)",
          applied_p == "#405060\n#102030", repr(applied_p))

    raised = None
    try:
        PixelPaletteApply.execute(image=a_img, palette="   \n  ", space="Lab", dither="None")
    except ValueError as exc:
        raised = exc
    check("an empty palette raises ValueError naming the input",
          raised is not None and "palette" in str(raised),
          type(raised).__name__ if raised else "no exception")
    check("the empty-palette error says this node does not derive one",
          raised is not None and "derive" in str(raised), str(raised)[:160])

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
    check("the method list has no 'apply a palette' entry left",
          not hasattr(pu, "FIXED_PALETTE") and not hasattr(pu, "QUANTIZE_METHODS"),
          "FIXED_PALETTE / QUANTIZE_METHODS are dead and must be gone")

    print("\n=== 8. dithering ===")
    # Every dither variant is measured against the palette that was actually
    # supplied -- which is now the only kind of palette there is.
    variants_out = {}
    dither_pal = None
    for mode in ("None", "Bayer 2x2", "Bayer 4x4", "Bayer 8x8", "Floyd-Steinberg"):
        out, applied, used = PixelPaletteApply.execute(
            image=a_img, palette=hex_a, space="Lab", dither=mode,
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

    print("\n=== 9. graph dependency (the real graph planner) ===")
    _section_graph(by_name=by_name, hex_a=hex_a)

    print("\n=== 10. determinism ===")
    hex_a2, _sw2, count_a2 = PixelPaletteExtract.execute(
        image=a_img, colors=32, method="Per-Tile k-means", space="Lab", sort="Luminance"
    ).result
    check("Extract: identical HEX on a second run", hex_a == hex_a2,
          "differs" if hex_a != hex_a2 else "")
    check("Extract: identical count on a second run", count_a == count_a2)

    q1, ap1, u1 = PixelPaletteApply.execute(
        image=a_img, palette=hex_a, space="Lab", dither="Bayer 4x4",
    ).result
    q2, ap2, u2 = PixelPaletteApply.execute(
        image=a_img, palette=hex_a, space="Lab", dither="Bayer 4x4",
    ).result
    check("Apply: applied_palette is byte-identical on a second run", ap1 == ap2)
    check("Apply: output tensor is byte-identical on a second run",
          bool(torch.equal(q1, q2)), f"max|d| = {float((q1 - q2).abs().max())}")
    check("Apply: colors_used is identical on a second run", u1 == u2)
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
    # Extract pools a batch into ONE palette (that is what keeps a sprite set
    # consistent), and Apply then holds every frame of the batch inside it.
    bhex_batch, _bsw, bcount = PixelPaletteExtract.execute(
        image=batch, colors=32, method="Per-Tile k-means", space="Lab", sort="Luminance",
    ).result
    _bo, bap, bused = PixelPaletteApply.execute(
        image=batch, palette=bhex_batch, space="Lab", dither="None"
    ).result
    check("Apply echoes a batch palette back verbatim", bap == bhex_batch,
          "differs" if bap != bhex_batch else "")
    bset = set(_pack(_to_u8(_bo)).tolist())
    print(f"  batch of {tuple(batch.shape)} -> {bused} colours, shared palette "
          f"{len(pu.parse_palette_hex(bap))} entries")
    check("a batch produces one shared palette for every frame",
          bset <= set(_pack(pu.parse_palette_hex(bap)).tolist()),
          f"{len(bset - set(_pack(pu.parse_palette_hex(bap)).tolist()))} outside")


def section_apply_validation(nodes: list) -> None:
    """Drive ``execution.validate_inputs`` -- the exact path a user's Run hits.

    This is the highest-fidelity guard in the file. The bug it exists for was
    invisible to every isolated unit check: calling ``validate_inputs(palette=...)``
    by hand passes a real string, but the framework passes ``None`` for any
    *linked* input because validation runs before execution. Only driving the
    real function reproduces that, so only this test could have caught it.

    It also pins the error COUNT. execution.py appends the validator's returned
    string once per entry of ``input_filtered``, and ``input_filtered`` is
    derived from the validator's parameter names -- so a ``**kwargs`` validator
    reports the same message against every socket on the node.
    """
    import asyncio

    import execution
    import nodes as comfy_nodes

    print("\n=== 9b. Apply validation through the real framework ===")
    for cls in nodes:
        comfy_nodes.NODE_CLASS_MAPPINGS.setdefault(cls.GET_SCHEMA().node_id, cls)

    def prompt(palette_input):
        return {
            "9": {"class_type": "LoadImage", "inputs": {"image": "fox_pixel_00001_.png"}},
            "3": {"class_type": "PixelGridRestore",
                  "inputs": {"image": ["9", 0], "sampling": "Majority Cluster",
                             "grid_mode": {"grid_mode": "Auto Detect"},
                             "backend": "Auto", "refine_intensity": 0.25,
                             "fix_square": True}},
            "1": {"class_type": "PixelPaletteExtract",
                  "inputs": {"image": ["3", 0], "colors": 32,
                             "method": "Per-Tile k-means", "space": "Lab",
                             "sort": "Luminance", "accent_protect": True, "tile_size": 16}},
            "2": {"class_type": "PixelPaletteApply",
                  "inputs": {"image": ["3", 0], "palette": palette_input,
                             "space": "Lab", "dither": "None"}},
        }

    async def validate(palette_input):
        validated: dict = {}
        await execution.validate_inputs("selftest", prompt(palette_input), "2", validated)
        ok, reasons, _ = validated.get("2", (None, None, None))
        return ok, (reasons or [])

    cases = (
        ("wired from Extract", ["1", 0], True),
        ("typed as a HEX constant", "#102030\n#405060", True),
        ("left empty", "", False),
    )
    for label, value, expect_valid in cases:
        ok, reasons = asyncio.run(validate(value))
        print(f"      {label:<24} valid={ok}  errors={len(reasons)}")
        check(f"Apply validates correctly when the palette is {label}",
              bool(ok) is expect_valid, f"valid={ok}, errors={[r.get('details') for r in reasons]}")
        if not expect_valid:
            # One error, named after the socket that is wrong -- not one per input.
            check("an empty palette produces exactly ONE error",
                  len(reasons) == 1, f"got {len(reasons)}: {[r.get('details') for r in reasons]}")
            check("that error is attributed to the palette input",
                  reasons and reasons[0].get("extra_info", {}).get("input_name") == "palette",
                  str([r.get("extra_info") for r in reasons]))


def section_video_schema(nodes: list) -> None:
    """The video node's contract, and the constraints that keep the pack loading."""
    from pixelpipe.nodes import grid_restore as gr
    from pixelpipe.nodes import video_grid_restore as vg

    print("\n=== 12. PixelVideoGridRestore schema ===")
    by_name = {c.__name__: c for c in nodes}
    schema = by_name["PixelVideoGridRestore"].GET_SCHEMA()
    ins, outs = _schema_rows(schema)

    check("video top-level input ids are exactly the contract",
          [r["id"] for r in ins] == ["video", "sampling", "grid_mode", "temporal_window",
                                     "backend", "chunk_frames", "fallback_pixel_size"],
          str([r["id"] for r in ins]))
    check("video input types",
          [r["type"] for r in ins] == ["VIDEO", "COMBO", "COMFY_DYNAMICCOMBO_V3", "INT",
                                       "COMBO", "INT", "INT"],
          str([r["type"] for r in ins]))
    check("only chunk_frames / fallback_pixel_size are advanced",
          [r["id"] for r in ins if r["advanced"]]
          == ["chunk_frames", "fallback_pixel_size"],
          str([r["id"] for r in ins if r["advanced"]]))
    check("temporal_window defaults to 0 (off) and can be turned off",
          (ins[3]["default"], ins[3]["min"], ins[3]["max"]) == (0, 0, 8),
          str((ins[3]["default"], ins[3]["min"], ins[3]["max"])))
    check("chunk_frames defaults to 12 and 0 means unchunked",
          (ins[5]["default"], ins[5]["min"]) == (12, 0),
          str((ins[5]["default"], ins[5]["min"])))
    check("video output list is exactly the 9 documented outputs",
          outs == [("video", "VIDEO"), ("images", "IMAGE"), ("audio", "AUDIO"),
                   ("fps", "FLOAT"), ("grid_w", "INT"), ("grid_h", "INT"),
                   ("pixel_size", "FLOAT"), ("detected", "BOOLEAN"),
                   ("confidence", "FLOAT")],
          str(outs))

    opts = _dynamic_options(schema)
    check("the video node's only DynamicCombo is grid_mode, with the three modes",
          list(opts) == ["grid_mode"]
          and list(opts["grid_mode"]) == ["Auto Detect", "By Pixel Size", "By Grid Count"],
          str(list(opts.get("grid_mode", {}))))
    # No `default` kwarg exists on DynamicCombo.Input, so option order IS the
    # default. Pin it or the node silently opens in a different mode.
    check("'Auto Detect' is FIRST on the video node too (that is the default lever)",
          next(iter(opts["grid_mode"])) == "Auto Detect")
    auto = {r["id"] for r in opts["grid_mode"]["Auto Detect"]}
    check("Auto Detect carries sample_stride plus the three detector parameters",
          auto == {"sample_stride", "min_size", "max_pixel_size", "peak_width"}, str(auto))
    check("no nested video input is optional, lazy or advanced",
          not any(r["optional"] or r["lazy"] or r["advanced"]
                  for mode in opts["grid_mode"].values() for r in mode),
          str([(r["id"], r["optional"], r["lazy"], r["advanced"])
               for mode in opts["grid_mode"].values() for r in mode]))

    defaults = {r["id"]: r["default"] for mode in opts["grid_mode"].values() for r in mode}
    for name, constant in (("min_size", gr.DEFAULT_MIN_SIZE),
                           ("max_pixel_size", gr.DEFAULT_MAX_PIXEL_SIZE),
                           ("peak_width", gr.DEFAULT_PEAK_WIDTH),
                           ("pixel_size", gr.DEFAULT_PIXEL_SIZE),
                           ("grid_w", gr.DEFAULT_GRID_W),
                           ("grid_h", gr.DEFAULT_GRID_H),
                           ("sample_stride", vg.DEFAULT_SAMPLE_STRIDE)):
        check(f"video nested default {name} matches its module constant",
              defaults[name] == constant, f"{defaults[name]} vs {constant}")

    # The two nodes must agree on these, or the same value means different things
    # depending on which node you are looking at.
    check("video sampling is independent; existing backend and grid choices stay compatible",
          set(gr.SAMPLING_OPTIONS).issubset(vg.SAMPLING_OPTIONS)
          and vg.SAMPLING_OPTIONS[0] == "Robust Cell"
          and vg.BACKEND_OPTIONS is gr.BACKEND_OPTIONS
          and vg.GRID_MODE_OPTIONS is gr.GRID_MODE_OPTIONS)
    from pixelpipe.nodes import _palette_utils as pu
    check("SAMPLE_SEED matches the palette nodes' KMEANS_SEED (one fixed seed in the pack)",
          gr.SAMPLE_SEED == pu.KMEANS_SEED, f"{hex(gr.SAMPLE_SEED)} vs {hex(pu.KMEANS_SEED)}")


def section_video_contract(nodes: list) -> None:
    """Drive the video node's DynamicCombo through comfy_api's own machinery."""
    from comfy_api.latest import _io
    from pixelpipe.nodes import video_grid_restore as vg

    print("\n=== 13. PixelVideoGridRestore runtime contract ===")
    by_name = {c.__name__: c for c in nodes}
    schema = by_name["PixelVideoGridRestore"].GET_SCHEMA()
    raw = _io.create_input_dict_v1(schema.inputs)

    check("grid_mode serialises as COMFY_DYNAMICCOMBO_V3 on the video node",
          raw["required"]["grid_mode"][0] == "COMFY_DYNAMICCOMBO_V3",
          str(raw["required"]["grid_mode"][0]))

    cases = (
        ("Auto Detect", {"sample_stride": 8, "min_size": 4.0, "max_pixel_size": 20.0,
                         "peak_width": 6}),
        ("By Pixel Size", {"pixel_size": 16}),
        ("By Grid Count", {"grid_w": 64, "grid_h": 64}),
    )
    every = {"sample_stride", "min_size", "max_pixel_size", "peak_width", "pixel_size",
             "grid_w", "grid_h"}
    for mode, nested in cases:
        flat = {"grid_mode": mode, **{f"grid_mode.{k}": v for k, v in nested.items()}}
        out, _missing, v3 = _io.get_finalized_class_inputs(raw, dict(flat))
        built = _io.build_nested_inputs(dict(flat), v3)["grid_mode"]
        prefixed = {k.split(".", 1)[1] for k in out["required"] if k.startswith("grid_mode.")}
        print(f"      {mode:<15} -> {built}")
        check(f"video {mode}: the selected key rides inside the input's own dict",
              built.get("grid_mode") == mode, str(built))
        check(f"video {mode}: exactly that mode's parameters become required",
              prefixed == set(nested), str(sorted(prefixed)))
        check(f"video {mode}: no other mode's parameters leak in",
              not (every - set(nested)) & prefixed, str(sorted(prefixed)))
        check(f"video {mode}: _unpack_grid_mode round-trips",
              vg._unpack_grid_mode(built)[0] == mode)

    check("a video prompt with no grid_mode at all behaves as Auto Detect",
          vg._unpack_grid_mode(None)[0] == "Auto Detect")


def section_video_behaviour(nodes: list) -> None:
    """Run the video v2 regressions against real ComfyUI components."""
    import unittest
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_video_pipeline.py")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    check("video v2 motion, alpha, timing and sampling regressions", result.wasSuccessful(),
          f"{result.testsRun} tests, {len(result.failures)} failures, {len(result.errors)} errors")


def execution_get_input_data_probe(node_cls) -> "list[tuple[str, object]]":
    """What ``validate_inputs`` sees for a wired vs a typed palette.

    Drives ``execution.get_input_data`` for real rather than hand-writing the
    expected value: the ``None`` this exposes is the entire reason a wired
    palette used to fail validation, and it is not documented anywhere obvious.
    Called without an ``execution_list``, exactly as ``execution.validate_inputs``
    calls it, so every linked input is marked missing.
    """
    import execution

    cases = (
        ("wired (link)", {"image": ["1", 0], "palette": ["2", 0],
                          "space": "Lab", "dither": "None"}),
        ("typed as a constant", {"image": ["1", 0], "palette": "#102030\n#405060",
                                 "space": "Lab", "dither": "None"}),
        ("empty constant", {"image": ["1", 0], "palette": "",
                            "space": "Lab", "dither": "None"}),
    )
    out = []
    for label, inputs in cases:
        data, _missing, _v3 = execution.get_input_data(inputs, node_cls, "3")
        # get_input_data wraps constants in a 1-element list; execution's
        # get_slice() unwraps it before calling the validator, so unwrap here too.
        seen = data.get("palette")
        if isinstance(seen, (list, tuple)):
            seen = seen[0]
        out.append((label, seen))
    return out


def _section_graph(by_name: dict, hex_a: str) -> None:
    """Drive the real ``TopologicalSort`` -- no ComfyUI server required.

    This section used to prove that the ``lazy`` palette input pruned the
    ``PixelPaletteExtract`` subtree out of the execution graph. That input is now
    an ordinary required one, so the property to prove is the mirror image: the
    palette is a *hard* dependency, and Extract is in the execution block from
    the start rather than being promoted into it by ``check_lazy_status`` on a
    second pass.

    Worth keeping rather than deleting, because "the node I wired always runs" is
    the thing a future lazy refactor would silently take away again.
    """
    import comfy_execution.graph as graph_mod
    import nodes as comfy_nodes

    apply_node = by_name["PixelPaletteApply"]
    extract = by_name["PixelPaletteExtract"]

    # The pruner looks node types up in the global registry.
    for cls in (by_name["PixelGridRestore"], extract, apply_node, by_name["PixelExportScale"]):
        comfy_nodes.NODE_CLASS_MAPPINGS.setdefault(cls.GET_SCHEMA().node_id, cls)

    def build(palette_value):
        return {
            "1": {
                # A non-link placeholder: the pruner only follows links, and this
                # keeps the test independent of whatever loader nodes happen to be
                # registered. Node 1 is the root of the graph here.
                "class_type": "PixelGridRestore",
                "inputs": {"image": None, "sampling": "Median",
                           "grid_mode": {"grid_mode": "By Grid Count",
                                         "grid_w": 125, "grid_h": 122}},
            },
            "2": {
                "class_type": "PixelPaletteExtract",
                "inputs": {"image": ["1", 0], "colors": 32, "method": "Per-Tile k-means",
                           "space": "Lab", "sort": "Luminance"},
            },
            "3": {
                "class_type": "PixelPaletteApply",
                "inputs": {"image": ["1", 0], "space": "Lab", "dither": "None",
                           "palette": palette_value},
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

    wired = plan(["2", 0])  # real link to the extract node
    print(f"  initial block, palette wired   : {sorted(wired.pendingNodes)}")
    check("a wired palette pulls PixelPaletteExtract into the INITIAL block",
          sorted(wired.pendingNodes) == ["1", "2", "3", "4"],
          f"block = {sorted(wired.pendingNodes)} -- no second pass should be needed")

    # A pasted palette needs no upstream at all, so Extract stays out -- but for
    # the ordinary reason (nothing links to it), not because a lazy input pruned it.
    pasted = plan("#102030\n#405060")
    print(f"  initial block, palette pasted  : {sorted(pasted.pendingNodes)}")
    check("a pasted palette needs no upstream node",
          "2" not in pasted.pendingNodes, f"block = {sorted(pasted.pendingNodes)}")
    check("a pasted palette still schedules Apply itself",
          "3" in pasted.pendingNodes, f"block = {sorted(pasted.pendingNodes)}")

    # validate_inputs is now the only thing standing between a prompt and an
    # empty palette, since there is no lazy re-queue to catch it later.
    #
    # REGRESSION, and the reason this block is shaped the way it is: validation
    # runs BEFORE any node executes, and execution.get_input_data() marks every
    # *linked* input missing when it is called without an execution_list -- which
    # is precisely how the validation phase calls it. So a wired palette arrives
    # at validate_inputs as None. Treating None as "empty" makes a wired palette
    # impossible to validate, i.e. it bricks the node for the only wiring the
    # pack is designed around. None must mean "not known yet; execute() decides".
    import inspect

    src = execution_get_input_data_probe(apply_node)
    for label, seen in src:
        print(f"      {label:<24} validate_inputs sees palette={seen!r}")
        if label.startswith("wired"):
            check("a WIRED palette passes validation (it arrives as None, not as empty)",
                  apply_node.validate_inputs(palette=seen) is True,
                  f"got {apply_node.validate_inputs(palette=seen)!r} for palette={seen!r}")
    check("validate_inputs tolerates palette=None (wired / not yet evaluated)",
          apply_node.validate_inputs(palette=None) is True,
          repr(apply_node.validate_inputs(palette=None)))
    check("validate_inputs accepts a real palette",
          apply_node.validate_inputs(palette=hex_a) is True)
    check("validate_inputs rejects an empty palette",
          isinstance(apply_node.validate_inputs(palette=""), str))
    check("validate_inputs rejects a whitespace-only palette",
          isinstance(apply_node.validate_inputs(palette="\n  \n"), str))

    # The validator must name ONLY palette, with no **kwargs. Both halves are
    # load-bearing in execution.py:
    #   input_filtered is built from argspec.args, and the returned string is
    #   appended to `errors` once per entry of input_filtered -- so a **kwargs
    #   validator reports the same message against EVERY input (this is what a
    #   user actually sees: four identical "palette is required" errors).
    #   validate_has_kwargs also gates ComfyUI's built-in min/max/options checks,
    #   so **kwargs silently DISABLES them for the whole node.
    spec = inspect.getfullargspec(apply_node.validate_inputs)
    check("validate_inputs takes only palette (no **kwargs), so its error is "
          "attributed to palette alone",
          spec.args == ["cls", "palette"] and spec.varkw is None,
          f"args={spec.args} varkw={spec.varkw}")
    check("no **kwargs on the validator means ComfyUI's built-in range and combo "
          "checks are NOT disabled for this node",
          spec.varkw is None)


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
    quant = by_name["PixelPaletteApply"]
    extract = by_name["PixelPaletteExtract"]
    scale = by_name["PixelExportScale"]

    rgba, inset, cell = _rgba_test_image()
    print(f"      input {tuple(rgba.shape)}, opaque square inset {inset}px, cell {cell}px")

    # -- 11a. RGBA must not raise, and alpha must survive to a 4-channel output
    try:
        out, gw, gh, px, det, conf = grid.execute(
            image=rgba, sampling="Majority Cluster", **_mode("Auto Detect")
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
        image=opaque, sampling="Majority Cluster", **_mode("Auto Detect")
    ).result
    shape_b = tuple(_to_u8(out_b).shape)
    print(f"      all-opaque input -> {shape_b}")
    check("11b all-opaque alpha is dropped (RGB out)", len(shape_b) == 4 and shape_b[-1] == 3,
          f"got {shape_b}")

    # -- 11d. Extract accepts RGBA (and must ignore alpha when choosing colours)
    # Run before 11c so Apply can be fed a real Extract output on an RGBA image.
    _h, _sw, count = extract.execute(
        image=out, colors=32, method="Per-Tile k-means", space="Lab",
        sort="Luminance", accent_protect=True, tile_size=cell,
    ).result
    check("11d Extract accepts RGBA", count > 0, f"count={count}")

    # -- 11c. Apply must pass alpha through byte-for-byte
    q_out, _q_pal, _q_used = quant.execute(
        image=out, palette=_h, space="Lab", dither="None",
    ).result
    q_u8 = _to_u8(q_out)
    check("11c Apply keeps 4 channels", q_u8.shape[-1] == 4, f"got {q_u8.shape}")
    if q_u8.shape[-1] == 4:
        check("11c Apply leaves alpha byte-identical",
              np.array_equal(q_u8[..., 3], out_u8[..., 3]), "alpha changed")

    # -- 11e. ExportScale carries alpha through untouched
    sc, sw_w, sc_h = scale.execute(image=out, scale=4, mode="Nearest").result
    sc_shape = tuple(sc.shape)
    print(f"      ExportScale -> {sc_shape} ({sw_w}x{sc_h})")
    check("11e ExportScale keeps 4 channels", sc_shape[-1] == 4, f"got {sc_shape}")
    check("11e ExportScale quadruples the grid", (sc_h, sw_w) == (gh * 4, gw * 4),
          f"got {sw_w}x{sc_h}")


# --------------------------------------------------------------------------- #
# 15. Sprite node schemas
# --------------------------------------------------------------------------- #
def section_sprite_schema(nodes: list) -> None:
    """Both sprite nodes' contracts; the sheet's layout DynamicCombo in particular."""
    print("\n=== 15. sprite node schemas ===")
    by_name = {c.__name__: c for c in nodes}
    check("both sprite nodes are registered",
          {"PixelSpritePrepare", "PixelSpriteSheet"} <= set(by_name), str(sorted(by_name)))
    if not {"PixelSpritePrepare", "PixelSpriteSheet"} <= set(by_name):
        return
    from pixelpipe.nodes import sprite_prepare as sp
    from pixelpipe.nodes import sprite_sheet as ss

    ins, outs = _schema_rows(by_name["PixelSpritePrepare"].GET_SCHEMA())
    check("prepare top-level input ids are exactly the contract",
          [r["id"] for r in ins] == ["image", "color_tolerance", "padding_ratio",
                                     "remove_background"],
          str([r["id"] for r in ins]))
    check("prepare top-level input types",
          [r["type"] for r in ins] == ["IMAGE", "INT", "FLOAT", "BOOLEAN"],
          str([r["type"] for r in ins]))
    check("prepare defaults and ranges",
          [(r["id"], r["default"], r["min"], r["max"]) for r in ins[1:]] == [
              ("color_tolerance", 12, 0, 255),
              ("padding_ratio", 0.05, 0.0, 1.0),
              ("remove_background", True, None, None)],
          str([(r["id"], r["default"], r["min"], r["max"]) for r in ins[1:]]))
    check("prepare schema defaults equal the module constants",
          ins[1]["default"] == sp.DEFAULT_COLOR_TOLERANCE
          and ins[2]["default"] == sp.DEFAULT_PADDING_RATIO)
    check("prepare constants match the reference script",
          (sp.DEFAULT_COLOR_TOLERANCE, sp.DEFAULT_PADDING_RATIO, sp.PADDING_MIN_PX)
          == (12, 0.05, 2),
          str((sp.DEFAULT_COLOR_TOLERANCE, sp.DEFAULT_PADDING_RATIO, sp.PADDING_MIN_PX)))
    check("prepare has no optional/lazy/advanced inputs",
          not any(r["optional"] or r["lazy"] or r["advanced"] for r in ins))
    check("prepare outputs are images + bg_color",
          outs == [("images", "IMAGE"), ("bg_color", "STRING")], str(outs))

    ins, outs = _schema_rows(by_name["PixelSpriteSheet"].GET_SCHEMA())
    check("sheet top-level input ids are exactly the contract",
          [r["id"] for r in ins] == ["image", "layout"], str([r["id"] for r in ins]))
    check("sheet top-level input types",
          [r["type"] for r in ins] == ["IMAGE", "COMFY_DYNAMICCOMBO_V3"],
          str([r["type"] for r in ins]))
    opts = _dynamic_options(by_name["PixelSpriteSheet"].GET_SCHEMA())
    check("one DynamicCombo named layout with the two documented layouts",
          list(opts) == ["layout"]
          and list(opts["layout"]) == ["Auto Square", "Fixed Columns"],
          str(list(opts.get("layout", {}))))
    # DynamicCombo.Input has no `default` kwarg, so option order is the only
    # lever (same pitfall as grid_mode). Pin it.
    check("'Auto Square' is the FIRST option (that is what makes it the default)",
          next(iter(opts["layout"])) == "Auto Square")
    check("Auto Square carries no nested parameters",
          opts["layout"]["Auto Square"] == [], str(opts["layout"]["Auto Square"]))
    fixed = {r["id"]: r for r in opts["layout"]["Fixed Columns"]}
    check("Fixed Columns carries only columns",
          list(fixed) == ["columns"], str(list(fixed)))
    check("columns is INT, schema default equals the module constant, with room to rise",
          (fixed["columns"]["type"], fixed["columns"]["default"],
           fixed["columns"]["min"], fixed["columns"]["max"])
          == ("INT", ss.DEFAULT_COLUMNS, 1, 1024), str(fixed["columns"]))
    check("no nested input is optional, lazy, or advanced",
          not any(r["optional"] or r["lazy"] or r["advanced"]
                  for mode in opts["layout"].values() for r in mode))
    check("sheet outputs are sheet + grid metadata",
          outs == [("sheet", "IMAGE"), ("columns", "INT"), ("rows", "INT"),
                   ("cell_width", "INT"), ("cell_height", "INT")], str(outs))


# --------------------------------------------------------------------------- #
# 16. Sprite runtime contract
# --------------------------------------------------------------------------- #
def section_sprite_contract(nodes: list) -> None:
    """The layout combo through ComfyUI's own machinery (mirrors 3g)."""
    from comfy_api.latest import _io

    print("\n=== 16. layout runtime contract (via comfy_api internals) ===")
    by_name = {c.__name__: c for c in nodes}
    if "PixelSpriteSheet" not in by_name:
        _fail("PixelSpriteSheet is registered", str(sorted(by_name)))
        return
    from pixelpipe.nodes import sprite_sheet as ss

    schema = by_name["PixelSpriteSheet"].GET_SCHEMA()
    raw = _io.create_input_dict_v1(schema.inputs)
    check("layout serialises as COMFY_DYNAMICCOMBO_V3",
          raw["required"]["layout"][0] == "COMFY_DYNAMICCOMBO_V3",
          str(raw["required"]["layout"][0]))

    flat = {"layout": "Fixed Columns", "layout.columns": 4}
    out, _missing, v3 = _io.get_finalized_class_inputs(raw, dict(flat))
    built = _io.build_nested_inputs(dict(flat), v3)
    got = built["layout"]
    prefixed = {k.split(".", 1)[1] for k in out["required"] if k.startswith("layout.")}
    print(f"      Fixed Columns -> {got}")
    check("Fixed Columns: execute receives the selected key inside the input's own dict",
          got.get("layout") == "Fixed Columns", str(got))
    check("Fixed Columns: columns arrives alongside it",
          {k: v for k, v in got.items() if k != "layout"} == {"columns": 4}, str(got))
    check("Fixed Columns: exactly its parameters become required",
          prefixed == {"columns"}, str(sorted(prefixed)))
    check("Fixed Columns: _unpack_layout round-trips",
          ss._unpack_layout(got) == ("Fixed Columns", got), str(ss._unpack_layout(got)))

    flat = {"layout": "Auto Square"}
    out, _missing, v3 = _io.get_finalized_class_inputs(raw, dict(flat))
    built = _io.build_nested_inputs(dict(flat), v3)
    got = built.get("layout")
    prefixed = {k.split(".", 1)[1] for k in out["required"] if k.startswith("layout.")}
    print(f"      Auto Square   -> {got}")
    # An option with zero nested inputs has no documented runtime shape (the
    # internals may hand back an empty dict); whatever arrives must unpack to
    # the mode, and no nested key may become required.
    check("Auto Square: whatever build_nested_inputs produces unpacks to Auto Square",
          ss._unpack_layout(got)[0] == "Auto Square", str(got))
    check("Auto Square: no nested parameters become required",
          prefixed == set(), str(sorted(prefixed)))

    check("a prompt with no layout at all behaves as Auto Square",
          ss._unpack_layout(None)[0] == "Auto Square", str(ss._unpack_layout(None)))
    check("a bare (unmatched) layout string is passed through for the caller to reject",
          ss._unpack_layout("Nonsense")[0] == "Nonsense")

    raised = None
    try:
        by_name["PixelSpriteSheet"].execute(image=None, layout="Nonsense")
    except ValueError as exc:
        raised = exc
    except Exception as exc:  # noqa: BLE001
        raised = exc
    check("an unknown layout raises ValueError naming the layout",
          isinstance(raised, ValueError) and "Nonsense" in str(raised),
          f"{type(raised).__name__}: {raised}")


# --------------------------------------------------------------------------- #
# 17. Sprite node behaviour
# --------------------------------------------------------------------------- #
def _sprite_fixture():
    """3 frames 24x24 RGB on a green background, one 4x4 red square each.

    f0 rows 6-9 cols 4-7 | f1 rows 6-9 cols 10-13 | f2 rows 14-17 cols 8-11.
    Union bbox (left 4, top 6, right 13, bottom 17); reference-script math:
    content 9x11 -> raw 11, padding max(round(0.55), 2) = 2, crop 15, centre
    (8.5, 11.5) -> window (left 1, top 4). Hand-computed, so the checks pin the
    port against the reference script rather than against itself.
    """
    import numpy as np
    import torch

    frames = np.tile(np.array([10, 200, 30], np.uint8), (3, 24, 24, 1))
    red = np.array([200, 30, 30], np.uint8)
    frames[0, 6:10, 4:8] = red
    frames[1, 6:10, 10:14] = red
    frames[2, 14:18, 8:12] = red
    return torch.from_numpy(frames).float() / 255.0


def section_sprite_behaviour(nodes: list) -> None:
    import numpy as np
    import torch

    print("\n=== 17. sprite node behaviour ===")
    by_name = {c.__name__: c for c in nodes}
    if not {"PixelSpritePrepare", "PixelSpriteSheet"} <= set(by_name):
        _fail("both sprite nodes are registered", str(sorted(by_name)))
        return
    prepare = by_name["PixelSpritePrepare"]
    sheet = by_name["PixelSpriteSheet"]
    fixture = _sprite_fixture()

    # -- 17a. default run: detection + shared square crop + keying
    out, bg_hex = prepare.execute(image=fixture).result
    u8 = _to_u8(out)
    print(f"      prepare -> {tuple(out.shape)}, bg {bg_hex}")
    check("17a crop is the hand-computed reference window", u8.shape == (3, 15, 15, 4),
          str(u8.shape))
    check("17a bg_color is the detected green, uppercase hex", bg_hex == "#0AC81E", bg_hex)
    check("17a border-connected background is keyed out",
          bool((u8[..., 3][:, 0, :] == 0).all() and (u8[..., 3][:, -1, :] == 0).all()
               and (u8[..., 3][:, :, 0] == 0).all() and (u8[..., 3][:, :, -1] == 0).all()))
    check("17a a content pixel stays opaque with its colour",
          bool((u8[0, 2:6, 3:7, 3] == 255).all()
               and (u8[0, 3, 4, :3] == [200, 30, 30]).all()),
          str(u8[0, 3, 4]))
    check("17a registration: frame 2's square lands at crop offset (7, 10)",
          bool((u8[2, 10:14, 7:11, 3] == 255).all()))

    # -- 17b. only border-connected background is keyed: an enclosed bg island survives
    island = np.tile(np.array([10, 200, 30], np.uint8), (1, 24, 24, 1))
    island[0, 8:16, 8:16] = [200, 30, 30]
    island[0, 11:13, 11:13] = [10, 200, 30]  # bg-coloured, fully enclosed by red
    out_i, _ = prepare.execute(
        image=torch.from_numpy(island).float() / 255.0).result
    i8 = _to_u8(out_i)
    # content = red ring rows/cols 8-15 -> raw 7, padding 2, crop 11 at (6, 6).
    check("17b crop is the hand-computed reference window", i8.shape == (1, 11, 11, 4),
          str(i8.shape))
    check("17b the enclosed background island keeps its alpha",
          bool((i8[0, 5:7, 5:7, 3] == 255).all()), str(i8[0, 5:7, 5:7, 3]))
    check("17b the outer background is keyed out", bool(i8[0, 0, 0, 3] == 0))

    # -- 17c. remove_background=False crops but creates no alpha (noclean variant)
    out_c, _ = prepare.execute(image=fixture, remove_background=False).result
    check("17c keying off keeps RGB, same crop", tuple(out_c.shape) == (3, 15, 15, 3),
          str(tuple(out_c.shape)))

    # -- 17d. no content anywhere: full-frame fallback, and keying erases everything
    empty = torch.from_numpy(np.tile(np.array([10, 200, 30], np.uint8), (2, 24, 24, 1))).float() / 255.0
    out_e, _ = prepare.execute(image=empty).result
    check("17d no content falls back to the full frame",
          tuple(out_e.shape) == (2, 24, 24, 4), str(tuple(out_e.shape)))
    check("17d an all-background batch keys fully transparent",
          bool((_to_u8(out_e)[..., 3] == 0).all()))
    out_e2, _ = prepare.execute(image=empty, remove_background=False).result
    check("17d no content with keying off is a full-frame RGB pass-through",
          tuple(out_e2.shape) == (2, 24, 24, 3)
          and torch.equal(out_e2, empty[..., :3]))

    # -- 17e. RGBA: the alpha mask drives the content box; keying composes onto it
    rgba = np.tile(np.array([10, 200, 30, 0], np.uint8), (2, 24, 24, 1))
    rgba[0, 6:10, 4:8, 3] = 255
    rgba[1, 6:10, 12:16, 3] = 255
    rgba[0, 7:9, 5:7, :3] = [200, 30, 30]
    rgba[1, 7:9, 13:15, :3] = [200, 30, 30]
    out_r, _ = prepare.execute(image=torch.from_numpy(rgba).float() / 255.0).result
    r8 = _to_u8(out_r)
    # alpha-marked union (left 4, top 6, right 15, bottom 9) -> content 11x3,
    # raw 11, padding 2, crop 15; centre (9.5, 7.5) -> left 2, top round(0.5)=0.
    print(f"      prepare(RGBA) -> {tuple(out_r.shape)}")
    check("17e the alpha mask drives the content box", r8.shape == (2, 15, 15, 4),
          str(r8.shape))
    check("17e the red cores survive opaque",
          bool((r8[0, 7:9, 3:5, 3] == 255).all() and (r8[1, 7:9, 11:13, 3] == 255).all()),
          f"{r8[0, 7, 3]} / {r8[1, 7, 11]}")
    check("17e bg-coloured pixels inside the alpha block are still keyed",
          bool(r8[0, 6, 2, 3] == 0))

    # -- 17f. empty batch raises on both nodes
    for node, kwargs in ((prepare, {"image": torch.zeros((0, 8, 8, 3))}),
                         (sheet, {"image": torch.zeros((0, 8, 8, 3)), "layout": None})):
        raised = None
        try:
            node.execute(**kwargs)
        except ValueError as exc:
            raised = exc
        check(f"17f {node.__name__} raises on an empty batch",
              isinstance(raised, ValueError), f"{type(raised).__name__ if raised else None}")

    # -- 17g. sheet, Auto Square: 5 frames -> 3 cols x 2 rows, row-major, zeros behind
    frames = torch.stack([torch.full((8, 8, 3), (i + 1) * 40 / 255.0) for i in range(5)])
    out_s, cols, rows, cw, ch = sheet.execute(image=frames, layout=None).result
    print(f"      sheet(auto, 5) -> {tuple(out_s.shape)} {cols}x{rows}")
    check("17g auto layout is the near-square reference grid",
          (cols, rows, cw, ch) == (3, 2, 8, 8), str((cols, rows, cw, ch)))
    check("17g sheet shape is rows*H x cols*W", tuple(out_s.shape) == (1, 16, 24, 3),
          str(tuple(out_s.shape)))
    check("17g placement is row-major",
          bool((out_s[0, 0:8, 8:16] == frames[1]).all()
               and (out_s[0, 8:16, 0:8] == frames[3]).all()))
    check("17g the trailing cell stays zero", bool((out_s[0, 8:16, 16:24] == 0).all()))

    # -- 17h. sheet, Fixed Columns, including columns > n
    out_s, cols, rows, _, _ = sheet.execute(
        image=frames, layout={"layout": "Fixed Columns", "columns": 2}).result
    check("17h fixed columns wraps after exactly that many",
          (cols, rows) == (2, 3) and tuple(out_s.shape) == (1, 24, 16, 3),
          str((cols, rows, tuple(out_s.shape))))
    out_s, cols, rows, _, _ = sheet.execute(
        image=frames, layout={"layout": "Fixed Columns", "columns": 8}).result
    check("17h more columns than frames is honoured (one row, trailing zeros)",
          (cols, rows) == (8, 1) and tuple(out_s.shape) == (1, 8, 64, 3)
          and bool((out_s[0, :, 40:] == 0).all()),
          str((cols, rows, tuple(out_s.shape))))

    # -- 17i. channels survive; a single frame is a 1x1 pass-through
    rgba5 = torch.stack([torch.full((8, 8, 4), (i + 1) * 40 / 255.0) for i in range(5)])
    out_s, *_ = sheet.execute(image=rgba5, layout=None).result
    check("17i RGBA stays RGBA, trailing alpha transparent",
          tuple(out_s.shape) == (1, 16, 24, 4) and bool((out_s[0, 8:16, 16:24, 3] == 0).all()),
          str(tuple(out_s.shape)))
    out_s, cols, rows, _, _ = sheet.execute(image=frames[:1], layout=None).result
    check("17i one frame is a 1x1 pass-through",
          (cols, rows) == (1, 1) and torch.equal(out_s[0], frames[0]))

    # -- 17j. the two nodes chain: prepare's frames become the sheet's cells
    out_p, _ = prepare.execute(image=fixture).result
    out_s, cols, rows, cw, ch = sheet.execute(image=out_p, layout=None).result
    check("17j prepare -> sheet chains with the cell size carried through",
          (cols, rows, cw, ch) == (2, 2, 15, 15) and tuple(out_s.shape) == (1, 30, 30, 4),
          str((cols, rows, cw, ch, tuple(out_s.shape))))

    # -- 17k. determinism
    out1, bg1 = prepare.execute(image=fixture).result
    out2, bg2 = prepare.execute(image=fixture).result
    s1 = sheet.execute(image=frames, layout=None).result
    s2 = sheet.execute(image=frames, layout=None).result
    check("17k both nodes are deterministic",
          torch.equal(out1, out2) and bg1 == bg2 and torch.equal(s1[0], s2[0])
          and s1[1:] == s2[1:])


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
    section_grid_schema(nodes)
    try:
        section_grid_contract(nodes)
    except Exception:
        traceback.print_exc()
        _fail("grid runtime-contract section crashed", "see traceback above")
    try:
        section_grid_port(nodes, DEFAULT_BASELINE)
    except Exception:
        traceback.print_exc()
        _fail("max_pixel_size port section crashed", "see traceback above")
    section_palette_schema(nodes)
    section_locale(nodes)
    for name, fn in (("video schema", section_video_schema),
                     ("video runtime contract", section_video_contract),
                     ("video behaviour", section_video_behaviour)):
        try:
            fn(nodes)
        except Exception:
            traceback.print_exc()
            _fail(f"{name} section crashed", "see traceback above")
    try:
        section_apply_validation(nodes)
    except Exception:
        traceback.print_exc()
        _fail("Apply framework-validation section crashed", "see traceback above")
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
    for name, fn in (("sprite schema", section_sprite_schema),
                     ("sprite runtime contract", section_sprite_contract),
                     ("sprite behaviour", section_sprite_behaviour)):
        try:
            fn(nodes)
        except Exception:
            traceback.print_exc()
            _fail(f"{name} section crashed", "see traceback above")

    print("\n=== summary ===")
    print(f"  checks: {CHECKS}  failures: {len(FAILURES)}")
    for name in FAILURES:
        print(f"    - {name}")
    print("  RESULT: " + ("ALL PASS" if not FAILURES else "FAILED"))
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    raise SystemExit(main())
