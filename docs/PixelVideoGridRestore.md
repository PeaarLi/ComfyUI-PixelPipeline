# PixelVideoGridRestore — video v2

Restores each frame onto one regular pixel grid. Image nodes and shared image
algorithms are unchanged. The nine output slots keep their previous ordering.

## Recommended starting point

For the detailed MiniMax knight clip used in verification:

- `sampling`: **Robust Cell**
- `grid_mode`: **By Pixel Size**, `pixel_size=3` (245×245 from 736×736)
- `temporal_window`: **0** for the baseline, **1** for conservative static-region stabilisation
- `chunk_frames`: **12**
- Downstream palette: one palette for the whole batch, dithering **None**
- Downstream export: integer **Nearest** scaling

Use a larger pitch for coarser art, or Center Sample if fine highlights matter
more than noise reduction. The direct `video` output is unquantised; connect
`images` to Palette Extract → Palette Apply → Export Scale to limit colours.

## What changed

1. **Exact regular geometry.** By Grid Count produces exactly the requested
   dimensions; By Pixel Size uses rounded width/pitch and height/pitch. Cells
   cover the full frame, including fractional boundaries. No reference-frame
   grid snapping, square correction or frame substitution remains.
2. **Current-frame sampling.** Robust Cell samples nine centre-weighted points,
   deduplicates repeated source positions in small cells, and selects a weighted
   medoid in CIE Lab. It returns an actual current-cell sample. Centre-first tie
   breaking has no RGB ordering bias. Transparent hidden colours have zero weight.
3. **Conservative temporal selection.** The filter rejects local source-patch
   changes, edges and their one-cell neighbourhood, transparency and cuts. It
   chooses among current-cell candidates with bounded colour/cost changes.
   It never averages temporal RGB or feeds processed history back into itself.
4. **Evidence-based detection.** Auto uses individual boundary spectra and paired
   grid candidates from at most 32 frames, including both endpoints. Insufficient
   consensus falls back to an explicit style pitch; it does not invent a detected grid.
5. **Both alpha routes work.** RGBA or `VideoComponents.alpha` are supported.
   The separate slot wins if both exist. RGB and alpha share spatial coordinates;
   alpha coverage is sampled in the current frame and never temporally filtered.

## Inputs

| Input | Meaning |
|---|---|
| `sampling` | Robust Cell (new default), Center Sample, Majority Cluster or Median |
| `grid_mode` | Auto Detect, By Pixel Size or By Grid Count; existing names retained |
| `sample_stride` (Auto) | Approximate probe spacing; uniform sampling, capped at 32, includes endpoints |
| `min_size`, `max_pixel_size` (Auto) | Accepted pitch interval in source pixels |
| `peak_width` (Auto) | **Changed meaning:** minimum repeated cells per axis; retained name for workflow compatibility |
| `pixel_size` (manual) | Requested pitch; actual width/grid_w can be fractional |
| `grid_w`, `grid_h` (manual) | Exact dimensions, including rectangular grids |
| `temporal_window` | 0 off; 1–8 frames each side, with consecutive pair validity required along each path |
| `backend` | Only selects the legacy Majority/Median implementation; other video operations use NumPy |
| `chunk_frames` | uint8 conversion batch size; 0 all frames. Does not change results |
| `fallback_pixel_size` | Auto fallback pitch, default 8; a style choice, not a detected property |

Majority/Median still use the chosen upstream sampler on a regular grid for
ordinary RGB downsampling. For alpha or grids larger than the source they use
visible bounded medoid candidates, avoiding hidden black pixels and empty cells.
No OpenCV backend is needed by Robust Cell, Center or the new detector itself.

## Temporal limitations

This release deliberately stabilises **flat, near-static regions**. It does not
estimate optical flow or stabilise moving silhouettes. Some flicker around moving
details will remain. Large windows cannot overwrite the current cell with old
black contours: eligible replacements are current-frame candidates within 3 Lab
units of the spatial baseline, with a limited increase in spatial cost.

A real single-frame event is protected when it changes local structure/colour;
very subtle changes in flat noisy regions may still be treated as noise. Compare
Off and 1. Leave Off for content where such changes are intentional. Motion
compensation remains a possible future extension, not an advertised feature.

## Migration

After restarting ComfyUI to load the Python changes, refresh the browser to load
`web/video_migration.js`. When loading a saved frontend workflow, it removes the
video-only `refine_intensity` and `fix_square` widgets before positional values
are assigned, inserts fallback size 8, and marks the node as v2. It supports named
widget values, regular positional saves and nested graph definitions. Connections
to removed video inputs are discarded; image nodes are never migrated.

The node ID, grid-mode names, existing sampling names, window values and nine
output slots remain unchanged. The **algorithm and peak_width meaning changed**,
so old renders are not expected to reproduce pixel-for-pixel. Save the migrated
workflow to a new filename if you need to retain an old-render configuration.

API prompt JSON should remove `refine_intensity` and `fix_square`; add
`fallback_pixel_size` only if a different fallback is desired. Direct Python
callers must remove those two obsolete keyword arguments. Workflow migration is
covered by Node.js fixtures; a full interactive frontend round trip is not part
of the automated Python tests.

## Outputs and preservation

`video, images, audio, fps, grid_w, grid_h, pixel_size, detected, confidence`

- Every input frame yields exactly one output frame in order. Original audio
  data and the in-graph rational frame rate are retained. `fps` is a float and a
  downstream CreateVideo may not preserve an exact non-integer Fraction.
- `detected=True` requires Auto evidence. Manual grids and Auto fallback return
  false. `confidence` is a spectral evidence diagnostic, not visual quality;
  manual pitches outside the detector's candidate interval report zero.
- Images are RGB or RGBA; fully opaque output drops alpha. VIDEO components use
  RGB images with alpha in its separate slot. The core encoder can discard alpha.
- Processing is **8-bit SDR oriented**. Source encoding bit depth and colour-space
  tags are forwarded, but this does not preserve HDR precision; HDR emits a notice.
- Encoded 4:2:0 video does not preserve exact palette colours. Verify palette
  membership on `images` or lossless PNGs before encoding.

## Memory and validation

Input decoding still materialises the full video and output IMAGE is a full
batch. Conversion is chunked, detection keeps only spectra, and temporal candidate
storage is bounded by the window rather than the video length. Larger grids and
windows increase candidate storage. This is not a streaming decoder.

Run with the ComfyUI environment (its venv interpreter, from this directory):

```powershell
python -m unittest discover -s tests -p test_video_pipeline.py -v
node tests/test_video_migration.mjs
python _selftest.py
python tools/verify_video.py <clip.mp4> --output _selftest_out/video_v2/check --pixel-size 3 --window 1
```

The real-video verifier saves pre-encoding frames, a nearest-scaled preview,
parameters, hashes, processing time and RSS measurements. See
[the design and review criteria](VideoPipelineRedesignPlan.md) and
[the implementation report](VideoPipelineV2Results.md).
