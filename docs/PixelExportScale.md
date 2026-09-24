# `PixelExportScale`

**Category:** `image/pixel pipeline`
**Output kind:** processing node (last stage)

Nearest-neighbour integer upscale of a pixel image, plus the resulting size as
outputs so a downstream Save/Resize node can be driven without guessing.

```
IMAGE -> image (IMAGE), width (INT), height (INT)
```

## Inputs

| id | type | default | range | notes |
|---|---|---|---|---|
| `image` | IMAGE | — | — | Pixel image to upscale, e.g. the output of `PixelGridRestore`. |
| `scale` | INT | `4` | 1–16 | Integer factor. `1` is a pass-through. Output is `input size * scale`. |
| `mode` | COMBO | `Nearest` | `Nearest` | Resampling filter. |

## Outputs

| display name | type | notes |
|---|---|---|
| `image` | IMAGE | Upscaled image, `[B, H*scale, W*scale, 3]`. |
| `width` | INT | Resulting width = input W × `scale`. |
| `height` | INT | Resulting height = input H × `scale`. |

## Why only `Nearest`

Every source pixel becomes an exact `scale × scale` block. Bilinear, bicubic or
Lanczos would reintroduce exactly the soft gradients that `PixelGridRestore`
spent effort removing, so those filters are not offered. The combo exists so
further filters can be added later without breaking saved workflows; passing
anything else raises a `ValueError`.

The implementation is `torch.nn.functional.interpolate(mode="nearest")`, the same
one the legacy V1 node used.

## Scaling is not a resample

`scale` is an integer block replication, not a resample to a target size. To reach
a specific output resolution, pick `scale` such that `input * scale` is close
enough, or follow this node with a normal Resize node — the pixel art is already
crisp at that point, so an area filter thereafter is harmless.

## Example

| input | `scale` | result |
|---|---|---|
| 125×122 | 4 | 500×488, `width=500`, `height=488` |
| 125×122 | 1 | pass-through, 125×122 |
| 64×64 | 8 | 512×512 |

## See also

- [`PixelGridRestore`](PixelGridRestore.md) — normally the producer of `image`
- [`PixelPaletteQuantize`](PixelPaletteQuantize.md) — insert before this node to
  quantise, or after it to quantise at the final resolution (quantising *before*
  the upscale is faster and gives identical colours, since the upscale is an
  exact block replication)
