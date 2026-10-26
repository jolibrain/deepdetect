# CPU image augmentation

`deepdetect.augmentation` wraps the same C++ image operators used by native
DeepDetect. It does not create a runtime/service, load images, normalize pixels,
convert tensors, or initialize CUDA. The extension links the separately installed
`DeepDetect::image_augmentation` SDK target and OpenCV, not LibTorch or the server.
It ships in the existing DeepDetect wheel; that wheel's Python dependencies have
not been split into a separate lightweight distribution.

```python
from deepdetect.augmentation import Augmenter, AugmentationPlan

augment = Augmenter({
    "mirror": True,
    "rotate": True,
    "crop_size": 256,
    "distort": {"prob": 0.5, "brightness": True,
                "brightness_range": [-32, 32], "contrast": True},
    "noise": {"probabilities": {"gauss_blur": 0.1, "inverse": 0.05}},
})
result = augment(image, seed=42, masks={"labels": labels})
saved = result.plan.to_json()
replayed = augment.apply(image, plan=AugmentationPlan.from_json(saved),
                        masks={"labels": labels})
clean = augment.apply(image, plan=result.plan, appearance=False)
# The caller chooses the supervision policy and ignore value:
eligible = result.in_image & ~result.occluded
```

Images must be NumPy uint8 arrays shaped HW (grayscale) or HWC with three
channels. RGB is the default; pass `color_order="BGR"` for OpenCV images.
Masks are optional named HW uint8/uint16 arrays. Sizes must agree. Unsupported
representations, including float images, int64 masks and soft target maps, fail
rather than silently changing dtype. Convert a mask explicitly only after
checking its IDs fit the chosen representation. Negative strides, noncontiguous
and read-only inputs are accepted through owned contiguous copies. Outputs never
alias inputs. Three-channel colour operations are rejected for grayscale inputs.

## Policies

`semantics="explicit_v1"` is the default. All operations start disabled. Empty
configuration is identity. A `noise`/`distort` family `prob` applies independently
to each explicitly enabled operation; it is not an outer gate. Use the optional
`probabilities` mapping for individual probabilities. A named entry enables that
operation unless its boolean is explicitly false (a conflicting positive
probability fails). Enabling a family probability without selecting operations
fails. Unknown keys and invalid ranges fail early.

Top-level options:

| Option | Behaviour |
|---|---|
| `mirror` | Horizontal mirror with probability 0.5 |
| `rotate` | Uniform 0/90/180/270 degrees; explicit mode requires square input |
| `crop_size` | Random square crop, positive integer fitting input; -1 disables |
| `cutout` | Probability of native random rectangular noise fill |
| `geometry` | Native perspective, translation and zoom policy |
| `noise`, `distort` | Independent operation gates described below |

Geometry keys are `prob`, `persp_horizontal`, `persp_vertical`,
`transl_horizontal`, `transl_vertical`, `zoom_out`, `zoom_in`, `pad_mode`,
`persp_factor`, `transl_factor`, and `zoom_factor`. Probability defaults to zero;
perspective and zoom flags default true, translation false. Factors default to
0.25, 0.5, and 0.25, respectively. Factors must be within [0,1]. Border modes are
`constant`, `mirrored`, and `repeat_nearest`.

Noise operation names and native magnitudes:

| Name | Magnitude |
|---|---|
| `decolorize` | Grayscale replicated across three channels |
| `gauss_blur` | Fixed 7×7, sigma 1.5 |
| `hist_eq` | Histogram equalization |
| `clahe` | Clip limit 4, OpenCV default tile grid |
| `jpg` | Uniform integer quality 0–99 |
| `erosion` | Fixed 3×3 elliptical element |
| `posterize` | Four levels, bin width 64 |
| `inverse` | Bitwise inversion |
| `saltpepper` | `saltpepper_fraction`, default 0.1, of position draws; repeats possible |
| `convert_to_hsv`, `convert_to_lab` | Direct transformed channel representations |

HSV/Lab conversion does **not** convert back to a conventional RGB image. These
are historical operators, not claims of plausible acquisition variation. Fixed
magnitudes above are deliberately documented rather than silently accepting
configuration fields that the C++ implementation ignores.

Distortion operations: `brightness`, `contrast`, `saturation`, `hue`, and
`channel_order`. Configurable sampling bounds are
`brightness_range` (default [-32,32] intensity codes), `contrast_range` and
`saturation_range` (default [0.5,1.5] multipliers), and `hue_range` (default
[-36,36] OpenCV hue codes). Contrast multiplies pixel values; it is not centred
on image mean. Explicit brightness supports both signs. Channel ordering uses a
random permutation. Noise follows native order; distortion randomly places
contrast before or after saturation/hue, as native DeepDetect does.

### Legacy compatibility

`semantics="legacy_v1"` accepts the native `mirror`, `rotate`, `crop_size`,
`cutout`, `geometry`, `noise: {prob: ...}`, and `distort: {prob: ...}` settings.
Noise/distortion use native default operation enablement (including grayscale
restrictions), rather than explicit-mode booleans/ranges. Non-square rotation is
disabled as in the native parser. Nonpositive integer crop sizes disable cropping.
Dataset/test-only options are outside this API.

The native training constructor leaves four distortion distributions at the C++
default [0,1]. The other constructor initializes the advertised ranges, but is
not used by that path. Also, the legacy brightness kernel ignores negative
offsets. Legacy mode preserves these quirks; explicit mode fixes both. Existing
native training callers retain their original behaviour.

## Geometry, labels and validity

Order: geometry → cutout → crop → mirror → rotation → noise → distortion.
Hard masks receive geometry/crop/mirror/rotation only. Perspective images and
masks retain native nearest-neighbour sampling. Constant mask padding retains
native zero values, **not** an implicit ignore label.

`result.in_image` is a boolean mask marking true source support. Reflected and
replicated synthetic image padding remains invalid. `result.occluded` marks
cutout support after crop/orientation, separately from source validity. If the
input already contains padding or untrusted regions, pass that support
map as an additional hard mask and combine its transformed values with
`in_image`; array dimensions alone cannot identify pre-existing invalid pixels.
The API does not overwrite labels or decide which training objective can use an occluded
pixel. It does not infer an occlusion mask for blur, compression or clipping.

`source_to_output` is a 3×3 homogeneous mapping of source pixel-centre coordinates
(x,y) to output coordinates, including the native enlarged-image translation.
Its inverse can be computed with `numpy.linalg.inv` for nonsingular mappings.
Warp validity is provided separately; a matrix alone does not establish support.

`appearance=False` returns the clean geometrically transformed image and zero
cutout occlusion with identical masks and geometric plan. It omits the effects
of cutout/noise/distortion. The original plan still records the sampled policy.
The initial implementation executes appearance on a temporary result to validate
complete replay, so requesting clean views does not avoid that CPU work.

## Plans, randomness and concurrency

`augment.sample(shape, seed=...)` returns a plan without reading source content.
The first version executes the operators on a synthetic image to capture exact
native sampling order. Prefer the one-call `augment(image, seed=...)` when only
one output is needed; this avoids an extra pass.

Plans contain actual scalar draws, operation order, crop/orientation, a geometric
matrix, compact engine snapshots for salt/pepper and permutations, and a cutout
field seed. Replay consumes recorded draws and per-field RNG state, never the
caller's Python/NumPy RNG. Cutout uses a local OpenCV RNG; no global RNG/thread
configuration is changed. A shared immutable `Augmenter` can be called from
multiple threads with independent plans. C++ processing releases the GIL.

JSON plans are immutable through the public API; metadata/config properties
return independent copies. Plans are validated against policy, input shape and
build fingerprint (source hashes, compiler and OpenCV). Do not promise portable
bitwise output across OpenCV/compiler upgrades. A changed fingerprint requires
new sampling or a new training run contract. New masks can be supplied when
replaying a view of matching shape. Plans do not support cross-resolution replay,
coordinate registration or probability/logit interpolation in this release.

There is no implicit clean-fraction or scanner clipping controller. Applications
own source sampling, record relationships, normalization, batching, resume
bookkeeping, augmentation telemetry and evaluation policies.

## Build and verification

Build and install the native SDK, then build the Python project against its
`DeepDetect_DIR`. The SDK exports `DeepDetect::image_augmentation` and its public
header. Wheel bundling includes `libdeepdetect_augmentation`; release-wheel repair
must also resolve OpenCV shared dependencies. Source distributions include both
binding sources. Missing extensions raise a dependency error when first used.
Wheel repair keeps GLVND dispatch libraries external, with the host GL stack,
to avoid mixing renamed dispatch libraries with system `libGL` dependencies.

Run `python -m pytest bindings/python/tests/test_augmentation.py` against the
built/installed package. Pre-refactor PNG fixtures use OpenCV 4.6.0 and BGR; those
binary checks are skipped on other OpenCV versions, while analytic/replay tests
remain active. Run native image-augmentation and affected connector regressions
when changing the shared implementation.
Set `DD_AUGMENTATION_NATIVE=1` when testing an installed full wheel to also check
that augmentation and the model runtime load correctly in either order.

`bindings/python/examples/augmentation_preview.py` creates a synthetic comparison
and JSON replay record; `--benchmark` additionally measures one-call and replay
throughput. Neither example requires the cracks worker or real datasets.
