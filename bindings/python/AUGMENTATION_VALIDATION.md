# Augmentation implementation validation

Validated on 2026-10-21 with Python 3.12, GCC 13.3.0 and OpenCV 4.6.0.
The implementation is described in [augmentation.md](deepdetect/augmentation.md).

## Correctness and compatibility

- 45 core Python augmentation tests pass against the compiled extension. These cover
  paired image/mask geometry, uint8/uint16 labels, source support for all padding
  modes, RGB/BGR conversion, grayscale restrictions, cutout support, clean views,
  replay/JSON, shared-policy threads, array ownership, rejected inputs and all
  individual noise/distortion operators.
- Three pre-refactor native image/mask fixtures match pixel-for-pixel in legacy
  mode, both on first application and replay. These binary fixtures are specific
  to OpenCV 4.6.0; other versions still run the analytic and replay tests.
- Nine native `imgdataaug` tests pass, including existing configuration, bounding
  box and crop tests, replay/mask parity and the opt-in signed brightness check.
- Eight native keypoint connector regressions and the reference detector's
  service-level connector-pull training test pass. The latter uses a tiny
  synthetic fixture and requires permission to bind a local worker socket.
- Worker runtime and semantic protocol regressions: 84 pass, including native
  semantic serialization through an external fixture worker.

The native tests were compiled against the configured GPU SDK using its existing
compiler/link settings. Enabling the full repository test build would fetch
unrelated datasets; only the affected fixture tests were executed.

## Isolation and packaging

The `_augmentation` dependency tree contains the shared augmentation library and
OpenCV dependencies, without LibTorch, CUDA or the DeepDetect server/runtime.
The import test also verifies that importing and using augmentation does not
import `torch` or `deepdetect._native`. The full DeepDetect wheel still declares
its existing model-runtime dependencies.

The source distribution includes the Python API, C++ binding, documentation,
preview example and legacy PNG/reference fixtures. The SDK exports the separate
`DeepDetect::image_augmentation` target and implementation fingerprint.

Installed-wheel checks exposed an import-order crash in the repaired wheel:
renamed GLVND libraries were mixed with the external host `libGL` stack. The
repair helper now excludes `libGLX`, `libOpenGL` and `libGLdispatch` together.
Two subprocess regression tests cover augmentation-first and runtime-first
loading. The augmentation operators themselves do not initialize a GUI or GPU.

The corrected wheel was installed into an isolated target directory, with no
development `LD_LIBRARY_PATH`. The complete affected Python suite passes all
131 tests (45 augmentation, two import-order, 84 worker/protocol), with no skips.
A separate smoke check exercises augmentation before and after loading PyTorch
2.12.1 and the DeepDetect runtime. Dependency inspection confirms the installed
augmentation extension resolves without LibTorch, CUDA or `libdeepdetect`.

The locally built artifact is
`dist/augmentation/deepdetect-0.1.0-cp312-cp312-manylinux_2_39_x86_64.whl`
(Python 3.12, x86-64, glibc 2.39 or newer). It includes the existing GPU-enabled
DeepDetect runtime; the augmentation API itself remains CPU-only. Both raw and
repaired wheels include the shared augmentation library and extension, and the
repaired wheel bundles OpenCV while retaining host GL dispatch dependencies.

To repeat the Python checks against an installed wheel:

```shell
DD_AUGMENTATION_NATIVE=1 DD_SEMANTIC_NATIVE=1 python -m pytest \
  bindings/python/tests/test_augmentation.py \
  bindings/python/tests/test_pytorch_worker_runtime.py \
  bindings/python/tests/test_semantic_worker.py -q
```

The semantic fixture requires permission to create a local worker socket.

## Bounded throughput measurement

The synthetic preview example measured 48 RGB 512×512 crops, with mirror,
rotation, contrast and occasional Gaussian blur:

| Path | Crops/second |
|---|---:|
| Identity including input/output copies | 3239 |
| One-call sampling/application | 213 |
| Replay application | 130 |
| One worker thread | 127 |
| Four worker threads | 413 |

These are short measurements taken while a native build was running, not an
isolated benchmark or a prediction of training throughput. The different
one-thread numbers show the scheduling variability. There is no comparison
against the cracks worker's previous pipeline.

## Deliberate first-version limits

`sample` captures native draws by executing on a synthetic image; use the one-call
API to avoid this extra pass for a single output. Clean replay currently executes
appearance on a temporary result to validate the complete plan. Plans are
restricted to matching shapes/policies/build fingerprints. Float images, soft
targets, cross-resolution replay, boxes/keypoints in the Python API and worker
integration remain separate work.
