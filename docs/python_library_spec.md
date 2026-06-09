# Embedded Python DeepDetect Library v0.1

## Status and scope

This document specifies version 0.1 of the in-process `deepdetect` Python
package. The existing `dd_client` package remains the supported HTTP client
and is not renamed, wrapped, or otherwise changed.

Version 0.1 supports Linux x86-64, CPython 3.10 through 3.13, and bundled
DeepDetect wheels built against `torch==2.12.*`. The wheel contains
`libdeepdetect.so`, the private `_native` extension, `libtorchvision.so`, and
protobuf runtime libraries built with DeepDetect. LibTorch and NVIDIA CUDA
runtime libraries are supplied by the PyTorch and NVIDIA Python wheels.

The goals are in-process service creation, training, status inspection,
cancellation, and prediction; stable Python errors; and a narrow native ABI.
Chains, resources, streams, pandas and Torch adapters, zero-copy tensors,
remote transport unification, isolated registries, and non-Linux wheels are
non-goals for v0.1.

## Public API

The distribution and import name are `deepdetect`; its version is `0.1.0`.
The native extension is private and imported as `deepdetect._native`.

```python
import deepdetect

dd = deepdetect.DeepDetect()
print(dd.build_info)
print(dd.info())

service = dd.create_service(
    "classifier",
    model={"repository": "/models/classifier"},
    mllib="torch",
    input_parameters={"connector": "image"},
    mllib_parameters={"template": "resnet18", "nclasses": 10, "gpu": True},
    output_parameters={},
)

result = service.predict(
    ["image.jpg"],
    input_parameters={"width": 224, "height": 224},
    output_parameters={"best": 5},
)
service.delete()
```

`DeepDetect.info()` returns the API `head`. `DeepDetect.build_info` is a
dictionary containing the DeepDetect version, commit, branch, build type,
enabled compile flags and dependency versions, CUDA mode, and cuDNN
availability.

`DeepDetect.create_service()` returns a `Service`. Names are normalized to
lowercase. `DeepDetect.service(name)` creates a handle to an existing
process-wide service without retaining a native service pointer.

`Service.info()` and `Service.predict()` return the API `body`.
`Service.delete(clear=None)` returns `None`; after successful deletion that
handle raises `RuntimeError` on reuse. `Service` is a context manager and
deletes the service when leaving the `with` block, including when prediction
or training raises:

```python
try:
    with dd.create_service(...) as service:
        result = service.predict(["image.jpg"])
except deepdetect.DeepDetectError as error:
    print(error.status_code, error.dd_code, error.message)
```

```python
metrics = service.train(
    ["/data/train"],
    input_parameters={"test_split": 0.1},
    mllib_parameters={"solver": {"iterations": 100}},
    output_parameters={"measure": ["acc"]},
)

job = service.train(
    ["/data/train"],
    mllib_parameters={"solver": {"iterations": 10000}},
    asynchronous=True,
)
current = job.status()
final = job.wait(timeout=3600, poll_interval=1.0)
job.cancel()
```

Blocking training returns the API `body`. Asynchronous training returns a
`TrainingJob`. Job status combines the status, job id, and elapsed time from
the API `head` with fields from the API `body`. `wait()` returns on
`finished`, `error`, `terminated`, or `cancelled`, and raises `TimeoutError`
when its deadline expires. Cancellation is administrative and returns
`None`.

## Inputs

Data accepts strings, `os.PathLike` objects, JSON-compatible scalars,
mappings, and lists. Path-like values are converted with `os.fspath`.

Image services also accept NumPy `uint8` arrays shaped `H x W`, `H x W x 3`,
or `H x W x 4`. The Python layer encodes each array as PNG and passes its
base64 text to the existing image connector. Other dtypes and shapes raise
`TypeError`. PIL objects, pandas objects, Torch tensors, shared batch
buffers, and zero-copy adapters are deferred.

## Responses and errors

The native façade always returns the complete DeepDetect JSON envelope.
Python validates `status.code` before normalizing the result. A non-2xx code
raises `DeepDetectError` with:

- `status_code`: HTTP-style DeepDetect status
- `dd_code`: optional DeepDetect-specific code
- `message`: `dd_msg`, then `msg`, then a generic fallback
- `response`: the complete decoded response

`CapabilityError` is a `DeepDetectError` subclass. A request with
`parameters.mllib.gpu=true` is rejected before dispatch when `build_info`
reports a CPU-only SDK. Invalid Python argument types raise `TypeError` or
`ValueError`. Invalid JSON or unexpected C++ exceptions are contained at the
native boundary and returned as stable 400 or 500 DeepDetect envelopes;
C++ exceptions never cross the extension boundary.

## Native architecture

`libdeepdetect.so` exports `DeepDetect::ddetect` as an installable CMake
target and installs one public PIMPL header. The façade accepts and returns
UTF-8 JSON strings. RapidJSON documents, service variants, backend classes,
generated protobuf types, and internal headers are not public ABI.

The extension uses one process-wide native `Runtime`. Every `DeepDetect`
instance references it, so the service registry is process-wide. `Service`
stores only a normalized name and Python runtime reference.

The façade has a reader/writer lock. Service creation and deletion take
exclusive access. Information, prediction, training, status, and cancellation
take shared access while existing service-level locks continue to serialize
backend conflicts. The extension releases the Python GIL for all native
operations, including service/model loading.

The runtime is not fork-safe after initialization. Multiprocessing programs
must select `spawn` before constructing `DeepDetect`. The native singleton is
intentionally retained until process exit rather than destructed during
Python finalization: CUDA may already be shutting down when extension statics
are destroyed. The operating system reclaims its process resources.
Applications must still delete services, preferably with `with`, and
explicitly wait for or cancel asynchronous jobs before shutdown.

## Build and installation

The SDK installation contains:

```text
include/deepdetect/runtime.h
lib/libdeepdetect.so.0
lib/cmake/DeepDetect/DeepDetectConfig.cmake
lib/cmake/DeepDetect/DeepDetectTargets.cmake
```

The Python project uses pybind11 and scikit-build-core. The DeepDetect build
tree is not itself an SDK prefix: DeepDetect must first be installed into a
staging prefix containing its library, public header, and CMake package.

The commands below start from the DeepDetect repository root and use
`build3` as a fresh native build directory.

### 1. Build DeepDetect

#### Recommended: official prebuilt LibTorch

DeepDetect supports the official LibTorch 2.12.0 distribution. Set
`LIBTORCH_ROOT` to the absolute extraction directory:

```shell
export LIBTORCH_ROOT=/absolute/path/to/libtorch

cmake -S . -B build3 \
  -DUSE_TORCH=ON \
  -DUSE_PREBUILT_TORCH=ON \
  -DCMAKE_PREFIX_PATH="$LIBTORCH_ROOT"
cmake --build build3 -j
```

For a CUDA build, use an official CUDA 12.6 or CUDA 13 LibTorch package.
CUDA, cuDNN 9, cuSPARSELt 0, NVSHMEM 3, and the NCCL runtime matching that
LibTorch package must be installed on the host. CMake checks required NCCL
symbols and rejects an older incompatible runtime. It also searches common
NVIDIA Python package locations such as the active virtualenv, conda
environment, and `$HOME/venv` for cuDNN 9, cuSPARSELt, and NVSHMEM. When
those libraries are elsewhere, pass their directories through
`CMAKE_LIBRARY_PATH`:

```shell
cmake -S . -B build3 \
  -DUSE_TORCH=ON \
  -DUSE_PREBUILT_TORCH=ON \
  -DCMAKE_PREFIX_PATH="$LIBTORCH_ROOT" \
  -DCMAKE_LIBRARY_PATH="/path/to/cudnn/lib;/path/to/nccl/lib;/path/to/cusparselt/lib;/path/to/nvshmem/lib"
```

The build downloads torchvision 0.27.0 and compiles it against the selected
LibTorch. To reuse an existing source checkout or build without network
access, add:

```shell
-DTORCHVISION_SOURCE_DIR=/absolute/path/to/vision
```

For the official CPU-only LibTorch package, add `-DUSE_CPU_ONLY=ON` and
`-DUSE_TORCH_CPU_ONLY=ON`. `Torch_DIR` may be passed directly as
`"$LIBTORCH_ROOT/share/cmake/Torch"` instead of using `CMAKE_PREFIX_PATH`.

#### Alternative: build PyTorch from source

```shell
cmake -S . -B build3 -DUSE_TORCH=ON [other DeepDetect build options]
cmake --build build3 -j
```

With manual CMake, omitting `USE_PREBUILT_TORCH` retains the existing patched
PyTorch source build. When using `build.sh`, set `USE_PREBUILT_TORCH=OFF` for
that older path. Use the same Torch CPU or CUDA mode intended for the Python
runtime. The Python package does not rebuild DeepDetect or change enabled
backends.

### 2. Install the DeepDetect SDK into a staging prefix

```shell
cmake --install build3 --prefix "$PWD/build3/install"
```

Verify that the install step produced at least:

```text
build3/install/include/deepdetect/runtime.h
build3/install/lib/libdeepdetect.so
build3/install/lib/cmake/DeepDetect/DeepDetectConfig.cmake
build3/install/lib/cmake/DeepDetect/DeepDetectTargets.cmake
```

`build3/DeepDetectConfig.cmake` alone is not sufficient. Consumers must use
the installed package under `build3/install/lib/cmake/DeepDetect`.

### 3. Build the bundled Python wheel

Use the Python interpreter matching the environment where the wheel will be
installed. Install `torch==2.12.*`, `auditwheel`, and the Python build tools,
then run the repository helper:

```shell
python -m pip install "torch==2.12.*" "auditwheel>=6" scikit-build-core pybind11
python bindings/python/scripts/build_wheel.py
```

The helper configures DeepDetect with `USE_PREBUILT_TORCH=ON` and
`CMAKE_PREFIX_PATH` from `torch.utils.cmake_prefix_path`, discovers NVIDIA
runtime paths from installed Python packages, installs a temporary SDK under
`build/python-wheel/install`, and builds the scikit-build wheel. The result is
a platform- and interpreter-specific wheel similar to:

```text
dist/python/deepdetect-0.1.0-cp312-cp312-linux_x86_64.whl
```

The exact `cpXY` tag depends on the Python interpreter used to build it.
Use `--cmake /path/to/cmake` or `CMAKE_COMMAND=/path/to/cmake` if the desired
CMake executable is not first on `PATH`.
Use `--cuda-architectures`, for example `--cuda-architectures 86`, when CMake
cannot infer a CUDA architecture for the installed PyTorch wheel.
Use `--repair` to run `auditwheel repair`; this is useful for inspection but
may over-vendor host OpenCV/GUI/system libraries on non-manylinux hosts. Use
`--reuse-raw-wheel --repair` to rerun only the repair step after a successful
raw wheel build.

To reuse an existing torchvision 0.27.0 checkout, add:

```shell
python bindings/python/scripts/build_wheel.py \
  --torchvision-source-dir /absolute/path/to/vision
```

### 4. Build CPU and GPU release variants

CPU and GPU wheels for the same Python version and platform cannot share the
same PyPI project name. The release helper builds two distribution names that
both install the `deepdetect` import package. The defaults are
`deepdetect-cpu` and `deepdetect-gpu`, and they must not be installed together
in one environment.

By default the release helper creates temporary build environments under
`build/python-wheel-envs`, installs the CPU and GPU PyTorch dependencies,
builds both wheels, and removes those environments when the build completes.

```shell
python bindings/python/scripts/build_release_wheels.py \
  --cmake /usr/bin/cmake \
  --jobs 4
```

Pass `--keep-envs` to keep those environments for inspection or reuse. The
CPU environment installs PyTorch from `https://download.pytorch.org/whl/cpu`.
The GPU environment installs PyTorch from the default pip indexes unless
`--gpu-torch-index-url` is provided.

The output directories are:

```text
dist/python/release/cpu/
dist/python/release/gpu/
```

Use `--cpu-python` and `--gpu-python` to build with existing environments
instead. `--cpu-python` must point to a CPU-only PyTorch environment, and
`--gpu-python` must point to a CUDA-enabled PyTorch environment.

Use `--cpu-name` and `--gpu-name` to select the actual PyPI project names.
Use `--skip-cpu` or `--skip-gpu` to build one variant. CUDA-specific options
such as `--cuda-architectures`, `--cuda-compiler`,
`--torchvision-source-dir`, and `--repair` are forwarded to the underlying
wheel builder.

For an existing staged SDK, the lower-level scikit-build command remains
available:

```shell
python -m pip wheel ./bindings/python \
  --no-deps \
  --wheel-dir dist/python \
  --config-settings=cmake.define.DeepDetect_DIR="$PWD/build3/install/lib/cmake/DeepDetect"
```

Set `cmake.define.DEEPDETECT_PYTHON_BUNDLE_NATIVE=OFF` only for the old thin
wheel that requires an external SDK and loader path configuration.

### 5. Install the wheel

Install the generated filename using the target environment's Python:

```shell
python -m pip install \
  dist/python/deepdetect-0.1.0-cp312-cp312-linux_x86_64.whl
```

Use `--force-reinstall` when replacing an existing development installation:

```shell
python -m pip install --force-reinstall --no-deps \
  dist/python/deepdetect-0.1.0-cp312-cp312-linux_x86_64.whl
```

For release variants, install exactly one of the CPU or GPU distribution
wheels:

```shell
python -m pip install --force-reinstall \
  dist/python/release/gpu/deepdetect_gpu-0.1.0-cp312-cp312-linux_x86_64.whl
```

### 6. Runtime library discovery

No `LD_LIBRARY_PATH` is required for the bundled DeepDetect libraries. The
extension imports `torch` before loading `_native`, validates that the
installed torch version is `2.12.*`, and relies on wheel RPATHs to find
DeepDetect libraries beside `_native` and libtorch under `site-packages/torch`.
Use `ldd` on the installed `deepdetect/_native*.so` to diagnose unusual host
CUDA runtime configurations.

Only the thin-wheel developer mode with
`DEEPDETECT_PYTHON_BUNDLE_NATIVE=OFF` requires manual loader configuration.
For the legacy PyTorch source build, a typical external-SDK loader path is:

```shell
export LD_LIBRARY_PATH="$PWD/build3/install/lib:\
$PWD/build3/opencv/opencv-4.13.0/build/lib:\
$PWD/build3/protobuf/src/protobuf-build:\
$PWD/build3/pytorch/src/pytorch/torch/lib:\
$PWD/build3/pytorch_vision/src/pytorch_vision-install/lib\
${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

In that mode, use `ldd build3/install/lib/libdeepdetect.so` to identify any
additional unresolved libraries required by the local CUDA or system
configuration.

### 7. Verify the installation

```shell
python -c "import deepdetect; print(deepdetect.DeepDetect().build_info)"
python bindings/python/examples/smoke_test.py
```

For an isolated virtual environment, use that environment's interpreter for
both wheel creation and installation, for example
`/path/to/venv/bin/python -m pip ...`.

Before publication, reserve the `deepdetect` project name on PyPI. If it is
unavailable, publish the distribution as `deepdetect-native` while preserving
`import deepdetect`.

## Design alternatives

Directly binding `JsonAPI` would require less glue but would expose unstable
RapidJSON and internal C++ types. A C ABI through ctypes or cffi would improve
language and compiler portability but requires explicit allocation, lifetime,
and error-memory protocols. An embedded HTTP server or subprocess would reuse
`dd_client` but adds serialization, ports, process management, and a second
failure domain. Bundling libtorch and CUDA directly is avoided because those
libraries are large and are already supplied by the PyTorch wheel stack.

The JSON façade leaves room for a future stable C ABI, isolated runtime
instances, transport-neutral clients, and optional buffer/tensor entry points
without changing the v0.1 Python objects.

## Acceptance criteria

Version 0.1 is accepted when:

- Python unit tests cover requests, normalization, errors, NumPy encoding,
  deleted handles, polling, timeout, and cancellation using a fake runtime.
- Native façade tests cover valid and malformed JSON, error responses, and
  exception containment.
- The package builds bundled Linux wheels against `torch==2.12.*` on CPython
  3.10 through 3.13.
- CPU integration creates a Torch service, trains briefly, predicts, checks
  status, and deletes it.
- A GPU runner verifies capability reporting and CUDA training/prediction
  without CPU fallback.
- Concurrent prediction is allowed while service deletion waits for active
  façade operations.
- `dd_client` packaging and imports are unchanged.
- An sdist installs using only the documented SDK discovery mechanism.
