# Embedded Python DeepDetect Library v0.1

## Status and scope

This document specifies version 0.1 of the in-process `deepdetect` Python
package. The existing `dd_client` package remains the supported HTTP client
and is not renamed, wrapped, or otherwise changed.

Version 0.1 supports Linux x86-64, CPython 3.10 through 3.13, and DeepDetect
SDK installations built with Torch CPU or Torch CUDA. Other compiled backends
may work but are outside the compatibility guarantee. CUDA, cuDNN, libtorch,
models, and `libdeepdetect.so` are supplied by the SDK and are not bundled in
the Python distribution.

The goals are in-process service creation, training, status inspection,
cancellation, and prediction; stable Python errors; and a narrow native ABI.
Chains, resources, streams, pandas and Torch adapters, zero-copy tensors,
remote transport unification, isolated registries, and binary wheels are
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

```shell
cmake -S . -B build3 [DeepDetect build options]
cmake --build build3 -j
```

Use the same Torch CPU or Torch CUDA options intended for the Python runtime.
The Python package does not rebuild DeepDetect or change its enabled
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

### 3. Build the Python wheel

Use the Python interpreter matching the environment where the wheel will be
installed. Build dependencies are installed automatically by pip in an
isolated build environment.

```shell
mkdir -p dist/python

CCACHE_DISABLE=1 python -m pip wheel ./bindings/python \
  --no-deps \
  --wheel-dir dist/python \
  --config-settings=cmake.define.DeepDetect_DIR="$PWD/build3/install/lib/cmake/DeepDetect"
```

The result is a platform- and interpreter-specific wheel similar to:

```text
dist/python/deepdetect-0.1.0-cp312-cp312-linux_x86_64.whl
```

The exact `cpXY` tag depends on the Python interpreter used to build it.

`CMAKE_PREFIX_PATH` is also supported:

```shell
CMAKE_PREFIX_PATH="$PWD/build3/install" \
CCACHE_DISABLE=1 \
python -m pip wheel ./bindings/python \
  --no-deps \
  --wheel-dir dist/python
```

All SDK paths must be absolute because scikit-build-core configures CMake in
a temporary build directory. `DeepDetect_DIR` is the preferred, unambiguous
form. `CCACHE_DISABLE=1` is needed only when ccache cannot write to its
configured cache directory.

### 4. Install the wheel

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

### 5. Configure runtime library discovery

The wheel contains `deepdetect._native` and Python sources only. It does not
bundle `libdeepdetect.so`, libtorch, torchvision, OpenCV, protobuf, CUDA, or
cuDNN. The dynamic loader must be able to find the exact libraries used by
the DeepDetect build.

At minimum, add the staged SDK library directory:

```shell
export LD_LIBRARY_PATH="$PWD/build3/install/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

When DeepDetect dependencies were built inside `build3` rather than installed
system-wide, add their library directories as well. For a typical source
build:

```shell
export LD_LIBRARY_PATH="$PWD/build3/install/lib:\
$PWD/build3/opencv/opencv-4.13.0/build/lib:\
$PWD/build3/protobuf/src/protobuf-build:\
$PWD/build3/pytorch/src/pytorch/torch/lib:\
$PWD/build3/pytorch_vision/src/pytorch_vision-install/lib\
${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Use `ldd build3/install/lib/libdeepdetect.so` to identify any additional
unresolved libraries required by the local CUDA or system configuration.

### 6. Verify the installation

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
failure domain. Bundled wheels simplify installation but are deferred because
libtorch and CUDA are large and strongly platform-specific.

The JSON façade leaves room for a future stable C ABI, isolated runtime
instances, transport-neutral clients, and optional buffer/tensor entry points
without changing the v0.1 Python objects.

## Acceptance criteria

Version 0.1 is accepted when:

- Python unit tests cover requests, normalization, errors, NumPy encoding,
  deleted handles, polling, timeout, and cancellation using a fake runtime.
- Native façade tests cover valid and malformed JSON, error responses, and
  exception containment.
- The package builds and imports against installed CPU Torch SDKs on CPython
  3.10 through 3.13.
- CPU integration creates a Torch service, trains briefly, predicts, checks
  status, and deletes it.
- A GPU runner verifies capability reporting and CUDA training/prediction
  without CPU fallback.
- Concurrent prediction is allowed while service deletion waits for active
  façade operations.
- `dd_client` packaging and imports are unchanged.
- An sdist installs using only the documented SDK discovery mechanism.
