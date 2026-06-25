# deepdetect

`deepdetect` embeds DeepDetect in the current Python process. The wheel
contains the Python package, the private `_native` extension, `libdeepdetect`,
`libtorchvision`, and protobuf runtime libraries built with DeepDetect.
LibTorch and NVIDIA CUDA runtime libraries are provided by the `torch==2.12.0`
wheel and its dependencies.

## Build a bundled wheel

Use the target Python interpreter from an environment that already contains
`torch==2.12.0` and the Python build tools:

```shell
python -m pip install "torch==2.12.0" "auditwheel>=6" scikit-build-core pybind11
python bindings/python/scripts/build_wheel.py
```

The helper configures DeepDetect with the prebuilt torch CMake package from
`torch.utils.cmake_prefix_path`, discovers NVIDIA runtime paths from installed
Python packages, installs a temporary SDK under `build/python-wheel/install`,
and builds the Python wheel. The final wheel is written to `dist/python`.
Use `--cmake /path/to/cmake` or `CMAKE_COMMAND=/path/to/cmake` if the desired
CMake executable is not first on `PATH`.
Use `--cuda-architectures`, for example `--cuda-architectures 86`, when CMake
cannot infer a CUDA architecture for the installed PyTorch wheel.
Use `--repair` to run `auditwheel repair`; this is useful for inspection but
may over-vendor host OpenCV/GUI/system libraries on non-manylinux hosts.
Use `--reuse-raw-wheel --repair` to rerun only the repair step after a
successful raw wheel build.

For offline torchvision builds, pass a matching torchvision 0.27.0 checkout:

```shell
python bindings/python/scripts/build_wheel.py \
  --torchvision-source-dir /absolute/path/to/vision
```

For non-standard CUDA runtime locations that are not provided by Python
packages, pass their directories to CMake:

```shell
python bindings/python/scripts/build_wheel.py \
  --cmake-library-path "/path/to/cudnn/lib;/path/to/nccl/lib"
```

## Build CPU and GPU release wheels

PyPI cannot publish two wheels with the same distribution name, version,
Python tag, ABI tag, and platform tag. Build the CPU and GPU artifacts as
separate distributions that both install the same import package,
`deepdetect`. The default release distribution names are `deepdetect-cpu` and
`deepdetect-gpu`; they are mutually exclusive in a Python environment because
both provide `import deepdetect`.

By default the release helper creates temporary build environments under
`build/python-wheel-envs`, installs the CPU and GPU PyTorch dependencies,
builds both wheels, and removes those environments when the build completes:

```shell
python bindings/python/scripts/build_release_wheels.py \
  --cmake /usr/bin/cmake \
  --jobs 4
```

Pass `--keep-envs` to keep the managed environments for inspection or reuse.
The CPU environment installs PyTorch from
`https://download.pytorch.org/whl/cpu`. The GPU environment installs PyTorch
from the default pip indexes unless `--gpu-torch-index-url` is provided.

The output is written under:

```text
dist/python/release/cpu/
dist/python/release/gpu/
```

Use `--cpu-python` and `--gpu-python` to build with existing environments
instead. `--cpu-python` must point to a CPU-only PyTorch environment, and
`--gpu-python` must point to a CUDA-enabled PyTorch environment.

Use `--cpu-name` and `--gpu-name` to choose different PyPI project names.
Use `--skip-cpu` or `--skip-gpu` to build only one variant. GPU builds accept
the same CUDA options as the single-wheel helper, for example:

```shell
python bindings/python/scripts/build_release_wheels.py \
  --cuda-architectures 86 \
  --cuda-compiler /usr/local/cuda/bin/nvcc
```

## Install and verify

Install the generated Linux wheel into the target Python environment. The
environment must provide the Python dependencies, especially `torch==2.12.0`,
`numpy`, and `Pillow`.

```shell
python -m pip install --force-reinstall \
  dist/python/deepdetect-0.1.0-*-linux_x86_64.whl
```

If you know the exact Python ABI tag, the wheel filename can be specified
directly:

```shell
python -m pip install --force-reinstall \
  dist/python/deepdetect-0.1.0-cp312-cp312-linux_x86_64.whl
```

For release variants, install exactly one of the CPU or GPU distributions:

```shell
python -m pip install --force-reinstall \
  dist/python/release/gpu/deepdetect_gpu-0.1.0-*-linux_x86_64.whl
```

Then verify import and native runtime initialization:

```shell
python -c "import deepdetect; print(deepdetect.DeepDetect().build_info)"
python bindings/python/examples/smoke_test.py
```

No `LD_LIBRARY_PATH` is required for the bundled DeepDetect libraries. If the
host uses a non-standard CUDA setup, verify missing dynamic dependencies with
`ldd` on the installed `deepdetect/_native*.so`.

## Developer SDK build

The lower-level scikit-build project can still build against an existing
staged SDK:

```shell
python -m pip wheel ./bindings/python \
  --no-deps \
  --wheel-dir dist/python \
  --config-settings=cmake.define.DeepDetect_DIR="$PWD/build/python-wheel/install/lib/cmake/DeepDetect"
```

Set `--config-settings=cmake.define.DEEPDETECT_PYTHON_BUNDLE_NATIVE=OFF` to
produce the old thin wheel that requires an external DeepDetect SDK and
runtime library path configuration.
