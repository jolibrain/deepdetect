# deepdetect

`deepdetect` embeds an installed DeepDetect SDK in the current Python process.
From the DeepDetect repository root, build against the official LibTorch
2.12.0 package and stage the native SDK first:

```shell
export LIBTORCH_ROOT=/absolute/path/to/libtorch

cmake -S . -B build3 \
  -DUSE_TORCH=ON \
  -DUSE_PREBUILT_TORCH=ON \
  -DCMAKE_PREFIX_PATH="$LIBTORCH_ROOT"
cmake --build build3 -j
cmake --install build3 --prefix "$PWD/build3/install"
```

The CUDA package requires CUDA 13, cuDNN 9, cuSPARSELt 0, NVSHMEM 3, and the
NCCL runtime matching the LibTorch package. CMake searches the active
virtualenv, conda environment, and `$HOME/venv` NVIDIA package library
directories for cuDNN, cuSPARSELt, and NVSHMEM. Pass other non-standard
runtime directories with `-DCMAKE_LIBRARY_PATH="/path/one;/path/two"`. Add
`-DTORCHVISION_SOURCE_DIR=/absolute/path/to/vision` to reuse a local
torchvision 0.27.0 checkout. Omitting `USE_PREBUILT_TORCH` retains the
existing PyTorch source build.

Build a wheel using the target Python interpreter:

```shell
mkdir -p dist/python
CCACHE_DISABLE=1 python -m pip wheel ./bindings/python \
  --no-deps \
  --wheel-dir dist/python \
  --config-settings=cmake.define.DeepDetect_DIR="$PWD/build3/install/lib/cmake/DeepDetect"
```

Install the generated wheel, substituting its actual Python ABI tag:

```shell
python -m pip install \
  dist/python/deepdetect-0.1.0-cp312-cp312-linux_x86_64.whl
```

The wheel does not bundle libtorch, OpenCV, protobuf, CUDA, cuDNN, or
`libdeepdetect.so`. Configure the dynamic loader before importing it:

```shell
export LD_LIBRARY_PATH="$PWD/build3/install/lib:\
$LIBTORCH_ROOT/lib:\
$PWD/build3/opencv/opencv-4.13.0/build/lib:\
$PWD/build3/protobuf/src/protobuf-build:\
${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

The SDK install includes the matching `libtorchvision.so`. Add any
non-system CUDA runtime directories to `LD_LIBRARY_PATH`, then check
`ldd build3/install/lib/libdeepdetect.so` for unresolved dependencies.
See `docs/python_library_spec.md` for the full build and installation
contract.

Run the native import and information smoke test with:

```shell
python bindings/python/examples/smoke_test.py
```

To create a real Torch image service and predict:

```shell
python bindings/python/examples/smoke_test.py \
  --model-repository /models/classifier \
  --image image.jpg \
  --template resnet18 \
  --nclasses 10 \
  --gpu
```
