# deepdetect

`deepdetect` embeds an installed DeepDetect SDK in the current Python process.
From the DeepDetect repository root, build and stage the native SDK first:

```shell
cmake -S . -B build3 [DeepDetect build options]
cmake --build build3 -j
cmake --install build3 --prefix "$PWD/build3/install"
```

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
$PWD/build3/opencv/opencv-4.13.0/build/lib:\
$PWD/build3/protobuf/src/protobuf-build:\
$PWD/build3/pytorch/src/pytorch/torch/lib:\
$PWD/build3/pytorch_vision/src/pytorch_vision-install/lib\
${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Check `ldd build3/install/lib/libdeepdetect.so` for other local dependencies.
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
