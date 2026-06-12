# Build DeepDetect From Source On Ubuntu 24.04

This guide documents the current source build flow used by `build.sh` and the
Dockerfiles. It is written as copy-pasteable steps for humans and automation
agents.

Supported build paths in this document:

- Torch CPU: training and inference on CPU.
- Torch GPU: training and inference with CUDA.
- TensorRT GPU: optimized inference from exported or compatible models.

Retired backends are not buildable. Do not pass `USE_CAFFE`, `USE_CAFFE2`,
`USE_TF`, or `USE_TF_CPU_ONLY`; CMake rejects those options. Use Torch or
ONNX/TensorRT instead.

## Build Layout

Run all commands from the repository root unless a step says otherwise:

```bash
cd /path/to/deepdetect
```

The main server binary is produced at:

```text
build/main/dede
```

`build.sh` is the recommended entry point. It configures CMake with the same
flags used by the Docker builds and uses prebuilt Torch by default.

## Common Ubuntu 24.04 Dependencies

Install CMake from Kitware and the system packages used by the Ubuntu 24.04
Docker builds:

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates gpg wget curl

wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc \
  | gpg --dearmor \
  | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null

echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ noble main' \
  | sudo tee /etc/apt/sources.list.d/kitware.list >/dev/null

sudo apt-get update
sudo rm -f /usr/share/keyrings/kitware-archive-keyring.gpg
sudo apt-get install -y kitware-archive-keyring
sudo apt-get install -y cmake
```

Install build dependencies:

```bash
sudo apt-get install -y \
  git \
  ccache \
  automake \
  build-essential \
  default-jdk \
  pkg-config \
  zip \
  g++ \
  gcc \
  zlib1g-dev \
  libgoogle-glog-dev \
  libgflags-dev \
  libeigen3-dev \
  libopencv-dev \
  libboost-dev \
  libboost-filesystem-dev \
  libboost-thread-dev \
  libboost-system-dev \
  libboost-iostreams-dev \
  libboost-program-options-dev \
  libboost-test-dev \
  libboost-regex-dev \
  libboost-date-time-dev \
  libboost-chrono-dev \
  libboost-stacktrace-dev \
  libssl-dev \
  libcurlpp-dev \
  libcurl4-openssl-dev \
  libopenblas-dev \
  libhdf5-dev \
  libleveldb-dev \
  libsnappy-dev \
  liblmdb-dev \
  libutfcpp-dev \
  rapidjson-dev \
  libmapbox-variant-dev \
  autoconf \
  libtool-bin \
  python3-numpy \
  python3-yaml \
  swig \
  unzip \
  python3-setuptools \
  python3-dev \
  python3-pip \
  python3-venv \
  python3-six \
  libgoogle-perftools-dev \
  libarchive-dev \
  libtcmalloc-minimal4 \
  bash-completion \
  libomp-dev \
  libomp5
```

Create a Python environment for the PyTorch package that CMake will discover:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install typing_extensions pyyaml numpy
```

Set the CMake compatibility policy used by the Docker builds:

```bash
export CMAKE_POLICY_VERSION_MINIMUM=3.5
```

## Torch CPU Build

Install CPU PyTorch and torchvision. DeepDetect currently builds against
PyTorch `2.12.0` and torchvision `0.27.0`.

```bash
. .venv/bin/activate
python -m pip install \
  torch==2.12.0 \
  torchvision==0.27.0 \
  --index-url https://download.pytorch.org/whl/cpu
```

Build DeepDetect:

```bash
rm -rf build
mkdir build
cd build

DEEPDETECT_ARCH=cpu \
DEEPDETECT_BUILD=default \
DEEPDETECT_RELEASE=OFF \
DEEPDETECT_DEFAULT_MODELS=false \
BUILD_TESTS=OFF \
USE_PREBUILT_TORCH=ON \
../build.sh

cd ..
```

Verify the binary:

```bash
./build/main/dede --help
```

## Torch GPU Build

Install the NVIDIA driver and CUDA toolkit before building. The default GPU
variant follows `docker/gpu.Dockerfile`: CUDA `13.0.2`, PyTorch CUDA index
`cu130`, and GPU architectures `7.5;8.0;8.6;8.9;9.0;10.0;12.0`.

Install CUDA PyTorch and torchvision:

```bash
. .venv/bin/activate
python -m pip install \
  torch==2.12.0 \
  torchvision==0.27.0 \
  --index-url https://download.pytorch.org/whl/cu130
```

Make CUDA tools and libraries visible:

```bash
export PATH="/usr/local/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
```

Build DeepDetect:

```bash
rm -rf build
mkdir build
cd build

DD_CUDA_VERSION=13.0.2 \
DEEPDETECT_ARCH=gpu \
DEEPDETECT_BUILD=default \
DEEPDETECT_GPU_VARIANT=default \
DEEPDETECT_RELEASE=OFF \
DEEPDETECT_DEFAULT_MODELS=false \
BUILD_TESTS=OFF \
USE_PREBUILT_TORCH=ON \
../build.sh

cd ..
```

For older GPUs that need compute capability 6.1 through 9.0, use the legacy
GPU variant and a CUDA 12.6 PyTorch wheel:

```bash
. .venv/bin/activate
python -m pip install \
  --force-reinstall \
  torch==2.12.0 \
  torchvision==0.27.0 \
  --index-url https://download.pytorch.org/whl/cu126

rm -rf build
mkdir build
cd build

DD_CUDA_VERSION=12.6.3 \
DEEPDETECT_ARCH=gpu \
DEEPDETECT_BUILD=default \
DEEPDETECT_GPU_VARIANT=legacy61 \
DEEPDETECT_RELEASE=OFF \
DEEPDETECT_DEFAULT_MODELS=false \
BUILD_TESTS=OFF \
USE_PREBUILT_TORCH=ON \
../build.sh

cd ..
```

## TensorRT Inference Build

TensorRT builds are GPU inference builds. The `build.sh` TensorRT profile uses:

- `DEEPDETECT_ARCH=gpu`
- `DEEPDETECT_BUILD=tensorrt`
- `USE_TENSORRT=ON`
- `USE_TORCH=OFF`
- `USE_CUDA_CV=ON`

You need CUDA, TensorRT headers, TensorRT libraries, and an OpenCV 4 build with
CUDA support.

### Recommended: Build In The TensorRT Container

The Docker build uses `docker/gpu_tensorrt.Dockerfile`, based on NVIDIA's
TensorRT image. This is the most reproducible way to build TensorRT support:

```bash
export DOCKER_BUILDKIT=1

docker build \
  -t jolibrain/deepdetect_gpu_tensorrt \
  -f docker/gpu_tensorrt.Dockerfile \
  .
```

### Native Host TensorRT Build

If building directly on Ubuntu 24.04, install TensorRT first using NVIDIA's
packages. The `build.sh` TensorRT profile expects TensorRT in the default
system paths:

- `NvInfer.h` and `NvInferVersion.h`
- `libnvinfer.so`
- `libnvinfer_plugin.so`
- `libnvonnxparser.so`

For x86_64 Ubuntu system packages, those paths are normally:

```text
/usr/include/x86_64-linux-gnu
/usr/lib/x86_64-linux-gnu
```

If TensorRT is installed from an SDK tarball or another non-system directory,
use the [Manual TensorRT](#manual-tensorrt) CMake commands so you can pass
`TENSORRT_DIR`, `TENSORRT_LIB_DIR`, or `TENSORRT_INC_DIR` explicitly.

Build with OpenCV CUDA support enabled. This makes `build.sh` download and build
OpenCV `4.13.0` under `build/opencv` before configuring DeepDetect:

```bash
rm -rf build
mkdir build
cd build

DEEPDETECT_ARCH=gpu \
DEEPDETECT_BUILD=tensorrt \
DEEPDETECT_RELEASE=OFF \
DEEPDETECT_DEFAULT_MODELS=false \
BUILD_TESTS=OFF \
BUILD_OPENCV=ON \
../build.sh

cd ..
```

If TensorRT version detection fails, pass it explicitly:

```bash
rm -rf build
mkdir build
cd build

DEEPDETECT_ARCH=gpu \
DEEPDETECT_BUILD=tensorrt \
DEEPDETECT_TENSORRT_VERSION=v10.12 \
DEEPDETECT_RELEASE=OFF \
DEEPDETECT_DEFAULT_MODELS=false \
BUILD_TESTS=OFF \
BUILD_OPENCV=ON \
../build.sh

cd ..
```

If you already have a CUDA-enabled OpenCV 4 build, use it instead of
`BUILD_OPENCV=ON`:

```bash
rm -rf build
mkdir build
cd build

DEEPDETECT_ARCH=gpu \
DEEPDETECT_BUILD=tensorrt \
DEEPDETECT_OPENCV4_BUILD_PATH=/path/to/opencv-4.13.0/build \
DEEPDETECT_RELEASE=OFF \
DEEPDETECT_DEFAULT_MODELS=false \
BUILD_TESTS=OFF \
../build.sh

cd ..
```

Do not set both `BUILD_OPENCV` and `DEEPDETECT_OPENCV4_BUILD_PATH`; `build.sh`
rejects that combination.

## Manual CMake Equivalents

Use these only when an agent needs exact CMake control. `build.sh` is preferred
for normal builds.

### Manual Torch CPU

```bash
. .venv/bin/activate
export CMAKE_PREFIX_PATH="$(python - <<'PY'
import torch
print(torch.utils.cmake_prefix_path)
PY
)"

rm -rf build
cmake -S . -B build \
  -DUSE_PREBUILT_TORCH=ON \
  -DUSE_TORCH=ON \
  -DUSE_CPU_ONLY=ON \
  -DUSE_TORCH_CPU_ONLY=ON \
  -DUSE_XGBOOST=OFF \
  -DUSE_SIMSEARCH=OFF \
  -DUSE_TSNE=OFF \
  -DUSE_NCNN=OFF \
  -DRELEASE=OFF \
  -DBUILD_TESTS=OFF

cmake --build build -j"$(nproc)"
```

### Manual Torch GPU

```bash
. .venv/bin/activate
export CMAKE_PREFIX_PATH="$(python - <<'PY'
import torch
print(torch.utils.cmake_prefix_path)
PY
)"
export PATH="/usr/local/cuda/bin:${PATH}"

rm -rf build
cmake -S . -B build \
  -DUSE_PREBUILT_TORCH=ON \
  -DUSE_TORCH=ON \
  -DUSE_CUDNN=ON \
  -DUSE_FAISS=OFF \
  -DUSE_XGBOOST=OFF \
  -DUSE_SIMSEARCH=OFF \
  -DUSE_TSNE=OFF \
  -DUSE_OPENCV_VERSION=4 \
  -DCUDA_ARCH="7.5;8.0;8.6;8.9;9.0;10.0;12.0" \
  -DCUDA_ARCH_FLAGS="-gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_100,code=sm_100 -gencode arch=compute_120,code=sm_120" \
  -DRELEASE=OFF \
  -DBUILD_TESTS=OFF

cmake --build build -j"$(nproc)"
```

### Manual TensorRT

Use this form when TensorRT is installed from an SDK directory:

```bash
export TENSORRT_DIR=/path/to/TensorRT
export PATH="/usr/local/cuda/bin:${PATH}"
export LD_LIBRARY_PATH="${TENSORRT_DIR}/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

rm -rf build
cmake -S . -B build \
  -DUSE_TENSORRT=ON \
  -DUSE_TORCH=OFF \
  -DUSE_CUDA_CV=ON \
  -DUSE_OPENCV_VERSION=4 \
  -DTENSORRT_DIR="${TENSORRT_DIR}" \
  -DCUDA_ARCH="7.5;8.0;8.6;8.9;9.0;10.0;12.0" \
  -DCUDA_ARCH_FLAGS="-gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_100,code=sm_100 -gencode arch=compute_120,code=sm_120" \
  -DRELEASE=OFF \
  -DBUILD_TESTS=OFF

cmake --build build -j"$(nproc)"
```

If TensorRT is installed in system multi-arch paths, use explicit directories
instead of `TENSORRT_DIR`:

```bash
rm -rf build
cmake -S . -B build \
  -DUSE_TENSORRT=ON \
  -DUSE_TORCH=OFF \
  -DUSE_CUDA_CV=ON \
  -DUSE_OPENCV_VERSION=4 \
  -DTENSORRT_LIB_DIR=/usr/lib/x86_64-linux-gnu \
  -DTENSORRT_INC_DIR=/usr/include/x86_64-linux-gnu \
  -DCUDA_ARCH="7.5;8.0;8.6;8.9;9.0;10.0;12.0" \
  -DCUDA_ARCH_FLAGS="-gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90 -gencode arch=compute_100,code=sm_100 -gencode arch=compute_120,code=sm_120" \
  -DRELEASE=OFF \
  -DBUILD_TESTS=OFF

cmake --build build -j"$(nproc)"
```

## Run The Server

Start the REST server:

```bash
./build/main/dede -host 0.0.0.0 -port 8080
```

In another terminal:

```bash
curl http://localhost:8080/info
```

Useful server options:

- `-host`: address to bind, default `localhost`.
- `-port`: HTTP port, default `8080`.
- `-nthreads`: number of HTTP threads, default `10`.
- `-service_start_list`: load services and optional predictions on startup.

Show all options:

```bash
./build/main/dede --help
```

## Run Tests

Install test dependencies:

```bash
sudo apt-get install -y libgtest-dev python3-numpy
```

Build with tests:

```bash
rm -rf build
mkdir build
cd build

DEEPDETECT_ARCH=cpu \
DEEPDETECT_BUILD=default \
DEEPDETECT_RELEASE=OFF \
DEEPDETECT_DEFAULT_MODELS=false \
BUILD_TESTS=ON \
USE_PREBUILT_TORCH=ON \
../build.sh

ctest --output-on-failure

cd ..
```

Some tests download fixtures and can take a long time, especially on CPU-only
machines.

## Service Auto-Start

Pass a service startup list to `dede`:

```bash
./build/main/dede -service_start_list /path/to/services.txt
```

File format:

```text
service_create;service_name;JSON string
service_predict;JSON string
```

The JSON string is the same request body used by the REST API, without external
quotes.

## Pure Command-Line JSON API

Use the same JSON API without running the HTTP server:

```bash
./build/main/dede --jsonapi 1 -info true
```

Show command-line JSON API options:

```bash
./build/main/dede --jsonapi 1 --help
```

## Troubleshooting

- `Could not find Torch`: activate `.venv` and verify `python -c "import torch; print(torch.utils.cmake_prefix_path)"`.
- `USE_TORCH_CPU_ONLY=ON requires a CPU-only LibTorch package`: install the CPU PyTorch wheel or remove CPU-only flags.
- `GPU Torch support requires a CUDA LibTorch package`: install the CUDA PyTorch wheel that matches your CUDA variant.
- `Could not find TensorRT libnvinfer.so`: set `TENSORRT_DIR`, or set both `TENSORRT_LIB_DIR` and `TENSORRT_INC_DIR`.
- `DEEPDETECT_GPU_VARIANT=default requires DD_CUDA_VERSION >= 13`: use CUDA 13 or switch to `DEEPDETECT_GPU_VARIANT=legacy61` with CUDA 12.x.
- OpenCV CUDA build failures: use the Docker TensorRT build, or provide a known-good OpenCV build with `DEEPDETECT_OPENCV4_BUILD_PATH`.

## Code Style

Run clang-format through CMake:

```bash
cmake --build build --target clang-format
```
