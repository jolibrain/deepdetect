# DeepDetect Docker images

## Installation

See https://github.com/jolibrain/deepdetect/tree/master/docs/docker.md

## Build

Dockerfiles are stored in the "docker" folder, but **you must launch build from root directory**.

We choose to prefix Dockerfiles with target architecture :

* cpu.Dockerfile
* cpu-armv7.Dockerfile
* gpu.Dockerfile

### Build script

Build script is available in docker path : build/build.sh

Docker build-arg : DEEPDETECT_BUILD

Description : DEEPDETECT_BUILD build argument change cmake arguments in build.sh script.

Expected values :

* CPU
  * tf
  * torch
  * default
* GPU
  * tf
  * tf-cpu
  * caffe-cpu-tf
  * caffe-tf
  * torch
  * default

#### Prepare build environment

Create build directory and put build script inside :

```bash
mkdir build
cd build
cp -a ../build.sh .
```

#### Launch build with environments variables

```bash
DEEPDETECT_ARCH=cpu,gpu DEEPDETECT_BUILD=default,caffe-tf,armv7,[...] ./build.sh
```

#### Launch build with build script parameters

```bash
Params usage: ./build.sh [options...]

   -a, --deepdetect-arch          Choose Deepdetect architecture : cpu,gpu
   -b, --deepdetect-build         Choose Deepdetect build profile : CPU (default,caffe-tf,armv7) / GPU (default,caffe-cpu-tf,caffe-tf,caffe2,p100,volta)
```

### Building an image

#### Docker build arguments

* DEEPDETECT_BUILD : Change cmake arguments, checkout build script documentation.
* DEEPDETECT_DEFAULT_MODELS : [**true**/false] Enable or disable default models in deepdetect docker image. Default models size is about 160MB.
* DEEPDETECT_GPU_VARIANT : [`default`/**legacy61**] GPU-only arch preset. `default` tracks the current Dockerfile CUDA line and targets compute capabilities `7.5+`, while `legacy61` switches the build targets to include compute capability `6.1`.
* DD_CUDA_VERSION / DD_CUDA_MAJOR_MINOR : Optional CUDA image overrides. They are only needed when building the `legacy61` GPU variant with the raw `docker build` command.

#### Build examples

> You must launch build from root directory

Example with CPU image:
```
# Build with default cmake
export DOCKER_BUILDKIT=1
docker build -t jolibrain/deepdetect_cpu --no-cache -f docker/cpu.Dockerfile .

# Build with default cmake and without default models
export DOCKER_BUILDKIT=1
docker build --build-arg DEEPDETECT_DEFAULT_MODELS=false -t jolibrain/deepdetect_cpu --no-cache -f cpu.Dockerfile .

# Build with custom cmake
export DOCKER_BUILDKIT=1
docker build --build-arg DEEPDETECT_BUILD=caffe-tf -t jolibrain/deepdetect_cpu --no-cache -f docker/cpu.Dockerfile .

```

Example with CPU (armv7) image:
```
# Build with default cmake
export DOCKER_BUILDKIT=1
docker build -t jolibrain/deepdetect_cpu:armv7 --no-cache -f docker/cpu-armv7.Dockerfile .

```

Example with GPU image:
```
# Build with default cmake
export DOCKER_BUILDKIT=1
docker build -t jolibrain/deepdetect_gpu --no-cache -f docker/gpu.Dockerfile .

# Build with default cmake and without default models
export DOCKER_BUILDKIT=1
docker build --build-arg DEEPDETECT_DEFAULT_MODELS=false -t jolibrain/deepdetect_gpu --no-cache -f docker/gpu.Dockerfile .

# Build with custom cmake
export DOCKER_BUILDKIT=1
docker build --build-arg DEEPDETECT_BUILD=caffe-tf -t jolibrain/deepdetect_gpu --no-cache -f docker/gpu.Dockerfile .

# Build the legacy61 variant with CUDA 12.8.1
export DOCKER_BUILDKIT=1
docker build \
  --build-arg DEEPDETECT_GPU_VARIANT=legacy61 \
  --build-arg DD_CUDA_VERSION=12.8.1 \
  --build-arg DD_CUDA_MAJOR_MINOR=12.8 \
  -t jolibrain/deepdetect_gpu:legacy61 \
  --no-cache \
  -f docker/gpu.Dockerfile .

# Same legacy61 build through the helper wrapper
export DOCKER_BUILDKIT=1
./docker/build_gpu_image.sh legacy61 -t jolibrain/deepdetect_gpu:legacy61 --no-cache
```
