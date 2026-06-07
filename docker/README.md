# DeepDetect Docker images

## Installation

See https://github.com/jolibrain/deepdetect/tree/master/docs/docker.md

## Build

Dockerfiles are stored in the "docker" folder, but **you must launch build from root directory**.

We choose to prefix Dockerfiles with target architecture :

* cpu.Dockerfile
* cpu-arm.Dockerfile
* gpu.Dockerfile
* gpu_tensorrt.Dockerfile

#### Docker build arguments

* DEEPDETECT_BUILD : Change cmake arguments, checkout build script documentation.
* DEEPDETECT_DEFAULT_MODELS : [**true**/false] Enable or disable default models in deepdetect docker image. Default models size is about 160MB.
* DEEPDETECT_GPU_VARIANT : [`default`/**legacy61**] GPU-only arch preset. `default` targets CUDA 13 and compute capabilities `7.5;8.0;8.6;8.9;9.0;10.0;12.0`. `legacy61` targets CUDA 12.x and preserves older compute capabilities `6.1;6.2;7.0;7.2;7.5;8.0;8.6;8.9;9.0`.
* DD_CUDA_VERSION / DD_CUDA_MAJOR_MINOR : Optional CUDA image overrides. The defaults are CUDA `13.0.2` / `13.0`; use CUDA `12.6.3` / `12.6` with `DEEPDETECT_GPU_VARIANT=legacy61`.

#### Build examples

> You must launch build from root directory

Example for CPU image:
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

Example with CPU arm image:
```
# Build with default cmake
export DOCKER_BUILDKIT=1
docker build -t jolibrain/deepdetect_cpu:arm --no-cache -f docker/cpu-arm.Dockerfile .

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

# Build the legacy61 compatibility alias
export DOCKER_BUILDKIT=1
docker build \
  --build-arg DEEPDETECT_GPU_VARIANT=legacy61 \
  --build-arg DD_CUDA_VERSION=12.6.3 \
  --build-arg DD_CUDA_MAJOR_MINOR=12.6 \
  -t jolibrain/deepdetect_gpu:legacy61 \
  --no-cache \
  -f docker/gpu.Dockerfile .

# Same legacy61 build through the helper wrapper
export DOCKER_BUILDKIT=1
./ci/build-docker-images.sh gpu_legacy61
```

Example with Jetson Orin
```
export DOCKER_BUILDKIT=1
docker build -t jolibrain/deepdetect_jetson_orin_runtime -f docker/jetson_orin.Dockerfile .
```
