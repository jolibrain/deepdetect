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
```

Example with Jetson Orin
```
export DOCKER_BUILDKIT=1
docker build -t jolibrain/deepdetect_jetson_orin_runtime -f docker/jetson_orin.Dockerfile .
```
