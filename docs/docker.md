# DeepDetect Docker Usage

DeepDetect Docker images run the `dede` REST server. Use them when you want a
long-running HTTP service, containerized serving, or a reproducible deployment
environment. For local in-process work and the `deepdetect` CLI, use the Python
wheel described in [../bindings/python/README.md](../bindings/python/README.md).

The current recommended server backends are:

- `torch` for training and inference.
- `tensorrt` for optimized inference from exported or compatible models.

The retired `caffe`, `caffe2`, `tf`, and `tensorflow` backends are rejected by
the server. Do not use old Docker examples that create services with
`"mllib":"caffe"`.

## Images

Official images are published at `docker.jolibrain.com`:

```bash
docker pull docker.jolibrain.com/deepdetect_cpu
docker pull docker.jolibrain.com/deepdetect_gpu
docker pull docker.jolibrain.com/deepdetect_gpu_tensorrt
```

Use `deepdetect_cpu` for CPU-only serving, `deepdetect_gpu` for CUDA Torch
serving, and `deepdetect_gpu_tensorrt` when you need TensorRT support.

To inspect the registry:

```bash
curl -X GET https://docker.jolibrain.com/v2/_catalog
curl -X GET https://docker.jolibrain.com/v2/deepdetect_cpu/tags/list
```

## Run The Server

Start the CPU image:

```bash
docker run -d \
  --name deepdetect \
  -p 8080:8080 \
  docker.jolibrain.com/deepdetect_cpu
```

Check that the server is responding:

```bash
curl http://localhost:8080/info
```

The response should contain a `status.code` of `200` and a `services` list.

Stop and remove the container:

```bash
docker stop deepdetect
docker rm deepdetect
```

## Run With GPUs

GPU images require Docker with NVIDIA Container Toolkit support. With recent
Docker versions, use `--gpus`:

```bash
docker run -d \
  --name deepdetect \
  --gpus all \
  -p 8080:8080 \
  docker.jolibrain.com/deepdetect_gpu
```

For TensorRT:

```bash
docker run -d \
  --name deepdetect-trt \
  --gpus all \
  -p 8080:8080 \
  docker.jolibrain.com/deepdetect_gpu_tensorrt
```

If your Docker installation still uses the old `nvidia-docker` wrapper, the
equivalent form is:

```bash
nvidia-docker run -d \
  --name deepdetect \
  -p 8080:8080 \
  docker.jolibrain.com/deepdetect_gpu
```

## Mount Models And Data

Mount host directories for model repositories and datasets. The container runs
as user `dd`, so the mounted paths must be readable by that user and writable
when training or saving models.

```bash
mkdir -p models data

docker run -d \
  --name deepdetect \
  -p 8080:8080 \
  -v "$PWD/models:/models" \
  -v "$PWD/data:/data" \
  docker.jolibrain.com/deepdetect_cpu
```

Inside API calls, use the container paths (`/models/...`, `/data/...`), not the
host paths.

## Create A Torch Service

Create services from model repositories mounted into the container. This
example assumes `/models/resnet18` contains a compatible Torch model repository:

```bash
curl -X PUT "http://localhost:8080/services/imageserv" \
  -H "Content-Type: application/json" \
  -d '{
    "mllib": "torch",
    "description": "Torch image classification service",
    "type": "supervised",
    "model": {
      "repository": "/models/resnet18"
    },
    "parameters": {
      "input": {
        "connector": "image",
        "width": 224,
        "height": 224,
        "rgb": true,
        "scale": 0.0039
      },
      "mllib": {
        "template": "resnet18",
        "nclasses": 2,
        "gpu": false
      },
      "output": {}
    }
  }'
```

Run prediction:

```bash
curl -X POST "http://localhost:8080/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "service": "imageserv",
    "parameters": {
      "input": {
        "width": 224,
        "height": 224
      },
      "output": {
        "best": 3
      }
    },
    "data": ["/data/image.jpg"]
  }'
```

For GPU inference, run a GPU image and set `"gpu": true` in the service
`parameters.mllib` object.

## Create A TensorRT Service

Use the TensorRT image with an exported or compatible model repository:

```bash
curl -X PUT "http://localhost:8080/services/yolox-trt" \
  -H "Content-Type: application/json" \
  -d '{
    "mllib": "tensorrt",
    "description": "TensorRT YOLOX inference service",
    "type": "supervised",
    "model": {
      "repository": "/models/yolox-trt"
    },
    "parameters": {
      "input": {
        "connector": "image",
        "width": 640,
        "height": 640,
        "rgb": true
      },
      "mllib": {
        "template": "yolox",
        "nclasses": 80,
        "datatype": "fp16",
        "maxBatchSize": 8,
        "gpuid": 0
      },
      "output": {}
    }
  }'
```

TensorRT can read an existing engine from the model repository or build and
write one depending on `readEngine`, `writeEngine`, and `maxBatchSize`. See the
TensorRT parameters in [api.md](api.md).

## Logs And Debugging

Follow server logs:

```bash
docker logs -f deepdetect
```

List running containers:

```bash
docker ps
```

Open a shell in the container:

```bash
docker exec -it deepdetect bash
```

Check loaded services:

```bash
curl "http://localhost:8080/info?status=true"
```

## Swagger UI

With a running server, Swagger UI is served at:

```text
http://localhost:8080/swagger/ui
```

If the server is behind a path prefix such as `/api/deepdetect`, pass the
prefix as an extra server argument after the image name:

```bash
docker run -d \
  --name deepdetect \
  -p 8080:8080 \
  docker.jolibrain.com/deepdetect_cpu \
  -swagger_api_prefix api/deepdetect/
```

## Build Images Locally

Build from the repository root, not from the `docker/` directory:

```bash
export DOCKER_BUILDKIT=1

docker build \
  -t jolibrain/deepdetect_cpu \
  -f docker/cpu.Dockerfile \
  .
```

Build the GPU image:

```bash
export DOCKER_BUILDKIT=1

docker build \
  -t jolibrain/deepdetect_gpu \
  -f docker/gpu.Dockerfile \
  .
```

Build the TensorRT image:

```bash
export DOCKER_BUILDKIT=1

docker build \
  -t jolibrain/deepdetect_gpu_tensorrt \
  -f docker/gpu_tensorrt.Dockerfile \
  .
```

Useful build arguments:

- `DEEPDETECT_DEFAULT_MODELS=false`: skip downloading bundled legacy default models.
- `USE_PREBUILT_TORCH=ON`: use official prebuilt PyTorch/LibTorch packages.
- `DEEPDETECT_GPU_VARIANT=legacy61`: build a GPU image for older compute capabilities.
- `PYTORCH_CUDA_INDEX=cu126`: select the matching PyTorch CUDA wheel index for legacy GPU builds.
- `DD_CUDA_VERSION` and `DD_CUDA_MAJOR_MINOR`: override CUDA image versions for GPU builds.

More build details are in [../docker/README.md](../docker/README.md).
