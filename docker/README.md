# DeepDetect Docker images

This repository contains the Dockerfiles for building the CPU and GPU images for deepdetect.

Also see https://hub.docker.com/u/jolibrain for the latest pre-built images

The docker images contain:
- a running `dede` server ready to be used, no install required
- `googlenet` and `resnet_50` pre-trained image classification models, in `/opt/models/`

This allows to run the container and set an image classification model based on deep (residual) nets in two short command line calls.

## Getting and running official images

```
docker pull jolibrain/deepdetect_cpu
```
or
```
docker pull jolibrain/deepdetect_gpu
```

### Running the CPU image

```
docker run -d -p 8080:8080 jolibrain/deepdetect_cpu
```

`dede` server is now listening on your port `8080`:

```
curl http://localhost:8080/info

{"status":{"code":200,"msg":"OK"},"head":{"method":"/info","version":"0.1","branch":"master","commit":"c8556f0b3e7d970bcd9861b910f9eae87cfd4b0c","services":[]}}
```

Here is how to do a simple image classification service and prediction test:
- service creation
```
curl -X PUT "http://localhost:8080/services/imageserv" -d "{\"mllib\":\"caffe\",\"description\":\"image classification service\",\"type\":\"supervised\",\"parameters\":{\"input\":{\"connector\":\"image\"},\"mllib\":{\"nclasses\":1000}},\"model\":{\"repository\":\"/opt/models/ggnet/\"}}"

{"status":{"code":201,"msg":"Created"}}
```
- image classification
```
curl -X POST "http://localhost:8080/predict" -d "{\"service\":\"imageserv\",\"parameters\":{\"input\":{\"width\":224,\"height\":224},\"output\":{\"best\":3},\"mllib\":{\"gpu\":false}},\"data\":[\"http://i.ytimg.com/vi/0vxOhd4qlnA/maxresdefault.jpg\"]}"

{"status":{"code":200,"msg":"OK"},"head":{"method":"/predict","time":852.0,"service":"imageserv"},"body":{"predictions":{"uri":"http://i.ytimg.com/vi/0vxOhd4qlnA/maxresdefault.jpg","classes":[{"prob":0.2255125343799591,"cat":"n03868863 oxygen mask"},{"prob":0.20917612314224244,"cat":"n03127747 crash helmet"},{"last":true,"prob":0.07399296760559082,"cat":"n03379051 football helmet"}]}}}
```

### Running the GPU image

This requires [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) in order for the local GPUs to be made accessible by the container.

The following steps are required:

- install `nvidia-docker`: https://github.com/NVIDIA/nvidia-docker
- run with
```
nvidia-docker run -d -p 8080:8080 jolibrain/deepdetect_gpu
```

Notes:
- `nvidia-docker` requires docker >= 1.9

To test on image classification on GPU:
```
curl -X PUT "http://localhost:8080/services/imageserv" -d "{\"mllib\":\"caffe\",\"description\":\"image classification service\",\"type\":\"supervised\",\"parameters\":{\"input\":{\"connector\":\"image\"},\"mllib\":{\"nclasses\":1000}},\"model\":{\"repository\":\"/opt/models/ggnet/\"}}"
{"status":{"code":201,"msg":"Created"}}
```
and
```
curl -X POST "http://localhost:8080/predict" -d "{\"service\":\"imageserv\",\"parameters\":{\"input\":{\"width\":224,\"height\":224},\"output\":{\"best\":3},\"mllib\":{\"gpu\":true}},\"data\":[\"http://i.ytimg.com/vi/0vxOhd4qlnA/maxresdefault.jpg\"]}"
```

Try the `POST` call twice: first time loads the net so it takes slightly below a second, then second call should yield a `time` around 100ms as reported in the output JSON.

### Access to server logs

To look at server logs, use 
```
docker logs -f <container name>
```
where <container name> can be obtained via `docker ps`

Example:


- start container and server:
```
> docker run -d -p 8080:8080 jolibrain/deepdetect_cpu
```

- look for container:
```
> docker ps
CONTAINER ID        IMAGE                  COMMAND                  CREATED              STATUS              PORTS                    NAMES
d9944734d5d6        jolibrain/deepdetect_cpu   "/bin/sh -c './dede -"   17 seconds ago       Up 16 seconds       0.0.0.0:8080->8080/tcp   loving_shaw
```

- access server logs:
```
> docker logs -f loving_shaw 

DeepDetect [ commit 4e2c9f4cbd55eeba3a93fae71d9d62377e91ffa5 ]
Running DeepDetect HTTP server on 0.0.0.0:8080
```

- share a volume with the image:
```
docker run -d -p 8080:8080 -v /path/to/volume:/mnt jolibrain/deepdetect_cpu
```
where `path/to/volume` is the path to your local volume that you'd like to attach to `/opt/deepdetect/`. This is useful for sharing / saving models, etc...

## Build Deepdetect Docker images

Dockerfiles  are stored in the "docker" folder, but **you must launch build from root directory**.

We choose to prefix Dockerfiles with target architecture :
* cpu-armv7.Dockerfile
* cpu.Dockerfile
* gpu.Dockerfile

### Build script 

Build script is avaliable in docker path : build/build.sh

Docker build-arg : DEEPDETECT_BUILD

Description : DEEPDETECT_BUILD build argument change cmake arguments in build.sh script.

Expected values :

* CPU
  * caffe-tf
  * default
* GPU
  * tf
  * tf-cpu
  * caffe-cpu-tf
  * caffe-tf
  * caffe2
  * p100
  * volta
  * volta-faiss
  * faiss
  * default

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

#### Build examples

> You must launch build from root directory

Example with CPU image:
```
# Build with default cmake 
docker build -t jolibrain/deepdetect_cpu --no-cache -f docker/cpu.Dockerfile .

# Build with default cmake and without default models
docker build --build-arg DEEPDETECT_DEFAULT_MODELS=false -t jolibrain/deepdetect_cpu --no-cache -f cpu.Dockerfile .

# Build with custom cmake
docker build --build-arg DEEPDETECT_BUILD=caffe-tf -t jolibrain/deepdetect_cpu --no-cache -f docker/cpu.Dockerfile .

```

Example with CPU (armv7) image:
```
# Build with default cmake 
docker build -t jolibrain/deepdetect_cpu:armv7 --no-cache -f docker/cpu-armv7.Dockerfile .

```

Example with GPU image:
```
# Build with default cmake 
docker build -t jolibrain/deepdetect_gpu --no-cache -f docker/gpu.Dockerfile .

# Build with default cmake and without default models
docker build --build-arg DEEPDETECT_DEFAULT_MODELS=false -t jolibrain/deepdetect_gpu --no-cache -f docker/gpu.Dockerfile .

# Build with custom cmake
docker build --build-arg DEEPDETECT_BUILD=caffe-tf -t jolibrain/deepdetect_gpu --no-cache -f docker/gpu.Dockerfile .
```