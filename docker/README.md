## DeepDetect Docker images

This repository contains the Dockerfiles for building the CPU and GPU images for deepdetect.

Also see https://hub.docker.com/u/jolibrain for the latest pre-built images

The docker images contain:
- a running `dede` server ready to be used, no install required
- `googlenet` and `resnet_50` pre-trained image classification models, in `/opt/models/`

This allows to run the container and set an image classification model based on deep (residual) nets in two short command line calls.

### Getting and running official images

```
docker pull jolibrain/deepdetect_cpu
```
or
```
docker pull jolibrain/deepdetect_gpu
```

#### Running the CPU image

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

#### Running the GPU image

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

#### Access to server logs

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

#### Building an image

Example goes with the CPU image:
```
cd cpu
docker build -t jolibrain/deepdetect_cpu --no-cache .
```
