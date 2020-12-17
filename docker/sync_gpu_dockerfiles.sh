#!/bin/bash

here=$(dirname $(readlink -f $0))

declare -A runtime_images
runtime_images[gpu]="nvidia/cuda:11.1-cudnn8-runtime-ubuntu20.04"
# Ubuntu 20.04+cuda 11.1.80 + cuDNN 8 + TensorRT 7.2.x
# https://docs.nvidia.com/deeplearning/tensorrt/container-release-notes/rel_20-12.html#rel_20-12
runtime_images[gpu_tensorrt]="nvcr.io/nvidia/tensorrt:20.12-py3"

declare -A devel_images
devel_images[gpu]="nvidia/cuda:11.1-cudnn8-devel-ubuntu20.04"
devel_images[gpu_tensorrt]="nvcr.io/nvidia/tensorrt:20.12-py3"

declare -A builds
builds[gpu_tensorrt]="tensorrt"

for dest in "${!runtime_images[@]}" ; do
    runtime_image=${runtime_images[$dest]}
    devel_image=${devel_images[$dest]}
    build=${builds[$dest]}

    [ ! "$build" ] && build="default"

    sed \
        -e "s,FROM [^ ]* AS build,FROM ${devel_image} AS build,g" \
        -e "s,FROM [^ ]* AS runtime,FROM ${runtime_image} AS runtime,g" \
        -e "s,ARG DEEPDETECT_ARCH=.*,ARG DEEPDETECT_ARCH=gpu,g" \
        -e "s,ARG DEEPDETECT_BUILD=.*,ARG DEEPDETECT_BUILD=${build},g" \
        $here/cpu.Dockerfile > $here/${dest}.Dockerfile
done
