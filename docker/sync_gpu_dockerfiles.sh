#!/bin/bash

here=$(dirname $(readlink -f $0))

declare -A images
images[gpu]="nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04"
# Ubuntu 18.04+cuda 10.2.80 + cuDNN 7.6.5 + TensorRT 7.0.0
# https://docs.nvidia.com/deeplearning/tensorrt/container-release-notes/rel_20-03.html#rel_20-03
images[gpu_tensorrt]="nvcr.io/nvidia/tensorrt:20.03-py3"

declare -A builds
builds[gpu_tensorrt]="tensorrt"

for dest in "${!images[@]}" ; do
    image=${images[$dest]}
    build=${builds[$dest]}

    [ ! "$build" ] && build="default"

    sed \
        -e "s,FROM [^ ]*,FROM ${image},g" \
        -e "s,ARG DEEPDETECT_ARCH=.*,ARG DEEPDETECT_ARCH=gpu,g" \
        -e "s,ARG DEEPDETECT_BUILD=.*,ARG DEEPDETECT_BUILD=${build},g" \
        -e "s/\(apt_\(cache\|lib\)\)_cpu/\1_${dest}/g" \
        $here/cpu.Dockerfile > $here/${dest}.Dockerfile
done
