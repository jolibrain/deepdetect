#!/bin/bash

set -e

mkdir -p /tmp/lib

libs=(
    /usr/local/lib/libcurlpp.*
    /opt/deepdetect/build/tensorrt-oss/bin/*
    /opt/deepdetect/build/protobuf/src/protobuf-build/lib*.so*
    /opt/deepdetect/build/pytorch/src/pytorch-build/build/lib/lib*.so*
    /opt/deepdetect/build/pytorch_vision/src/pytorch_vision-install/lib/lib*.so*
    /opt/deepdetect/build/Multicore-TSNE/src/Multicore-TSNE-build/libtsne_multicore.so
    /opt/deepdetect/build/faiss/src/faiss/libfaiss.so
)

for index in ${!libs[*]}; do

    if [ -f "${libs[index]}" ]; then
        echo "Copy ${libs[index]} to /tmp/lib/"
        cp -a ${libs[index]} /tmp/lib/
    else
        echo ${libs[index]} 'is not found.'
    fi

done
