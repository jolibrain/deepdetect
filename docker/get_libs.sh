#!/bin/bash

set -e

mkdir -p /tmp/lib

copy_glob() {
    local pattern="$1"
    local found=0

    for lib in $pattern; do
        if [ -e "$lib" ]; then
            echo "Copy $lib to /tmp/lib/"
            cp -a "$lib" /tmp/lib/
            found=1
        fi
    done

    if [ "$found" = 0 ]; then
        echo "$pattern is not found."
    fi
}

libs=(
    "/usr/local/lib/libcurlpp.*"
    "/opt/deepdetect/build/tensorrt-oss/bin/*"
    "/opt/deepdetect/build/protobuf/src/protobuf-build/lib*.so*"
    "/opt/deepdetect/build/pytorch/src/pytorch-build/build/lib/lib*.so*"
    "/opt/deepdetect/build/pytorch_vision/src/pytorch_vision-install/lib/lib*.so*"
    "/opt/deepdetect/build/Multicore-TSNE/src/Multicore-TSNE-build/libtsne_multicore.so"
    "/opt/deepdetect/build/faiss/src/faiss/libfaiss.so"
    "/usr/local/lib/python*/dist-packages/torch/lib/lib*.so*"
    "/usr/local/lib/python*/site-packages/torch/lib/lib*.so*"
    "/usr/lib/python*/dist-packages/torch/lib/lib*.so*"
    "/usr/lib/python*/site-packages/torch/lib/lib*.so*"
    "/usr/local/lib/python*/dist-packages/nvidia/*/lib/lib*.so*"
    "/usr/local/lib/python*/site-packages/nvidia/*/lib/lib*.so*"
    "/usr/lib/python*/dist-packages/nvidia/*/lib/lib*.so*"
    "/usr/lib/python*/site-packages/nvidia/*/lib/lib*.so*"
)

for lib_glob in "${libs[@]}"; do
    copy_glob "$lib_glob"
done
