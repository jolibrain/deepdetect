#!/bin/bash

set -e
set -x

export DOCKER_BUILDKIT=1

selected=$1

declare -A TARGETS
TARGETS[cpu]="cpu/default"
TARGETS[gpu]="gpu/default"
TARGETS[gpu_tf]="gpu/tf"
TARGETS[gpu_tensorrt]="gpu_tensorrt/tensorrt"

NAMES=${!TARGETS[@]}
if [ "$1" ]; then
    NAMES="$1"
fi

if [ "$TAG_NAME" ]; then
    TMP_TAG="ci-$TAG_NAME"
elif [ "$GIT_BRANCH" == "master" ]; then
    TMP_TAG="ci-$GIT_BRANCH"
else
    TMP_TAG="trash"
fi

image_url_prefix_release="jolibrain/deepdetect"
image_url_prefix_ci="ceres:5000/${image_url_prefix_release}"
images_to_push=

for name in $NAMES; do
    target=${TARGETS[$name]}
    if [ ! "$target" ]; then
        echo "$name target doesn't exists"
        exit 1
    fi

    arch=${target%%/*}
    build=${target##*/}
    image_url_release="${image_url_prefix_release}_${name}"
    image_url_ci="${image_url_prefix_ci}_${name}"

    docker build \
        -t $image_url_ci:$TMP_TAG \
        --build-arg DEEPDETECT_BUILD=$build \
        -f docker/${arch}.Dockerfile \
        .

    if [ "$TAG_NAME" ]; then
        docker tag $image_url_ci:$TMP_TAG $image_url_release:${TAG_NAME}
        docker push $image_url_release:${TAG_NAME}
    else
        docker push $image_url_ci:$TMP_TAG
    fi
done
