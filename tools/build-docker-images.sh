#!/bin/bash

export DOCKER_BUILDKIT=1

set -e

[ "${JENKINS_URL}" ] && set -x

if [ ! "$@" ]; then
  echo "usage: $(basename $0) [cpu|gpu|...]"
  exit 1
fi
NAMES="$@"

declare -A TARGETS
TARGETS[cpu]="cpu/default"
TARGETS[gpu]="gpu/default"
TARGETS[gpu_tf]="gpu/tf"
TARGETS[gpu_tensorrt]="gpu_tensorrt/tensorrt"

if [ "$TAG_NAME" ]; then
    TMP_TAG="ci-$TAG_NAME"
elif [ "$GIT_BRANCH" == "master" ]; then
    TMP_TAG="ci-$GIT_BRANCH"
else
    # Not built with Jenkins
    TMP_TAG="trash"
fi

image_url_prefix="jolibrain/deepdetect"

for name in $NAMES; do
    target=${TARGETS[$name]}
    if [ ! "$target" ]; then
        echo "$name target doesn't exists"
        exit 1
    fi

    arch=${target%%/*}
    build=${target##*/}
    image_url="${image_url_prefix}_${name}"

    docker build \
        -t $image_url:$TMP_TAG \
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
