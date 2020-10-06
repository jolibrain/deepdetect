#!/bin/bash

export DOCKER_BUILDKIT=1

TARGETS="cpu/default gpu/default gpu/tf gpu/tensorrt"

if [ "$TAG_NAME" ]; then
    TMP_TAG=${TAG_NAME#v}
elif [ "$GIT_BRANCH" == "master" ]; then
    TMP_TAG="ci"
else
    TMP_TAG="trash"
fi

image_url_prefix_release="jolibrain/deepdetect"
image_url_prefix_ci="ceres:5000"
images_to_push=

for target in $TARGETS ; do
    arch=${target%%/*}
    build=${target##*/}
    image_url_release="${image_url_prefix_release}_${arch}"
    [ "$build" != "default" ] && image_url_release="${image_url}_build"

    image_url_ci="${image_url_prefix_ci}/${image_url_prefix_release}"

    docker build \
        -t $image_url_ci:$TMP_TAG \
        --build-arg DEEPDETECT_BUILD=$build \
        -f docker/${arch}.Dockerfile \
        .

    if [ "$TAG_NAME" ]; then
        docker tag $image_url_ci:$TMP_TAG $image_url_release:$TMP_TAG
        docker tag $image_url_ci:$TMP_TAG $image_url_release:latest
    fi

    images_to_push="${images_to_push} $image_url:$TMP_TAG"
done

for image in $images_to_push; do
    docker push $image
done
