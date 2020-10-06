#!/bin/bash

export DOCKER_BUILDKIT=1

TARGETS="cpu/default gpu/default gpu/tf gpu/tensorrt"

if [ "$GIT_BRANCH" == "master" ]; then
    TMP_TAG="ci"
else
    TMP_TAG="trash"
fi

image_url_prefix="ceres:5000/jolibrain/deepdetect"
images_to_push=

for target in $TARGETS ; do
    arch=${target%%/*}
    build=${target##*/}
    image_url="${image_url_prefix}_${arch}"
    [ "$build" != "default" ] && image_url="${image_url}_build"

    docker build \
        -t $image_url:$TMP_TAG \
        --build-arg DEEPDETECT_BUILD=$build \
        -f docker/${arch}.Dockerfile \
        .

    images_to_push="${images_to_push} $image_url:$TMP_TAG"
done

for image in $images_to_push; do
    docker push $image
done
