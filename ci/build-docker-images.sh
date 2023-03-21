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
TARGETS[cpu_torch]="cpu/torch"
TARGETS[gpu]="gpu/default"
TARGETS[gpu_tf]="gpu/tf"
TARGETS[gpu_torch]="gpu/torch"
TARGETS[gpu_tensorrt]="gpu_tensorrt/tensorrt"

PR_NUMBER=$(echo $GIT_BRANCH | sed -n '/^PR-/s/PR-//gp')
if [ "$TAG_NAME" ]; then
    TMP_TAG="ci-$TAG_NAME"
elif [ "$GIT_BRANCH" == "master" ]; then
    TMP_TAG="ci-$GIT_BRANCH"
elif [ "$PR_NUMBER" ]; then
    TMP_TAG=ci-pr-$PR_NUMBER
else
    # Not built with Jenkins
    TMP_TAG="trash"
fi

image_url_prefix="docker.jolibrain.com/deepdetect"

for name in $NAMES; do
    target=${TARGETS[$name]}
    if [ ! "$target" ]; then
        echo "$name target doesn't exists"
        exit 1
    fi

    arch=${target%%/*}
    build=${target##*/}
    image_url="${image_url_prefix}_${name}"
    release="OFF"
    if [ "$TAG_NAME" ]; then
        already_exists=$(DOCKER_CLI_EXPERIMENTAL=enabled docker manifest inspect ${image_url}:$TAG_NAME 2>/dev/null || true)
        if [ "$already_exists" ]; then
            echo "${image_url}:$TAG_NAME already built skipping..."
            continue
        fi
        release="ON"
    fi

    # BUILD
    docker build \
        -t $image_url:$TMP_TAG \
        --progress plain \
        --build-arg DEEPDETECT_BUILD=$build \
        --build-arg DEEPDETECT_RELEASE=$release \
        -f docker/${arch}.Dockerfile \
        .

    # TEST
    CONTAINER_NAME="ci_testing_deepdetect_${name}_${TMP_TAG}"
    docker run -p 1025-65535:8080 -d --name $CONTAINER_NAME $image_url:$TMP_TAG
    log_and_cleanup () {
        docker logs $CONTAINER_NAME
        docker stop $CONTAINER_NAME
        docker rm $CONTAINER_NAME
    }
    trap log_and_cleanup EXIT

    PORT=$(docker port $CONTAINER_NAME 8080/tcp | awk -F: '{print $2}')

    timeout 60 sh -c "until nc -z localhost $PORT; do sleep 1; done"

    sleep 10  # Wait dd start

    curl -s --head --request GET http://localhost:$PORT/info | head -1 | grep 'HTTP/1.1 200'
    trap - EXIT

    docker stop $CONTAINER_NAME
    docker rm $CONTAINER_NAME

    # PUSH
    if [ "$TMP_TAG" != "trash" ]; then
        if [ "$TAG_NAME" ]; then
            docker tag $image_url:$TMP_TAG $image_url:${TAG_NAME}
            docker tag $image_url:$TMP_TAG $image_url:latest
            docker push $image_url:${TAG_NAME}
            docker push $image_url:latest
            docker image rm $image_url:${TAG_NAME}
            docker image rm $image_url:latest
        elif [ "$GIT_BRANCH" == "master" ]; then
            docker push $image_url:$TMP_TAG
        fi
    fi
    docker image rm $image_url:$TMP_TAG
done
