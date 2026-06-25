#!/bin/bash

export DOCKER_BUILDKIT=1

set -e

[ "${JENKINS_URL}" ] && set -x

usage() {
    local exit_code="${1:-1}"
    cat >&2 <<EOF
Usage: $(basename "$0") [options] [cpu|gpu|gpu_tensorrt|...]

Options:
  --tag TAG           Local image tag to build or push, e.g. v0.28.1
  --release          Build with DEEPDETECT_RELEASE=ON
  --no-push          Build and test images, but keep them local
  --push-only        Push existing local images tagged with --tag
  --no-test          Skip the container smoke test
  --image-prefix P   Image prefix (default: docker.jolibrain.com/deepdetect)
  -h, --help         Show this help

Environment:
  DOCKER_GPU_RUN_ARGS  Extra docker run args for GPU images (default: --gpus all)
EOF
    exit "$exit_code"
}

die() {
    echo "Error: $*" >&2
    exit 1
}

NAMES=()
OVERRIDE_TAG=""
FORCE_RELEASE=0
NO_PUSH=0
PUSH_ONLY=0
NO_TEST=0
image_url_prefix="docker.jolibrain.com/deepdetect"

while [ "$#" -gt 0 ]; do
    case "$1" in
        --tag)
            OVERRIDE_TAG="${2:-}"
            [ -n "$OVERRIDE_TAG" ] || die "--tag requires a value"
            shift 2
            ;;
        --release)
            FORCE_RELEASE=1
            shift
            ;;
        --no-push)
            NO_PUSH=1
            shift
            ;;
        --push-only)
            PUSH_ONLY=1
            shift
            ;;
        --no-test)
            NO_TEST=1
            shift
            ;;
        --image-prefix)
            image_url_prefix="${2:-}"
            [ -n "$image_url_prefix" ] || die "--image-prefix requires a value"
            shift 2
            ;;
        -h|--help)
            usage 0
            ;;
        --)
            shift
            while [ "$#" -gt 0 ]; do
                NAMES+=("$1")
                shift
            done
            ;;
        -*)
            die "unsupported argument: $1"
            ;;
        *)
            NAMES+=("$1")
            shift
            ;;
    esac
done

[ "${#NAMES[@]}" -gt 0 ] || usage 1
[ "$PUSH_ONLY" -eq 0 ] || [ -n "$OVERRIDE_TAG" ] || die "--push-only requires --tag"
[ "$PUSH_ONLY" -eq 0 ] || [ "$NO_PUSH" -eq 0 ] || die "--push-only and --no-push are mutually exclusive"

declare -A TARGETS
TARGETS[cpu]="cpu/default"
TARGETS[gpu]="gpu/default"
TARGETS[gpu_legacy61]="gpu/default"
TARGETS[gpu_tensorrt]="gpu_tensorrt/tensorrt"

declare -A GPU_VARIANTS
declare -A CUDA_VERSIONS
declare -A CUDA_MAJOR_MINORS
declare -A PYTORCH_CUDA_INDEXES
GPU_VARIANTS[gpu]="default"
GPU_VARIANTS[gpu_legacy61]="legacy61"
CUDA_VERSIONS[gpu_legacy61]="12.6.3"
CUDA_MAJOR_MINORS[gpu_legacy61]="12.6"
PYTORCH_CUDA_INDEXES[gpu]="cu130"
PYTORCH_CUDA_INDEXES[gpu_legacy61]="cu126"

PR_NUMBER=$(echo ${GIT_BRANCH:-} | sed -n '/^PR-/s/PR-//gp')
if [ "$OVERRIDE_TAG" ]; then
    TMP_TAG="$OVERRIDE_TAG"
elif [ "${TAG_NAME:-}" ]; then
    TMP_TAG="ci-$TAG_NAME"
elif [ "${GIT_BRANCH:-}" == "master" ]; then
    TMP_TAG="ci-$GIT_BRANCH"
elif [ "$PR_NUMBER" ]; then
    TMP_TAG=ci-pr-$PR_NUMBER
else
    # Not built with Jenkins
    TMP_TAG="trash"
fi

push_image() {
    local image_url="$1"
    local built_tag="$2"
    local publish_tag="$3"

    if [ "$built_tag" != "$publish_tag" ]; then
        docker tag "$image_url:$built_tag" "$image_url:$publish_tag"
    fi
    docker tag "$image_url:$built_tag" "$image_url:latest"
    docker push "$image_url:$publish_tag"
    docker push "$image_url:latest"
    docker image rm "$image_url:latest"
}

docker_gpu_run_args() {
    printf '%s\n' "${DOCKER_GPU_RUN_ARGS:---gpus all}"
}

target_needs_gpu_runtime() {
    case "$1" in
        gpu|gpu_tensorrt)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

for name in "${NAMES[@]}"; do
    target=${TARGETS[$name]}
    if [ ! "$target" ]; then
        echo "$name target doesn't exists"
        exit 1
    fi

    arch=${target%%/*}
    build=${target##*/}
    image_url="${image_url_prefix}_${name}"
    gpu_variant=${GPU_VARIANTS[$name]:-default}
    cuda_version=${CUDA_VERSIONS[$name]}
    cuda_major_minor=${CUDA_MAJOR_MINORS[$name]}
    pytorch_cuda_index=${PYTORCH_CUDA_INDEXES[$name]}
    release="OFF"
    if [ "$FORCE_RELEASE" -eq 1 ]; then
        release="ON"
    elif [ "${TAG_NAME:-}" ]; then
        already_exists=$(DOCKER_CLI_EXPERIMENTAL=enabled docker manifest inspect ${image_url}:$TAG_NAME 2>/dev/null || true)
        if [ "$already_exists" ]; then
            echo "${image_url}:$TAG_NAME already built skipping..."
            continue
        fi
        release="ON"
    fi

    build_args=(
        --build-arg DEEPDETECT_BUILD=$build
        --build-arg DEEPDETECT_RELEASE=$release
    )
    if [ "$arch" = "gpu" ]; then
        build_args+=(--build-arg DEEPDETECT_GPU_VARIANT=$gpu_variant)
        [ "$cuda_version" ] && build_args+=(--build-arg DD_CUDA_VERSION=$cuda_version)
        [ "$cuda_major_minor" ] && build_args+=(--build-arg DD_CUDA_MAJOR_MINOR=$cuda_major_minor)
        [ "$pytorch_cuda_index" ] && build_args+=(--build-arg PYTORCH_CUDA_INDEX=$pytorch_cuda_index)
    fi

    if [ "$PUSH_ONLY" -eq 0 ]; then
        # BUILD
        docker build \
            -t $image_url:$TMP_TAG \
            --progress plain \
            "${build_args[@]}" \
            -f docker/${arch}.Dockerfile \
            .

        # TEST
        if [ "$NO_TEST" -eq 0 ]; then
            safe_tag=$(printf '%s' "$TMP_TAG" | tr -c '[:alnum:]_.-' '_')
            CONTAINER_NAME="ci_testing_deepdetect_${name}_${safe_tag}"
            docker_run_args=(-p 1025-65535:8080 -d --name "$CONTAINER_NAME")
            if target_needs_gpu_runtime "$arch"; then
                gpu_run_args="$(docker_gpu_run_args)"
                if [ -n "$gpu_run_args" ]; then
                    read -r -a gpu_run_args_array <<< "$gpu_run_args"
                    docker_run_args+=("${gpu_run_args_array[@]}")
                fi
            fi

            echo "+ docker run ${docker_run_args[*]} $image_url:$TMP_TAG"
            docker run "${docker_run_args[@]}" "$image_url:$TMP_TAG"
            log_and_cleanup () {
                docker logs "$CONTAINER_NAME" || true
                docker stop "$CONTAINER_NAME" || true
                docker rm "$CONTAINER_NAME" || true
            }
            trap log_and_cleanup EXIT

            PORT=$(docker port $CONTAINER_NAME 8080/tcp | awk -F: '{print $2}')
            if [ -z "$PORT" ]; then
                echo "Container $CONTAINER_NAME did not expose 8080/tcp; logs follow:" >&2
                docker logs "$CONTAINER_NAME" >&2 || true
                exit 1
            fi

            timeout 60 sh -c "until nc -z localhost $PORT; do sleep 1; done"

            sleep 10  # Wait dd start

            curl -s --head --request GET http://localhost:$PORT/info | head -1 | grep 'HTTP/1.1 200'
            trap - EXIT

            docker stop $CONTAINER_NAME
            docker rm $CONTAINER_NAME
        fi
    fi

    # PUSH
    if [ "$TMP_TAG" != "trash" ] && [ "$NO_PUSH" -eq 0 ]; then
        if [ "$OVERRIDE_TAG" ]; then
            push_image "$image_url" "$TMP_TAG" "$OVERRIDE_TAG"
        elif [ "${TAG_NAME:-}" ]; then
            push_image "$image_url" "$TMP_TAG" "$TAG_NAME"
            docker image rm $image_url:${TAG_NAME}
        elif [ "${GIT_BRANCH:-}" == "master" ]; then
            docker push $image_url:$TMP_TAG
        fi
        docker image rm $image_url:$TMP_TAG
    elif [ "$TMP_TAG" = "trash" ] && [ "$NO_PUSH" -eq 0 ] && [ "$PUSH_ONLY" -eq 0 ]; then
        docker image rm $image_url:$TMP_TAG
    fi
done
