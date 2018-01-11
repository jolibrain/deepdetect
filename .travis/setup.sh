#!/bin/bash
# This script should be sourced, not executed
set -e

export BUILD_CUDA=false
export BUILD_TESTS=false

if [ "$BUILD" = 'linux' ]; then
    :
elif [ "$BUILD" = 'linux-cuda' ]; then
    export BUILD_CUDA=true
    export BUILD_TESTS=false
else
    echo "BUILD \"$BUILD\" is unknown"
    exit 1
fi

