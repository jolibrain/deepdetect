#!/bin/bash

set -x

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/deepdetect/build/lib"

DEDE="/opt/deepdetect/build/main/dede"

echo "Checking dede dependencies"

if [ ! -e "$DEDE" ]; then
    echo "$DEDE not found"
    exit 1
fi

LIBS=$(ldd $DEDE | grep 'not found')
if [ "$LIBS" ] ; then
    echo "* missing libs"
    echo $LIBS
    exit 1
fi
