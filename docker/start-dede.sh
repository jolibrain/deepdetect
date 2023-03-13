#!/bin/bash
set -x
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/deepdetect/build/lib"
exec /opt/deepdetect/build/main/dede "$@"
