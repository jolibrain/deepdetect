#!/bin/bash

set -e

if [ "$DEEPDETECT_DEFAULT_MODELS" = false ] ; then
    echo "Downloading default Deepdetect models is disable"
else
    echo "Downloading default Deepdetect models is enable"
    echo "To disable it set DEEPDETECT_DEFAULT_MODELS env variable to false : DEEPDETECT_DEFAULT_MODELS=false"
    mkdir -p ggnet resnet_50

    pushd ggnet
        wget https://www.deepdetect.com/models/ggnet/bvlc_googlenet.caffemodel
    popd

    pushd resnet_50
        wget https://www.deepdetect.com/models/resnet/ResNet-50-model.caffemodel
        wget https://www.deepdetect.com/models/resnet/ResNet_mean.binaryproto
        mv ResNet_mean.binaryproto mean.binaryproto
    popd
fi
