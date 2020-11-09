#!/bin/bash
#
# Workaround due to:
# https://github.com/NVIDIA/nvidia-docker/issues/1399
# Until nvidia-docker get fixed we need this hack, so dede can see the GPUs

gen_dockerfile(){
    base=$1
    cat << EOF
FROM $base
USER root
RUN export DEBIAN_FRONTEND=noninteractive && apt-get update -y && apt-get install -y sudo
CMD bash -c "ldconfig; nvidia-smi ; sudo -u dd ./dede -host 0.0.0.0"
EOF
}

for image in jolibrain/deepdetect_gpu jolibrain/deepdetect_gpu_tensorrt ; do
    docker pull ${image}
    gen_dockerfile ${image} | docker build -t ${image}_buster_workaround -
done
