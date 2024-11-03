# syntax = docker/dockerfile:1.0-experimental

ARG DD_UBUNTU_VERSION=22.04
ARG DD_CUDA_VERSION=12.1.1
ARG DD_CUDNN_VERSION=8
FROM nvidia/cuda:${DD_CUDA_VERSION}-cudnn${DD_CUDNN_VERSION}-devel-ubuntu${DD_UBUNTU_VERSION} AS build


RUN echo UBUNTU_VERSION=${DD_UBUNTU_VERSION} >> /image-info
RUN echo CUDA_VERSION=${DD_CUDA_VERSION} >> /image-info
RUN echo CUDNN_VERSION=${DD_CUDNN_VERSION} >> /image-info

ARG DEEPDETECT_RELEASE=OFF
ARG DEEPDETECT_ARCH=gpu
ARG DEEPDETECT_BUILD=default
ARG DEEPDETECT_DEFAULT_MODELS=true

# Install build dependencies
RUN rm -f /etc/apt/apt.conf.d/docker-clean; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache

RUN --mount=type=cache,id=dede_cache_lib,sharing=locked,target=/var/cache/apt \
    --mount=type=cache,id=dede_apt_lib,sharing=locked,target=/var/lib/apt \
    export DEBIAN_FRONTEND=noninteractive && \
    apt-get update -y && apt-get install -y python3-dev apt-transport-https ca-certificates gnupg software-properties-common wget curl

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update -y && apt-get upgrade -y && apt-get install -y ca-certificates gpg  wget 
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
RUN echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main' |  tee /etc/apt/sources.list.d/kitware.list >/dev/null
RUN export DEBIAN_FRONTEND=noninteractive && apt-get update -y 
RUN rm /usr/share/keyrings/kitware-archive-keyring.gpg
RUN export DEBIAN_FRONTEND=noninteractive && apt-get install -y kitware-archive-keyring
RUN export DEBIAN_FRONTEND=noninteractive && apt-get install -y cmake

# Don't install opencv-ml-dev, it will install libprotobuf dans link dede to 2 versions of protobuf
RUN --mount=type=cache,id=dede_cache_lib,sharing=locked,target=/var/cache/apt \
    --mount=type=cache,id=dede_apt_lib,sharing=locked,target=/var/lib/apt \
    export DEBIAN_FRONTEND=noninteractive && \
    apt-get install -y \
    git \
    ccache \
    automake \
    build-essential \
    openjdk-8-jdk \
    pkg-config \
    zip \
    g++ \
    zlib1g-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libeigen3-dev \
    libopencv-dev \
    libboost-dev \
    libboost-filesystem-dev \
    libboost-thread-dev \
    libboost-system-dev \
    libboost-iostreams-dev \
    libboost-program-options-dev \
    libboost-test-dev \
    libboost-regex-dev \
    libboost-date-time-dev \
    libboost-chrono-dev \
    libboost-stacktrace-dev \
    libssl-dev \
    libcurlpp-dev \
    libcurl4-openssl-dev \
    libopenblas-dev \
    libhdf5-dev \
    libleveldb-dev \
    libsnappy-dev \
    liblmdb-dev \
    libutfcpp-dev \
    rapidjson-dev \
    libmapbox-variant-dev \
    autoconf \
    libtool-bin \
    python3-numpy \
    python3-yaml \
    swig \
    curl \
    unzip \
    python-setuptools \
    python3-dev \
    python3-dev \
    python3-pip \
    python-six \
    python3-yaml \
    unzip \
    libgoogle-perftools-dev \
    curl \
    libarchive-dev \
    libtcmalloc-minimal4 \
    bash-completion


# Fix "ImportError: No module named builtins"
# RUN pip install future pyyaml typing

# Fix  ModuleNotFoundError: No module named 'dataclasses', 'typing_extensions' for torch 1.8.0
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN python -m pip install --upgrade pip
RUN python -m pip install dataclasses typing_extensions

ADD . /opt/deepdetect
WORKDIR /opt/deepdetect/

ENV CCACHE_DIR=/ccache
ENV PATH=/usr/lib/ccache:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# Build Deepdetect
ENV TERM=xterm
RUN --mount=type=cache,target=/ccache/ mkdir build && cd build && ../build.sh

# Copy libs to /tmp/libs for next build stage
RUN ./docker/get_libs.sh

# Build final Docker image
FROM nvidia/cuda:${DD_CUDA_VERSION}-cudnn${DD_CUDNN_VERSION}-runtime-ubuntu${DD_UBUNTU_VERSION} AS runtime

ARG DEEPDETECT_ARCH=gpu

LABEL description="DeepDetect deep learning server & API / ${DEEPDETECT_ARCH} version"
LABEL maintainer="emmanuel.benazera@jolibrain.com"

# Install tools and dependencies
RUN --mount=type=cache,id=dede_cache_lib,sharing=locked,target=/var/cache/apt \
    --mount=type=cache,id=dede_apt_lib,sharing=locked,target=/var/lib/apt \
    export DEBIAN_FRONTEND=noninteractive && \
    apt-get update -y && apt-get install -y \
    wget \
    curl \
	libopenblas-base \
	liblmdb0 \
	libleveldb1d \
    libboost-regex1.74.0 \
	libgoogle-glog0v5 \
    libopencv-core4.5d \
    libopencv-contrib4.5d \
    libopencv-video4.5d \
    libopencv-videoio4.5d \
	libgflags2.2 \
	libcurl4 \
	libcurlpp0 \
	libhdf5-cpp-103 \
    libboost-atomic1.74.0 \
    libboost-chrono1.74.0 \
    libboost-date-time1.74.0 \
	libboost-filesystem1.74.0 \
	libboost-thread1.74.0 \
	libboost-iostreams1.74.0 \
    libboost-regex1.74.0 \
    libboost-stacktrace1.74.0 \
    libboost-system1.74.0 \
	libarchive13 \
	libtcmalloc-minimal4

# Fix permissions
RUN ln -sf /dev/stdout /var/log/deepdetect.log && \
    ln -sf /dev/stderr /var/log/deepdetect.log

RUN useradd -ms /bin/bash dd && \
    chown -R dd:dd /opt
USER dd

# Copy Deepdetect binaries from previous step
COPY --from=build /opt/deepdetect/build/main /opt/deepdetect/build/main
COPY --from=build /opt/deepdetect/build/oatpp-swagger/src/oatpp-swagger/res/ /opt/deepdetect/build/oatpp-swagger/src/oatpp-swagger/res/
COPY --from=build --chown=dd /opt/deepdetect/datasets/imagenet/corresp_ilsvrc12.txt /opt/models/ggnet/corresp.txt
COPY --from=build --chown=dd /opt/deepdetect/datasets/imagenet/corresp_ilsvrc12.txt /opt/models/resnet_50/corresp.txt
COPY --from=build --chown=dd /opt/deepdetect/templates/caffe/googlenet/*prototxt /opt/models/ggnet/
COPY --from=build --chown=dd /opt/deepdetect/templates/caffe/resnet_50/*prototxt /opt/models/resnet_50/
COPY --from=build /tmp/lib/* /opt/deepdetect/build/lib/
COPY --from=build /opt/deepdetect/templates /opt/deepdetect/build/templates

COPY --from=build /opt/deepdetect/get_models.sh /opt/deepdetect/
COPY --from=build /opt/deepdetect/docker/check-dede-deps.sh /opt/deepdetect/
COPY --from=build /opt/deepdetect/docker/start-dede.sh /opt/deepdetect/

COPY --from=build /usr/local/cuda-12.1/targets/x86_64-linux/lib/libcupti* /usr/local/cuda-12.1/targets/x86_64-linux/lib/

# External volume to be mapped, e.g. for models or training data
WORKDIR /opt/models

USER root
RUN chown -R dd:dd /opt/models

USER dd
RUN /opt/deepdetect/get_models.sh

# Ensure all libs are presents
RUN /opt/deepdetect/check-dede-deps.sh

WORKDIR /opt/deepdetect/build/main
ENTRYPOINT ["/opt/deepdetect/start-dede.sh", "-host", "0.0.0.0"]
VOLUME ["/data"]
EXPOSE 8080
