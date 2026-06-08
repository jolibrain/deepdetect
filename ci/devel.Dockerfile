# syntax = docker/dockerfile:1.0-experimental

ARG DD_UBUNTU_VERSION=24.04
ARG DD_CUDA_VERSION=13.0.2
ARG PYTORCH_CUDA_INDEX=cu130
ARG DEEPDETECT_GPU_VARIANT=default
FROM nvidia/cuda:${DD_CUDA_VERSION}-cudnn-devel-ubuntu${DD_UBUNTU_VERSION}

ARG DD_UBUNTU_VERSION
ARG DD_CUDA_VERSION
ARG PYTORCH_CUDA_INDEX
ARG DEEPDETECT_GPU_VARIANT


RUN echo UBUNTU_VERSION=${DD_UBUNTU_VERSION} >> /image-info
RUN echo CUDA_VERSION=${DD_CUDA_VERSION} >> /image-info
RUN echo GPU_VARIANT=${DEEPDETECT_GPU_VARIANT} >> /image-info
RUN echo PYTORCH_CUDA_INDEX=${PYTORCH_CUDA_INDEX} >> /image-info


RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update -y && apt-get install -y ca-certificates gpg wget
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
RUN echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ noble main' |  tee /etc/apt/sources.list.d/kitware.list >/dev/null
RUN export DEBIAN_FRONTEND=noninteractive && apt-get update -y 
RUN rm /usr/share/keyrings/kitware-archive-keyring.gpg
RUN export DEBIAN_FRONTEND=noninteractive && apt-get install -y kitware-archive-keyring
RUN export DEBIAN_FRONTEND=noninteractive && apt-get install -y cmake


RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get install -y \
    build-essential \
    git \
    ccache \
    automake \
    rsync \
    clang-format-14 \
    build-essential \
    default-jdk \
    pkg-config \
    zip \
    zlib1g-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libeigen3-dev \
    libopencv-dev \
    libprotobuf-dev \
    protobuf-compiler \
    libboost-dev \
    libboost-filesystem-dev \
    libboost-thread-dev \
    libboost-system-dev \
    libboost-stacktrace-dev \
    libboost-iostreams-dev \
    libboost-program-options-dev \
    libboost-test-dev \
    libboost-regex-dev \
    libboost-date-time-dev \
    libboost-chrono-dev \
    libssl-dev \
    libgtest-dev \
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
    swig \
    curl \
    unzip \
    python3-setuptools \
    python3-dev \
    python3-pip \
    tox \
    python3-six \
    unzip \
    libgoogle-perftools-dev \
    curl \
    git \
    libarchive-dev \
    bash-completion \
    schedtool \
    util-linux \
    googletest \
    python3-yaml \
    python3-numpy 

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1
RUN python -m pip install --break-system-packages torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/${PYTORCH_CUDA_INDEX}
RUN python -m pip install --break-system-packages onnx onnxscript


RUN apt clean -y
ADD ci/gitconfig /etc/gitconfig
WORKDIR /root

# Workaround for dependencies with old cmake_minimum_required
ENV CMAKE_POLICY_VERSION_MINIMUM=3.5
