# syntax = docker/dockerfile:1.0-experimental

ARG DD_UBUNTU_VERSION=22.04
ARG DD_CUDA_VERSION=12.1.1
ARG DD_CUDNN_VERSION=8
FROM nvidia/cuda:${DD_CUDA_VERSION}-cudnn${DD_CUDNN_VERSION}-devel-ubuntu${DD_UBUNTU_VERSION}


RUN echo UBUNTU_VERSION=${DD_UBUNTU_VERSION} >> /image-info
RUN echo CUDA_VERSION=${DD_CUDA_VERSION} >> /image-info
RUN echo CUDNN_VERSION=${DD_CUDNN_VERSION} >> /image-info


RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update -y && apt-get upgrade -y && apt-get install -y ca-certificates gpg  wget 
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
RUN echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ jammy main' |  tee /etc/apt/sources.list.d/kitware.list >/dev/null
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
    clang-format\
    build-essential \
    openjdk-8-jdk \
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
    python-setuptools \
    python3-dev \
    python3-pip \
    tox \
    python-six \
    unzip \
    libgoogle-perftools-dev \
    curl \
    git \
    libarchive-dev \
    bash-completion \
    schedtool \
    util-linux \
    googletest \
    googletest-tools \
    python3-yaml \
    python3-numpy 

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
RUN python -m pip install --upgrade pip
RUN python -m pip install  torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN python -m pip install onnx


RUN apt clean -y
ADD ci/gitconfig /etc/gitconfig
WORKDIR /root
