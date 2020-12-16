# syntax = docker/dockerfile:1.0-experimental

ARG DD_UBUNTU_VERSION=18.04
ARG DD_CUDA_VERSION=10.2
ARG DD_CUDNN_VERSION=7
ARG DD_TENSORRT_VERSION=7.1.3-1+cuda10.2

FROM nvidia/cuda:${DD_CUDA_VERSION}-cudnn${DD_CUDNN_VERSION}-devel-ubuntu${DD_UBUNTU_VERSION}

ARG DD_UBUNTU_VERSION
ARG DD_CUDA_VERSION
ARG DD_CUDNN_VERSION
ARG DD_TENSORRT_VERSION

RUN echo UBUNTU_VERSION=${DD_UBUNTU_VERSION} >> /image-info
RUN echo CUDA_VERSION=${DD_CUDA_VERSION} >> /image-info
RUN echo CUDNN_VERSION=${DD_CUDNN_VERSION} >> /image-info
RUN echo TENSORRT_VERSION=${DD_TENSORRT_VERSION} >> /image-info

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update -y && apt-get install -y python-dev apt-transport-https ca-certificates gnupg software-properties-common wget curl
RUN curl https://apt.kitware.com/keys/kitware-archive-latest.asc | apt-key add -
RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'

# python2 pycompile + docker-buildkit is a bit buggy, it's slow as hell, so disable it for dev
# bug closed as won't fix as python2 is eol: https://github.com/docker/for-linux/issues/502
RUN cp /bin/true /usr/bin/pycompile

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update -y && apt-get install -y \
    git \
    ccache \
    automake \
    clang-format-10 \
    build-essential \
    openjdk-8-jdk \
    pkg-config \
    cmake \
    zip \
    g++ \
    gcc-7 g++-7 \
    zlib1g-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libeigen3-dev \
    libopencv-dev \
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
    protobuf-compiler \
    libopenblas-dev \
    libhdf5-dev \
    libprotobuf-dev \
    libleveldb-dev \
    libsnappy-dev \
    liblmdb-dev \
    libutfcpp-dev \
    rapidjson-dev \
    libmapbox-variant-dev \
    autoconf \
    libtool-bin \
    python-numpy \
    python-future \
    python-yaml \
    python-typing \
    swig \
    curl \
    unzip \
    libspdlog-dev \
    python-setuptools \
    python-dev \
    python-tox \
    python3-dev \
    python3-pip \
    python-wheel \
    python-pip \
    python-six \
    python-enum34 \
    python3-yaml \
    unzip \
    libgoogle-perftools-dev \
    curl \
    libspdlog-dev \
    libarchive-dev \
    bash-completion \
    schedtool \
    util-linux \
    libnvparsers7=${DD_TENSORRT_VERSION} \
    libnvparsers-dev=${DD_TENSORRT_VERSION} \
    libnvinfer7=${DD_TENSORRT_VERSION} \
    libnvinfer-dev=${DD_TENSORRT_VERSION}
    libnvinfer-plugin7=${DD_TENSORRT_VERSION} \
    libnvinfer-plugin-dev=${DD_TENSORRT_VERSION}

RUN for url in \
        https://github.com/bazelbuild/bazel/releases/download/0.24.1/bazel_0.24.1-linux-x86_64.deb \
        ; do curl -L -s -o /tmp/p.deb $url && dpkg -i /tmp/p.deb && rm -rf /tmp/p.deb; done

RUN pip3 install torch

# ubuntu GTEST ugly packaging
WORKDIR /usr/src/gtest
RUN cmake .
RUN make -j8
RUN make install

RUN apt clean -y
RUN echo "[user]\n    email = jenkins@deepdetect.com\n    name = Deepdetect Jenkins Bot\n" > /etc/gitconfig
WORKDIR /root
