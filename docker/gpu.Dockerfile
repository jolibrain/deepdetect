# Default CUDA version
ARG CUDA_VERSION=9.0-cudnn7

# Download default Deepdetect models
ARG DEEPDETECT_DEFAULT_MODELS=true

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu16.04 AS build

ARG DEEPDETECT_ARCH=gpu
ARG DEEPDETECT_BUILD=default

# Install build dependencies
RUN apt-get update && \
    apt-get install -y git \
    cmake \
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
    libcppnetlib-dev \
    libboost-dev \
    libboost-iostreams-dev \
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
    wget \
    autoconf \
    libtool-bin \
    python-numpy \
    swig \
    curl \
    unzip \
    libspdlog-dev \
    python-setuptools \
    python-dev \
    python-wheel \
    python-pip \
    unzip \
    libgoogle-perftools-dev \
    curl \
    libspdlog-dev \
    libarchive-dev \
    bash-completion && \
    wget -O /tmp/bazel.deb https://github.com/bazelbuild/bazel/releases/download/0.24.1/bazel_0.24.1-linux-x86_64.deb && \
    dpkg -i /tmp/bazel.deb && \
    apt-get remove -y libcurlpp0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Fix "ImportError: No module named builtins"
RUN pip install future pyyaml typing six enum enum34

# Git config
RUN git config --global user.email "build@local.local" && \
    git config --global user.name "Build"

WORKDIR /opt
RUN git clone https://github.com/jpbarrette/curlpp.git
WORKDIR /opt/curlpp
RUN cmake . && \
    make install && \
    cp /usr/local/lib/libcurlpp.* /usr/lib/

# Build Deepdetect
ADD ./ /opt/deepdetect
WORKDIR /opt/deepdetect/

RUN ./build.sh
# Copy libs to /tmp/libs for next build stage
RUN ./get_libs.sh

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu16.04

# Download default Deepdetect models
ARG DEEPDETECT_DEFAULT_MODELS=true

# Copy Deepdetect binaries from previous step
COPY --from=build /opt/deepdetect/main /opt/deepdetect/main

LABEL maintainer="emmanuel.benazera@jolibrain.com"
LABEL description="DeepDetect deep learning server & API / GPU version"

# Install tools and dependencies
RUN apt-get update && \ 
    apt-get install -y wget \
    libopenblas-base \
    liblmdb0 \
    libleveldb1v5 \
    libboost-regex1.58.0 \
    libgoogle-glog0v5 \
    libopencv-highgui2.4v5 \
    libcppnetlib0 \
    libgflags2v5 \
    libcurl3 \
    libhdf5-cpp-11 \
    libboost-filesystem1.58.0 \
    libboost-thread1.58.0 \
    libboost-iostreams1.58.0 \
    libarchive13 \
    libprotobuf9v5 && \
    rm -rf /var/lib/apt/lists/*

# Fix permissions
RUN ln -sf /dev/stdout /var/log/deepdetect.log && \
    ln -sf /dev/stderr /var/log/deepdetect.log

RUN useradd -ms /bin/bash dd && \
    chown dd:dd /opt
USER dd

# External volume to be mapped, e.g. for models or training data
RUN mkdir /opt/models

# Include a few image models within the image
WORKDIR /opt/models
RUN /opt/deepdetect/get_models.sh

COPY --chown=dd --from=build /opt/deepdetect/datasets/imagenet/corresp_ilsvrc12.txt /opt/models/ggnet/corresp.txt
COPY --chown=dd --from=build /opt/deepdetect/datasets/imagenet/corresp_ilsvrc12.txt /opt/models/resnet_50/corresp.txt
COPY --chown=dd --from=build /opt/deepdetect/templates/caffe/googlenet/*prototxt /opt/models/ggnet/
COPY --chown=dd --from=build /opt/deepdetect/templates/caffe/resnet_50/*prototxt /opt/models/resnet_50/
COPY --from=build /tmp/lib/* /usr/lib/

WORKDIR /opt/deepdetect/main
VOLUME ["/data"]

# Set entrypoint
CMD ./dede -host 0.0.0.0
EXPOSE 8080
