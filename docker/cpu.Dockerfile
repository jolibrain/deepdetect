# syntax = docker/dockerfile:1.0-experimental
FROM ubuntu:18.04 AS build

ARG DEEPDETECT_ARCH=cpu
ARG DEEPDETECT_BUILD=default
ARG DEEPDETECT_DEFAULT_MODELS=true

# Install build dependencies
RUN rm -f /etc/apt/apt.conf.d/docker-clean; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache

RUN --mount=type=cache,target=/var/cache/apt --mount=type=cache,target=/var/lib/apt \
    export DEBIAN_FRONTEND=noninteractive && \
    apt-get update -y && apt-get install -y python-dev apt-transport-https ca-certificates gnupg software-properties-common wget curl

# CMake
RUN curl https://apt.kitware.com/keys/kitware-archive-latest.asc | apt-key add -
RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'

# python2 pycompile + docker-buildkit is a bit buggy, it's slow as hell, so disable it for dev
# bug closed as won't fix as python2 is eol: https://github.com/docker/for-linux/issues/502
RUN cp /bin/true /usr/bin/pycompile

RUN --mount=type=cache,target=/var/cache/apt --mount=type=cache,target=/var/lib/apt \
    export DEBIAN_FRONTEND=noninteractive && \
    apt-get update -y && apt-get install -y \
    git \
    ccache \
    automake \
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
    libboost-iostreams-dev \
    libboost-program-options-dev \
    libboost-test-dev \
    libssl-dev \
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
    python-wheel \
    python-pip \
    python-six \
    python-enum34 \
    unzip \
    libgoogle-perftools-dev \
    curl \
    libspdlog-dev \
    libarchive-dev \
    bash-completion

## Need recent cmake version for cuda 10
#RUN mkdir /tmp/cmake && cd /tmp/cmake && \
#    apt remove cmake && \
#    wget https://cmake.org/files/v3.14/cmake-3.14.0.tar.gz && \
#    tar xf cmake-3.14.0.tar.gz && \
#    cd cmake-3.14.0 && \
#    ./configure && \
#    make install && \
#    rm -rf /tmp/cmake

RUN wget -O /tmp/bazel.deb https://github.com/bazelbuild/bazel/releases/download/0.24.1/bazel_0.24.1-linux-x86_64.deb && \
    dpkg -i /tmp/bazel.deb && rm -rf /tmp/bazel.deb

# Fix "ImportError: No module named builtins"
# RUN pip install future pyyaml typing

ADD . /opt/deepdetect
WORKDIR /opt/deepdetect/

ENV CCACHE_DIR=/ccache
ENV PATH=/usr/lib/ccache:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# Build cpp-netlib
# TODO(sileht): move to a package version
RUN --mount=type=cache,target=/ccache/ \
    wget https://github.com/cpp-netlib/cpp-netlib/archive/cpp-netlib-0.11.2-final.tar.gz && \
    tar xvzf cpp-netlib-0.11.2-final.tar.gz && \
    cd cpp-netlib-cpp-netlib-0.11.2-final && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j && \
    make install

# Build Deepdetect
ENV TERM=xterm
RUN --mount=type=cache,target=/ccache/ mkdir build && cd build && ../build.sh

# Copy libs to /tmp/libs for next build stage
RUN ./docker/get_libs.sh

# Build final Docker image
FROM ubuntu:18.04
ARG DEEPDETECT_ARCH=cpu

# Copy Deepdetect binaries from previous step
COPY --from=build /opt/deepdetect/build/main /opt/deepdetect/build/main
COPY --from=build /opt/deepdetect/get_models.sh /opt/deepdetect/get_models.sh

LABEL description="DeepDetect deep learning server & API / ${DEEPDETECT_ARCH} version"
LABEL maintainer="emmanuel.benazera@jolibrain.com"

# Install tools and dependencies
RUN --mount=type=cache,target=/var/cache/apt --mount=type=cache,target=/var/lib/apt \
    apt-get update -y && \ apt-get install -y \
    wget \
	libopenblas-base \
	liblmdb0 \
	libleveldb1v5 \
    libboost-regex1.62.0 \
	libgoogle-glog0v5 \
	libopencv-highgui3.2 \
	libgflags2.2 \
	libcurl4 \
	libcurlpp0 \
	libhdf5-cpp-100 \
	libboost-filesystem1.65.1 \
	libboost-thread1.65.1 \
	libboost-iostreams1.65.1 \
    libboost-regex1.65.1 \
	libarchive13 \
	libprotobuf10 && \

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
COPY --from=build /opt/deepdetect/templates /opt/deepdetect/build/templates

WORKDIR /opt/deepdetect/build/main
VOLUME ["/data"]

# Set entrypoint
CMD ./dede -host 0.0.0.0
EXPOSE 8080
