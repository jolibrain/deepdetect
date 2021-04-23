# syntax = docker/dockerfile:1.0-experimental
FROM ubuntu:20.04 AS build

ARG DEEPDETECT_RELEASE=OFF
ARG DEEPDETECT_ARCH=cpu
ARG DEEPDETECT_BUILD=default
ARG DEEPDETECT_DEFAULT_MODELS=true

# Install build dependencies
RUN rm -f /etc/apt/apt.conf.d/docker-clean; echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache

RUN --mount=type=cache,id=dede_cache_lib,sharing=locked,target=/var/cache/apt \
    --mount=type=cache,id=dede_apt_lib,sharing=locked,target=/var/lib/apt \
    export DEBIAN_FRONTEND=noninteractive && \
    apt-get update -y && apt-get install -y python-dev apt-transport-https ca-certificates gnupg software-properties-common wget curl

# CMake
RUN curl https://apt.kitware.com/keys/kitware-archive-latest.asc | apt-key add -
RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ focal main'

# python2 pycompile + docker-buildkit is a bit buggy, it's slow as hell, so disable it for dev
# bug closed as won't fix as python2 is eol: https://github.com/docker/for-linux/issues/502
RUN cp /bin/true /usr/bin/pycompile

# Don't install opencv-ml-dev, it will install libprotobuf dans link dede to 2 versions of protobuf
RUN --mount=type=cache,id=dede_cache_lib,sharing=locked,target=/var/cache/apt \
    --mount=type=cache,id=dede_apt_lib,sharing=locked,target=/var/lib/apt \
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
    python-numpy \
    python-yaml \
    python-typing \
    swig \
    unzip \
    python-setuptools \
    python-dev \
    python3-dev \
    python3-pip \
    python-six \
    python-enum34 \
    python3-yaml \
    unzip \
    libgoogle-perftools-dev \
    curl \
    libarchive-dev \
    libtcmalloc-minimal4 \
    bash-completion \
    libomp-10-dev \
    libomp5-10

RUN for url in \
        https://github.com/bazelbuild/bazel/releases/download/0.24.1/bazel_0.24.1-linux-x86_64.deb \
        ; do curl -L -s -o /tmp/p.deb $url && dpkg -i /tmp/p.deb && rm -rf /tmp/p.deb; done

# Fix "ImportError: No module named builtins"
# RUN pip install future pyyaml typing

# Fix  ModuleNotFoundError: No module named 'dataclasses', 'typing_extensions' for torch 1.8.0
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install dataclasses typing_extensions

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
FROM ubuntu:20.04 AS runtime

ARG DEEPDETECT_ARCH=cpu

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
    libboost-regex1.71.0 \
    libgoogle-glog0v5 \
    libopencv4.2 \
    libgflags2.2 \
    libcurl4 \
    libcurlpp0 \
    libhdf5-cpp-103 \
    libboost-atomic1.71.0 \
    libboost-chrono1.71.0 \
    libboost-date-time1.71.0 \
    libboost-filesystem1.71.0 \
    libboost-thread1.71.0 \
    libboost-iostreams1.71.0 \
    libboost-regex1.71.0 \
    libboost-stacktrace1.71.0 \
    libboost-system1.71.0 \
    libarchive13 \
    libtcmalloc-minimal4 \
    libomp-10-dev \
    libomp5-10

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

# External volume to be mapped, e.g. for models or training data
WORKDIR /opt/models

USER root
RUN chown -R dd:dd /opt/models

USER dd
RUN /opt/deepdetect/get_models.sh

# Ensure all libs are presents
RUN /opt/deepdetect/check-dede-deps.sh

WORKDIR /opt/deepdetect/build/main
CMD /opt/deepdetect/start-dede.sh -host 0.0.0.0
VOLUME ["/data"]
EXPOSE 8080
