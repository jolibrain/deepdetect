# syntax = docker/dockerfile:1.0-experimental
#FROM registry.access.redhat.com/ubi7/ubi AS build
FROM centos:7 AS build

ARG DEEPDETECT_RELEASE=OFF
ARG DEEPDETECT_ARCH=cpu
ARG DEEPDETECT_BUILD=default
ARG DEEPDETECT_DEFAULT_MODELS=true

# Official
RUN yum install -y \
    make \
    gcc \
    gcc-c++ \
    git \
    java-1.8.0-openjdk \
    zip \
    zlib-devel \
    opencv-core \
    boost-devel \
    openssl-devel \
    libomp-devel \
    curl-devel \
    openblas \
    atlas-devel \
    protobuf \
    protobuf-lite \
    protobuf-c \
    libtool \
    python2-devel \
    python2-setuptools \
    python2-numpy \
    python2-pyyaml \
    python2-wheel \
    python2-pip \
    python2-six \
    swig \
    curl \
    unzip \
    libarchive

# Extra official centos repos
RUN yum install -y 'dnf-command(config-manager)'
RUN yum config-manager --set-enabled PowerTools
RUN yum install -y \
    snappy-devel \
    gflags-devel \
    glog-devel \
    eigen3-devel \
    opencv-devel \
    libarchive-devel \
    protobuf-compiler \
    protobuf-devel \
    python3-pyyaml \
    openblas-devel


# Community
RUN yum install -y https://download-ib01.fedoraproject.org/pub/epel/7/x86_64/Packages/e/epel-release-7-12.noarch.rpm
RUN yum install -y \
    leveldb-devel \
    hdf5-devel \
    lmdb-devel \
    utf8cpp-devel \
    rapidjson-devel \
    python2-enum34 \
    ccache \
    gperftools-devel

# Grabbed from fedora 33
RUN dnf install -y \
    https://download-ib01.fedoraproject.org/pub/fedora/linux/releases/33/Everything/x86_64/os/Packages/m/mapbox-variant-devel-1.2.0-1.fc33.x86_64.rpm \
    https://download-ib01.fedoraproject.org/pub/fedora/linux/releases/33/Everything/x86_64/os/Packages/c/curlpp-0.8.1-13.fc33.x86_64.rpm \
    https://download-ib01.fedoraproject.org/pub/fedora/linux/releases/33/Everything/x86_64/os/Packages/c/curlpp-devel-0.8.1-13.fc33.x86_64.rpm

# Untrusted sources for tensorflow only, so we don't care
# RUN yum config-manager --add-repo https://copr.fedorainfracloud.org/coprs/vbatts/bazel/repo/epel-8/vbatts-bazel-epel-8.repo
# RUN yum install -y bazel

# No packaged on centos 8
RUN pip2 install typing

# centos 8 version 3.11, we need 3.14
RUN pip2 install cmake


ADD . /opt/deepdetect
WORKDIR /opt/deepdetect/

ENV CCACHE_DIR=/ccache
ENV PATH=/usr/lib/ccache:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# Build Deepdetect
ENV TERM=xterm
RUN --mount=type=cache,target=/ccache/ mkdir build && cd build && ../build.sh
# TODO(sileht):
# * Must allow to pass -DBUILD_SPDLOG=ON -DWARNING=OFF
# * Replace cpp-netlib by aot++

# Copy libs to /tmp/libs for next build stage
#RUN ./docker/get_libs.sh

## Build final Docker image
#FROM ubuntu:18.04 AS runtime
#
#ARG DEEPDETECT_ARCH=cpu
#
## Copy Deepdetect binaries from previous step
#COPY --from=build /opt/deepdetect/build/main /opt/deepdetect/build/main
#COPY --from=build /opt/deepdetect/get_models.sh /opt/deepdetect/get_models.sh
#
#LABEL description="DeepDetect deep learning server & API / ${DEEPDETECT_ARCH} version"
#LABEL maintainer="emmanuel.benazera@jolibrain.com"
#
## Install tools and dependencies
#RUN --mount=type=cache,id=apt_cache_cpu,target=/var/cache/apt --mount=type=cache,id=apt_lib_cpu,target=/var/lib/apt \
#    export DEBIAN_FRONTEND=noninteractive && \
#    apt-get update -y && apt-get install -y \
#    wget \
#    curl \
#	libopenblas-base \
#	liblmdb0 \
#	libleveldb1v5 \
#    libboost-regex1.62.0 \
#	libgoogle-glog0v5 \
#	libopencv3.2 \
#	libgflags2.2 \
#	libcurl4 \
#	libcurlpp0 \
#	libhdf5-cpp-100 \
#    libboost-atomic1.65.1 \
#    libboost-chrono1.65.1 \
#    libboost-date-time1.65.1 \
#	libboost-filesystem1.65.1 \
#	libboost-thread1.65.1 \
#	libboost-iostreams1.65.1 \
#    libboost-regex1.65.1 \
#	libarchive13 \
#	libprotobuf10

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

#COPY --chown=dd --from=build /opt/deepdetect/datasets/imagenet/corresp_ilsvrc12.txt /opt/models/ggnet/corresp.txt
#COPY --chown=dd --from=build /opt/deepdetect/datasets/imagenet/corresp_ilsvrc12.txt /opt/models/resnet_50/corresp.txt
#COPY --chown=dd --from=build /opt/deepdetect/templates/caffe/googlenet/*prototxt /opt/models/ggnet/
#COPY --chown=dd --from=build /opt/deepdetect/templates/caffe/resnet_50/*prototxt /opt/models/resnet_50/
#COPY --from=build /tmp/lib/* /usr/lib/
#COPY --from=build /opt/deepdetect/templates /opt/deepdetect/build/templates
#COPY --from=build /opt/deepdetect/docker/check-dede-deps.sh /opt/deepdetect/

RUN /opt/deepdetect/check-dede-deps.sh

WORKDIR /opt/deepdetect/build/main
VOLUME ["/data"]

# Set entrypoint
CMD ./dede -host 0.0.0.0
EXPOSE 8080
