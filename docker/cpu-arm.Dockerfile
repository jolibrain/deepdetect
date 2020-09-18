# Default ARM version
ARG ARM_VERSION=arm32v7

FROM ${ARM_VERSION}/ubuntu:18.04 AS build

ARG DEEPDETECT_ARCH=cpu
ARG DEEPDETECT_BUILD=armv7

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get install -y git \
    build-essential \
    cmake \
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
    libcurl4-openssl-dev \
    protobuf-compiler \
    libopenblas-dev \
    libprotobuf-dev \
    libleveldb-dev \
    libsnappy-dev \
    liblmdb-dev \
    libutfcpp-dev \
    rapidjson-dev \
    wget \
    unzip \
    libspdlog-dev \
    python-setuptools \
    python-dev \
    libhdf5-dev \
    libarchive-dev \
    apt-transport-https \
    ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Build cpp-netlib
RUN wget https://github.com/cpp-netlib/cpp-netlib/archive/cpp-netlib-0.11.2-final.tar.gz && \
    tar xvzf cpp-netlib-0.11.2-final.tar.gz && \
    cd cpp-netlib-cpp-netlib-0.11.2-final && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make && \
    make install

WORKDIR /opt
RUN git clone https://github.com/jpbarrette/curlpp.git
WORKDIR /opt/curlpp
RUN cmake . && \
    make install && \
    cp /usr/local/lib/libcurlpp.* /usr/lib/

# Copy Deepdetect sources files
ADD ./ /opt/deepdetect
WORKDIR /opt/deepdetect/

# Build Deepdetect
RUN mkdir build && \
    cd build && \
    cp -a ../build.sh . && \
    ./build.sh

# Copy libs to /tmp/libs for next build stage
RUN ./docker/get_libs.sh

# Build final Docker image
FROM ${ARM_VERSION}/ubuntu:18.04

# Copy Deepdetect binaries from previous step
COPY --from=build /opt/deepdetect/build/main /opt/deepdetect/build/main

LABEL maintainer="emmanuel.benazera@jolibrain.com"
LABEL description="DeepDetect deep learning server & API / CPU NCNN-only RPi3 version"

# Install tools and dependencies
RUN apt-get update && \
    apt-get install -y wget \
	libopenblas-base \
	liblmdb0 \
	libleveldb1v5 \
    libboost-regex1.62.0 \
	libgoogle-glog0v5 \
	libopencv-highgui3.2 \
	libgflags2.2 \
	libcurl4 \
	libhdf5-cpp-100 \
	libboost-filesystem1.65.1 \
	libboost-thread1.65.1 \
	libboost-iostreams1.65.1 \
    libboost-regex1.65.1 \
	libarchive13 \
	libprotobuf10 && \
    rm -rf /var/lib/apt/lists/*

# Fix permissions
RUN ln -sf /dev/stdout /var/log/deepdetect.log && \
    ln -sf /dev/stderr /var/log/deepdetect.log

RUN useradd -ms /bin/bash dd && \
    chown dd:dd /opt
USER dd

# External volume to be mapped, e.g. for models or training data
RUN mkdir /opt/models

# Copy missings libs from build step
COPY --from=build /tmp/lib/* /usr/lib/

WORKDIR /opt/deepdetect/build/main
VOLUME ["/data"]

# Set entrypoint
CMD ./dede -host 0.0.0.0
EXPOSE 8080
