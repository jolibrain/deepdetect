FROM armv7/armhf-ubuntu:16.04 AS build

ARG DEEPDETECT_ARCH=cpu
ARG DEEPDETECT_BUILD=armv7

RUN apt-get update && \
    apt-get install -y git \
    build-essential \
    libgoogle-glog-dev \
    libgflags-dev \
    libeigen3-dev \
    libopencv-dev \
    libcppnetlib-dev \
    libboost-dev \
    libboost-iostreams-dev \
    libcurl4-openssl-dev \
    protobuf-compiler \
    libopenblas-dev \
    libprotobuf-dev \
    libleveldb-dev \
    libsnappy-dev \
    liblmdb-dev \
    libutfcpp-dev \
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

# Need cmake > 3.10 : https://github.com/jolibrain/ncnn/blob/master/CMakeLists.txt#L14
RUN mkdir /tmp/cmake && cd /tmp/cmake && \
    apt remove cmake && \
    wget https://cmake.org/files/v3.10/cmake-3.10.3.tar.gz && \
    tar xf cmake-3.10.3.tar.gz && \
    cd cmake-3.10.3 && \
    ./configure && \
    make install && \
    rm -rf /tmp/cmake

WORKDIR /opt
RUN git clone https://github.com/jpbarrette/curlpp.git
WORKDIR /opt/curlpp
RUN cmake . && \
    make install && \
    cp /usr/local/lib/libcurlpp.* /usr/lib/

# Build Deepdetect
ADD ./ /opt/deepdetect
WORKDIR /opt/deepdetect/build
RUN ./build.sh
# Copy libs to /tmp/libs for next build stage
RUN ./get_libs.sh

FROM armv7/armhf-ubuntu:16.04

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

# Copy missings libs from build step
COPY --from=build /tmp/lib/* /usr/lib/

WORKDIR /opt/deepdetect/build/main
VOLUME ["/data"]

# Set entrypoint
CMD ./dede -host 0.0.0.0
EXPOSE 8080