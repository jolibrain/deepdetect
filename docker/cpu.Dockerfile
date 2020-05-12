FROM ubuntu:16.04 AS build

ARG DEEPDETECT_ARCH=cpu
ARG DEEPDETECT_BUILD=default

# Add gcc7 repository
RUN apt update && \
    apt install -y software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test -y

# Install build dependencies
RUN apt-get update -y && \
    apt-get install -y git \
    automake \
    build-essential \
    openjdk-8-jdk \
    pkg-config \
    zip \
    g++ \
    gcc-7 g++-7 \
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
    python-future \
    swig \
    curl \
    unzip \
    libspdlog-dev \
    python-setuptools \
    python-dev \
    python-wheel \
    unzip \
    libgoogle-perftools-dev \
    libarchive-dev \
    bash-completion && \
    wget -O /tmp/bazel.deb https://github.com/bazelbuild/bazel/releases/download/0.24.1/bazel_0.24.1-linux-x86_64.deb && \
    dpkg -i /tmp/bazel.deb && \
    apt-get remove -y libcurlpp0 && \
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
FROM ubuntu:16.04

# Download default Deepdetect models
ARG DEEPDETECT_DEFAULT_MODELS=true

# Copy Deepdetect binaries from previous step
COPY --from=build /opt/deepdetect/build/main /opt/deepdetect/build/main
COPY --from=build /opt/deepdetect/get_models.sh /opt/deepdetect/get_models.sh

LABEL maintainer="emmanuel.benazera@jolibrain.com"
LABEL description="DeepDetect deep learning server & API / CPU version"

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

WORKDIR /opt/deepdetect/build/main
VOLUME ["/data"]

# Set entrypoint
CMD ./dede -host 0.0.0.0
EXPOSE 8080
