FROM ubuntu:18.04 AS build

ARG DEEPDETECT_ARCH=cpu
ARG DEEPDETECT_BUILD=default

# Install build dependencies
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update -y && \
    apt-get install -y git \
    automake \
    cmake \
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
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Build cpp-netlib
RUN wget https://github.com/cpp-netlib/cpp-netlib/archive/cpp-netlib-0.11.2-final.tar.gz && \
    tar xvzf cpp-netlib-0.11.2-final.tar.gz && \
    cd cpp-netlib-cpp-netlib-0.11.2-final && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j && \
    make install

# Copy Deepdetect sources files
WORKDIR /opt
RUN git clone https://github.com/jolibrain/deepdetect.git /opt/deepdetect
WORKDIR /opt/deepdetect/

# Build Deepdetect
RUN mkdir build && \
    cd build && \
    cp -a ../build.sh . && \
    ./build.sh

# Copy libs to /tmp/libs for next build stage
RUN ./docker/get_libs.sh

# Build final Docker image
FROM ubuntu:18.04

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
COPY --from=build /opt/deepdetect/templates /opt/deepdetect/build/templates

WORKDIR /opt/deepdetect/build/main
VOLUME ["/data"]

# Set entrypoint
CMD ./dede -host 0.0.0.0
EXPOSE 8080
