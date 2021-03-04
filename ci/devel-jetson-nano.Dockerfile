# Special docker image for Jetson nano
FROM nvcr.io/nvidia/deepstream-l4t:5.1-21.02-base

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update -y && apt-get install -y \
    python-dev apt-transport-https ca-certificates gnupg software-properties-common wget curl \
    cmake \
    git \
    ccache \
    automake \
    rsync \
    clang-format-10 \
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
    python-numpy \
    python-yaml \
    python-typing \
    swig \
    curl \
    unzip \
    python-setuptools \
    python-dev \
    python3-dev \
    python3-pip \
    tox \
    python-six \
    python-enum34 \
    python3-yaml \
    unzip \
    libgoogle-perftools-dev \
    curl \
    git \
    libarchive-dev \
    bash-completion \
    schedtool \
    util-linux

    # Not packaged by nvidia, but already installed in base images
    #libnvparsers7=${DD_TENSORRT_VERSION} \
    #libnvparsers-dev=${DD_TENSORRT_VERSION} \
    #libnvinfer7=${DD_TENSORRT_VERSION} \
    #libnvinfer-dev=${DD_TENSORRT_VERSION} \
    #libnvinfer-plugin7=${DD_TENSORRT_VERSION} \
    #libnvinfer-plugin-dev=${DD_TENSORRT_VERSION}

RUN apt clean -y

RUN curl -L https://github.com/Kitware/CMake/releases/download/v3.19.6/cmake-3.19.6-Linux-aarch64.sh -o cmake-install.sh
RUN chmod +x ./cmake-install.sh
RUN ./cmake-install.sh --prefix=/usr/local --skip-license

# ubuntu GTEST ugly packaging
WORKDIR /usr/src/gtest
RUN cmake .
RUN make -j8
RUN make install

ADD ci/gitconfig /etc/gitconfig
WORKDIR /root
