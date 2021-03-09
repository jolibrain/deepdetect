# Special docker image for Jetson nano
FROM nvcr.io/nvidia/deepstream-l4t:5.1-21.02-base

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update -y && apt-get install -y \
    wget \
    curl \
    cmake \
    git \
    ccache \
    automake \
    rsync \
    build-essential \
    pkg-config \
    zip \
    g++ \
    gcc-7 g++-7 \
    zlib1g-dev \
    protobuf-compiler \
    libprotobuf-dev \
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
    swig \
    unzip \
    libgoogle-perftools-dev \
    libarchive-dev \
    bash-completion \
    schedtool \
    python-numpy \
    util-linux

RUN apt clean -y

RUN curl -L https://github.com/Kitware/CMake/releases/download/v3.19.6/cmake-3.19.6-Linux-aarch64.sh -o cmake-install.sh
RUN chmod +x ./cmake-install.sh
RUN ./cmake-install.sh --prefix=/usr/local --skip-license

# ubuntu GTEST ugly packaging
WORKDIR /usr/src/gtest
RUN cmake .
RUN make -j8
RUN make install

# NOTE(sileht): docker nvidia on jetson is bugged with non-root account, we
# have to configure the device access manually. The UID and GID must be the
# same as the jenkins user on the jetson used by the CI.
RUN addgroup --gid 1001 jenkins
RUN useradd -M -s /bin/bash --uid 1001 --gid 1001 jenkins
RUN usermod -a -G video jenkins

ADD ci/deviceQuery /deviceQuery
ADD ci/gitconfig /etc/gitconfig
WORKDIR /root
