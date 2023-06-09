# syntax = docker/dockerfile:1.0-experimental

ARG DD_UBUNTU_VERSION=22.04
ARG DD_CUDA_VERSION=12.1
ARG DD_CUDNN_VERSION=8
ARG DD_TENSORRT_VERSION=8.6.1+cuda12.1

# FROM nvidia/cuda:${DD_CUDA_VERSION}-cudnn${DD_CUDNN_VERSION}-devel-ubuntu${DD_UBUNTU_VERSION}
FROM nvcr.io/nvidia/tensorrt:23.05-py3 AS build

ARG DD_UBUNTU_VERSION
ARG DD_CUDA_VERSION
ARG DD_CUDNN_VERSION
ARG DD_TENSORRT_VERSION

RUN echo UBUNTU_VERSION=${DD_UBUNTU_VERSION} >> /image-info
RUN echo CUDA_VERSION=${DD_CUDA_VERSION} >> /image-info
RUN echo CUDNN_VERSION=${DD_CUDNN_VERSION} >> /image-info
RUN echo TENSORRT_VERSION=${DD_TENSORRT_VERSION} >> /image-info

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update -y && apt-get install -y apt-transport-https ca-certificates gnupg software-properties-common wget curl

# python2 pycompile + docker-buildkit is a bit buggy, it's slow as hell, so disable it for dev
# bug closed as won't fix as python2 is eol: https://github.com/docker/for-linux/issues/502
# RUN cp /bin/true /usr/bin/pycompile

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update -y && apt-get install -y \
    git \
    ccache \
    automake \
    rsync \
    clang-format-14 \
    build-essential \
    openjdk-8-jdk \
    pkg-config \
    cmake \
    zip \
    gcc-11 g++-11 \
    zlib1g-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libeigen3-dev \
    libprotobuf-dev \
    protobuf-compiler \
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
    curl \
    unzip \
    python-setuptools \
    tox \
    unzip \
    libgoogle-perftools-dev \
    curl \
    git \
    libarchive-dev \
    bash-completion \
    schedtool \
    util-linux \
    libgstreamer1.0-dev

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install torch==2.0.1

WORKDIR /tmp

# Install NVidia video codec
RUN wget http://www.deepdetect.com/stuff/Video_Codec_SDK_11.1.5.zip && unzip Video_Codec_SDK_11.1.5.zip
RUN cd Video_Codec_SDK_11.1.5 && cp Interface/* /usr/local/cuda/targets/x86_64-linux/include/ && cp Lib/linux/stubs/x86_64/* /usr/local/cuda/targets/x86_64-linux/lib/stubs/

# Build OpenCV 4 with CUDA
RUN mkdir opencv && cd opencv && wget -O opencv.zip https://github.com/opencv/opencv/archive/refs/tags/4.7.0.zip && wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/refs/tags/4.7.0.zip && unzip opencv.zip && unzip opencv_contrib.zip
RUN cd /tmp/opencv/opencv-4.7.0 && mkdir build && cd build && cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local/ \
-D CMAKE_LIBRARY_PATH=/usr/local/cuda/lib64/stubs \
-D CMAKE_CXX_FLAGS="-Wl,--allow-shlib-undefined" \
-D WITH_TBB=ON \
-D ENABLE_FAST_MATH=1 \
-D CUDA_FAST_MATH=1 \
-D WITH_CUBLAS=1 \
-D WITH_CUDA=ON \
-D BUILD_opencv_cudacodec=ON \
-D WITH_CUDNN=ON \
-D OPENCV_DNN_CUDA=OFF \
-D CUDA_ARCH_BIN="6.1 7.5 8.6" \
-D WITH_V4L=ON \
-D WITH_QT=OFF \
-D WITH_OPENGL=ON \
-D WITH_GSTREAMER=ON \
-D WITH_NVCUVID=ON \
-D OPENCV_GENERATE_PKGCONFIG=ON \
-D OPENCV_PC_FILE_NAME=opencv.pc \
-D OPENCV_ENABLE_NONFREE=ON \
-D BUILD_opencv_python2=OFF \
-D BUILD_opencv_python3=OFF \
-D OPENCV_EXTRA_MODULES_PATH=/tmp/opencv/opencv_contrib-4.7.0/modules \
-D INSTALL_PYTHON_EXAMPLES=OFF \
-D INSTALL_C_EXAMPLES=OFF \
-D BUILD_EXAMPLES=OFF ..

WORKDIR /tmp/opencv/opencv-4.7.0/build
RUN make -j20
RUN make install

RUN apt clean -y
ADD ci/gitconfig /etc/gitconfig
WORKDIR /root
