# syntax = docker/dockerfile:1.0-experimental

FROM nvcr.io/nvidia/tensorrt:26.02-py3 AS build

ARG DD_CUDA_VERSION
ARG DEEPDETECT_OPENCV4_BUILD_PATH=/tmp/opencv/opencv-4.13.0/build
ARG DEEPDETECT_TENSORRT_VERSION=

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update -y && apt-get install -y apt-transport-https ca-certificates gnupg software-properties-common wget curl

RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update -y && apt-get install -y \
    git \
    ccache \
    automake \
    rsync \
    clang-format-14 \
    build-essential \
    default-jdk \
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
    python3-setuptools \
    tox \
    unzip \
    libgoogle-perftools-dev \
    curl \
    git \
    libarchive-dev \
    libgstreamer-plugins-base1.0-dev \
    libgstreamer-plugins-bad1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    gstreamer1.0-x \
    gstreamer1.0-gl \
    libgtk-3-dev \
    bash-completion \
    schedtool \
    util-linux \
    libgstreamer1.0-dev

RUN python3 -m pip install --break-system-packages --upgrade pip

WORKDIR /tmp/

ENV CCACHE_DIR=/ccache
ENV PATH=/usr/lib/ccache:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs
ENV DEEPDETECT_OPENCV4_BUILD_PATH=${DEEPDETECT_OPENCV4_BUILD_PATH}
ENV DEEPDETECT_TENSORRT_VERSION=${DEEPDETECT_TENSORRT_VERSION}

# Install NVidia video codec
RUN wget http://www.deepdetect.com/stuff/Video_Codec_SDK_11.1.5.zip && unzip Video_Codec_SDK_11.1.5.zip
RUN cd Video_Codec_SDK_11.1.5 && cp Interface/* /usr/local/cuda/targets/x86_64-linux/include/ && \
    cp Lib/linux/stubs/x86_64/* /usr/local/cuda/targets/x86_64-linux/lib/stubs/ && \
    cd /usr/local/cuda/targets/x86_64-linux/lib/stubs/ && \
    ln -s libcuda.so libcuda.so.1 && ln -s libnvcuvid.so libnvcuvid.so.1 && ln -s libnvidia-encode.so libnvidia-encode.so.1

# Workaround for dependencies with old cmake_minimum_required
ENV CMAKE_POLICY_VERSION_MINIMUM=3.5

# Build OpenCV 4 with CUDA
RUN mkdir opencv && cd opencv && wget -O opencv.zip https://github.com/opencv/opencv/archive/refs/tags/4.13.0.zip && wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/refs/tags/4.13.0.zip && unzip opencv.zip && unzip opencv_contrib.zip
RUN cd /tmp/opencv/opencv-4.13.0 && mkdir build && cd build && cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_CXX_STANDARD=17 \
-D CMAKE_CXX_STANDARD_REQUIRED=ON \
-D CMAKE_CUDA_STANDARD=17 \
-D CMAKE_CUDA_STANDARD_REQUIRED=ON \
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
-D CUDA_ARCH_BIN="7.5 8.0 8.6 8.9 9.0 10.0 12.0" \
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
    -D OPENCV_EXTRA_MODULES_PATH=/tmp/opencv/opencv_contrib-4.13.0/modules \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D INSTALL_C_EXAMPLES=OFF \
    -D BUILD_EXAMPLES=OFF ..

WORKDIR /tmp/opencv/opencv-4.13.0/build
RUN make -j20
RUN make install

RUN apt clean -y
ADD ci/gitconfig /etc/gitconfig
WORKDIR /root
